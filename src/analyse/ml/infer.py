from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.overall.config import PACKAGE_ROOT, cfg_get
from src.analyse.detection.detector import ParticleCandidate
from src.analyse.ml.features import FEATURE_COLUMNS, candidates_to_matrix

KIND_RESIDUAL = "residual"

DEFAULT_MODEL_PATH = "src/analyse/models/particle_clf_v4_residual.joblib"

_WORKER_MODEL: dict[str, Any] = {"path": None, "artifact": None}


def resolve_model_path(path: str | Path | None) -> Path:
    raw = Path(str(path or DEFAULT_MODEL_PATH))
    if raw.is_absolute():
        return raw
    return (PACKAGE_ROOT / raw).resolve()


def load_artifact(path: str | Path) -> dict[str, Any]:
    import joblib

    destination = Path(path)
    try:
        artifact = joblib.load(destination)
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        if "TinyPatchCNN" in text or "src.ml.cnn" in text:
            raise ValueError(
                f"{destination.name} is a cascade model (TinyCNN). "
                "Those models are no longer supported."
            ) from exc
        raise
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Expected a dict with a 'model' key at {destination}")
    kind = str(artifact.get("kind") or KIND_RESIDUAL)
    if kind == "cascade" or artifact.get("cnn") is not None:
        raise ValueError(
            f"{destination.name} is a cascade model (TinyCNN). "
            "Those models are no longer supported."
        )
    if kind == "patch":
        raise ValueError(
            f"{destination.name} is a patch model (HOG). "
            "Those models are no longer supported."
        )
    if kind != KIND_RESIDUAL:
        raise ValueError(f"Unknown ML artifact kind {kind!r}")
    names = tuple(artifact.get("feature_names") or FEATURE_COLUMNS)
    if names != FEATURE_COLUMNS:
        raise ValueError(
            f"Model feature names {names} do not match {FEATURE_COLUMNS}"
        )
    artifact["kind"] = kind
    return artifact


def cached_artifact(path: str | Path) -> dict[str, Any]:
    resolved = str(Path(path).resolve())
    if _WORKER_MODEL.get("path") == resolved and _WORKER_MODEL.get("artifact") is not None:
        return _WORKER_MODEL["artifact"]
    artifact = load_artifact(path)
    _WORKER_MODEL["path"] = resolved
    _WORKER_MODEL["artifact"] = artifact
    return artifact


def _positive_proba(model: Any, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    proba = model.predict_proba(matrix)
    classes = list(model.classes_)
    if 1 in classes:
        return np.asarray(proba[:, classes.index(1)], dtype=np.float64)
    return np.zeros(len(matrix), dtype=np.float64)


def calibrate_scores(artifact: dict[str, Any], scores: np.ndarray) -> np.ndarray:
    
    raw = np.asarray(scores, dtype=np.float64)
    calibrator = artifact.get("calibrator")
    if calibrator is None or raw.size == 0:
        return raw
    return np.asarray(calibrator.predict(raw), dtype=np.float64)


def predict_proba(
    candidates: list[ParticleCandidate],
    artifact: dict[str, Any],
    pixel_size_nm: float,
) -> np.ndarray:
    if not candidates:
        return np.empty((0,), dtype=np.float64)
    matrix = candidates_to_matrix(candidates, pixel_size_nm)
    return calibrate_scores(artifact, _positive_proba(artifact["model"], matrix))


def apply_ml_filter(
    candidates: list[ParticleCandidate],
    config: dict[str, Any],
) -> list[ParticleCandidate]:
    
    if not bool(cfg_get(config, "ml.enabled", False)):
        return candidates
    path = resolve_model_path(cfg_get(config, "ml.model_path", DEFAULT_MODEL_PATH))
    if not path.is_file():
        raise FileNotFoundError(
            f"ml.enabled is true but the classifier was not found at {path}. "
            "Point ml.model_path at a model in src/analyse/models/ or turn ml.enabled off."
        )
    artifact = cached_artifact(path)
    threshold = cfg_get(config, "ml.threshold", None)
    if threshold is None:
        threshold = artifact.get("threshold", 0.35)
    threshold = float(threshold)
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    scores = predict_proba(candidates, artifact, pixel_size)
    return [cand for cand, score in zip(candidates, scores) if score >= threshold]
