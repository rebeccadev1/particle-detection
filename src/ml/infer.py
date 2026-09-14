"""Load and apply the optional post-filter on detector proposals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import PACKAGE_ROOT, cfg_get
from src.detection.detector import ParticleCandidate, compute_tile_residual
from src.io.tile_loader import DEFAULT_FILENAME_PATTERN
from src.ml.features import FEATURE_COLUMNS, candidates_to_matrix
from src.ml.patch_features import layout_from_candidate, patches_to_matrix
from src.ml.patches import PATCH_SIZE, extract_unmarked_channels

KIND_RESIDUAL = "residual"
KIND_PATCH = "patch"
KIND_CASCADE = "cascade"

_WORKER_MODEL: dict[str, Any] = {"path": None, "artifact": None}


def resolve_model_path(path: str | Path | None) -> Path:
    raw = Path(str(path or "models/particle_clf.joblib"))
    if raw.is_absolute():
        return raw
    return (PACKAGE_ROOT / raw).resolve()


def load_artifact(path: str | Path) -> dict[str, Any]:
    import joblib

    destination = Path(path)
    artifact = joblib.load(destination)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Expected a dict with a 'model' key at {destination}")
    kind = str(artifact.get("kind") or KIND_RESIDUAL)
    if kind == KIND_RESIDUAL:
        names = tuple(artifact.get("feature_names") or FEATURE_COLUMNS)
        if names != FEATURE_COLUMNS:
            raise ValueError(
                f"Model feature names {names} do not match {FEATURE_COLUMNS}"
            )
    elif kind not in (KIND_PATCH, KIND_CASCADE):
        raise ValueError(f"Unknown ML artifact kind {kind!r}")
    elif kind == KIND_CASCADE and artifact.get("cnn") is None:
        raise ValueError("Cascade artifact is missing the CNN weights")
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


def predict_proba(
    candidates: list[ParticleCandidate],
    artifact: dict[str, Any],
    pixel_size_nm: float,
) -> np.ndarray:
    if not candidates:
        return np.empty((0,), dtype=np.float64)
    matrix = candidates_to_matrix(candidates, pixel_size_nm)
    return _positive_proba(artifact["model"], matrix)


def combine_cascade_scores(
    hog_scores: np.ndarray,
    cnn_scores: np.ndarray,
    band_low: float,
    band_high: float,
    uncertain: np.ndarray | None = None,
) -> np.ndarray:
    """Use CNN scores on the uncertain HOG band; otherwise keep HOG."""
    final = np.asarray(hog_scores, dtype=np.float64).copy()
    if uncertain is None:
        mask = (final >= float(band_low)) & (final < float(band_high))
    else:
        mask = np.asarray(uncertain, dtype=bool)
    if mask.any():
        final[mask] = np.asarray(cnn_scores, dtype=np.float64).reshape(-1)
    return final


def _candidate_patches(
    candidates: list[ParticleCandidate],
    image: np.ndarray,
    config: dict[str, Any],
    patch_size: int,
) -> np.ndarray:
    _array, residual = compute_tile_residual(image, config)
    corrected = np.asarray(image, dtype=np.float32)
    patches = [
        extract_unmarked_channels(
            corrected, residual, float(cand.x_local), float(cand.y_local), size=patch_size
        )
        for cand in candidates
    ]
    return np.stack(patches, axis=0)


def predict_patch_scores(
    candidates: list[ParticleCandidate],
    artifact: dict[str, Any],
    config: dict[str, Any],
    image: np.ndarray,
    source_tile: str = "",
) -> np.ndarray:
    """HOG ExtraTrees scores, optionally replaced by CNN on the uncertain band."""
    if not candidates:
        return np.empty((0,), dtype=np.float64)
    patch_size = int(artifact.get("patch_size") or PATCH_SIZE)
    channels = _candidate_patches(candidates, image, config, patch_size)
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    pattern = str(cfg_get(config, "filename_pattern", DEFAULT_FILENAME_PATTERN))
    layouts = [
        layout_from_candidate(cand, pixel_size, source_tile, pattern)
        for cand in candidates
    ]
    matrix = patches_to_matrix(channels, layouts)
    hog_scores = _positive_proba(artifact["model"], matrix)
    kind = str(artifact.get("kind") or KIND_PATCH)
    if kind != KIND_CASCADE:
        return hog_scores
    band_low = artifact.get("band_low")
    band_high = artifact.get("band_high")
    if band_low is None:
        band_low = cfg_get(config, "ml.band_low", 0.20)
    if band_high is None:
        band_high = cfg_get(config, "ml.band_high", 0.80)
    band_low = float(band_low)
    band_high = float(band_high)
    uncertain = (hog_scores >= band_low) & (hog_scores < band_high)
    cnn = artifact["cnn"]
    cnn_subset = (
        cnn.predict_proba(channels[uncertain])
        if uncertain.any()
        else np.empty((0,), dtype=np.float64)
    )
    return combine_cascade_scores(
        hog_scores, cnn_subset, band_low, band_high, uncertain=uncertain
    )


def apply_ml_filter(
    candidates: list[ParticleCandidate],
    config: dict[str, Any],
    image: np.ndarray | None = None,
    source_tile: str = "",
) -> list[ParticleCandidate]:
    """Drop proposals whose P(particle) is below the configured threshold.

    No-op when ``ml.enabled`` is false. Raises if enabled but the model file
    is missing so a silent classical run cannot be mistaken for a filter.
    Patch and cascade artifacts need the corrected tile ``image``.
    """
    if not bool(cfg_get(config, "ml.enabled", False)):
        return candidates
    path = resolve_model_path(cfg_get(config, "ml.model_path", "models/particle_clf.joblib"))
    if not path.is_file():
        raise FileNotFoundError(
            f"ml.enabled is true but the classifier was not found at {path}. "
            "Train with `python -m src.ml.train` or turn ml.enabled off."
        )
    artifact = cached_artifact(path)
    threshold = cfg_get(config, "ml.threshold", None)
    if threshold is None:
        threshold = artifact.get("threshold", 0.35)
    threshold = float(threshold)
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    kind = str(artifact.get("kind") or KIND_RESIDUAL)
    if kind in (KIND_PATCH, KIND_CASCADE):
        if image is None:
            raise ValueError(
                "Patch/cascade ML filter needs the corrected tile image. "
                "Pass image= from the pipeline worker."
            )
        scores = predict_patch_scores(
            candidates, artifact, config, image, source_tile=source_tile
        )
    else:
        scores = predict_proba(candidates, artifact, pixel_size)
    return [cand for cand, score in zip(candidates, scores) if score >= threshold]
