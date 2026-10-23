"""Unmarked crop extraction and patch FP filter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.analyse.detection.detector import ParticleCandidate, compute_tile_residual
from src.analyse.ml.infer import apply_ml_filter
from src.analyse.ml.patch_features import (
    PATCH_FEATURE_NAMES,
    patch_feature_vector,
    patches_to_matrix,
)
from src.analyse.ml.patches import (
    PATCH_SIZE,
    extract_centered,
    extract_unmarked_channels,
    stack_channels,
)
from src.analyse.preprocessing.corrections import apply_corrections
from tests.fixtures.synthetic import detector_test_config


def _blob_patch(rng: np.random.Generator, size: int = PATCH_SIZE) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    cy = size / 2 + rng.normal(0, 1.5)
    cx = size / 2 + rng.normal(0, 1.5)
    sigma = 7.0 + rng.random() * 2.0
    blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))
    residual = blob + 0.02 * rng.random((size, size))
    raw = 0.2 + 0.8 * blob
    return stack_channels(raw, residual)


def _letter_patch(rng: np.random.Generator, size: int = PATCH_SIZE) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.float32)
    cx = size // 2 + int(rng.integers(-4, 5))
    cy = size // 2 + int(rng.integers(-4, 5))
    image[cy - 22 : cy + 22, cx - 4 : cx + 4] = 1.0
    image[cy - 4 : cy + 4, cx - 18 : cx + 18] = 1.0
    image[cy + 10 : cy + 14, cx - 16 : cx + 16] = 1.0
    noise = 0.03 * rng.random((size, size))
    return stack_channels(image + noise, image * 0.7 + noise)


def _separable_dataset(n_pos: int = 12, n_neg: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    patches = []
    labels = []
    groups = []
    for i in range(n_pos):
        patches.append(_blob_patch(rng))
        labels.append(1)
        groups.append(f"R3_1_{i % 4}_5X.tif")
    for i in range(n_neg):
        patches.append(_letter_patch(rng))
        labels.append(0)
        groups.append(f"R3_2_{i % 4}_5X.tif")
    return np.stack(patches), np.asarray(labels), np.asarray(groups)


def _save_patch_model(path: Path) -> Path:
    import joblib
    from sklearn.ensemble import ExtraTreesClassifier

    channels, y, _groups = _separable_dataset()
    model = ExtraTreesClassifier(n_estimators=50, random_state=0).fit(patches_to_matrix(channels), y)
    joblib.dump(
        {
            "kind": "patch",
            "model": model,
            "feature_names": PATCH_FEATURE_NAMES,
            "patch_size": PATCH_SIZE,
            "threshold": 0.5,
        },
        path,
    )
    return path


def test_extract_centered_is_fixed_size_and_padded() -> None:
    image = np.arange(20, dtype=np.float32).reshape(4, 5)
    patch = extract_centered(image, x_local=0.0, y_local=0.0, size=8)
    assert patch.shape == (8, 8)
    assert patch.dtype == np.float32


def test_unmarked_channels_have_no_red_marker() -> None:
    raw = np.zeros((96, 96), dtype=np.float32)
    raw[40:55, 40:55] = 0.9
    residual = raw * 0.8
    channels = extract_unmarked_channels(raw, residual, 48.0, 48.0, size=96)
    assert channels.shape == (2, 96, 96)
    assert channels.max() <= 1.0 + 1e-5
    assert channels.ndim == 3 and channels.shape[0] == 2


def test_patch_feature_length_matches_names() -> None:
    rng = np.random.default_rng(0)
    vector = patch_feature_vector(_blob_patch(rng))
    assert vector.shape == (len(PATCH_FEATURE_NAMES),)


def test_compute_tile_residual_matches_blob_maps() -> None:
    from tests.fixtures.synthetic import make_structured_tile

    image = make_structured_tile(particles=[(40.0, 36.0, 3.8)])
    config = detector_test_config()
    corrected = apply_corrections(image, config)
    array, residual = compute_tile_residual(corrected, config)
    assert array.shape == residual.shape
    assert float(residual.max()) > 0


def test_patch_filter_uses_threshold(tmp_path: Path) -> None:
    path = _save_patch_model(tmp_path / "patch.joblib")
    image = np.zeros((128, 128), dtype=np.float32)
    image[54:74, 54:74] = 1.0
    blob = ParticleCandidate(y_local=64.0, x_local=64.0, size=20.0, confidence=0.9)
    config = detector_test_config()
    config["ml"] = {"enabled": True, "model_path": str(path), "threshold": 0.0}
    kept = apply_ml_filter([blob], config, image=image, source_tile="R3_1_1_5X.tif")
    assert blob in kept
    config["ml"]["threshold"] = 1.1
    assert apply_ml_filter([blob], config, image=image, source_tile="R3_1_1_5X.tif") == []


def test_patch_filter_requires_image(tmp_path: Path) -> None:
    path = _save_patch_model(tmp_path / "patch.joblib")
    config = {"ml": {"enabled": True, "model_path": str(path)}}
    with pytest.raises(ValueError, match="corrected tile image"):
        apply_ml_filter(
            [ParticleCandidate(y_local=1.0, x_local=1.0, size=10.0, confidence=0.5)],
            config,
        )

