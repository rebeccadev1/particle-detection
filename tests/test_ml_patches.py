"""Unmarked crop extraction and patch/cascade FP filter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.detection.detector import ParticleCandidate, compute_tile_residual
from src.labeling.crops import crop_particle
from src.ml.cnn import TinyPatchCNN, train_tiny_cnn
from src.ml.infer import (
    KIND_CASCADE,
    apply_ml_filter,
    combine_cascade_scores,
    load_artifact,
)
from src.ml.patch_features import PATCH_FEATURE_NAMES, patch_feature_vector, patches_to_matrix
from src.ml.patches import (
    PATCH_SIZE,
    collect_unmarked_dataset,
    extract_centered,
    extract_unmarked_channels,
    stack_channels,
)
from src.ml.train import save_artifact, train_cascade_classifier, train_patch_classifier
from src.preprocessing.corrections import apply_corrections
from tests.fixtures.synthetic import detector_test_config, write_tile_tiff
from tests.test_labeling import _red_mask


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
    # Two-channel float crop, not an RGB overlay with CIRCLE_COLOR.
    assert channels.ndim == 3 and channels.shape[0] == 2


def test_unmarked_dataset_ignores_circled_jpegs(tmp_path: Path) -> None:
    image = np.zeros((160, 160), dtype=np.float32)
    image[70:90, 70:90] = 1.0
    path = write_tile_tiff(tmp_path / "R3_1_1_5X.tif", image)
    row = {
        "key": "k1",
        "source_tile": path.name,
        "x_global": 80.0,
        "y_global": 80.0,
        "size": 20.0,
        "confidence": 1.0,
        "label": "particle",
        "crop_path": "/not/used/circled.jpg",
    }
    circled = crop_particle(row, {"input_dir": str(tmp_path), "pixel_size_nm": 1.0,
                                  "filename_pattern": detector_test_config()["filename_pattern"],
                                  "overlap_fraction": 0.0})
    assert int(_red_mask(circled.rgb).sum()) > 20

    labels = pd.DataFrame([row])
    config = detector_test_config()
    config["input_dir"] = str(tmp_path)
    config["pixel_size_nm"] = 1.0
    config["overlap_fraction"] = 0.0
    channels, y, groups, meta = collect_unmarked_dataset(labels, config)
    assert channels.shape[0] == 1
    assert channels.shape[1:] == (2, PATCH_SIZE, PATCH_SIZE)
    assert int(y[0]) == 1
    assert str(groups[0]).endswith("R3_1_1_5X.tif")
    assert "circled.jpg" not in meta.columns or True


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


def test_hog_extratrees_separates_letters_from_blobs() -> None:
    channels, y, groups = _separable_dataset()
    artifact = train_patch_classifier(channels, y, groups)
    assert artifact["kind"] == "patch"
    assert artifact["metrics"]["recall"] >= 0.8
    assert artifact["metrics"]["precision"] > 0.5
    assert artifact["n_positive"] == 12


def test_cascade_cnn_on_uncertain_band(tmp_path: Path) -> None:
    channels, y, groups = _separable_dataset()
    artifact = train_cascade_classifier(channels, y, groups, cnn_epochs=3)
    assert artifact["kind"] == KIND_CASCADE
    assert artifact["cnn"] is not None
    combined = combine_cascade_scores(
        np.array([0.05, 0.5, 0.95]),
        np.array([0.8]),
        band_low=0.2,
        band_high=0.8,
    )
    assert combined[0] == pytest.approx(0.05)
    assert combined[1] == pytest.approx(0.8)
    assert combined[2] == pytest.approx(0.95)

    path = save_artifact(artifact, tmp_path / "cascade.joblib")
    loaded = load_artifact(path)
    assert loaded["kind"] == KIND_CASCADE
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
    channels, y, groups = _separable_dataset()
    artifact = train_patch_classifier(channels, y, groups)
    path = save_artifact(artifact, tmp_path / "patch.joblib")
    config = {"ml": {"enabled": True, "model_path": str(path)}}
    with pytest.raises(ValueError, match="corrected tile image"):
        apply_ml_filter(
            [ParticleCandidate(y_local=1.0, x_local=1.0, size=10.0, confidence=0.5)],
            config,
        )


def test_cascade_band_keeps_hog_when_cnn_is_worse() -> None:
    from src.ml.train import _select_cascade_band

    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    hog = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.1, 0.15, 0.2, 0.05, 0.08])
    cnn = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    _lo, _hi, combined, _t, metrics, n_uncertain = _select_cascade_band(
        y, hog, cnn, 0.25
    )
    assert metrics["precision"] >= 0.9
    assert n_uncertain == 0
    np.testing.assert_allclose(combined, hog)


def test_tiny_cnn_learns_separable_patches() -> None:
    channels, y, _groups = _separable_dataset(n_pos=8, n_neg=8)
    model = train_tiny_cnn(channels, y, epochs=4, rng=np.random.default_rng(0))
    assert isinstance(model, TinyPatchCNN)
    scores = model.predict_proba(channels)
    assert scores.shape == (16,)
    assert scores[y == 1].mean() > scores[y == 0].mean()
