"""Tests for the optional sklearn post-filter. Does not use circled crop JPEGs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.detection.detector import CANDIDATE_FEATURE_FIELDS, ParticleCandidate
from src.ml.features import FEATURE_COLUMNS, dataframe_feature_matrix
from src.ml.infer import apply_ml_filter, load_artifact, next_versioned_model_path
from src.ml.train import save_artifact, train_classifier


def _feature_table(n_pos: int = 12, n_neg: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for i in range(n_pos):
        rows.append(
            {
                "label": "particle",
                "source_tile": f"T{i % 4}.tif",
                "size": 30_000.0 + rng.normal(0, 500),
                "confidence": 0.95 + 0.04 * rng.random(),
                "circularity": 0.75 + 0.15 * rng.random(),
                "support_over_area": 1.1 + 0.2 * rng.random(),
                "edge_distance_px": 80.0 + rng.random() * 20,
                "tile_border_dist_px": 40.0,
                "n_neighbors_48": 0.0,
                "local_peak_snr": 6.0 + rng.random(),
                "radial_inner": 0.8,
                "radial_mid": 0.4,
                "radial_outer": 0.1,
            }
        )
    for i in range(n_neg):
        rows.append(
            {
                "label": "not_particle",
                "source_tile": f"T{i % 4}.tif",
                "size": 25_000.0 + rng.normal(0, 500),
                "confidence": 0.6 + 0.1 * rng.random(),
                "circularity": 0.2 + 0.1 * rng.random(),
                "support_over_area": 4.0 + rng.random(),
                "edge_distance_px": 4.0 + rng.random() * 4,
                "tile_border_dist_px": 5.0,
                "n_neighbors_48": 4.0 + rng.integers(0, 3),
                "local_peak_snr": 2.6 + 0.3 * rng.random(),
                "radial_inner": 0.3,
                "radial_mid": 0.4,
                "radial_outer": 0.5,
            }
        )
    return pd.DataFrame(rows)


def test_train_rejects_tables_without_features() -> None:
    df = pd.DataFrame({"size": [1.0], "confidence": [1.0]})
    with pytest.raises(ValueError, match="missing feature columns"):
        dataframe_feature_matrix(df)


def test_apply_ml_filter_is_noop_when_disabled() -> None:
    candidates = [
        ParticleCandidate(y_local=1.0, x_local=2.0, size=10.0, confidence=0.9)
    ]
    kept = apply_ml_filter(candidates, {"ml": {"enabled": False}})
    assert kept is candidates or kept == candidates


def test_next_versioned_model_path_increments(tmp_path: Path) -> None:
    assert next_versioned_model_path(tmp_path).name == "particle_clf_v1.joblib"
    (tmp_path / "particle_clf_v3.joblib").write_bytes(b"x")
    (tmp_path / "particle_clf.joblib").write_bytes(b"x")
    assert next_versioned_model_path(tmp_path).name == "particle_clf_v4.joblib"


def test_apply_ml_filter_errors_when_enabled_without_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ml.enabled"):
        apply_ml_filter(
            [],
            {
                "ml": {
                    "enabled": True,
                    "model_path": str(tmp_path / "missing.joblib"),
                }
            },
        )


def test_keep_all_threshold_is_min_positive_score() -> None:
    from src.ml.train import keep_all_threshold, hard_negative_weights

    y = np.array([1, 1, 0, 0])
    proba = np.array([0.4, 0.9, 0.3, 0.8])
    assert keep_all_threshold(y, proba) == pytest.approx(0.4)
    weights = hard_negative_weights(y, np.array([0.1, 0.9, 0.25, 0.5]), cutoff=0.20, heavy=3.0)
    np.testing.assert_allclose(weights, [1.0, 1.0, 3.0, 3.0])


def test_train_and_filter_separable_features(tmp_path: Path) -> None:
    table = _feature_table()
    artifact = train_classifier(table)
    assert artifact["n_positive"] == 12
    assert artifact["n_negative"] == 24
    assert artifact["feature_names"] == FEATURE_COLUMNS
    path = save_artifact(artifact, tmp_path / "particle_clf.joblib")
    loaded = load_artifact(path)
    assert loaded["threshold"] == artifact["threshold"]

    particle = ParticleCandidate(
        y_local=10.0,
        x_local=10.0,
        size=30.0,
        confidence=0.98,
        circularity=0.85,
        support_over_area=1.1,
        edge_distance_px=90.0,
        tile_border_dist_px=40.0,
        n_neighbors_48=0.0,
        local_peak_snr=6.5,
        radial_inner=0.8,
        radial_mid=0.4,
        radial_outer=0.1,
    )
    corner = ParticleCandidate(
        y_local=12.0,
        x_local=12.0,
        size=25.0,
        confidence=0.62,
        circularity=0.22,
        support_over_area=4.5,
        edge_distance_px=5.0,
        tile_border_dist_px=5.0,
        n_neighbors_48=5.0,
        local_peak_snr=2.7,
        radial_inner=0.3,
        radial_mid=0.4,
        radial_outer=0.5,
    )
    config = {
        "pixel_size_nm": 1000.0,
        "ml": {"enabled": True, "model_path": str(path), "threshold": 0.5},
    }
    kept = apply_ml_filter([particle, corner], config)
    assert particle in kept
    assert corner not in kept
    for name in CANDIDATE_FEATURE_FIELDS:
        assert name in table.columns
