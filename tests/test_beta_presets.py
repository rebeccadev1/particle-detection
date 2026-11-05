from __future__ import annotations

import pytest

from src.overall.config import (
    PACKAGE_ROOT,
    apply_beta_ml,
    apply_nsew_settings,
    beta_preset_rows,
    load_config,
    preset_for_beta,
)


def test_preset_for_beta_returns_model_and_threshold() -> None:
    presets = [
        {"beta": 1, "model_path": "models/particle_clf_v5.joblib", "threshold": "0.39"},
        {"beta": 2.0, "model_path": "models/particle_clf_v6.joblib", "threshold": 0.55},
    ]
    chosen = preset_for_beta(presets, 1.0)
    assert chosen == {
        "beta": 1.0,
        "model_path": "models/particle_clf_v5.joblib",
        "threshold": 0.39,
    }
    assert preset_for_beta(presets, 2)["model_path"] == "models/particle_clf_v6.joblib"


def test_preset_for_beta_rejects_unknown_beta() -> None:
    presets = [
        {"beta": 1.0, "model_path": "models/particle_clf_v5.joblib", "threshold": 0.39}
    ]
    with pytest.raises(ValueError, match="No ML preset for beta 0.50"):
        preset_for_beta(presets, 0.5)


def test_beta_preset_rows_require_a_complete_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        beta_preset_rows({})
    with pytest.raises(ValueError, match="missing model_path"):
        beta_preset_rows({"report": {"beta_presets": [{"beta": 1.0, "threshold": 0.39}]}})
    with pytest.raises(ValueError, match="more than once"):
        beta_preset_rows(
            {
                "report": {
                    "beta_presets": [
                        {
                            "beta": 1.0,
                            "model_path": "models/a.joblib",
                            "threshold": 0.2,
                        },
                        {
                            "beta": 1.0,
                            "model_path": "models/b.joblib",
                            "threshold": 0.4,
                        },
                    ]
                }
            }
        )


_SHIPPED_PRESETS = (
    {
        "beta": 0.5,
        "model_path": "src/analyse/models/particle_clf_v7.joblib",
        "threshold": 0.6,
    },
    {
        "beta": 1.0,
        "model_path": "src/analyse/models/particle_clf_v8.joblib",
        "threshold": 0.31,
    },
    {
        "beta": 1.5,
        "model_path": "src/analyse/models/particle_clf_v8.joblib",
        "threshold": 0.31,
    },
    {
        "beta": 2.0,
        "model_path": "src/analyse/models/particle_clf_v7.joblib",
        "threshold": 0.21,
    },
)


def test_shipped_config_default_beta_has_a_preset() -> None:
    config = load_config(PACKAGE_ROOT / "config" / "config.yaml")
    rows = beta_preset_rows(config)
    assert rows == [dict(row) for row in _SHIPPED_PRESETS]
    chosen = preset_for_beta(rows, config["report"]["f_beta"])
    assert chosen["model_path"] == "src/analyse/models/particle_clf_v8.joblib"
    assert chosen["threshold"] == pytest.approx(0.31)


def test_apply_beta_ml_leaves_configs_without_presets_alone() -> None:
    config = {"ml": {"enabled": True, "model_path": "models/kept.joblib", "threshold": 0.5}}
    apply_beta_ml(config)
    assert config["ml"]["model_path"] == "models/kept.joblib"
    assert config["ml"]["threshold"] == 0.5


def test_uniform_beta_one_selects_v8_for_the_run() -> None:
    config = load_config(PACKAGE_ROOT / "config" / "config.yaml")
    config["ml"]["model_path"] = "models/other.joblib"
    config["ml"]["threshold"] = 0.9
    apply_beta_ml(config)
    assert config["ml"]["model_path"] == "src/analyse/models/particle_clf_v8.joblib"
    assert config["ml"]["threshold"] == pytest.approx(0.31)
    assert config["ml"]["enabled"] is True


_NSEW_PRESETS = (
    {
        "beta": 0.5,
        "model_path": "src/analyse/models/particle_clf_v4_residual.joblib",
        "threshold": 0.51,
    },
    {
        "beta": 1.0,
        "model_path": "src/analyse/models/particle_clf_v8.joblib",
        "threshold": 0.31,
    },
    {
        "beta": 1.5,
        "model_path": "src/analyse/models/particle_clf_v4_residual.joblib",
        "threshold": 0.34,
    },
    {
        "beta": 2.0,
        "model_path": "src/analyse/models/particle_clf_v8.joblib",
        "threshold": 0.3,
    },
)


def test_nsew_beta_one_selects_v8_for_the_run() -> None:
    merged = apply_nsew_settings(load_config(PACKAGE_ROOT / "config" / "config.yaml"))
    assert beta_preset_rows(merged) == [dict(row) for row in _NSEW_PRESETS]
    apply_beta_ml(merged)
    assert merged["ml"]["model_path"] == "src/analyse/models/particle_clf_v8.joblib"
    assert merged["ml"]["threshold"] == pytest.approx(0.31)
