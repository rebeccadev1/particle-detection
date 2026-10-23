"""Beta presets that pick an ML model and threshold."""

from __future__ import annotations

import pytest

from src.overall.config import beta_preset_rows, preset_for_beta


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


def test_shipped_config_default_beta_has_a_preset() -> None:
    from src.overall.config import PACKAGE_ROOT, load_config

    config = load_config(PACKAGE_ROOT / "config" / "config.yaml")
    rows = beta_preset_rows(config)
    chosen = preset_for_beta(rows, config["report"]["f_beta"])
    assert chosen["model_path"] == "src/analyse/models/particle_clf_v4_residual.joblib"
    assert chosen["threshold"] == 0.23
