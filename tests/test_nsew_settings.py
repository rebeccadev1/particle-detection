"""NSEW standard-values overlay used by the sidebar NSEW toggle."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.overall.config import (
    PACKAGE_ROOT,
    apply_nsew_settings,
    deep_update,
    load_nsew_overlay,
)


def test_deep_update_overlays_nested_keys() -> None:
    merged = deep_update(
        {"preprocessing": {"denoise_sigma": 1.0, "flatten_sigma": 160.0}, "keep": 1},
        {"preprocessing": {"denoise_sigma": 2.0}, "ml": {"threshold": 0.4}},
    )
    assert merged["preprocessing"]["denoise_sigma"] == 2.0
    assert merged["preprocessing"]["flatten_sigma"] == 160.0
    assert merged["keep"] == 1
    assert merged["ml"]["threshold"] == 0.4


def test_apply_nsew_settings_reads_overlay_file(tmp_path: Path) -> None:
    path = tmp_path / "nsew_config.yaml"
    path.write_text(
        yaml.safe_dump({"pixel_size_nm": 3500.0, "ml": {"threshold": 0.55, "enabled": True}}),
        encoding="utf-8",
    )
    overlay = load_nsew_overlay(path)
    assert overlay["pixel_size_nm"] == 3500.0
    assert overlay["ml"]["threshold"] == 0.55
    merged = apply_nsew_settings(
        {"pixel_size_nm": 960.0, "ml": {"threshold": 0.0165, "enabled": False}},
        path=path,
    )
    assert merged["pixel_size_nm"] == 3500.0
    assert merged["ml"]["threshold"] == 0.55
    assert merged["ml"]["enabled"] is True


def test_uniform_and_nsew_list_the_same_preprocess_and_detect_keys() -> None:
    from src.overall.config import load_config

    uniform = load_config(PACKAGE_ROOT / "config" / "config.yaml")
    nsew = load_nsew_overlay()
    assert set(nsew["preprocessing"]) == set(uniform["preprocessing"])
    assert set(nsew["detection"]) == set(uniform["detection"])
    assert "min_prominence" in uniform["detection"]


def test_nsew_config_file_has_sidebar_keys() -> None:
    overlay = load_nsew_overlay()
    merged = apply_nsew_settings(
        {
            "pixel_size_nm": 960.0,
            "preprocessing": {"denoise_sigma": 9.0},
            "detection": {"min_confidence": 0.01, "tophat_radius": 1},
            "ml": {
                "enabled": False,
                "threshold": 0.23,
                "model_path": "models/particle_clf_v4_residual.joblib",
            },
            "report": {
                "f_beta": 2.0,
                "beta_presets": [
                    {
                        "beta": 1.0,
                        "model_path": "models/particle_clf_v4_residual.joblib",
                        "threshold": 0.23,
                    }
                ],
            },
        }
    )
    assert merged["pixel_size_nm"] == overlay["pixel_size_nm"]
    for section in ("preprocessing", "detection", "ml", "report"):
        for key, value in overlay[section].items():
            assert merged[section][key] == value


def test_app_sidebar_has_nsew_toggle() -> None:
    text = (PACKAGE_ROOT / "app.py").read_text(encoding="utf-8")
    assert "Settings profile" in text
    assert "Uniform" in text
    assert "ui_settings_nsew" in text
    assert "apply_nsew_settings" in text
    assert '["Loading", "Preprocessing", "Detection", "Output"]' in text
    assert '["Capture", "Analyse"]' in text
    assert "Verify" not in text
    assert "Additional tools" not in text
    assert "Training" not in text
    assert "Last run" in text
    assert "render_labeled_tiles_tab(" not in text
    assert "Summary of last run" not in text
    assert "_nsew_config(base)" in text
    assert "_apply_standard_settings" in text.split("def _on_reset_standard")[1].split("def ")[0]
    assert 'if st.session_state.get("ui_settings_nsew"):' in text
