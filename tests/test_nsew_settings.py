"""NSEW standard-values overlay used by the sidebar checkbox."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from src.config import (
    PACKAGE_ROOT,
    apply_nsew_settings,
    deep_update,
    load_nsew_overlay,
    save_nsew_overlay,
)


def _load_sweep():
    path = PACKAGE_ROOT / "scripts" / "nsew_param_sweep.py"
    spec = importlib.util.spec_from_file_location("nsew_param_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    save_nsew_overlay(
        {"pixel_size_nm": 3500.0, "ml": {"threshold": 0.55, "enabled": True}},
        path,
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


def test_apply_knob_writes_nsew_preprocess_and_detect() -> None:
    sweep = _load_sweep()
    cfg = sweep.nsew_base_config()
    cfg = sweep.apply_knob(cfg, "denoise_sigma", "off")
    cfg = sweep.apply_knob(cfg, "blob_threshold", 0.20)
    cfg = sweep.apply_knob(cfg, "method", "tophat")
    cfg = sweep.apply_knob(cfg, "flatten_sigma", "off")
    cfg = sweep.apply_knob(cfg, "contrast", "3_97")
    cfg = sweep.apply_knob(cfg, "tophat_radius", 35)
    assert cfg["preprocessing"]["denoise"] is False
    assert cfg["preprocessing"]["flatten_illumination"] is False
    assert cfg["preprocessing"]["contrast_percentiles"] == [3.0, 97.0]
    assert cfg["detection"]["blob_threshold"] == 0.20
    assert cfg["detection"]["method"] == "tophat"
    assert cfg["detection"]["tophat_radius"] == 35
    assert cfg["pixel_size_nm"] == 3500.0
    overlay = sweep.overlay_from_config(cfg)
    assert overlay["ml"]["enabled"] is True
    assert overlay["detection"]["tophat_radius"] == 35
    assert "input_dir" not in overlay


def test_score_table_matches_groundup_v3_particles_only_last_run() -> None:
    sweep = _load_sweep()
    from src.config import WORKSPACE_ROOT, load_config
    from src.labeling.queue import load_particles_csv
    from src.labeling.store import LabelStore

    csv_path = WORKSPACE_ROOT / "Outputs" / "Groundup v3 Particles only" / "particles.csv"
    if not csv_path.is_file():
        return
    cfg = load_config(PACKAGE_ROOT / "config.yaml")
    cfg["pixel_size_nm"] = 3500.0
    cfg["detection"]["min_size_nm"] = 20000.0
    labels = LabelStore(PACKAGE_ROOT / "labels").load()
    po_labels, po_scene = sweep._scene_labels(
        labels, sweep.PO_INPUT, list(sweep.PO_TILES)
    )
    table = load_particles_csv(csv_path)
    two = sweep.score_table(
        table, po_labels, ["particles_only.png"], po_scene, cfg
    )
    three = sweep.score_table(
        table, po_labels, ["particles_only_3of4.png"], po_scene, cfg
    )
    four = sweep.score_table(
        table, po_labels, ["particles_only_4of4.png"], po_scene, cfg
    )
    assert int(two["n_detected"]) == 256
    assert int(two["n_real"]) == 88
    assert int(two["n_undetected"]) == 41
    assert int(three["n_detected"]) == 1
    assert int(four["n_detected"]) == 423
    assert round(float(two["unhappiness"]), 2) == 43.93


def test_nsew_config_file_has_sidebar_keys() -> None:
    overlay = load_nsew_overlay()
    assert overlay["pixel_size_nm"] == 3500.0
    assert overlay["ml"]["enabled"] is True
    assert "threshold" in overlay["ml"]
    assert "min_confidence" in overlay["detection"]
    merged = apply_nsew_settings(
        {"pixel_size_nm": 960.0, "ml": {"threshold": 0.0165}, "detection": {"min_confidence": 0.5}}
    )
    assert merged["pixel_size_nm"] == 3500.0
    assert merged["ml"]["enabled"] is True
    assert merged["ml"]["threshold"] == overlay["ml"]["threshold"]


def test_app_sidebar_has_apply_nsew_settings_checkbox() -> None:
    text = (PACKAGE_ROOT / "app.py").read_text(encoding="utf-8")
    assert "Apply NSEW settings" in text
    assert "ui_apply_nsew_settings" in text
    assert "apply_nsew_settings" in text


def test_trial_rank_prefers_high_nsew_recall_then_low_unhappiness() -> None:
    sweep = _load_sweep()
    weak = {
        "nsew_recall": 0.70,
        "nsew_unhappiness": 10.0,
        "mean_unhappiness": 10.0,
        "nsew_precision": 0.9,
        "nsew_hits": 100,
    }
    strong = {
        "nsew_recall": 0.96,
        "nsew_unhappiness": 20.0,
        "mean_unhappiness": 40.0,
        "nsew_precision": 0.4,
        "nsew_hits": 200,
    }
    better = {
        "nsew_recall": 0.96,
        "nsew_unhappiness": 15.0,
        "mean_unhappiness": 30.0,
        "nsew_precision": 0.5,
        "nsew_hits": 180,
    }
    assert sweep._trial_rank(strong) > sweep._trial_rank(weak)
    assert sweep._trial_rank(better) > sweep._trial_rank(strong)


def test_trial_rank_po2_minimizes_unhappiness() -> None:
    sweep = _load_sweep()
    worse = {
        "po2_unhappiness": 44.0,
        "po2_precision": 0.34,
        "po2_recall": 0.68,
        "po2_hits": 256,
    }
    better = {
        "po2_unhappiness": 30.0,
        "po2_precision": 0.56,
        "po2_recall": 0.55,
        "po2_hits": 122,
    }
    assert sweep._trial_rank_po2(better) > sweep._trial_rank_po2(worse)
    same_u = {**better, "ml_threshold": 0.55}
    same_u_low_ml = {**better, "ml_threshold": 0.0}
    assert sweep._trial_rank_po2(same_u) > sweep._trial_rank_po2(same_u_low_ml)


def test_po2_search_includes_sigma_cliff_and_local_combos() -> None:
    sweep = _load_sweep()
    sigma_vals = dict(sweep.PO2_ONE_FACTOR)["blob_min_sigma"]
    assert 8.0 in sigma_vals
    assert 8.5 in sigma_vals
    assert 9.5 in sigma_vals
    assert 10.0 in sigma_vals
    assert any(c.get("blob_min_sigma") == 9.0 and c.get("min_circularity") == 0.10 for c in sweep.PO2_SEED_COMBOS)


def test_combine_po_scored_empty() -> None:
    sweep = _load_sweep()
    cfg = sweep.nsew_base_config()
    table = sweep._combine_po_scored([], cfg, 0.5)
    assert table is not None
    assert table.empty or "source_tile" in table.columns
