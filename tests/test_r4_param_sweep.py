"""Gold-set lock for the R4 preprocess/detect sweep."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from src.config import PACKAGE_ROOT


def _load_sweep():
    path = PACKAGE_ROOT / "scripts" / "r4_param_sweep.py"
    spec = importlib.util.spec_from_file_location("r4_param_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r4_gold_is_39_tinder_plus_two_inspect() -> None:
    sweep = _load_sweep()
    gold = sweep.load_gold()
    assert len(gold) == 41
    keys = set(gold["key"].astype(str))
    assert sweep.INSPECT_KEYS[0] in keys
    assert sweep.INSPECT_KEYS[1] in keys
    inspect = gold.loc[gold["key"].astype(str).isin(sweep.INSPECT_KEYS)]
    assert (inspect["size"].astype(float) >= 20_000).all()
    tinder = gold.loc[~gold["key"].astype(str).isin(sweep.INSPECT_KEYS)]
    assert len(tinder) == 39
    assert (tinder["size"].astype(float) >= 20_000).all()


def test_apply_knob_writes_recall_overlay() -> None:
    sweep = _load_sweep()
    cfg = sweep.r4_base_config()
    cfg = sweep.apply_knob(cfg, "recall.min_confidence", 0.60)
    cfg = sweep.apply_knob(cfg, "blob_min_sigma", 7.4)
    cfg = sweep.apply_knob(cfg, "denoise_sigma", "off")
    assert cfg["detection"]["recall"]["min_confidence"] == 0.60
    assert cfg["detection"]["min_confidence"] == 0.60
    assert cfg["detection"]["blob_min_sigma"] == 7.4
    assert cfg["preprocessing"]["denoise"] is False
    assert cfg["ml"]["threshold"] == 0.20
    assert cfg["ml"]["model_path"] == sweep.V4_PATH
    assert cfg["detection"]["min_size_nm"] == 20_000.0
    assert cfg["detection"]["min_circularity"] == 0.0
    cfg = sweep.apply_knob(cfg, "min_circularity", 0.30)
    assert cfg["detection"]["min_circularity"] == 0.30
