"""Score frozen R4 winner proposals: v4 residual @ 0.20 vs v5 keep-all / precision."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from r4_param_sweep import (  # noqa: E402
    INPUT_DIR,
    PATTERN,
    load_gold,
    score_table,
)
from src.config import load_config  # noqa: E402
from src.detection.detector import CANDIDATE_FEATURE_FIELDS, ParticleCandidate  # noqa: E402
from src.io.tile_loader import load_tile_image  # noqa: E402
from src.labeling.crops import find_tile_path, global_nm_to_local_px, placement_for_tile  # noqa: E402
from src.labeling.queue import detection_key  # noqa: E402
from src.ml.features import dataframe_feature_matrix  # noqa: E402
from src.ml.infer import _positive_proba, load_artifact, predict_patch_scores  # noqa: E402
from src.preprocessing.corrections import apply_corrections  # noqa: E402

WINNER = ROOT.parent / "Outputs" / "R4 16-09 param sweep" / "particles.csv"
OUT = ROOT.parent / "Outputs" / "R4 16-09 param sweep" / "v5_bakeoff.json"
V4_PATH = ROOT / "models" / "particle_clf_v4_residual.joblib"
V5_PATH = ROOT / "models" / "particle_clf_v5.joblib"
V4_THRESHOLD = 0.20


def _r4_config() -> dict:
    cfg = load_config(ROOT / "config.yaml")
    cfg["input_dir"] = INPUT_DIR
    cfg["filename_pattern"] = PATTERN
    cfg["ml"] = dict(cfg.get("ml") or {})
    cfg["ml"]["enabled"] = False
    return cfg


def _candidate(row: pd.Series, placement, pixel: float) -> ParticleCandidate:
    x_local, y_local = global_nm_to_local_px(
        float(row["x_global"]), float(row["y_global"]), placement, pixel
    )
    kwargs = {
        name: float(row[name])
        for name in CANDIDATE_FEATURE_FIELDS
        if name in row.index and pd.notna(row[name])
    }
    return ParticleCandidate(
        y_local=y_local,
        x_local=x_local,
        size=float(row["size"]) / pixel,
        confidence=float(row["confidence"]),
        **kwargs,
    )


def v4_scores(table: pd.DataFrame, artifact: dict) -> list[float]:
    matrix = dataframe_feature_matrix(table)
    return _positive_proba(artifact["model"], matrix).tolist()


def v5_scores(table: pd.DataFrame, artifact: dict, config: dict) -> list[float]:
    pixel = float(config["pixel_size_nm"])
    origin_cache: dict[str, tuple[int, int]] = {}
    scores = [0.0] * len(table)
    grouped = table.groupby(table["source_tile"].map(lambda s: Path(str(s)).name), sort=False)
    for tile_name, group in grouped:
        path = find_tile_path(tile_name, input_dir=config["input_dir"])
        placement = placement_for_tile(path, config, origin_cache=origin_cache)
        image = apply_corrections(load_tile_image(path), config)
        cands: list[ParticleCandidate] = []
        indexes: list[int] = []
        for idx, row in group.iterrows():
            cands.append(_candidate(row, placement, pixel))
            indexes.append(int(idx))
        tile_scores = predict_patch_scores(
            cands, artifact, config, image, source_tile=tile_name
        )
        for loc, value in zip(group.index.tolist(), tile_scores):
            scores[int(loc)] = float(value)
    return scores


def metrics_for(table: pd.DataFrame, scores: list[float], threshold: float, gold, config) -> dict:
    keep = table.loc[[s >= threshold for s in scores]].copy()
    stats = score_table(keep, gold, config)
    stats["threshold"] = float(threshold)
    stats["n_scored"] = int(len(table))
    return stats


def main() -> int:
    config = _r4_config()
    gold = load_gold()
    table = pd.read_csv(WINNER)
    table["key"] = [detection_key(row) for _, row in table.iterrows()]
    table = table.reset_index(drop=True)
    v4 = load_artifact(V4_PATH)
    rows = {
        "v4_0.20": metrics_for(table, v4_scores(table, v4), V4_THRESHOLD, gold, config),
    }
    if not V5_PATH.is_file():
        raise SystemExit(f"Missing {V5_PATH}. Train v5 first.")
    v5 = load_artifact(V5_PATH)
    v5_raw = v5_scores(table, v5, config)
    keep_all = float(v5.get("threshold_keep_all", v5.get("threshold", 0.2)))
    precision_t = float(v5.get("threshold_precision", keep_all))
    rows["v5_keep_all"] = metrics_for(table, v5_raw, keep_all, gold, config)
    rows["v5_precision"] = metrics_for(table, v5_raw, precision_t, gold, config)
    payload = {"rows": rows, "v5_keep_all": keep_all, "v5_precision": precision_t}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
