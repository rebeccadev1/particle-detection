"""Fit particle_clf_v6 on R3 + R6 + R7 rows 4–11.

R7 rows 12–14 are scored but not fitted. The shipped threshold minimizes
unhappiness on that holdout. R4 stays out of the fit. Does not replace
particle_clf_v5.joblib.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.train import (  # noqa: E402
    _layouts_for_meta,
    collect_unmarked_dataset,
    default_train_config,
    hard_negative_weights,
    save_artifact,
    train_patch_classifier,
    v4_scores_for_layouts,
)

LABELS = ROOT / "labels" / "labels.csv"
MANIFEST = ROOT.parent / "Outputs" / "v6_train_keys.csv"
MODEL = ROOT / "models" / "particle_clf_v6.joblib"
V4_MODEL = ROOT / "models" / "particle_clf_v4_residual.joblib"
V5_MODEL = ROOT / "models" / "particle_clf_v5.joblib"
R3_DIR = ROOT.parent / "Inputs" / "R3 04-08"
R6_DIR = ROOT.parent / "Inputs" / "R6 16-09"
R7_DIR = ROOT.parent / "Inputs" / "R7 17 and 18-09"
N2_PATTERN = r"n2_(?P<row>\d+)_(?P<col>\d+)\.tiff?"
N2_V2_PATTERN = r"n2_v2_(?P<row>\d+)_(?P<col>\d+)\.tiff?"
R7_ROW = re.compile(r"n2_v2_(\d+)_\d+\.tiff?$", re.IGNORECASE)
MIN_SIZE_NM = 20_000.0
R3_FP_WEIGHT = 0.4
R7_TRAIN_ROWS = range(4, 12)
R7_HOLDOUT_ROWS = range(12, 15)


def _eligible(labels: pd.DataFrame) -> pd.DataFrame:
    keep = labels.copy()
    keep["key"] = keep["key"].astype(str)
    keep["source_tile"] = keep["source_tile"].astype(str)
    jpeg = keep["source_tile"].str.lower().str.endswith((".jpg", ".jpeg"))
    labeled = keep["label"].astype(str).isin(("particle", "not_particle"))
    big = keep["size"].astype(float) >= MIN_SIZE_NM
    return keep.loc[~jpeg & labeled & big].copy()


def _r3_rows(labels: pd.DataFrame) -> pd.DataFrame:
    tile = labels["source_tile"].astype(str)
    mask = tile.str.match(r"(?:.*[/\\])?R3_\d+_\d+_5X\.tiff?$")
    return labels.loc[mask].copy()


def _tiles_in(folder: Path) -> set[str]:
    names: set[str] = set()
    for path in folder.iterdir():
        if path.suffix.lower() in {".tif", ".tiff"}:
            names.add(path.name)
    return names


def _r6_rows(labels: pd.DataFrame, tiles: set[str]) -> pd.DataFrame:
    src = labels["source_csv"].astype(str)
    tile = labels["source_tile"].astype(str)
    on_disk = tile.isin(tiles)
    from_run = src.str.endswith("R6 ML off 20 micron/particles.csv")
    inspect = (src == "tile_inspect") & on_disk
    n2 = tile.str.match(r"(?:.*[/\\])?n2_\d+_\d+\.tiff?$")
    return labels.loc[(from_run | inspect) & n2 & on_disk].copy()


def _r7_rows(labels: pd.DataFrame, tiles: set[str], rows: range) -> pd.DataFrame:
    src = labels["source_csv"].astype(str)
    tile = labels["source_tile"].astype(str)
    row = tile.map(_r7_row)
    wanted = row.isin(set(rows))
    on_disk = tile.isin(tiles)
    from_run = src.str.endswith("R7 ML off 20 micron/particles.csv")
    inspect = src == "tile_inspect"
    return labels.loc[wanted & on_disk & (from_run | inspect)].copy()


def _r7_row(name: str) -> int | None:
    match = R7_ROW.search(Path(str(name)).name)
    if match is None:
        return None
    return int(match.group(1))


def _extract(frame: pd.DataFrame, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    channels, y, groups, meta = collect_unmarked_dataset(frame, config)
    layouts = _layouts_for_meta(meta, frame, config)
    return channels, y, groups, meta, layouts


def _describe(name: str, frame: pd.DataFrame) -> None:
    particles = int((frame["label"] == "particle").sum())
    print(
        f"{name} {len(frame)} rows / {frame['source_tile'].nunique()} tiles "
        f"(particle {particles})",
        flush=True,
    )


def main() -> int:
    v5_before = V5_MODEL.stat().st_mtime_ns
    labels = _eligible(pd.read_csv(LABELS))
    r6_tiles = _tiles_in(R6_DIR)
    r7_tiles = _tiles_in(R7_DIR)
    r3 = _r3_rows(labels).drop_duplicates("key", keep="last")
    r6 = _r6_rows(labels, r6_tiles).drop_duplicates("key", keep="last")
    r7_train = _r7_rows(labels, r7_tiles, R7_TRAIN_ROWS).drop_duplicates("key", keep="last")
    r7_hold = _r7_rows(labels, r7_tiles, R7_HOLDOUT_ROWS).drop_duplicates("key", keep="last")
    if r3.empty or r6.empty or r7_train.empty or r7_hold.empty:
        raise SystemExit(
            f"Empty split: R3 {len(r3)} R6 {len(r6)} "
            f"R7 train {len(r7_train)} R7 hold {len(r7_hold)}"
        )
    _describe("R3", r3)
    _describe("R6", r6)
    _describe("R7 train rows 4-11", r7_train)
    _describe("R7 holdout rows 12-14", r7_hold)
    base = default_train_config(ROOT / "config.yaml")
    r3_cfg = dict(base)
    r3_cfg["input_dir"] = str(R3_DIR)
    r6_cfg = dict(base)
    r6_cfg["input_dir"] = str(R6_DIR)
    r6_cfg["filename_pattern"] = N2_PATTERN
    r7_cfg = dict(base)
    r7_cfg["input_dir"] = str(R7_DIR)
    r7_cfg["filename_pattern"] = N2_V2_PATTERN
    print("Extracting R3 patches…", flush=True)
    c3, y3, g3, m3, l3 = _extract(r3, r3_cfg)
    print("Extracting R6 patches…", flush=True)
    c6, y6, g6, m6, l6 = _extract(r6, r6_cfg)
    print("Extracting R7 train patches…", flush=True)
    c7, y7, g7, m7, l7 = _extract(r7_train, r7_cfg)
    print("Extracting R7 holdout patches…", flush=True)
    ch, yh, gh, mh, lh = _extract(r7_hold, r7_cfg)
    channels = np.concatenate([c3, c6, c7, ch], axis=0)
    y = np.concatenate([y3, y6, y7, yh])
    groups = np.concatenate([g3, g6, g7, gh])
    layouts = np.vstack([l3, l6, l7, lh])
    meta = pd.concat([m3, m6, m7, mh], ignore_index=True)
    holdout_mask = np.zeros(len(y), dtype=bool)
    holdout_mask[-len(yh) :] = True
    manifest = pd.concat(
        [
            r3.assign(split="train"),
            r6.assign(split="train"),
            r7_train.assign(split="train"),
            r7_hold.assign(split="holdout_tile"),
        ],
        ignore_index=True,
    )
    manifest = manifest.loc[manifest["key"].isin(set(meta["key"].astype(str)))]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest[["key", "label", "source_tile", "size", "split"]].to_csv(MANIFEST, index=False)
    print(f"Wrote {MANIFEST} ({len(manifest)} keys).", flush=True)

    v4_scores = v4_scores_for_layouts(layouts, V4_MODEL)
    weights = hard_negative_weights(y, v4_scores, cutoff=0.20, heavy=3.0)
    r3_neg = np.array([str(tile).find("R3_") >= 0 for tile in groups]) & (y == 0)
    weights[r3_neg] *= R3_FP_WEIGHT
    print(
        f"R3 not-particle weight {R3_FP_WEIGHT} on {int(r3_neg.sum())} rows. "
        f"v4 hard negatives {int(((y == 0) & (v4_scores >= 0.20) & ~holdout_mask).sum())}.",
        flush=True,
    )
    print("Training HOG ExtraTrees…", flush=True)
    artifact = train_patch_classifier(
        channels,
        y,
        groups,
        layouts,
        sample_weight=weights,
        holdout_mask=holdout_mask,
    )
    artifact["train_days"] = "R3 + R6 + R7 rows 4-11"
    artifact["held_out"] = "R7 rows 12-14; R4 not in the fit"
    saved = save_artifact(artifact, MODEL)
    metrics = artifact["metrics"]
    print(
        f"Wrote {saved.name}  n={artifact['n_samples']} "
        f"(particle {artifact['n_positive']} / not {artifact['n_negative']})  "
        f"threshold={artifact['threshold']:.2f}  "
        f"OOF P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"U={metrics['unhappiness']:.2f} F1={metrics['f1']:.3f}",
        flush=True,
    )
    print(
        f"  keep_all={artifact['threshold_keep_all']:.3f}  "
        f"max_f1={artifact['threshold_f1']:.3f}",
        flush=True,
    )
    if V5_MODEL.stat().st_mtime_ns != v5_before:
        raise SystemExit("particle_clf_v5.joblib was modified")
    if not V5_MODEL.is_file() or V5_MODEL.stat().st_size < 1000:
        raise SystemExit("particle_clf_v5.joblib is missing")
    print(f"Left {V5_MODEL.name} unchanged.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
