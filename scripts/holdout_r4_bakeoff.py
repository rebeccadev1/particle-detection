"""R4 holdout bake-off: classical ≥20 µm, then residual vs cascade scores."""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import cfg_get, load_config, resolve_output_dir, with_recall_profile
from src.detection.detector import CANDIDATE_FEATURE_FIELDS, ParticleCandidate
from src.io.results_writer import write_csv, write_xlsx
from src.io.tile_loader import load_tile_image
from src.labeling.crops import find_tile_path, global_nm_to_local_px, placement_for_tile
from src.ml.features import dataframe_feature_matrix
from src.ml.infer import _positive_proba, load_artifact, predict_patch_scores
from src.pipeline.runner import run_pipeline
from src.preprocessing.corrections import apply_corrections
from src.report.report_generator import (
    encode_overlay_jpeg,
    overlay_markers,
    summary_stats,
    write_overlay_image,
)

PATTERN = r"n2_(?P<row>\d+)_(?P<col>\d+)\.tiff?"
INPUT_DIR = "R4 14-09"
OUTPUT_NAME = "R4 14-09 holdout ML off 20um"
RESIDUAL_PATH = Path("models/particle_clf_v4_residual.joblib")
CASCADE_PATH = Path("models/particle_clf_v3.joblib")
THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


def _holdout_config() -> dict:
    cfg = with_recall_profile(deepcopy(load_config("config.yaml")))
    cfg["input_dir"] = INPUT_DIR
    cfg["output_dir"] = OUTPUT_NAME
    cfg["filename_pattern"] = PATTERN
    cfg["run"] = ""
    cfg["magnification"] = ""
    cfg.setdefault("ml", {})["enabled"] = False
    cfg.setdefault("detection", {})["recall_mode"] = True
    cfg["detection"]["min_size_nm"] = 20000.0
    return cfg


def run_classical(cfg: dict) -> Path:
    out = resolve_output_dir(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output -> {out}", flush=True)
    print(
        f"min_size={cfg['detection']['min_size_nm']/1000:.0f} µm  "
        f"pixel={cfg['pixel_size_nm']/1000:.3f} µm/px  "
        f"recall_mode={cfg['detection']['recall_mode']}",
        flush=True,
    )

    def progress(done: int, total: int, msg: str = "") -> None:
        if done == 0 or done == total or done % 5 == 0:
            print(f"  [{done}/{total}] {msg}", flush=True)

    t0 = time.time()
    particles, mosaic = run_pipeline(cfg, progress_cb=progress)
    print(f"particles={len(particles)} in {time.time() - t0:.0f}s", flush=True)
    write_csv(particles, out / "particles.csv")
    write_xlsx(particles, out / "particles.xlsx")
    print("stats", summary_stats(particles), flush=True)
    try:
        write_overlay_image(mosaic, particles, cfg, out / "mosaic_overlay.jpg")
        print("wrote mosaic_overlay.jpg", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"mosaic failed: {exc}", flush=True)
        try:
            rgb = overlay_markers(mosaic, particles, cfg)
            (out / "mosaic_overlay.jpg").write_bytes(encode_overlay_jpeg(rgb))
            print("wrote mosaic_overlay.jpg (fallback)", flush=True)
        except Exception as exc2:  # noqa: BLE001
            print(f"mosaic fallback failed: {exc2}", flush=True)
    return out


def score_models(cfg: dict, out: Path) -> pd.DataFrame:
    """Score the exact classical CSV rows with residual + cascade."""
    t0 = time.time()
    df = pd.read_csv(out / "particles.csv")
    print(f"Loaded {len(df)} proposals", flush=True)
    sizes = df["size"].astype(float) / 1000.0
    print(
        f"size µm: min={sizes.min():.2f} median={sizes.median():.2f} "
        f"max={sizes.max():.2f}  below20={(sizes < 20).sum()}",
        flush=True,
    )

    residual = load_artifact(RESIDUAL_PATH)
    cascade = load_artifact(CASCADE_PATH)
    print(
        f"residual OOF thr={residual.get('threshold')} "
        f"P={residual.get('metrics', {}).get('precision')} "
        f"R={residual.get('metrics', {}).get('recall')}",
        flush=True,
    )
    print(
        f"cascade OOF thr={cascade.get('threshold')} "
        f"band=[{cascade.get('band_low')},{cascade.get('band_high')}] "
        f"P={cascade.get('metrics', {}).get('precision')} "
        f"R={cascade.get('metrics', {}).get('recall')}",
        flush=True,
    )

    df["score_residual"] = _positive_proba(
        residual["model"], dataframe_feature_matrix(df)
    )
    print(f"residual scored in {time.time() - t0:.1f}s", flush=True)

    pixel_size = float(cfg_get(cfg, "pixel_size_nm", 960.0))
    cascade_scores = np.full(len(df), np.nan, dtype=np.float64)
    origin_cache: dict[str, tuple[int, int]] = {}
    groups = list(df.groupby(df["source_tile"].astype(str), sort=False))
    print(f"Scoring cascade on {len(groups)} tiles…", flush=True)

    for i, (tile_name, sub) in enumerate(groups, 1):
        tile_path = find_tile_path(tile_name, input_dir=cfg.get("input_dir"))
        corrected = apply_corrections(load_tile_image(tile_path), cfg)
        placement = placement_for_tile(tile_path, cfg, origin_cache=origin_cache)
        cands: list[ParticleCandidate] = []
        row_indices: list[int] = []
        for row_i, row in sub.iterrows():
            x_local, y_local = global_nm_to_local_px(
                float(row["x_global"]),
                float(row["y_global"]),
                placement,
                pixel_size,
            )
            size_px = float(row["size"]) / pixel_size
            kwargs = {
                name: float(row[name]) if name in row.index else 0.0
                for name in CANDIDATE_FEATURE_FIELDS
            }
            cands.append(
                ParticleCandidate(
                    y_local=float(y_local),
                    x_local=float(x_local),
                    size=size_px,
                    confidence=float(row["confidence"]),
                    **kwargs,
                )
            )
            row_indices.append(int(row_i))
        scores = predict_patch_scores(
            cands, cascade, cfg, corrected, source_tile=str(tile_name)
        )
        for row_i, score in zip(row_indices, scores):
            cascade_scores[row_i] = float(score)
        if i % 10 == 0 or i == len(groups):
            print(f"  [{i}/{len(groups)}] {tile_name} n={len(cands)}", flush=True)

    df["score_cascade"] = cascade_scores
    scored_path = out / "proposals_scored.csv"
    df.to_csv(scored_path, index=False)
    print(
        f"Wrote {scored_path} nan={int(np.isnan(cascade_scores).sum())} "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )
    return df


def write_sweep(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    n = len(df)
    for name, col in (("residual", "score_residual"), ("cascade", "score_cascade")):
        scores = df[col].to_numpy(dtype=np.float64)
        valid = np.isfinite(scores)
        for thr in THRESHOLDS:
            kept = int(((scores >= thr) & valid).sum())
            rows.append(
                {
                    "model": name,
                    "threshold": thr,
                    "kept": kept,
                    "scored": int(valid.sum()),
                    "total": n,
                }
            )
    sweep = pd.DataFrame(rows)
    path = out / "threshold_sweep_counts.csv"
    sweep.to_csv(path, index=False)
    print("\n=== Kept counts (need GT for TP/FP/FN) ===", flush=True)
    print(sweep.pivot(index="threshold", columns="model", values="kept").to_string())
    print(f"Wrote {path}", flush=True)
    return sweep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-classical",
        action="store_true",
        help="Reuse existing particles.csv under the 20um output folder.",
    )
    args = parser.parse_args()
    cfg = _holdout_config()
    out = resolve_output_dir(cfg["output_dir"])
    if not args.skip_classical:
        out = run_classical(cfg)
    elif not (out / "particles.csv").is_file():
        raise SystemExit(
            f"Missing {out / 'particles.csv'}; run without --skip-classical"
        )
    df = score_models(cfg, out)
    write_sweep(df, out)
    print(f"\nDONE -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
