"""Run the FN-recovery pipeline and recount the ≥20 µm ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.io.results_writer import write_csv
from src.labeling.crops import find_tile_path, global_nm_to_local_px, placement_for_tile
from src.labeling.queue import write_last_pipeline
from src.pipeline.runner import default_output_paths, run_pipeline

# Compact flakes marked in tile_inspect; all ≥20 µm. Smaller inspect marks
# (4_23 13.7, 4_5 15.1, both 5_9 18.3) are out of spec, not FN.
INSPECT_KEYS = (
    "R3_2_30_5X_81544816_2394400",
    "R3_2_30_5X_80834576_2903920",
    "R3_5_16_5X_39563456_8950736",
    "R3_5_16_5X_40212423_8699958",
)
NEAR_PX = 20.0


def _progress(current: int, total: int, name: str) -> None:
    print(f"  {current}/{total} {name}", flush=True)


def _ground_truth(labels: pd.DataFrame) -> pd.DataFrame:
    off = labels[
        labels["source_csv"]
        .astype(str)
        .str.contains(r"less dirty \(20 um, ML off\)", regex=True)
    ]
    gt7 = off.loc[off["label"] == "particle"].copy()
    gt4 = labels.loc[labels["key"].astype(str).isin(INSPECT_KEYS)].copy()
    return pd.concat([gt7, gt4], ignore_index=True)


def _localize(
    row: pd.Series,
    config: dict,
    origin_cache: dict[str, tuple[int, int]],
) -> tuple[str, float, float]:
    tile = Path(str(row["source_tile"])).name
    path = find_tile_path(tile, input_dir=config["input_dir"])
    placement = placement_for_tile(path, config, origin_cache=origin_cache)
    pixel_size = float(config.get("pixel_size_nm", 960.0))
    x_local, y_local = global_nm_to_local_px(
        float(row["x_global"]), float(row["y_global"]), placement, pixel_size
    )
    return tile, x_local, y_local


def recount(
    table: pd.DataFrame,
    gt: pd.DataFrame,
    config: dict,
    min_size_nm: float,
) -> tuple[int, int, int, int, list[tuple]]:
    origin_cache: dict[str, tuple[int, int]] = {}
    det_local: dict[str, list[tuple[float, float]]] = {}
    for _, row in table.iterrows():
        if float(row["size"]) < min_size_nm:
            continue
        tile, x_local, y_local = _localize(row, config, origin_cache)
        det_local.setdefault(tile, []).append((x_local, y_local))
    counted = sum(len(pts) for pts in det_local.values())
    real = 0
    missed: list[tuple] = []
    for _, row in gt.iterrows():
        tile, x_local, y_local = _localize(row, config, origin_cache)
        pts = det_local.get(tile, [])
        if not pts:
            missed.append((tile, "no dets", str(row["key"])))
            continue
        dist = min(((px - x_local) ** 2 + (py - y_local) ** 2) ** 0.5 for px, py in pts)
        if dist <= NEAR_PX:
            real += 1
        else:
            missed.append(
                (tile, round(dist, 1), float(row["size"]) / 1000.0, str(row["key"]))
            )
    return counted, real, counted - real, len(gt) - real, missed


def _print_row(name: str, counted: int, real: int, fake: int, missed_n: int, n_gt: int) -> None:
    precision = real / counted if counted else 0.0
    recall = real / n_gt if n_gt else 0.0
    fn_rate = missed_n / n_gt if n_gt else 0.0
    fp_rate = fake / counted if counted else 0.0
    print(
        f"{name}: counted={counted} real={real} fake={fake} missed={missed_n}  "
        f"P={precision:.1%} R={recall:.1%} FN={fn_rate:.1%} FP={fp_rate:.1%}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detections",
        type=Path,
        nargs="*",
        default=None,
        help="Existing particles.csv files to recount instead of running the pipeline.",
    )
    args = parser.parse_args(argv)
    config = load_config("config.yaml")
    labels = pd.read_csv("labels/labels.csv")
    gt = _ground_truth(labels)
    print(f"GT rows {len(gt)} (leftover ML-off particles + 4 inspect ≥20 µm)", flush=True)

    tables: list[tuple[str, pd.DataFrame]] = []
    if args.detections:
        for path in args.detections:
            tables.append((path.name, pd.read_csv(path)))
    else:
        print(
            "run",
            config["input_dir"],
            "recall",
            config["detection"].get("min_confidence"),
            "ml",
            config["ml"],
            flush=True,
        )
        table, _mosaic = run_pipeline(config, progress_cb=_progress)
        csv_path, _json_path, _xlsx_path = default_output_paths(config)
        write_csv(table, csv_path)
        write_last_pipeline(config, csv_path)
        print(f"wrote {csv_path} n={len(table)}", flush=True)
        tables.append(("this run", table))

    for name, table in tables:
        print(f"\n=== {name} n={len(table)} ===")
        for label, min_size in (("10um", 10_000.0), ("20um", 20_000.0)):
            counted, real, fake, missed_n, missed = recount(table, gt, config, min_size)
            _print_row(f"  {label}", counted, real, fake, missed_n, len(gt))
            for row in missed:
                print("    miss", row, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
