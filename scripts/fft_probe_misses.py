"""Diagnose FFT vs top-hat at the four inspect-mark flakes.

Usage (from particle_detection/):

    PYTHONPATH=. python scripts/fft_probe_misses.py
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import cfg_get, load_config, with_recall_profile
from src.detection.detector import (
    _background_residual,
    _gate_blob,
    _probe_has_lattice,
    _residual_and_blobs,
    detect_particles,
    trace_locations,
)
from src.io.tile_loader import load_tile_image
from src.labeling.crops import find_tile_path, global_nm_to_local_px, placement_for_tile
from src.preprocessing.corrections import apply_corrections

INSPECT_KEYS = (
    "R3_2_30_5X_81544816_2394400",
    "R3_2_30_5X_80834576_2903920",
    "R3_5_16_5X_39563456_8950736",
    "R3_5_16_5X_40212423_8699958",
)
THRESHOLDS = (0.35, 0.25, 0.15, 0.10)
WINDOW_PX = 40


def _marks(config: dict) -> list[dict]:
    labels = pd.read_csv(Path("labels/labels.csv"))
    rows = labels.loc[labels["key"].astype(str).isin(INSPECT_KEYS)]
    origin_cache: dict[str, tuple[int, int]] = {}
    marks: list[dict] = []
    for record in rows.to_dict(orient="records"):
        tile = Path(str(record["source_tile"])).name
        path = find_tile_path(tile, input_dir=config["input_dir"])
        placement = placement_for_tile(path, config, origin_cache=origin_cache)
        pixel_size = float(config.get("pixel_size_nm", 960.0))
        x_local, y_local = global_nm_to_local_px(
            float(record["x_global"]),
            float(record["y_global"]),
            placement,
            pixel_size,
        )
        marks.append(
            {
                "key": str(record["key"]),
                "tile": tile,
                "path": path,
                "x_local": float(x_local),
                "y_local": float(y_local),
            }
        )
    return marks


def main() -> int:
    config = with_recall_profile(load_config("config.yaml"))
    notch = int(cfg_get(config, "detection.fft_notch_radius", 3))
    for mark in _marks(config):
        image = apply_corrections(load_tile_image(mark["path"]), config)
        iy, ix = int(round(mark["y_local"])), int(round(mark["x_local"]))
        print(
            f"\n=== {mark['tile']} click=({mark['x_local']:.0f},{mark['y_local']:.0f}) ===",
            flush=True,
        )
        for threshold in THRESHOLDS:
            trial = deepcopy(config)
            trial.setdefault("detection", {})["fft_peak_threshold"] = float(threshold)
            probe = _probe_has_lattice(image, float(threshold), notch)
            _residual, used_tophat = _background_residual(image, trial, None)
            print(
                f"  thresh={threshold:.2f}  probe_lattice={probe}  "
                f"residual={'tophat' if used_tophat else 'fft'}",
                flush=True,
            )
        residual, used = _background_residual(image, config, None)
        y0, x0 = max(0, iy - WINDOW_PX), max(0, ix - WINDOW_PX)
        y1, x1 = min(residual.shape[0], iy + WINDOW_PX + 1), min(
            residual.shape[1], ix + WINDOW_PX + 1
        )
        patch = residual[y0:y1, x0:x1]
        py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
        peak_y, peak_x = y0 + int(py), x0 + int(px)
        array, res_maps, edge, blobs, protect = _residual_and_blobs(image, config)
        traces = trace_locations(
            image, config, [(mark["y_local"], mark["x_local"]), (float(peak_y), float(peak_x))]
        )
        gated = []
        for y, x, sigma in blobs:
            cand, reason = _gate_blob(
                float(y),
                float(x),
                float(sigma),
                res_maps,
                array,
                edge,
                config,
                protect=protect,
            )
            if cand is not None:
                gated.append(cand)
        kept = detect_particles(image, config)
        print(
            f"  default residual={'tophat' if used else 'fft'}  "
            f"click_res={float(residual[iy, ix]):.4f}  "
            f"win_max={float(patch.max()):.4f} at ({peak_x},{peak_y})  "
            f"gated={len(gated)} kept={len(kept)}",
            flush=True,
        )
        print(
            f"  click gate={traces[0]['reason']} blob={traces[0]['nearest_blob_px']:.1f} "
            f"kept={traces[0]['nearest_kept_px']:.1f}",
            flush=True,
        )
        print(
            f"  peak  gate={traces[1]['reason']} blob={traces[1]['nearest_blob_px']:.1f} "
            f"kept={traces[1]['nearest_kept_px']:.1f}",
            flush=True,
        )
        if kept:
            dist = min(
                ((c.x_local - mark["x_local"]) ** 2 + (c.y_local - mark["y_local"]) ** 2) ** 0.5
                for c in kept
            )
            print(f"  nearest kept to click {dist:.1f} px", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
