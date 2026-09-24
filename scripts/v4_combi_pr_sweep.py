"""One-factor preprocess / detection sweep on Groundup v4 2-of-4 combined images.

ML stays off. Precision is real hits / hits. Recall is real hits / (real + missed),
using the same 24 px label match as the v1–v3 combined counts.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts.v4_mloff_recall_sweep import ONE_FACTOR, apply_knob
from src.config import apply_nsew_settings, load_config, resolve_output_dir
from src.detection.detector import detect_particles
from src.io.tile_loader import load_tile_image
from src.labeling.inspect import _labels_for_pointer_set
from src.labeling.store import LabelStore
from src.measurement.measurer import measure_and_dedupe, measure_candidates
from src.preprocessing.corrections import apply_corrections

PIXEL = 3500.0
SETS = ("v1", "v2", "v3")
MATCH_PX = 24.0
SIZE_FRAC = 0.4
FLOOR_NM = 20000.0
STAGE = Path("/Users/rebeccajekel/Desktop/ASML SE/Inputs/Groundup v4 combi v1-3")
OUTPUT_NAME = "Groundup v4 ML off combi 2of4 sweep"


def config_from_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = apply_nsew_settings(load_config(ROOT / "config.yaml"))
    cfg["pixel_size_nm"] = PIXEL
    cfg["ml"]["enabled"] = False
    cfg["report"]["downsample"] = 1
    for name, value in overrides.items():
        cfg = apply_knob(cfg, name, value)
    return cfg


def detect_set(set_id: str, cfg: dict[str, Any]) -> np.ndarray:
    path = STAGE / f"{set_id}_2of4.bmp"
    corrected = apply_corrections(load_tile_image(path), cfg)
    candidates = detect_particles(corrected, cfg, structure_filters=True)
    particles = measure_candidates(
        candidates,
        origin_x=0.0,
        origin_y=0.0,
        pixel_size_nm=PIXEL,
        source_tile=path.name,
    )
    table = measure_and_dedupe(particles, cfg)
    if table.empty:
        return np.empty((0, 3), dtype=np.float64)
    return np.column_stack(
        [
            table["x_global"].to_numpy(float) / PIXEL,
            table["y_global"].to_numpy(float) / PIXEL,
            table["size"].to_numpy(float),
        ]
    )


def load_labels() -> dict[str, dict[str, np.ndarray]]:
    labels = LabelStore(ROOT / "labels").load()
    packed: dict[str, dict[str, np.ndarray]] = {}
    for set_id in SETS:
        extra = _labels_for_pointer_set(labels, set_id)
        packed[set_id] = {
            "xy": extra[["x_global", "y_global"]].astype(float).to_numpy() / PIXEL,
            "size": pd.to_numeric(extra["size"], errors="coerce").fillna(0).to_numpy(),
            "particle": (extra["label"].astype(str) == "particle").to_numpy(),
        }
    return packed


def score_points(
    labels: dict[str, dict[str, np.ndarray]],
    found: dict[str, np.ndarray],
) -> dict[str, Any]:
    hits = real = fake = missed = 0
    per_set: dict[str, dict[str, int]] = {}
    for set_id in SETS:
        lab = labels[set_id]
        lab_xy = lab["xy"]
        lab_size = lab["size"]
        lab_particle = lab["particle"]
        dets = found[set_id]
        n_real = n_fake = 0
        matched = np.zeros(len(lab_xy), dtype=bool)
        if len(dets) and len(lab_xy):
            dist, nn = cKDTree(lab_xy).query(dets[:, :2], k=1)
            for i in range(len(dets)):
                j = int(nn[i])
                limit = max(MATCH_PX, SIZE_FRAC * max(float(dets[i, 2]), float(lab_size[j])) / PIXEL)
                if dist[i] <= limit:
                    matched[j] = True
                    if lab_particle[j]:
                        n_real += 1
                    else:
                        n_fake += 1
                else:
                    n_fake += 1
        elif len(dets):
            n_fake = int(len(dets))
        part_idx = [
            i
            for i in range(len(lab_xy))
            if lab_particle[i] and lab_size[i] >= FLOOR_NM and not matched[i]
        ]
        n_missed = 0
        if part_idx:
            pts = lab_xy[part_idx]
            used = np.zeros(len(part_idx), dtype=bool)
            tree = cKDTree(pts)
            for i in range(len(part_idx)):
                if used[i]:
                    continue
                n_missed += 1
                for k in tree.query_ball_point(pts[i], MATCH_PX):
                    used[k] = True
        per_set[set_id] = {
            "hits": int(len(dets)),
            "real": n_real,
            "fake": n_fake,
            "missed": n_missed,
        }
        hits += int(len(dets))
        real += n_real
        fake += n_fake
        missed += n_missed
    precision = (real / hits) if hits else 0.0
    recall = (real / (real + missed)) if (real + missed) else 0.0
    return {
        "hits": hits,
        "real": real,
        "fake": fake,
        "missed": missed,
        "precision": precision,
        "recall": recall,
        "per_set": per_set,
    }


def _trial_worker(payload: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    import cv2

    cv2.setNumThreads(1)
    trial_id, overrides = payload
    cfg = config_from_overrides(overrides)
    t0 = time.time()
    points = {set_id: detect_set(set_id, cfg) for set_id in SETS}
    return {
        "trial_id": trial_id,
        "overrides": overrides,
        "seconds": round(time.time() - t0, 2),
        "points": {set_id: arr.tolist() for set_id, arr in points.items()},
    }


def main() -> int:
    out = resolve_output_dir(OUTPUT_NAME)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "sweep_results.csv"
    labels = load_labels()
    trials: list[tuple[str, dict[str, Any]]] = [("baseline", {})]
    for knob, values in ONE_FACTOR:
        for value in values:
            trials.append((f"1f:{knob}={value}", {knob: value}))

    rows: list[dict[str, Any]] = []
    print(f"Running {len(trials)} trials", flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_trial_worker, item): item[0] for item in trials}
        for future in as_completed(futures):
            payload = future.result()
            points = {
                set_id: np.asarray(payload["points"][set_id], dtype=np.float64)
                for set_id in SETS
            }
            scored = score_points(labels, points)
            row = {
                "trial_id": payload["trial_id"],
                "overrides": payload["overrides"],
                "seconds": payload["seconds"],
                **{k: scored[k] for k in ("hits", "real", "fake", "missed", "precision", "recall", "per_set")},
            }
            rows.append(row)
            print(
                f"{row['trial_id']}  P={row['precision']:.3f} R={row['recall']:.3f} "
                f"hits={row['hits']} real={row['real']} fake={row['fake']} missed={row['missed']} "
                f"({row['seconds']}s)",
                flush=True,
            )
    flat = []
    for row in rows:
        item = {k: v for k, v in row.items() if k not in ("overrides", "per_set")}
        item["overrides_json"] = json.dumps(row["overrides"], sort_keys=True)
        item["per_set_json"] = json.dumps(row["per_set"])
        flat.append(item)
    pd.DataFrame(flat).to_csv(results_path, index=False)
    print("wrote", results_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
