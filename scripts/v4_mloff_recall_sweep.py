"""Sweep Groundup v4 preprocess / detection settings for recall. ML stays off.

Gold is the v1–v3 labeled hits from ``Outputs/Groundup v4 ML off`` plus the
inspect marks that measured as real particles at least 20 µm and were not
already within 20 px of a hit. Recall is the fraction of those locations a
trial detects (within 30 px). Precision is recorded and not used to rank.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from src.config import apply_nsew_settings, cfg_get, load_config, resolve_output_dir
from src.detection.detector import (
    blob_equivalent_diameter,
    blob_local_peak_snr,
    detect_particles,
)
from src.io.tile_loader import load_tile_image
from src.measurement.measurer import measure_and_dedupe, measure_candidates
from src.preprocessing.corrections import apply_corrections

PIXEL = 3500.0
SETS = ("v1", "v2", "v3")
MATCH_PX = 30.0
STAGE = Path("/Users/rebeccajekel/Desktop/ASML SE/Inputs/Groundup v4 ML off")
ANGLES = Path("/Users/rebeccajekel/Desktop/ASML SE/Inputs/Groundup v4")
LABELS = ROOT / "labels" / "labels.csv"
OUTPUT_NAME = "Groundup v4 ML off recall sweep"

# Current Groundup v4 ML-off operating point (NSEW overlay, ML disabled).
BASELINE: dict[str, Any] = {
    "denoise_sigma": 0.5,
    "flatten_sigma": 160.0,
    "contrast": "1_99",
    "method": "fft",
    "fft_peak_threshold": 0.35,
    "tophat_radius": 50,
    "blob_threshold": 0.30,
    "min_prominence": 0.30,
    "min_confidence": 0.50,
    "blob_min_sigma": 9.0,
    "blob_max_sigma": 36.0,
    "blob_sigma_ratio": 1.4,
    "edge_exclude_px": 12.0,
    "edge_soften_sigma": 12.0,
    "edge_soften_strength": 2.0,
    "min_circularity": 0.0,
    "structure_neighbor_px": 48.0,
    "structure_line_min_run": 3,
    "local_snr_sigma": 0.0,
    "min_size_nm": 20000.0,
    "fft_notch_radius": 3,
}

ONE_FACTOR: list[tuple[str, list[Any]]] = [
    ("denoise_sigma", ["off", 0.25, 1.0, 1.5]),
    ("flatten_sigma", ["off", 40.0, 80.0, 240.0]),
    ("contrast", ["off", "0.5_99.5", "2_98", "5_95"]),
    ("method", ["tophat"]),
    ("fft_peak_threshold", [0.15, 0.25, 0.55]),
    ("tophat_radius", [30, 80, 120]),
    ("blob_threshold", [0.02, 0.05, 0.08, 0.12, 0.20]),
    ("min_prominence", [0.0, 0.10, 0.20, 0.45]),
    ("min_confidence", [0.0, 0.20, 0.35, 0.65]),
    ("blob_min_sigma", [1.5, 2.5, 3.7, 5.5, 7.4]),
    ("blob_max_sigma", [24.0, 48.0, 60.0]),
    ("blob_sigma_ratio", [1.2, 1.6]),
    ("edge_exclude_px", [0.0, 4.0, 24.0]),
    ("edge_soften_sigma", [0.0, 6.0, 24.0]),
    ("edge_soften_strength", [0.0, 1.0, 4.0]),
    ("min_circularity", [0.15]),
    ("structure_neighbor_px", [0.0, 24.0, 72.0]),
    ("structure_line_min_run", [0]),
    ("local_snr_sigma", [-1.0, 40.0, 80.0]),
    ("min_size_nm", [10000.0, 15000.0]),
    ("fft_notch_radius", [1, 6]),
]


def apply_knob(cfg: dict[str, Any], knob: str, value: Any) -> dict[str, Any]:
    out = deepcopy(cfg)
    pre = out.setdefault("preprocessing", {})
    det = out.setdefault("detection", {})
    if knob == "denoise_sigma":
        if value == "off":
            pre["denoise"] = False
        else:
            pre["denoise"] = True
            pre["denoise_sigma"] = float(value)
        return out
    if knob == "flatten_sigma":
        if value == "off":
            pre["flatten_illumination"] = False
        else:
            pre["flatten_illumination"] = True
            pre["flatten_sigma"] = float(value)
        return out
    if knob == "contrast":
        if value == "off":
            pre["contrast_stretch"] = False
        else:
            pre["contrast_stretch"] = True
            lo, hi = str(value).split("_")
            pre["contrast_percentiles"] = [float(lo), float(hi)]
        return out
    if knob == "method":
        det["method"] = str(value)
        return out
    if knob == "tophat_radius":
        det["tophat_radius"] = int(value)
        return out
    det[knob] = value
    return out


def config_from_knobs(knobs: dict[str, Any]) -> dict[str, Any]:
    cfg = apply_nsew_settings(load_config(ROOT / "config.yaml"))
    cfg["pixel_size_nm"] = PIXEL
    cfg["ml"]["enabled"] = False
    cfg["report"]["downsample"] = 1
    for name, value in knobs.items():
        cfg = apply_knob(cfg, name, value)
    return cfg


def _snap_peak(image: np.ndarray, y: float, x: float, search: int = 8) -> tuple[float, float]:
    height, width = image.shape
    iy, ix = int(round(y)), int(round(x))
    y0, y1 = max(iy - search, 0), min(iy + search + 1, height)
    x0, x1 = max(ix - search, 0), min(ix + search + 1, width)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    peaks = np.argwhere(patch == maximum_filter(patch, size=3))
    cy, cx = iy - y0, ix - x0
    best: tuple[float, float, float] | None = None
    span = float(np.ptp(patch)) + 1.0
    for py, px in peaks:
        dist = float(np.hypot(py - cy, px - cx))
        if dist > search:
            continue
        score = float(patch[py, px]) - 0.05 * dist * span
        if best is None or score > best[0]:
            best = (score, float(y0 + py), float(x0 + px))
    if best is None:
        return y, x
    return best[1], best[2]


def _radial_um(image: np.ndarray, y: float, x: float, max_r: int = 36) -> float:
    height, width = image.shape
    iy, ix = int(round(y)), int(round(x))
    y0, y1 = max(iy - max_r, 0), min(iy + max_r + 1, height)
    x0, x1 = max(ix - max_r, 0), min(ix + max_r + 1, width)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    cy, cx = iy - y0, ix - x0
    yy, xx = np.indices(patch.shape)
    dist = np.hypot(yy - cy, xx - cx)
    peak = float(patch[min(max(cy, 0), patch.shape[0] - 1), min(max(cx, 0), patch.shape[1] - 1)])
    far = patch[dist >= 0.7 * max_r]
    bg = float(np.median(far)) if far.size else float(np.median(patch))
    contrast = peak - bg
    if contrast <= 1e-6:
        return 0.0
    thresh = bg + 0.4 * contrast
    last = 0.0
    for radius in range(1, max_r):
        ring = patch[(dist >= radius - 0.5) & (dist < radius + 0.5)]
        if ring.size and float(np.median(ring)) >= thresh:
            last = float(radius)
        elif radius > 3 and last > 0:
            break
    return 2.0 * last * PIXEL / 1000.0


def build_gold() -> dict[str, np.ndarray]:
    """Pixel coordinates of labeled reals and size-checked misses, per set."""
    labels = pd.read_csv(LABELS)
    src = labels["source_csv"].astype(str)
    tile = labels["source_tile"].astype(str)
    hit_labels = labels.loc[
        src.str.contains("Groundup v4 ML off")
        & tile.str.match(r"v[123]_2of4")
        & (labels["label"].astype(str) == "particle")
    ]
    real_rows: list[tuple[str, float, float]] = []
    for rec in hit_labels.to_dict(orient="records"):
        set_id = str(rec["source_tile"]).split("_")[0]
        real_rows.append((set_id, float(rec["x_global"]) / PIXEL, float(rec["y_global"]) / PIXEL))

    misses = labels.loc[
        (src == "tile_inspect")
        & tile.str.match(r"v[123](?:/|_2of4)")
        & (labels["label"].astype(str) == "particle")
    ]
    cfg = config_from_knobs(BASELINE)
    cache: dict[str, np.ndarray] = {}

    def corrected(rel: str) -> np.ndarray:
        if rel not in cache:
            cache[rel] = apply_corrections(load_tile_image(ANGLES / rel), cfg)
        return cache[rel]

    miss_rows: list[tuple[str, float, float]] = []
    for rec in misses.to_dict(orient="records"):
        source = str(rec["source_tile"])
        set_id = source.split("/")[0].split("_")[0]
        x = float(rec["x_global"]) / PIXEL
        y = float(rec["y_global"]) / PIXEL
        if "/" in source:
            rel = source
        else:
            best_snr = None
            rel = f"{set_id}/N.bmp"
            for direction in "NSEW":
                image = corrected(f"{set_id}/{direction}.bmp")
                snr = blob_local_peak_snr(image, y, x, 12.0, bright=True)
                if best_snr is None or snr > best_snr:
                    best_snr = snr
                    rel = f"{set_id}/{direction}.bmp"
        image = corrected(rel)
        py, px = _snap_peak(image, y, x)
        core = _radial_um(image, py, px)
        snr = blob_local_peak_snr(image, py, px, 12.0, bright=True)
        if snr >= 5.0 and 20.0 <= core <= 140.0:
            miss_rows.append((set_id, px, py))

    gold: dict[str, np.ndarray] = {}
    for set_id in SETS:
        rows = [(x, y) for name, x, y in real_rows + miss_rows if name == set_id]
        gold[set_id] = np.asarray(rows, dtype=np.float64).reshape(-1, 2)
    return gold


def detect_set(set_id: str, cfg: dict[str, Any]) -> np.ndarray:
    path = STAGE / f"{set_id}_2of4.bmp"
    image = load_tile_image(path)
    corrected = apply_corrections(image, cfg)
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
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack(
        [
            table["x_global"].to_numpy(float) / PIXEL,
            table["y_global"].to_numpy(float) / PIXEL,
        ]
    )


def score_points(gold: dict[str, np.ndarray], found: dict[str, np.ndarray]) -> dict[str, Any]:
    per_set: dict[str, dict[str, int]] = {}
    hit_gold = 0
    gold_n = 0
    hits = 0
    for set_id in SETS:
        points = gold[set_id]
        dets = found[set_id]
        gold_n += int(len(points))
        hits += int(len(dets))
        if len(points) == 0 or len(dets) == 0:
            matched = 0
        else:
            dist, _ = cKDTree(dets).query(points, k=1)
            matched = int(np.sum(dist <= MATCH_PX))
        hit_gold += matched
        per_set[set_id] = {"gold": int(len(points)), "found": matched, "hits": int(len(dets))}
    recall = (hit_gold / gold_n) if gold_n else 0.0
    return {
        "found": hit_gold,
        "gold": gold_n,
        "recall": recall,
        "hits": hits,
        "per_set": per_set,
    }


def _trial_worker(payload: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    import cv2

    cv2.setNumThreads(1)
    trial_id, knobs = payload
    cfg = config_from_knobs(knobs)
    t0 = time.time()
    points = {set_id: detect_set(set_id, cfg) for set_id in SETS}
    return {
        "trial_id": trial_id,
        "knobs": knobs,
        "seconds": round(time.time() - t0, 2),
        "points": {set_id: arr.tolist() for set_id, arr in points.items()},
    }


def _rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (float(row["recall"]), int(row["found"]), -int(row["hits"]))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for row in rows:
        item = {k: v for k, v in row.items() if k != "points"}
        item["knobs_json"] = json.dumps(row.get("knobs") or {}, sort_keys=True)
        item["per_set_json"] = json.dumps(row.get("per_set") or {})
        flat.append(item)
    pd.DataFrame(flat).to_csv(path, index=False)


def _load_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    done: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        record["knobs"] = json.loads(record.pop("knobs_json") or "{}")
        record["per_set"] = json.loads(record.pop("per_set_json") or "{}")
        done[str(record["trial_id"])] = record
    return done


def phase2_trials(done: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """One-factor and small combinations around the denoise-σ 1.5 gain."""
    seed = {**BASELINE, "denoise_sigma": 1.5}
    grid: list[tuple[str, list[Any]]] = [
        ("denoise_sigma", [2.0, 2.5, 3.0, 4.0]),
        ("blob_min_sigma", [1.5, 2.0, 2.5, 3.7, 5.5]),
        ("blob_threshold", [0.02, 0.05, 0.08, 0.12, 0.20]),
        ("contrast", ["0.5_99.5", "off"]),
        ("local_snr_sigma", [80.0, 120.0, 160.0]),
        ("min_prominence", [0.35, 0.45, 0.55]),
        ("flatten_sigma", ["off", 80.0, 240.0]),
        ("fft_peak_threshold", [0.45, 0.50]),
        ("blob_max_sigma", [24.0, 48.0]),
        ("structure_neighbor_px", [0.0]),
        ("min_confidence", [0.0]),
        ("edge_exclude_px", [0.0]),
        ("fft_notch_radius", [2, 4]),
        ("blob_sigma_ratio", [1.2, 1.25]),
    ]
    pairs: list[dict[str, Any]] = [
        {"blob_min_sigma": 2.5, "blob_threshold": 0.05},
        {"blob_min_sigma": 2.5, "blob_threshold": 0.08},
        {"blob_min_sigma": 1.5, "blob_threshold": 0.05},
        {"blob_min_sigma": 2.5, "contrast": "0.5_99.5"},
        {"blob_min_sigma": 2.5, "local_snr_sigma": 80.0},
        {"blob_threshold": 0.05, "contrast": "0.5_99.5"},
        {"denoise_sigma": 2.0, "blob_min_sigma": 2.5},
        {"denoise_sigma": 2.0, "blob_threshold": 0.05},
        {"denoise_sigma": 2.0, "blob_min_sigma": 2.5, "blob_threshold": 0.05},
        {"denoise_sigma": 2.5, "blob_min_sigma": 2.5},
        {"denoise_sigma": 2.5, "blob_threshold": 0.05},
        {"denoise_sigma": 3.0, "blob_min_sigma": 2.5, "blob_threshold": 0.05},
    ]
    trials: list[tuple[str, dict[str, Any]]] = []
    for knob, values in grid:
        for value in values:
            knobs = {**seed, knob: value}
            trials.append((f"d15:{knob}={value}", knobs))
    for extra in pairs:
        knobs = {**seed, **extra}
        trial_id = "d15p:" + ",".join(f"{key}={knobs[key]}" for key in sorted(extra))
        trials.append((trial_id, knobs))
    return [item for item in trials if item[0] not in done]


def phase3_trials(done: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Combinations of the knobs that raised recall on top of denoise σ 1.5."""
    seed = {**BASELINE, "denoise_sigma": 1.5, "blob_min_sigma": 2.5}
    extras: list[dict[str, Any]] = [
        {"flatten_sigma": 60.0},
        {"flatten_sigma": 80.0},
        {"flatten_sigma": 100.0},
        {"flatten_sigma": 120.0},
        {"min_prominence": 0.40},
        {"min_prominence": 0.55},
        {"local_snr_sigma": 60.0},
        {"local_snr_sigma": 100.0},
        {"blob_min_sigma": 2.2},
        {"blob_min_sigma": 2.8},
        {"blob_min_sigma": 3.0},
        {"denoise_sigma": 1.25},
        {"denoise_sigma": 1.75},
        {"contrast": "off"},
        {"blob_max_sigma": 24.0},
        {"flatten_sigma": 80.0, "local_snr_sigma": 80.0},
        {"flatten_sigma": 80.0, "min_prominence": 0.45},
        {"flatten_sigma": 80.0, "local_snr_sigma": 80.0, "min_prominence": 0.45},
        {"flatten_sigma": 80.0, "contrast": "off"},
        {"local_snr_sigma": 80.0, "min_prominence": 0.55},
        {"local_snr_sigma": 80.0, "flatten_sigma": 80.0, "blob_max_sigma": 24.0},
        {"denoise_sigma": 1.25, "flatten_sigma": 80.0},
        {"denoise_sigma": 1.75, "flatten_sigma": 80.0},
        {"denoise_sigma": 1.25, "local_snr_sigma": 80.0},
        {"blob_threshold": 0.25},
        {"blob_threshold": 0.15},
        {"structure_neighbor_px": 0.0, "flatten_sigma": 80.0},
        {"local_snr_sigma": 50.0},
        {"local_snr_sigma": 70.0},
        {"local_snr_sigma": 60.0, "structure_neighbor_px": 0.0},
        {"local_snr_sigma": 60.0, "flatten_sigma": 60.0},
        {"local_snr_sigma": 60.0, "flatten_sigma": 80.0},
        {"local_snr_sigma": 60.0, "structure_neighbor_px": 0.0, "flatten_sigma": 60.0},
        {"local_snr_sigma": 60.0, "min_prominence": 0.40},
        {"local_snr_sigma": 60.0, "contrast": "off"},
        {"local_snr_sigma": 60.0, "blob_max_sigma": 24.0},
        {"local_snr_sigma": 60.0, "denoise_sigma": 1.25},
        {"local_snr_sigma": 60.0, "denoise_sigma": 1.35},
        {"local_snr_sigma": 40.0},
        {"local_snr_sigma": 60.0, "edge_soften_strength": 0.0},
        {"local_snr_sigma": 60.0, "fft_peak_threshold": 0.45},
        {"local_snr_sigma": 60.0, "structure_neighbor_px": 8.0},
        {"local_snr_sigma": 60.0, "structure_neighbor_px": 16.0},
        {"local_snr_sigma": 60.0, "structure_neighbor_px": 32.0},
        {"local_snr_sigma": 60.0, "structure_min_neighbors": 4},
        {"local_snr_sigma": 60.0, "structure_min_neighbors": 6},
        {"local_snr_sigma": 60.0, "structure_neighbor_px": 0.0, "structure_line_min_run": 0},
        {"local_snr_sigma": 60.0, "structure_neighbor_px": 0.0, "edge_soften_strength": 0.0},
        {"local_snr_sigma": 60.0, "structure_neighbor_px": 0.0, "blob_threshold": 0.15},
        {"local_snr_sigma": 60.0, "structure_neighbor_px": 0.0, "min_size_nm": 10000.0},
        {"local_snr_sigma": 60.0, "structure_neighbor_px": 36.0},
        {"local_snr_sigma": 60.0, "structure_neighbor_px": 40.0},
        {"local_snr_sigma": 60.0, "structure_neighbor_px": 44.0},
        {"local_snr_sigma": 60.0, "structure_min_neighbors": 3},
    ]
    trials: list[tuple[str, dict[str, Any]]] = []
    for extra in extras:
        knobs = {**seed, **extra}
        trial_id = "d15b:" + ",".join(f"{key}={knobs[key]}" for key in sorted(extra))
        trials.append((trial_id, knobs))
    return [item for item in trials if item[0] not in done]


def _run_trials(
    pending: list[tuple[str, dict[str, Any]]],
    gold: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    done: dict[str, dict[str, Any]],
    results_path: Path,
    workers: int,
) -> None:
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_trial_worker, item): item[0] for item in pending}
        for future in as_completed(futures):
            payload = future.result()
            points = {
                set_id: np.asarray(payload["points"][set_id], dtype=np.float64)
                for set_id in SETS
            }
            scored = score_points(gold, points)
            row = {
                "trial_id": payload["trial_id"],
                "knobs": payload["knobs"],
                "seconds": payload["seconds"],
                **{k: scored[k] for k in ("found", "gold", "recall", "hits", "per_set")},
            }
            rows.append(row)
            done[row["trial_id"]] = row
            _write(results_path, rows)
            bits = " ".join(
                f"{set_id} {row['per_set'][set_id]['found']}/{row['per_set'][set_id]['gold']}"
                for set_id in SETS
            )
            print(
                f"{row['trial_id']}  R={row['recall']:.3f} found={row['found']}/{row['gold']} "
                f"hits={row['hits']} {bits} ({row['seconds']}s)",
                flush=True,
            )


def phase4_trials(done: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Smaller steps around the recall peak: denoise 1.5, blob min σ 2.5, SNR 60, neighbor 32."""
    seed = {
        **BASELINE,
        "denoise_sigma": 1.5,
        "blob_min_sigma": 2.5,
        "local_snr_sigma": 60.0,
        "structure_neighbor_px": 32.0,
    }
    trials: list[tuple[str, dict[str, Any]]] = []
    for denoise in (1.2, 1.3, 1.4, 1.5, 1.6, 1.7):
        for blob_min in (2.2, 2.3, 2.4, 2.5, 2.6, 2.7):
            if denoise == 1.5 and blob_min == 2.5:
                continue
            knobs = {**seed, "denoise_sigma": denoise, "blob_min_sigma": blob_min}
            trials.append((f"fine:denoise_sigma={denoise},blob_min_sigma={blob_min}", knobs))
    singles: list[tuple[str, list[Any]]] = [
        ("local_snr_sigma", [54.0, 56.0, 58.0, 62.0, 64.0, 66.0, 68.0]),
        ("structure_neighbor_px", [24.0, 26.0, 28.0, 30.0, 33.0, 34.0, 35.0]),
        ("blob_threshold", [0.22, 0.26, 0.34, 0.38]),
        ("min_prominence", [0.25, 0.35, 0.40, 0.45]),
    ]
    for knob, values in singles:
        for value in values:
            knobs = {**seed, knob: value}
            trials.append((f"fine:{knob}={value}", knobs))
    for extra in (
        {"local_snr_sigma": 65.0},
        {"local_snr_sigma": 67.0},
        {"local_snr_sigma": 66.0, "structure_neighbor_px": 33.0},
        {"structure_neighbor_px": 33.0},
        {"local_snr_sigma": 66.0, "blob_min_sigma": 2.3},
    ):
        knobs = {**seed, **extra}
        trial_id = "fine:" + ",".join(f"{key}={knobs[key]}" for key in sorted(extra))
        trials.append((trial_id, knobs))
    return [item for item in trials if item[0] not in done]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--phase2", action="store_true", help="Search around denoise sigma 1.5.")
    parser.add_argument("--phase3", action="store_true", help="Combine the phase-2 recall gainers.")
    parser.add_argument(
        "--phase4",
        action="store_true",
        help="Finer steps around denoise, blob min σ, local SNR, and structure neighbor.",
    )
    args = parser.parse_args(argv)

    out = resolve_output_dir(OUTPUT_NAME)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "sweep_results.csv"
    print("Building gold…", flush=True)
    gold = build_gold()
    for set_id in SETS:
        print(f"  {set_id} gold {len(gold[set_id])}", flush=True)
    print(f"  total {sum(len(v) for v in gold.values())}", flush=True)

    done = _load_done(results_path)
    rows = list(done.values())

    trials: list[tuple[str, dict[str, Any]]] = [("baseline", dict(BASELINE))]
    if not args.combine_only:
        for knob, values in ONE_FACTOR:
            for value in values:
                knobs = {**BASELINE, knob: value}
                trials.append((f"1f:{knob}={value}", knobs))

    pending = [item for item in trials if item[0] not in done]
    if args.phase2:
        pending = phase2_trials(done)
    if args.phase3:
        pending = phase3_trials(done)
    if args.phase4:
        pending = phase4_trials(done)
    print(f"Running {len(pending)} trials with {args.workers} workers", flush=True)
    _run_trials(pending, gold, rows, done, results_path, args.workers)

    one_factor_rows = [row for row in rows if str(row["trial_id"]).startswith("1f:")]
    best_by_knob: dict[str, dict[str, Any]] = {}
    baseline_row = done.get("baseline")
    base_recall = float(baseline_row["recall"]) if baseline_row else 0.0
    for row in one_factor_rows:
        name = str(row["trial_id"])[3:].split("=", 1)[0]
        prev = best_by_knob.get(name)
        if prev is None or _rank(row) > _rank(prev):
            best_by_knob[name] = row
    improved = {
        name: row
        for name, row in best_by_knob.items()
        if float(row["recall"]) > base_recall + 1e-9
    }
    print("\nOne-factor recall gains:", flush=True)
    for name, row in sorted(improved.items(), key=lambda item: _rank(item[1]), reverse=True):
        print(
            f"  {row['trial_id']}  R={float(row['recall']):.3f} hits={int(row['hits'])}",
            flush=True,
        )

    combo = dict(BASELINE)
    for name, row in improved.items():
        value = json.loads(row["knobs_json"])[name] if "knobs_json" in row else row["knobs"][name]
        combo[name] = value
    combo_id = "combo:" + ",".join(
        f"{name}={combo[name]}" for name in sorted(improved)
    )
    if improved and combo_id not in done:
        print(f"\nCombined gainers: {combo_id}", flush=True)
        payload = _trial_worker((combo_id, combo))
        points = {
            set_id: np.asarray(payload["points"][set_id], dtype=np.float64) for set_id in SETS
        }
        scored = score_points(gold, points)
        row = {
            "trial_id": combo_id,
            "knobs": combo,
            "seconds": payload["seconds"],
            **{k: scored[k] for k in ("found", "gold", "recall", "hits", "per_set")},
        }
        rows.append(row)
        done[combo_id] = row
        _write(results_path, rows)
        print(
            f"{combo_id}  R={row['recall']:.3f} found={row['found']}/{row['gold']} hits={row['hits']}",
            flush=True,
        )

    best = max(rows, key=_rank)
    summary = {
        "best_trial": best["trial_id"],
        "recall": best["recall"],
        "found": best["found"],
        "gold": best["gold"],
        "hits": best["hits"],
        "per_set": best["per_set"],
        "knobs": best["knobs"],
        "baseline_recall": base_recall,
    }
    (out / "winner.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nWINNER", best["trial_id"], flush=True)
    print(json.dumps({k: summary[k] for k in ("recall", "found", "gold", "hits", "per_set")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
