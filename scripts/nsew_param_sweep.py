"""Sweep Groundup v3 preprocess / detect / ML settings against labeled NSEW tiles.

Orange (unlabeled) detector hits count as fake. Best settings are written to
``nsew_config.yaml`` for the sidebar NSEW toggle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import (
    PACKAGE_ROOT,
    apply_nsew_settings,
    cfg_get,
    load_config,
    resolve_output_dir,
    save_nsew_overlay,
)
from src.detection.detector import detect_particles
from src.io.tile_loader import load_tile_image, resolve_input_dir
from src.labeling.inspect import (
    last_run_class_counts,
    related_scene_tile_names,
    undetected_size_floor_nm,
)
from src.labeling.queue import snap_detection_keys_to_labels, tag_table
from src.labeling.store import LabelStore
from src.measurement.measurer import (
    DEFAULT_NSEW_MERGE_RADIUS_PX,
    DEFAULT_NSEW_SIZE_MATCH_FRACTION,
    measure_and_dedupe,
    measure_candidates,
)
from src.ml.infer import (
    KIND_CASCADE,
    KIND_PATCH,
    cached_artifact,
    predict_patch_scores,
    predict_proba,
    resolve_model_path,
)
from src.preprocessing.corrections import apply_corrections

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

NSEW_INPUT = "Groundup v3"
PO_INPUT = "Groundup v3/Output Groundup v3/Particles only"
OUTPUT_NAME = "Groundup v3 nsew sweep"
PIXEL_SIZE_NM = 3500.0
NSEW_TILES = ("N.png", "S.png", "E.png", "W.png")
PO_TILES = (
    "particles_only.png",
    "particles_only_3of4.png",
    "particles_only_4of4.png",
)
SAMPLES = (
    ("NSEW individually", NSEW_TILES),
    ("2-of-4", ("particles_only.png",)),
    ("3-of-4", ("particles_only_3of4.png",)),
    ("4-of-4", ("particles_only_4of4.png",)),
)
ML_THRESHOLDS = tuple(round(x, 2) for x in np.linspace(0.05, 0.85, 17))
PO2_ML_THRESHOLDS = tuple(
    sorted(
        {
            *tuple(round(x, 2) for x in np.linspace(0.0, 0.90, 19)),
            0.58,
            0.62,
            0.78,
            0.82,
            0.84,
        }
    )
)
MIN_NSEW_RECALL = 0.90
TARGET_NSEW_RECALL = 0.95
PO2_OUTPUT_NAME = "Groundup v3 2of4 sweep"
PO2_TILES_ONLY = ("particles_only.png",)

ONE_FACTOR: list[tuple[str, list[Any]]] = [
    ("denoise_sigma", [0.5, 1.0, 1.5, 2.0, "off"]),
    ("flatten_sigma", [80.0, 120.0, 160.0, 220.0]),
    ("contrast", ["1_99", "2_98", "off"]),
    ("method", ["fft", "tophat"]),
    ("fft_peak_threshold", [0.25, 0.35, 0.50]),
    ("blob_threshold", [0.08, 0.10, 0.12, 0.16, 0.20]),
    ("min_prominence", [0.15, 0.30, 0.45]),
    ("min_confidence", [0.35, 0.50, 0.65]),
    ("blob_min_sigma", [3.7, 5.5, 7.4]),
    ("edge_exclude_px", [8.0, 12.0, 24.0]),
    ("structure_neighbor_px", [0.0, 48.0]),
    ("min_circularity", [0.0, 0.15, 0.30]),
]
BASELINE_KNOBS: dict[str, Any] = {
    "denoise_sigma": 1.0,
    "flatten_sigma": 160.0,
    "contrast": "1_99",
    "method": "fft",
    "fft_peak_threshold": 0.35,
    "blob_threshold": 0.12,
    "min_prominence": 0.30,
    "min_confidence": 0.50,
    "blob_min_sigma": 3.7,
    "edge_exclude_px": 12.0,
    "structure_neighbor_px": 48.0,
    "min_circularity": 0.0,
}
# 2-of-4 particles-only masks: milder denoise already cut unhappiness from ~44
# to ~31. Search around that, plus tighter contrast and larger DoG scale.
PO2_BASELINE_KNOBS: dict[str, Any] = {
    **BASELINE_KNOBS,
    "denoise_sigma": 0.5,
}
PO2_ONE_FACTOR: list[tuple[str, list[Any]]] = [
    ("denoise_sigma", [0.25, 0.4, 0.5, 0.75, 1.0, "off"]),
    ("flatten_sigma", [40.0, 60.0, 80.0, 100.0, 160.0, "off"]),
    ("contrast", ["1_99", "2_98", "3_97", "5_95", "off"]),
    ("method", ["fft", "tophat"]),
    ("tophat_radius", [20, 35, 50, 80]),
    ("fft_peak_threshold", [0.20, 0.25, 0.35, 0.45, 0.55]),
    ("blob_threshold", [0.04, 0.08, 0.12, 0.18, 0.24, 0.30]),
    ("min_prominence", [0.05, 0.20, 0.30, 0.50, 0.70]),
    ("min_confidence", [0.20, 0.35, 0.50, 0.70, 0.85]),
    # 9.0 beat 7.4; 11.0 collapsed. Fill the cliff plus a bit below 9.
    ("blob_min_sigma", [3.7, 5.5, 7.4, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 14.0]),
    ("blob_max_sigma", [24.0, 36.0, 48.0]),
    ("edge_exclude_px", [0.0, 4.0, 12.0, 24.0, 36.0]),
    ("min_circularity", [0.0, 0.10, 0.20, 0.35, 0.50]),
    ("structure_neighbor_px", [0.0, 24.0, 48.0, 72.0]),
]
PO2_SEED_COMBOS: list[dict[str, Any]] = [
    {"denoise_sigma": 0.5, "contrast": "2_98"},
    {"denoise_sigma": 0.5, "contrast": "off"},
    {"denoise_sigma": 0.5, "contrast": "3_97"},
    {"denoise_sigma": 0.4, "contrast": "2_98"},
    {"denoise_sigma": 0.5, "blob_min_sigma": 7.4},
    {"denoise_sigma": 0.5, "contrast": "2_98", "blob_min_sigma": 7.4},
    {"denoise_sigma": 0.5, "contrast": "off", "blob_min_sigma": 7.4},
    {"denoise_sigma": 0.5, "contrast": "2_98", "blob_min_sigma": 9.0},
    {"denoise_sigma": 0.5, "min_circularity": 0.20},
    {"denoise_sigma": 0.5, "contrast": "2_98", "min_circularity": 0.20},
    {"denoise_sigma": 0.5, "flatten_sigma": "off"},
    {"denoise_sigma": 0.5, "method": "tophat", "tophat_radius": 35},
    {"denoise_sigma": 0.5, "contrast": "2_98", "blob_threshold": 0.18},
    {"denoise_sigma": 0.5, "contrast": "2_98", "min_confidence": 0.70},
    # Local search around blob_min_sigma=9 (best 1-factor so far, 2-of-4 U=23.43).
    {"blob_min_sigma": 8.0, "min_circularity": 0.10},
    {"blob_min_sigma": 8.5, "min_circularity": 0.10},
    {"blob_min_sigma": 9.0, "min_circularity": 0.10},
    {"blob_min_sigma": 9.5, "min_circularity": 0.10},
    {"blob_min_sigma": 9.0, "structure_neighbor_px": 72.0},
    {"blob_min_sigma": 9.0, "blob_threshold": 0.18},
    {"blob_min_sigma": 9.0, "blob_threshold": 0.24},
    {"blob_min_sigma": 9.0, "blob_threshold": 0.30},
    {"blob_min_sigma": 9.0, "min_prominence": 0.20},
    {"blob_min_sigma": 9.0, "min_confidence": 0.70},
    {"blob_min_sigma": 9.0, "denoise_sigma": 0.4},
    {"blob_min_sigma": 9.0, "blob_max_sigma": 24.0},
    {"blob_min_sigma": 9.0, "blob_max_sigma": 48.0},
    {"blob_min_sigma": 9.0, "fft_peak_threshold": 0.25},
    {"blob_min_sigma": 9.0, "min_circularity": 0.10, "blob_threshold": 0.18},
    {"blob_min_sigma": 8.5, "edge_exclude_px": 24.0},
    {"blob_min_sigma": 9.5, "edge_exclude_px": 24.0},
    {"blob_min_sigma": 9.0, "edge_exclude_px": 24.0, "min_circularity": 0.10},
    {"blob_min_sigma": 9.0, "contrast": "2_98", "min_circularity": 0.10},
]


def _labels_path() -> Path:
    return PACKAGE_ROOT / "labels"


def apply_knob(cfg: dict[str, Any], knob: str, value: Any) -> dict[str, Any]:
    """Mutate a copy of ``cfg`` for one named trial value."""
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
        elif value == "1_99":
            pre["contrast_stretch"] = True
            pre["contrast_percentiles"] = [1.0, 99.0]
        elif value == "2_98":
            pre["contrast_stretch"] = True
            pre["contrast_percentiles"] = [2.0, 98.0]
        elif value == "3_97":
            pre["contrast_stretch"] = True
            pre["contrast_percentiles"] = [3.0, 97.0]
        elif value == "5_95":
            pre["contrast_stretch"] = True
            pre["contrast_percentiles"] = [5.0, 95.0]
        else:
            raise ValueError(f"Unknown contrast value {value!r}")
        return out
    if knob == "method":
        det["method"] = str(value)
        return out
    if knob == "tophat_radius":
        det["method"] = "tophat"
        det["tophat_radius"] = int(value)
        return out
    det[knob] = value
    return out


def nsew_base_config() -> dict[str, Any]:
    cfg = deepcopy(load_config(PACKAGE_ROOT / "config.yaml"))
    cfg["pixel_size_nm"] = PIXEL_SIZE_NM
    cfg["filename_pattern"] = str(cfg.get("filename_pattern") or "")
    cfg["run"] = ""
    cfg["magnification"] = ""
    report = cfg.setdefault("report", {})
    report["downsample"] = 1
    ml = cfg.setdefault("ml", {})
    ml["enabled"] = True
    ml["model_path"] = "models/particle_clf_v5.joblib"
    ml["score_before_structure"] = False
    for knob, value in BASELINE_KNOBS.items():
        cfg = apply_knob(cfg, knob, value)
    return cfg


def overlay_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compact NSEW standard-values overlay (sidebar-relevant fields only)."""
    pre = cfg.get("preprocessing") or {}
    det = cfg.get("detection") or {}
    ml = cfg.get("ml") or {}
    return {
        "pixel_size_nm": float(cfg.get("pixel_size_nm", PIXEL_SIZE_NM)),
        "preprocessing": {
            "denoise": bool(pre.get("denoise", True)),
            "denoise_sigma": float(pre.get("denoise_sigma", 1.0)),
            "flatten_illumination": bool(pre.get("flatten_illumination", True)),
            "flatten_sigma": float(pre.get("flatten_sigma", 160.0)),
            "contrast_stretch": bool(pre.get("contrast_stretch", True)),
            "contrast_percentiles": list(pre.get("contrast_percentiles") or [1.0, 99.0]),
        },
        "detection": {
            "method": str(det.get("method", "fft")),
            "fft_peak_threshold": float(det.get("fft_peak_threshold", 0.35)),
            "blob_threshold": float(det.get("blob_threshold", 0.12)),
            "min_prominence": float(det.get("min_prominence", 0.30)),
            "min_confidence": float(det.get("min_confidence", 0.50)),
            "blob_min_sigma": float(det.get("blob_min_sigma", 3.7)),
            "blob_max_sigma": float(det.get("blob_max_sigma", 36.0)),
            "edge_exclude_px": float(det.get("edge_exclude_px", 12.0)),
            "min_circularity": float(det.get("min_circularity", 0.0)),
            "structure_neighbor_px": float(det.get("structure_neighbor_px", 48.0)),
            "tophat_radius": int(det.get("tophat_radius", 50)),
        },
        "ml": {
            "enabled": True,
            "model_path": str(ml.get("model_path", "models/particle_clf_v5.joblib")),
            "threshold": float(ml.get("threshold", 0.4)),
        },
    }


def _ml_scores(
    candidates: list[Any],
    config: dict[str, Any],
    image: np.ndarray,
    source_tile: str,
) -> np.ndarray:
    if not candidates:
        return np.empty((0,), dtype=np.float64)
    path = resolve_model_path(cfg_get(config, "ml.model_path", "models/particle_clf_v5.joblib"))
    artifact = cached_artifact(path)
    kind = str(artifact.get("kind") or "")
    if kind in (KIND_PATCH, KIND_CASCADE):
        return np.asarray(
            predict_patch_scores(
                candidates, artifact, config, image, source_tile=source_tile
            ),
            dtype=np.float64,
        )
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    return np.asarray(predict_proba(candidates, artifact, pixel_size), dtype=np.float64)


def detect_tile_scored(
    path: Path, config: dict[str, Any]
) -> list[tuple[Any, float]]:
    """Classical detections on one tile, each with an ML P(particle) score."""
    image = load_tile_image(path)
    corrected = apply_corrections(image, config)
    candidates = detect_particles(corrected, config, structure_filters=True)
    scores = _ml_scores(candidates, config, corrected, path.name)
    pixel_size = float(cfg_get(config, "pixel_size_nm", PIXEL_SIZE_NM))
    particles = measure_candidates(
        candidates,
        origin_x=0.0,
        origin_y=0.0,
        pixel_size_nm=pixel_size,
        source_tile=path.name,
    )
    return list(zip(particles, (float(s) for s in scores)))


def _combine_po_scored(
    scored: list[tuple[Any, float]],
    config: dict[str, Any],
    threshold: float,
) -> pd.DataFrame:
    """Deduplicate each particles-only file on its own (they are masks, not views)."""
    by_tile: dict[str, list[tuple[Any, float]]] = {}
    for particle, score in scored:
        name = _basename(getattr(particle, "source_tile", "") or "particles_only.png")
        by_tile.setdefault(name, []).append((particle, score))
    frames = [
        combine_scored(items, config, threshold, directional=False)
        for items in by_tile.values()
    ]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return combine_scored([], config, threshold, directional=False)
    return pd.concat(frames, ignore_index=True)


def combine_scored(
    scored: list[tuple[Any, float]],
    config: dict[str, Any],
    threshold: float,
    *,
    directional: bool,
) -> pd.DataFrame:
    kept = [particle for particle, score in scored if float(score) >= float(threshold)]
    if directional:
        return measure_and_dedupe(
            kept,
            config,
            size_match_fraction=float(
                cfg_get(config, "measurement.nsew_size_match_fraction", DEFAULT_NSEW_SIZE_MATCH_FRACTION)
            ),
            directional=True,
        )
    return measure_and_dedupe(kept, config)


def _basename(name: object) -> str:
    return Path(str(name)).name


def score_table(
    table: pd.DataFrame,
    labels: pd.DataFrame,
    tile_names: list[str],
    folder_tiles: list[str],
    config: dict[str, Any],
) -> dict[str, float]:
    tagged = tag_table(table)
    pixel_size = float(cfg_get(config, "pixel_size_nm", PIXEL_SIZE_NM))
    merge_nm = DEFAULT_NSEW_MERGE_RADIUS_PX * pixel_size
    snapped = snap_detection_keys_to_labels(
        tagged,
        labels,
        merge_nm,
        float(
            cfg_get(
                config,
                "measurement.nsew_size_match_fraction",
                DEFAULT_NSEW_SIZE_MATCH_FRACTION,
            )
        ),
    )
    floor = undetected_size_floor_nm(snapped, config)
    return last_run_class_counts(
        snapped,
        labels,
        tile_names=tile_names,
        min_size_nm=floor,
        folder_tiles=folder_tiles,
    )


def _scene_labels(labels: pd.DataFrame, folder: str, tiles: list[str]) -> tuple[pd.DataFrame, list[str]]:
    scene = related_scene_tile_names(folder, tiles)
    if labels is None or labels.empty or not scene:
        return labels, list(scene) if scene else list(tiles)
    tile = labels["source_tile"].astype(str).map(_basename)
    return labels.loc[tile.isin(scene)].copy(), list(scene)


def evaluate_tables(
    nsew_table: pd.DataFrame,
    po_table: pd.DataFrame,
    nsew_labels: pd.DataFrame,
    po_labels: pd.DataFrame,
    nsew_scene: list[str],
    po_scene: list[str],
    config: dict[str, Any],
) -> dict[str, dict[str, float]]:
    tables = {
        "NSEW individually": (nsew_table, nsew_labels, list(NSEW_TILES), nsew_scene),
        "2-of-4": (po_table, po_labels, ["particles_only.png"], po_scene),
        "3-of-4": (po_table, po_labels, ["particles_only_3of4.png"], po_scene),
        "4-of-4": (po_table, po_labels, ["particles_only_4of4.png"], po_scene),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (table, labels, tiles, scene) in tables.items():
        out[name] = score_table(table, labels, tiles, scene, config)
    return out


def _flatten_sample_metrics(samples: dict[str, dict[str, float]]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    unhappiness: list[float] = []
    for name, stats in samples.items():
        prefix = {
            "NSEW individually": "nsew",
            "2-of-4": "po2",
            "3-of-4": "po3",
            "4-of-4": "po4",
        }[name]
        row[f"{prefix}_hits"] = int(stats["n_detected"])
        row[f"{prefix}_real"] = int(stats["n_real"])
        row[f"{prefix}_missed"] = int(stats["n_undetected"])
        row[f"{prefix}_precision"] = float(stats["precision"])
        row[f"{prefix}_recall"] = float(stats["recall"])
        row[f"{prefix}_unhappiness"] = float(stats["unhappiness"])
        unhappiness.append(float(stats["unhappiness"]))
    row["mean_unhappiness"] = float(np.mean(unhappiness)) if unhappiness else 0.0
    return row


def _trial_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    nsew_r = float(row.get("nsew_recall") or 0.0)
    target = 1 if nsew_r >= TARGET_NSEW_RECALL else 0
    feasible = 1 if nsew_r >= MIN_NSEW_RECALL else 0
    return (
        target,
        feasible,
        -float(row.get("nsew_unhappiness") or 1e9),
        -float(row.get("mean_unhappiness") or 1e9),
        float(row.get("nsew_precision") or 0.0),
        -int(row.get("nsew_hits") or 0),
    )


def _trial_rank_po2(row: dict[str, Any]) -> tuple[Any, ...]:
    """Minimize 2-of-4 unhappiness; break ties on precision, then recall, then stronger ML."""
    return (
        -float(row.get("po2_unhappiness") or 1e9),
        float(row.get("po2_precision") or 0.0),
        float(row.get("po2_recall") or 0.0),
        float(row.get("ml_threshold") or 0.0),
        -int(row.get("po2_hits") or 0),
    )


def _combo_id(knobs: dict[str, Any]) -> str:
    parts = [f"{key}={knobs[key]}" for key in knobs]
    if len(parts) == 1:
        return "1f:" + parts[0]
    if len(parts) == 2:
        return "2f:" + ",".join(parts)
    return "nf:" + ",".join(parts)


def detect_folder_scored(folder: str, names: tuple[str, ...], config: dict[str, Any]) -> list[tuple[Any, float]]:
    root = resolve_input_dir(folder)
    paths = [root / name for name in names if (root / name).is_file()]
    if not paths:
        return []
    if len(paths) == 1:
        return detect_tile_scored(paths[0], config)
    if cv2 is not None:
        cv2.setNumThreads(1)
    scored: list[tuple[Any, float]] = []
    with ThreadPoolExecutor(max_workers=min(4, len(paths))) as pool:
        futures = {pool.submit(detect_tile_scored, path, config): path for path in paths}
        for future in as_completed(futures):
            scored.extend(future.result())
    return scored


def run_detection_config(
    trial_id: str,
    cfg: dict[str, Any],
    nsew_labels: pd.DataFrame,
    po_labels: pd.DataFrame,
    nsew_scene: list[str],
    po_scene: list[str],
    thresholds: tuple[float, ...],
    knobs: dict[str, Any] | None = None,
    *,
    include_nsew: bool = True,
    po_names: tuple[str, ...] = PO_TILES,
) -> list[dict[str, Any]]:
    print(f"\n=== detect {trial_id} ===", flush=True)
    t0 = time.time()
    nsew_scored = (
        detect_folder_scored(NSEW_INPUT, NSEW_TILES, cfg) if include_nsew else []
    )
    po_scored = detect_folder_scored(PO_INPUT, po_names, cfg)
    detect_s = time.time() - t0
    print(
        f"  candidates NSEW={len(nsew_scored)} PO={len(po_scored)} in {detect_s:.1f}s",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    knobs_json = json.dumps(knobs or {}, sort_keys=True)
    for threshold in thresholds:
        trial_cfg = deepcopy(cfg)
        trial_cfg.setdefault("ml", {})["threshold"] = float(threshold)
        nsew_table = combine_scored(nsew_scored, trial_cfg, threshold, directional=True)
        po_table = _combine_po_scored(po_scored, trial_cfg, threshold)
        samples = evaluate_tables(
            nsew_table, po_table, nsew_labels, po_labels, nsew_scene, po_scene, trial_cfg
        )
        payload = _flatten_sample_metrics(samples)
        payload["trial_id"] = f"{trial_id}|ml={threshold:.2f}"
        payload["detect_id"] = trial_id
        payload["ml_threshold"] = float(threshold)
        payload["seconds"] = round(detect_s, 1)
        payload["nsew_raw"] = int(len(nsew_scored))
        payload["po_raw"] = int(len(po_scored))
        payload["knobs_json"] = knobs_json
        rows.append(payload)
        print(
            f"  ml={threshold:.2f}  NSEW hits={payload['nsew_hits']} "
            f"P={payload['nsew_precision']:.3f} R={payload['nsew_recall']:.3f} "
            f"U={payload['nsew_unhappiness']:.2f}  "
            f"2of4 hits={payload['po2_hits']} P={payload['po2_precision']:.3f} "
            f"R={payload['po2_recall']:.3f} U={payload['po2_unhappiness']:.2f}",
            flush=True,
        )
    return rows


def _results_path(out: Path) -> Path:
    return out / "sweep_results.csv"


def _load_done(out: Path) -> dict[str, dict[str, Any]]:
    path = _results_path(out)
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    return {str(row["trial_id"]): row.to_dict() for _, row in df.iterrows()}


def _write_results(out: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(_results_path(out), index=False)


def _table_rows(best: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = (
        ("NSEW individually", "nsew"),
        ("2-of-4", "po2"),
        ("3-of-4", "po3"),
        ("4-of-4", "po4"),
    )
    rows = []
    for sample, prefix in mapping:
        rows.append(
            {
                "Sample": sample,
                "Hits": int(best[f"{prefix}_hits"]),
                "Real": int(best[f"{prefix}_real"]),
                "Missed": int(best[f"{prefix}_missed"]),
                "P": float(best[f"{prefix}_precision"]),
                "R": float(best[f"{prefix}_recall"]),
                "Unhappiness": float(best[f"{prefix}_unhappiness"]),
            }
        )
    return rows


def _parse_detect_knobs(detect_id: str, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    knobs = dict(baseline or BASELINE_KNOBS)
    if detect_id in {"baseline", "winner"}:
        return knobs
    body = detect_id.split(":", 1)[-1] if ":" in detect_id else ""
    if detect_id.startswith(("2f:", "nf:", "3f:")):
        for part in body.split(","):
            if "=" not in part:
                continue
            name, raw = part.split("=", 1)
            knobs[name] = _coerce_knob(name, raw)
        return knobs
    if detect_id.startswith("1f:"):
        name, raw = detect_id[3:].split("=", 1)
        knobs[name] = _coerce_knob(name, raw)
    return knobs


def _coerce_knob(name: str, raw: str) -> Any:
    if raw == "off" or name in {"contrast", "method"}:
        if name == "denoise_sigma" and raw == "off":
            return "off"
        if name in {"contrast", "method"}:
            return raw
        if raw == "off":
            return "off"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def config_from_detect_id(
    base: dict[str, Any],
    detect_id: str,
    baseline_knobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(base)
    knobs = _parse_detect_knobs(detect_id, baseline_knobs)
    start = baseline_knobs or BASELINE_KNOBS
    for name, value in knobs.items():
        if name == "ml_threshold":
            continue
        if start.get(name) == value and detect_id == "baseline":
            continue
        cfg = apply_knob(cfg, name, value)
    return cfg


def winner_overlay(
    best: dict[str, Any],
    base: dict[str, Any],
    baseline_knobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = best.get("knobs_json")
    cfg = deepcopy(base)
    if isinstance(raw, str) and raw.strip():
        knobs = json.loads(raw)
        for name, value in knobs.items():
            cfg = apply_knob(cfg, name, value)
    else:
        cfg = config_from_detect_id(
            base, str(best.get("detect_id") or "baseline"), baseline_knobs
        )
    cfg.setdefault("ml", {})["threshold"] = float(best["ml_threshold"])
    return overlay_from_config(cfg)


def apply_knobs(cfg: dict[str, Any], knobs: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(cfg)
    for name, value in knobs.items():
        out = apply_knob(out, name, value)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-one-factor", action="store_true")
    parser.add_argument("--ml-only", action="store_true", help="Baseline detection, sweep ML threshold only.")
    parser.add_argument("--skip-two-factor", action="store_true")
    parser.add_argument(
        "--objective",
        choices=("nsew", "po2"),
        default="nsew",
        help="nsew: keep NSEW recall ≥ 95%%. po2: minimize 2-of-4 unhappiness.",
    )
    parser.add_argument(
        "--one-factor-knob",
        default="",
        help="If set, run only this one-factor knob (e.g. blob_threshold).",
    )
    args = parser.parse_args(argv)

    po2 = args.objective == "po2"
    out = resolve_output_dir(PO2_OUTPUT_NAME if po2 else OUTPUT_NAME)
    out.mkdir(parents=True, exist_ok=True)
    labels = LabelStore(_labels_path()).load()
    nsew_labels, nsew_scene = _scene_labels(labels, NSEW_INPUT, list(NSEW_TILES))
    po_labels, po_scene = _scene_labels(labels, PO_INPUT, list(PO_TILES))
    print(
        f"Labels NSEW scene={len(nsew_labels)} tiles={sorted(nsew_scene)} "
        f"objective={args.objective}",
        flush=True,
    )

    done = _load_done(out)
    rows = list(done.values())
    baseline_knobs = dict(PO2_BASELINE_KNOBS if po2 else BASELINE_KNOBS)
    one_factor = PO2_ONE_FACTOR if po2 else ONE_FACTOR
    rank = _trial_rank_po2 if po2 else _trial_rank
    thresholds = PO2_ML_THRESHOLDS if po2 else ML_THRESHOLDS
    include_nsew = not po2
    po_names = PO2_TILES_ONLY if po2 else PO_TILES
    base = nsew_base_config()
    if po2:
        base = apply_knobs(base, baseline_knobs)

    def _record(batch: list[dict[str, Any]]) -> None:
        for payload in batch:
            if payload["trial_id"] in done:
                continue
            rows.append(payload)
            done[payload["trial_id"]] = payload
        _write_results(out, rows)

    def _run(detect_id: str, knobs: dict[str, Any], cfg: dict[str, Any]) -> None:
        _record(
            run_detection_config(
                detect_id,
                cfg,
                nsew_labels,
                po_labels,
                nsew_scene,
                po_scene,
                thresholds,
                knobs=knobs,
                include_nsew=include_nsew,
                po_names=po_names,
            )
        )

    baseline_done = any(str(r.get("detect_id")) == "baseline" for r in rows)
    if not baseline_done:
        _run("baseline", dict(baseline_knobs), deepcopy(base))
    else:
        print("skip baseline detection (already in sweep_results.csv)", flush=True)

    if not args.ml_only and not args.skip_one_factor:
        selected = one_factor
        if args.one_factor_knob:
            selected = [item for item in one_factor if item[0] == args.one_factor_knob]
            if not selected:
                raise ValueError(f"Unknown one-factor knob {args.one_factor_knob!r}")
        for knob, values in selected:
            for value in values:
                knobs = {**baseline_knobs, knob: value}
                detect_id = f"1f:{knob}={value}"
                if any(str(r.get("detect_id")) == detect_id for r in rows):
                    print(f"skip {detect_id}", flush=True)
                    continue
                if baseline_knobs.get(knob) == value:
                    print(f"alias {detect_id} -> baseline", flush=True)
                    continue
                _run(detect_id, knobs, apply_knob(deepcopy(base), knob, value))

    if po2 and not args.ml_only:
        for knobs in PO2_SEED_COMBOS:
            detect_id = _combo_id(knobs)
            if any(str(r.get("detect_id")) == detect_id for r in rows):
                print(f"skip {detect_id}", flush=True)
                continue
            merged = {**baseline_knobs, **knobs}
            _run(detect_id, merged, apply_knobs(base, knobs))

    if (
        not args.ml_only
        and not args.skip_one_factor
        and not args.one_factor_knob
        and not args.skip_two_factor
    ):
        best_by_knob: dict[str, dict[str, Any]] = {}
        for row in rows:
            detect_id = str(row.get("detect_id") or "")
            if not detect_id.startswith("1f:"):
                continue
            name = detect_id[3:].split("=", 1)[0]
            prev = best_by_knob.get(name)
            if prev is None or rank(row) > rank(prev):
                best_by_knob[name] = row
        ranked = sorted(best_by_knob.items(), key=lambda item: rank(item[1]), reverse=True)
        print("\nOne-factor winners:", flush=True)
        for name, row in ranked[:8]:
            print(
                f"  {name}: {row['detect_id']}|ml={float(row['ml_threshold']):.2f} "
                f"2of4 U={float(row['po2_unhappiness']):.2f} "
                f"P={float(row['po2_precision']):.3f} R={float(row['po2_recall']):.3f}",
                flush=True,
            )
        if len(ranked) >= 2:
            k1, _r1 = ranked[0]
            k2, _r2 = ranked[1]
            vals1 = [v for k, v in one_factor if k == k1][0]
            vals2 = [v for k, v in one_factor if k == k2][0]
            print(f"\n2-factor grid: {k1} × {k2}", flush=True)
            for a in vals1:
                for b in vals2:
                    knobs = {k1: a, k2: b}
                    detect_id = _combo_id(knobs)
                    if any(str(r.get("detect_id")) == detect_id for r in rows):
                        print(f"skip {detect_id}", flush=True)
                        continue
                    if knobs[k1] == baseline_knobs.get(k1) and knobs[k2] == baseline_knobs.get(k2):
                        print(f"alias {detect_id} -> baseline", flush=True)
                        continue
                    merged = {**baseline_knobs, **knobs}
                    _run(detect_id, merged, apply_knobs(base, knobs))

    ranked_rows = [r for r in rows if str(r.get("detect_id")) != "winner"]
    if not ranked_rows:
        raise RuntimeError("Sweep produced no rows.")
    best = max(ranked_rows, key=rank)
    overlay = winner_overlay(best, nsew_base_config(), baseline_knobs)
    save_nsew_overlay(overlay)

    if po2:
        print("\nRe-scoring winner on NSEW + all particles-only files…", flush=True)
        winner_cfg = apply_nsew_settings(nsew_base_config(), overlay)
        full = run_detection_config(
            "winner",
            winner_cfg,
            nsew_labels,
            po_labels,
            nsew_scene,
            po_scene,
            (float(best["ml_threshold"]),),
            knobs=json.loads(best.get("knobs_json") or "{}"),
            include_nsew=True,
            po_names=PO_TILES,
        )[0]
        best = full
        for key in [k for k, v in done.items() if str(v.get("detect_id")) == "winner"]:
            done.pop(key)
        rows[:] = [r for r in rows if str(r.get("detect_id")) != "winner"]
        _record([full])

    table = _table_rows(best)
    (out / "winner.json").write_text(
        json.dumps({"best": best, "table": table, "overlay": overlay}, indent=2, default=str),
        encoding="utf-8",
    )
    pd.DataFrame(table).to_csv(out / "winner_table.csv", index=False)
    print("\nWINNER", best["trial_id"], flush=True)
    print(pd.DataFrame(table).to_string(index=False), flush=True)
    print(f"Wrote NSEW standard values to {PACKAGE_ROOT / 'nsew_config.yaml'}", flush=True)
    print(
        f"Apply overlay check: {apply_nsew_settings(load_config(PACKAGE_ROOT / 'config.yaml'))['ml']['threshold']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
