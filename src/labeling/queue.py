"""Build the unlabeled review queue from detection CSVs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import DEFAULT_OUTPUT_DIR, cfg_get, resolve_output_dir
from src.measurement.measurer import (
    DEFAULT_NSEW_MERGE_RADIUS_PX,
    DEFAULT_NSEW_SIZE_MATCH_FRACTION,
    DIRECTION_STEMS,
    Particle,
    deduplicate_nsew,
    is_nsew_family_tile,
)

MIN_SIZE_NM = 10_000.0
RECALL_CSV_NAME = "recall/particles.csv"
FALLBACK_CSV_NAME = "allemaal 08/09/particles.csv"
LAST_RUN_CSV_NAME = RECALL_CSV_NAME
LAST_PIPELINE_POINTER = "last_pipeline.json"

REQUIRED = ("id", "x_global", "y_global", "size", "confidence", "source_tile")
_NSEW_KEY = re.compile(
    r"^(?:NSEW|N|S|E|W|particles_only(?:_\d+of4)?)_(-?\d+)_(-?\d+)$",
    re.IGNORECASE,
)
_SET_NSEW_KEY = re.compile(r"^(v\d+)_NSEW_(-?\d+)_(-?\d+)$", re.IGNORECASE)
_SET_2OF4_KEY = re.compile(r"^(v\d+)_2of4_(-?\d+)_(-?\d+)$", re.IGNORECASE)
_SET_2OF4_STEM = re.compile(r"^(v\d+)_2of4$", re.IGNORECASE)
_SET_DIR_NAME = re.compile(r"v\d+$", re.IGNORECASE)
# Same physical field at two exposures. Other folders keep every vN separate:
# Groundup v4 v3 and v4 are different fields.
EXPOSURE_FIELD_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "Groundup v5 combi": (("v3", "v4"),),
}
_PARTICLES_ONLY_ALIASES = (
    "particles_only",
    "particles_only_2of4",
    "particles_only_3of4",
    "particles_only_4of4",
)


def _set_id_from_tile(source_tile: str) -> str:
    parent = Path(str(source_tile).replace("\\", "/")).parent.name
    if _SET_DIR_NAME.fullmatch(parent):
        return parent
    return ""


def _set_nsew_aliases(set_id: str, x_nm: str, y_nm: str) -> set[str]:
    """One field's N/S/E/W and its 2-of-4 detection share a key. Other fields do not."""
    aliases = {f"{set_id}_NSEW_{x_nm}_{y_nm}", f"{set_id}_2of4_{x_nm}_{y_nm}"}
    aliases.update(f"{set_id}_{stem}_{x_nm}_{y_nm}" for stem in DIRECTION_STEMS)
    return aliases


def detection_key(row: Mapping[str, Any] | pd.Series) -> str:
    """Stable id across runs: tile plus rounded global coordinates.

    N/S/E/W and particles-only versions of the same location share one key
    so a label on any of them applies to the others. A ``v1/N.bmp`` label
    stays on that field (``v1_NSEW_…``) and is not mixed with ``v2``.
    """
    raw = str(row["source_tile"]).replace("\\", "/")
    tile = Path(raw).name
    x_nm = int(round(float(row["x_global"])))
    y_nm = int(round(float(row["y_global"])))
    if is_nsew_family_tile(tile):
        set_id = _set_id_from_tile(raw)
        if set_id:
            return f"{set_id}_NSEW_{x_nm}_{y_nm}"
        return f"NSEW_{x_nm}_{y_nm}"
    return f"{Path(tile).stem}_{x_nm}_{y_nm}"


def nsew_key_aliases(key: str) -> set[str]:
    """All keys that mean the same N/S/E/W / particles-only location."""
    text = str(key)
    set_match = _SET_NSEW_KEY.fullmatch(text) or _SET_2OF4_KEY.fullmatch(text)
    if set_match is not None:
        return _set_nsew_aliases(set_match.group(1), set_match.group(2), set_match.group(3))
    match = _NSEW_KEY.fullmatch(text)
    if match is None:
        return {text}
    x_nm, y_nm = match.group(1), match.group(2)
    aliases = {f"NSEW_{x_nm}_{y_nm}"}
    aliases.update(f"{stem}_{x_nm}_{y_nm}" for stem in DIRECTION_STEMS)
    aliases.update(f"{stem}_{x_nm}_{y_nm}" for stem in _PARTICLES_ONLY_ALIASES)
    return aliases


def expand_labeled_keys(keys: Iterable[str]) -> set[str]:
    """Union of each labeled key and its N/S/E/W aliases."""
    expanded: set[str] = set()
    for key in keys:
        expanded.update(nsew_key_aliases(str(key)))
    return expanded


def snap_detection_keys_to_labels(
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    merge_radius_nm: float,
    size_match_fraction: float = DEFAULT_NSEW_SIZE_MATCH_FRACTION,
) -> pd.DataFrame:
    """Reuse an NSEW / particles-only label key when a hit is at the same place.

    Detector centroids rarely round to the exact labelled nanometre, so exact
    ``NSEW_x_y`` equality misses most transfers. Nearest labelled location
    within the N/S/E/W merge radius is treated as the same particle.
    """
    if detections is None or detections.empty:
        return detections if detections is not None else pd.DataFrame()
    tagged = tagged_if_needed(detections).copy()
    if labels is None or labels.empty or "key" not in labels.columns:
        return tagged
    if "source_tile" not in tagged.columns or "x_global" not in tagged.columns:
        return tagged
    lab = labels.copy()
    if "source_tile" in lab.columns:
        lab = lab.loc[lab["source_tile"].astype(str).map(is_nsew_family_tile)]
    if lab.empty or "x_global" not in lab.columns:
        return tagged
    det_mask = tagged["source_tile"].astype(str).map(is_nsew_family_tile)
    if not det_mask.any():
        return tagged
    lab_xy = lab[["x_global", "y_global"]].astype(float).to_numpy()
    lab_size = (
        lab["size"].astype(float).to_numpy()
        if "size" in lab.columns
        else np.zeros(len(lab))
    )
    lab_keys: list[str] = []
    for record in lab.to_dict(orient="records"):
        lab_keys.append(detection_key(record))
    tree = cKDTree(lab_xy)
    det_idx = np.flatnonzero(det_mask.to_numpy())
    det_xy = tagged.loc[det_mask, ["x_global", "y_global"]].astype(float).to_numpy()
    det_size = (
        tagged.loc[det_mask, "size"].astype(float).to_numpy()
        if "size" in tagged.columns
        else np.zeros(len(det_idx))
    )
    search_r = float(merge_radius_nm)
    if size_match_fraction > 0 and len(lab_size):
        search_r = max(
            search_r,
            float(size_match_fraction) * float(np.nanmax(np.concatenate([det_size, lab_size]))),
        )
    dist, nn = tree.query(det_xy, k=1, distance_upper_bound=search_r + 1.0)
    keys = tagged["key"].astype(str).to_numpy()
    n_lab = len(lab_keys)
    for i, row_i in enumerate(det_idx):
        j = int(nn[i])
        if j < 0 or j >= n_lab or not np.isfinite(dist[i]):
            continue
        limit = float(merge_radius_nm)
        if size_match_fraction > 0:
            limit = max(
                limit,
                float(size_match_fraction) * max(float(det_size[i]), float(lab_size[j])),
            )
        if float(dist[i]) <= limit:
            keys[row_i] = lab_keys[j]
    tagged["key"] = keys
    return tagged


def label_map_with_aliases(labels: pd.DataFrame | None) -> dict[str, str]:
    """Map detection keys (and N/S/E/W aliases) onto stored labels."""
    mapping: dict[str, str] = {}
    if labels is None or labels.empty or "key" not in labels.columns:
        return mapping
    for record in labels.to_dict(orient="records"):
        name = str(record.get("label", "") or "")
        if not name:
            continue
        for alias in nsew_key_aliases(str(record["key"])):
            mapping[alias] = name
        tile = record.get("source_tile", "")
        if tile not in (None, "") and is_nsew_family_tile(str(tile)):
            mapping[detection_key(record)] = name
            for alias in nsew_key_aliases(detection_key(record)):
                mapping[alias] = name
    return mapping


def last_run_csv(project_root: str | Path, config: Mapping[str, Any] | None = None) -> Path:
    """CSV used as the default labeling source (under ``Outputs``).

    Prefers a high-recall run (``recall/particles.csv``) so previously labeled
    keys are skipped and only new loosened-gate hits need review. Falls back
    to the last classical run when the recall table is missing.
    """
    names: list[str] = []
    if config is not None:
        custom = cfg_get(dict(config), "labeling.source_csv", None)
        if custom not in (None, ""):
            names.append(str(custom))
    for name in (RECALL_CSV_NAME, FALLBACK_CSV_NAME):
        if name not in names:
            names.append(name)
    for name in names:
        for folder in (DEFAULT_OUTPUT_DIR, Path(project_root)):
            path = folder / name
            if path.is_file():
                return path
    return DEFAULT_OUTPUT_DIR / names[0]


def pipeline_particles_csv(config: Mapping[str, Any] | None = None) -> Path:
    """``particles.csv`` written by Detection → Run pipeline."""
    output_dir = None if config is None else cfg_get(dict(config), "output_dir", None)
    return resolve_output_dir(output_dir) / "particles.csv"


def write_last_pipeline(
    config: Mapping[str, Any],
    csv_path: str | Path,
    pointer_dir: str | Path | None = None,
) -> Path:
    """Remember which CSV + folder the latest pipeline run used."""
    destination = Path(csv_path).resolve()
    payload = {
        "csv": str(destination),
        "input_dir": str(cfg_get(dict(config), "input_dir", "") or ""),
        "output_dir": str(cfg_get(dict(config), "output_dir", "") or ""),
        "pixel_size_nm": cfg_get(dict(config), "pixel_size_nm", 960.0),
        "overlap_fraction": cfg_get(dict(config), "overlap_fraction", 0.0),
        "filename_pattern": cfg_get(dict(config), "filename_pattern", None),
    }
    folder = Path(pointer_dir) if pointer_dir is not None else DEFAULT_OUTPUT_DIR
    pointer = folder / LAST_PIPELINE_POINTER
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return pointer


def load_last_pipeline(
    config: Mapping[str, Any] | None = None,
    pointer_dir: str | Path | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """CSV and run settings for the latest pipeline write.

    Prefers ``last_pipeline.json``, then the current ``output_dir``
    ``particles.csv``. Does not use the labeling recall table.
    """
    run_config: dict[str, Any] = dict(config or {})
    folder = Path(pointer_dir) if pointer_dir is not None else DEFAULT_OUTPUT_DIR
    pointer = folder / LAST_PIPELINE_POINTER
    if pointer.is_file():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            payload = {}
        if isinstance(payload, dict):
            csv_text = payload.get("csv")
            csv_path = Path(str(csv_text)) if csv_text else None
            for key in (
                "input_dir",
                "output_dir",
                "pixel_size_nm",
                "overlap_fraction",
                "filename_pattern",
            ):
                value = payload.get(key)
                if value not in (None, ""):
                    run_config[key] = value
            if csv_path is not None and csv_path.is_file():
                return csv_path, run_config
    csv_path = pipeline_particles_csv(run_config)
    if csv_path.is_file():
        return csv_path, run_config
    return None, run_config


def load_particles_csv(path: str | Path) -> pd.DataFrame:
    """Read a detector table and tag it with ``source_csv`` and ``key``."""
    destination = Path(path)
    df = pd.read_csv(destination)
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"Results table missing columns: {missing}")
    tagged = df.copy()
    tagged["source_csv"] = str(destination)
    tagged["key"] = [_row_key(row) for _, row in tagged.iterrows()]
    return tagged


def filter_min_size(df: pd.DataFrame, min_size_nm: float = MIN_SIZE_NM) -> pd.DataFrame:
    """Keep detections at least ``min_size_nm`` (default 10 µm)."""
    if df.empty:
        return df.copy()
    return df.loc[df["size"].astype(float) >= float(min_size_nm)].reset_index(drop=True)


def tag_table(df: pd.DataFrame, source_csv: str | Path | None = None) -> pd.DataFrame:
    """Add ``key`` / ``source_csv`` to a pipeline table without mutating it."""
    tagged = df.copy()
    if "source_csv" not in tagged.columns:
        tagged["source_csv"] = "" if source_csv is None else str(source_csv)
    elif source_csv is not None:
        tagged["source_csv"] = tagged["source_csv"].where(
            tagged["source_csv"].astype(str).str.len() > 0, str(source_csv)
        )
    tagged["key"] = [_row_key(row) for _, row in tagged.iterrows()]
    return tagged


def merge_detection_tables(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate tables. Later rows update the same key; first-seen order is kept."""
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for frame in tables:
        if frame is None or frame.empty:
            continue
        tagged = tagged_if_needed(frame)
        for record in tagged.to_dict(orient="records"):
            key = _row_key(record)
            record["key"] = key
            if key not in merged:
                order.append(key)
            merged[key] = record
    if not order:
        return pd.DataFrame()
    return pd.DataFrame([merged[key] for key in order])


def _exposure_set_id(source_tile: str) -> str:
    stem = Path(str(source_tile).replace("\\", "/")).stem
    match = _SET_2OF4_STEM.fullmatch(stem)
    if match is None:
        return ""
    return match.group(1).lower()


def _exposure_folder_name(folder: str) -> str:
    return Path(str(folder).replace("\\", "/")).name


def share_exposure_keys(
    detections: pd.DataFrame | None,
    folder: str,
    merge_radius_nm: float,
    size_match_fraction: float = DEFAULT_NSEW_SIZE_MATCH_FRACTION,
) -> pd.DataFrame:
    """Give nearby hits on two exposures of one field the same label key.

    Both rows stay, so each exposure still draws the circle. ``exposure_keep``
    is false on the partner row; the queue keeps the anchor only.
    """
    if detections is None or detections.empty:
        return detections if detections is not None else pd.DataFrame()
    pairs = EXPOSURE_FIELD_PAIRS.get(_exposure_folder_name(folder))
    if not pairs or "source_tile" not in detections.columns:
        return detections
    tagged = tagged_if_needed(detections).copy()
    if "exposure_keep" not in tagged.columns:
        tagged["exposure_keep"] = True
    if "exposure_with" not in tagged.columns:
        tagged["exposure_with"] = ""
    set_ids = tagged["source_tile"].astype(str).map(_exposure_set_id)
    xy = tagged[["x_global", "y_global"]].astype(float).to_numpy()
    sizes = (
        tagged["size"].astype(float).to_numpy()
        if "size" in tagged.columns
        else np.zeros(len(tagged))
    )
    keys = tagged["key"].astype(str).to_numpy()
    keep = tagged["exposure_keep"].fillna(True).astype(bool).to_numpy()
    with_set = tagged["exposure_with"].astype(str).to_numpy()
    for left_id, right_id in pairs:
        left = np.flatnonzero((set_ids == left_id.lower()).to_numpy())
        right = np.flatnonzero((set_ids == right_id.lower()).to_numpy())
        if len(left) == 0 or len(right) == 0:
            continue
        left_nn_dist, left_nn = cKDTree(xy[right]).query(xy[left], k=1)
        right_nn = cKDTree(xy[left]).query(xy[right], k=1)[1]
        for local_i, local_j in enumerate(np.atleast_1d(left_nn)):
            i = int(left[int(local_i)])
            j = int(right[int(local_j)])
            if int(np.atleast_1d(right_nn)[int(local_j)]) != int(local_i):
                continue
            limit = float(merge_radius_nm)
            if size_match_fraction > 0:
                limit = max(limit, float(size_match_fraction) * max(float(sizes[i]), float(sizes[j])))
            if float(np.atleast_1d(left_nn_dist)[int(local_i)]) > limit:
                continue
            keys[j] = keys[i]
            keep[j] = False
            with_set[i] = right_id
            with_set[j] = left_id
    tagged["key"] = keys
    tagged["exposure_keep"] = keep
    tagged["exposure_with"] = with_set
    return tagged


def _drop_exposure_partners(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "exposure_keep" not in frame.columns:
        return frame
    keep = frame["exposure_keep"].map(lambda value: value not in (False, 0, "False", "false"))
    return frame.loc[keep.fillna(True)].reset_index(drop=True)


def unlabeled_queue(
    tables: Iterable[pd.DataFrame],
    labeled_keys: Iterable[str],
    min_size_nm: float = MIN_SIZE_NM,
    nsew_merge_radius_nm: float | None = None,
    nsew_size_match_fraction: float = DEFAULT_NSEW_SIZE_MATCH_FRACTION,
    labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detections at or above the size floor that are not yet in the label store."""
    tables = [_drop_exposure_partners(frame) for frame in tables]
    merged = merge_detection_tables(tables)
    if merged.empty:
        return merged
    radius = (
        DEFAULT_NSEW_MERGE_RADIUS_PX * 960.0
        if nsew_merge_radius_nm is None
        else float(nsew_merge_radius_nm)
    )
    merged = collapse_nsew_detections(
        tagged_if_needed(merged),
        merge_radius_nm=radius,
        size_match_fraction=nsew_size_match_fraction,
    )
    if labels is not None and not labels.empty:
        merged = snap_detection_keys_to_labels(
            merged, labels, radius, nsew_size_match_fraction
        )
    above = filter_min_size(tagged_if_needed(merged), min_size_nm)
    skip = expand_labeled_keys(str(key) for key in labeled_keys)
    if not skip:
        return above.reset_index(drop=True)
    return above.loc[~above["key"].astype(str).isin(skip)].reset_index(drop=True)


def tagged_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if "key" in df.columns:
        return df
    return tag_table(df)


def _row_key(row: Mapping[str, Any] | pd.Series) -> str:
    tile = ""
    if "source_tile" in row:
        tile = str(row["source_tile"] or "")
    if is_nsew_family_tile(tile):
        return detection_key(row)
    existing = None
    if "key" in row:
        existing = row["key"]
    if existing is None or (isinstance(existing, float) and pd.isna(existing)):
        return detection_key(row)
    text = str(existing)
    if text in ("", "nan"):
        return detection_key(row)
    return text


def collapse_nsew_detections(
    df: pd.DataFrame,
    merge_radius_nm: float,
    size_match_fraction: float = DEFAULT_NSEW_SIZE_MATCH_FRACTION,
) -> pd.DataFrame:
    """Keep one row per N/W/E/S / particles-only location so Tinder does not ask four times."""
    if df is None or df.empty or "source_tile" not in df.columns:
        return df
    tagged = tagged_if_needed(df)
    direction = tagged["source_tile"].astype(str).map(is_nsew_family_tile)
    if not direction.any():
        return tagged.reset_index(drop=True)
    rest = tagged.loc[~direction].copy()
    dirs = tagged.loc[direction].copy().reset_index(drop=True)
    particles = [
        Particle(
            id=_particle_id(record.get("id")),
            x_global=float(record["x_global"]),
            y_global=float(record["y_global"]),
            size=float(record["size"]),
            confidence=float(record.get("confidence") or 0.0),
            source_tile=str(record["source_tile"]),
        )
        for record in dirs.to_dict(orient="records")
    ]
    merged = deduplicate_nsew(
        particles,
        merge_radius=float(merge_radius_nm),
        size_aggregation="max",
        size_match_fraction=float(size_match_fraction),
    )
    kept: list[dict[str, Any]] = []
    taken = np.zeros(len(dirs), dtype=bool)
    x_vals = dirs["x_global"].astype(float).to_numpy()
    y_vals = dirs["y_global"].astype(float).to_numpy()
    tiles = dirs["source_tile"].astype(str).to_numpy()
    for particle in merged:
        distances = np.hypot(x_vals - particle.x_global, y_vals - particle.y_global)
        distances[(tiles != particle.source_tile) | taken] = np.inf
        index = int(np.argmin(distances))
        if not np.isfinite(distances[index]):
            record = {
                "id": particle.id,
                "x_global": particle.x_global,
                "y_global": particle.y_global,
                "size": particle.size,
                "confidence": particle.confidence,
                "source_tile": particle.source_tile,
            }
        else:
            taken[index] = True
            record = dirs.iloc[index].to_dict()
            record["size"] = particle.size
            record["confidence"] = particle.confidence
            record["x_global"] = particle.x_global
            record["y_global"] = particle.y_global
        record["nsew_count"] = particle.nsew_count
        record["nsew_dirs"] = particle.nsew_dirs
        record["key"] = detection_key(record)
        kept.append(record)
    collapsed = pd.DataFrame(kept)
    if rest.empty:
        return collapsed.reset_index(drop=True) if not collapsed.empty else rest
    if collapsed.empty:
        return rest.reset_index(drop=True)
    return pd.concat([rest, collapsed], ignore_index=True)


def _particle_id(value: Any) -> int:
    if value in (None, "") or (isinstance(value, float) and pd.isna(value)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
