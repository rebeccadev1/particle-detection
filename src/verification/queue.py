"""Build the unlabeled review queue from detection CSVs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.overall.config import DEFAULT_OUTPUT_DIR, cfg_get, resolve_output_dir
from src.overall.io.naming import (
    DIRECTION_STEMS,
    LIGHTING_ROLES,
    combi_key_family,
    is_combi_tile,
    is_nsew_family_tile,
    is_submap_lighting_tile,
    nsew_merge_stem,
    submap_id,
)
from src.analyse.measurement.measurer import (
    DEFAULT_NSEW_MERGE_RADIUS_PX,
    DEFAULT_NSEW_SIZE_MATCH_FRACTION,
    Particle,
    deduplicate_nsew,
)

MIN_SIZE_NM = 10_000.0
RECALL_CSV_NAME = "recall/particles.csv"
LAST_PIPELINE_POINTER = "last_pipeline.json"
LABEL_QUEUE_CLEARED = "label_queue_cleared.json"

REQUIRED = ("id", "x_global", "y_global", "size", "confidence", "source_tile")
_NSEW_KEY = re.compile(
    r"^(?:NSEW|N|S|E|W|particles_only(?:_\d+of4)?)_(-?\d+)_(-?\d+)$",
    re.IGNORECASE,
)
_SET_NSEW_KEY = re.compile(r"^(v\d+)_NSEW_(-?\d+)_(-?\d+)$", re.IGNORECASE)
_SET_2OF4_KEY = re.compile(r"^(v\d+)_2of4_(-?\d+)_(-?\d+)$", re.IGNORECASE)
_SET_LIGHT_KEY = re.compile(
    r"^(v\d+)_(?:LIGHT|quad[0-3]|all)_(-?\d+)_(-?\d+)$",
    re.IGNORECASE,
)
_LIGHT_KEY = re.compile(r"^(?:LIGHT|quad[0-3]|all)_(-?\d+)_(-?\d+)$", re.IGNORECASE)
_SET_2OF4_STEM = re.compile(r"^(v\d+)_2of4$", re.IGNORECASE)
_SET_DIR_NAME = re.compile(r"v\d+$", re.IGNORECASE)
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


def _light_aliases(set_id: str, x_nm: str, y_nm: str) -> set[str]:
    """One submap's four quads and its uniform ``all`` shot share a key."""
    prefix = f"{set_id}_" if set_id else ""
    roles = ("LIGHT", *LIGHTING_ROLES)
    return {f"{prefix}{role}_{x_nm}_{y_nm}" for role in roles}


def detection_key(row: Mapping[str, Any] | pd.Series) -> str:
    """Stable id across runs: tile plus rounded global coordinates.

    N/S/E/W and particles-only versions of the same location share one key
    so a label on any of them applies to the others. A ``v1/N.bmp`` label
    stays on that field (``v1_NSEW_…``) and is not mixed with ``v2``.

    Groundup submaps do the same for ``quad0``–``quad3`` and ``all`` inside
    one ``vN`` folder. ``v7`` is not mixed with ``v8``.
    """
    return _key_from_tile_xy(row["source_tile"], row["x_global"], row["y_global"])


def _key_from_tile_xy(source_tile: object, x_global: object, y_global: object) -> str:
    raw = str(source_tile).replace("\\", "/")
    tile = Path(raw).name
    x_nm = int(round(float(x_global)))
    y_nm = int(round(float(y_global)))
    family = combi_key_family(raw)
    if is_submap_lighting_tile(tile) or family == "light":
        set_id = submap_id(raw)
        if set_id:
            return f"{set_id}_LIGHT_{x_nm}_{y_nm}"
        return f"LIGHT_{x_nm}_{y_nm}"
    if is_nsew_family_tile(tile) or family == "nsew":
        set_id = _set_id_from_tile(raw)
        if set_id:
            return f"{set_id}_NSEW_{x_nm}_{y_nm}"
        return f"NSEW_{x_nm}_{y_nm}"
    return f"{Path(tile).stem}_{x_nm}_{y_nm}"


def label_covers_uniform(key: object) -> bool:
    """True when this key is the shared quad, combi, and uniform identity.

    Letter N/S/E/W keys do not cover the uniform ``all`` shot.
    """
    text = str(key or "")
    return (
        _SET_LIGHT_KEY.fullmatch(text) is not None
        or _LIGHT_KEY.fullmatch(text) is not None
    )


def nsew_key_aliases(key: str) -> set[str]:
    """All keys that mean the same N/S/E/W / particles-only / submap lighting location."""
    text = str(key)
    light_match = _SET_LIGHT_KEY.fullmatch(text)
    if light_match is not None:
        return _light_aliases(light_match.group(1), light_match.group(2), light_match.group(3))
    bare_light = _LIGHT_KEY.fullmatch(text)
    if bare_light is not None:
        return _light_aliases("", bare_light.group(1), bare_light.group(2))
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


def _snap_family(
    tagged: pd.DataFrame,
    labels: pd.DataFrame,
    det_mask: pd.Series,
    merge_radius_nm: float,
    size_match_fraction: float,
) -> pd.DataFrame:
    """Point family hits at the nearest labelled key inside the merge radius."""
    if labels.empty or "x_global" not in labels.columns or not bool(det_mask.any()):
        return tagged
    lab_xy = labels[["x_global", "y_global"]].astype(float).to_numpy()
    if len(lab_xy) == 0:
        return tagged
    lab_size = (
        labels["size"].astype(float).to_numpy()
        if "size" in labels.columns
        else np.zeros(len(labels))
    )
    lab_keys = [detection_key(record) for record in labels.to_dict(orient="records")]
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
    tagged = tagged.copy()
    tagged["key"] = keys
    return tagged


def _uses_light_keys(source_tile: str) -> bool:
    """Quads, the uniform shot, and a combi written beside them share LIGHT keys."""
    return is_submap_lighting_tile(source_tile) or combi_key_family(source_tile) == "light"


def _uses_nsew_keys(source_tile: str) -> bool:
    """Letter directions and a combi written beside them share NSEW keys."""
    return is_nsew_family_tile(source_tile) or combi_key_family(source_tile) == "nsew"


def snap_detection_keys_to_labels(
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    merge_radius_nm: float,
    size_match_fraction: float = DEFAULT_NSEW_SIZE_MATCH_FRACTION,
) -> pd.DataFrame:
    """Reuse a shared label key when a hit is at the same place.

    Detector centroids rarely round to the exact labelled nanometre, so exact
    key equality misses most transfers. Nearest labelled location within the
    merge radius is treated as the same particle. Submap quads and the uniform
    ``all`` shot snap only inside that ``vN`` folder.
    """
    if detections is None or detections.empty:
        return detections if detections is not None else pd.DataFrame()
    tagged = tagged_if_needed(detections).copy()
    if labels is None or labels.empty or "key" not in labels.columns:
        return tagged
    if "source_tile" not in tagged.columns or "x_global" not in tagged.columns:
        return tagged
    if "x_global" not in labels.columns:
        return tagged
    nsew_labels = labels
    if "source_tile" in labels.columns:
        nsew_labels = labels.loc[labels["source_tile"].astype(str).map(_uses_nsew_keys)]
    tagged = _snap_family(
        tagged,
        nsew_labels,
        tagged["source_tile"].astype(str).map(_uses_nsew_keys),
        merge_radius_nm,
        size_match_fraction,
    )
    if "source_tile" not in labels.columns:
        return tagged
    light_labels = labels.loc[labels["source_tile"].astype(str).map(_uses_light_keys)]
    light_hits = tagged["source_tile"].astype(str).map(_uses_light_keys)
    if light_labels.empty or not bool(light_hits.any()):
        return tagged
    set_ids = set(light_labels["source_tile"].astype(str).map(submap_id))
    set_ids.update(tagged.loc[light_hits, "source_tile"].astype(str).map(submap_id))
    for set_id in set_ids:
        lab = light_labels.loc[light_labels["source_tile"].astype(str).map(submap_id) == set_id]
        mask = light_hits & (tagged["source_tile"].astype(str).map(submap_id) == set_id)
        tagged = _snap_family(tagged, lab, mask, merge_radius_nm, size_match_fraction)
        light_hits = tagged["source_tile"].astype(str).map(_uses_light_keys)
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
        if tile not in (None, "") and (
            is_nsew_family_tile(str(tile))
            or is_submap_lighting_tile(str(tile))
            or bool(combi_key_family(str(tile)))
        ):
            mapping[detection_key(record)] = name
            for alias in nsew_key_aliases(detection_key(record)):
                mapping[alias] = name
    return mapping


def pipeline_particles_csv(config: Mapping[str, Any] | None = None) -> Path:
    """``particles.csv`` written by Detection → Run pipeline."""
    output_dir = None if config is None else cfg_get(dict(config), "output_dir", None)
    return resolve_output_dir(output_dir) / "particles.csv"


def write_last_pipeline(
    config: Mapping[str, Any],
    csv_path: str | Path,
    pointer_dir: str | Path | None = None,
) -> Path:
    """Save this run's settings as ``last_pipeline.json`` inside the run folder."""
    from src.overall.io.results_writer import write_run_settings

    del pointer_dir
    destination = Path(csv_path).resolve()
    output_dir = cfg_get(dict(config), "output_dir", None) or destination.parent
    run_folder = resolve_output_dir(output_dir)
    return write_run_settings(
        config, run_folder / LAST_PIPELINE_POINTER, csv_path=destination
    )


def use_last_pipeline_queue(
    session_source: str | None,
    pipeline_csv: str | Path | None,
) -> bool:
    """Use the latest pipeline CSV only after Load last run.

    Opening Label used to fall back to ``allemaal 08/09/particles.csv``, the
    old R3 review set, and then to whatever ``last_pipeline.json`` pointed at.
    Neither loads until the session asks for the last run.
    """
    if pipeline_csv is None or not Path(pipeline_csv).is_file():
        return False
    return session_source == "last_pipeline"


def _label_queue_cleared_path(pointer_dir: str | Path | None = None) -> Path:
    folder = Path(pointer_dir) if pointer_dir is not None else DEFAULT_OUTPUT_DIR
    return folder / LABEL_QUEUE_CLEARED


def _csv_stamp(csv_path: Path) -> dict[str, Any]:
    stat = csv_path.stat()
    return {
        "csv": str(csv_path.resolve()),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def label_queue_is_cleared(
    csv_path: str | Path | None,
    pointer_dir: str | Path | None = None,
) -> bool:
    """True when Empty queue dropped this exact results file."""
    if csv_path is None:
        return False
    path = Path(csv_path)
    if not path.is_file():
        return False
    pointer = _label_queue_cleared_path(pointer_dir)
    if not pointer.is_file():
        return False
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    stamp = _csv_stamp(path)
    try:
        saved_mtime = int(payload.get("mtime_ns"))
        saved_size = int(payload.get("size"))
    except (TypeError, ValueError):
        return False
    return payload.get("csv") == stamp["csv"] and saved_mtime == stamp["mtime_ns"] and saved_size == stamp["size"]


def clear_label_queue(
    csv_path: str | Path,
    pointer_dir: str | Path | None = None,
) -> Path:
    """Remember that this results file's unlabeled hits should not be shown."""
    path = Path(csv_path)
    pointer = _label_queue_cleared_path(pointer_dir)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps(_csv_stamp(path), indent=2), encoding="utf-8")
    return pointer


def restore_label_queue(pointer_dir: str | Path | None = None) -> None:
    """Put a cleared Label queue back. A rewritten results file does this on its own."""
    _label_queue_cleared_path(pointer_dir).unlink(missing_ok=True)


def _newest_run_settings(folder: Path) -> Path | None:
    """``last_pipeline.json`` in the run folder that was written most recently."""
    if not folder.is_dir():
        return None
    newest: Path | None = None
    newest_mtime = -1
    try:
        children = list(folder.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_dir():
            continue
        candidate = child / LAST_PIPELINE_POINTER
        try:
            if not candidate.is_file():
                continue
            mtime = candidate.stat().st_mtime_ns
        except OSError:
            continue
        if mtime >= newest_mtime:
            newest = candidate
            newest_mtime = mtime
    return newest


def load_last_pipeline(
    config: Mapping[str, Any] | None = None,
    pointer_dir: str | Path | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """CSV and run settings for the latest pipeline write.

    Reads ``last_pipeline.json`` inside the newest run folder, then the
    current ``output_dir`` ``particles.csv``. Does not use the labeling
    recall table.
    """
    run_config: dict[str, Any] = dict(config or {})
    folder = Path(pointer_dir) if pointer_dir is not None else DEFAULT_OUTPUT_DIR
    pointer = _newest_run_settings(folder)
    if pointer is not None and pointer.is_file():
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
    for source, dest in (
        ("x_global_um", "x_global"),
        ("y_global_um", "y_global"),
        ("size_um", "size"),
    ):
        if dest not in df.columns and source in df.columns:
            df[dest] = df[source].astype(float) * 1000.0
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"Results table missing columns: {missing}")
    tagged = df.copy()
    tagged["source_csv"] = str(destination)
    tagged["key"] = _row_keys(tagged)
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
    tagged["key"] = _row_keys(tagged)
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
    keys = tagged["key"].astype(str).to_numpy(copy=True)
    keep = tagged["exposure_keep"].fillna(True).astype(bool).to_numpy(copy=True)
    with_set = tagged["exposure_with"].astype(str).to_numpy(copy=True)
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
        labeled_keys=labeled_keys,
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
    if is_nsew_family_tile(tile) or is_submap_lighting_tile(tile):
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


def _row_keys(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    tiles = frame["source_tile"].tolist() if "source_tile" in frame.columns else [""] * len(frame)
    xs = frame["x_global"].tolist() if "x_global" in frame.columns else [0] * len(frame)
    ys = frame["y_global"].tolist() if "y_global" in frame.columns else [0] * len(frame)
    computed = [
        _key_from_tile_xy(tile, x_global, y_global)
        for tile, x_global, y_global in zip(tiles, xs, ys)
    ]
    if "key" not in frame.columns:
        return computed
    keep: list[str] = []
    for tile, value, fallback in zip(tiles, frame["key"].tolist(), computed):
        text = str(tile or "")
        if is_nsew_family_tile(text) or is_submap_lighting_tile(text):
            keep.append(fallback)
            continue
        if value is None or (isinstance(value, float) and pd.isna(value)):
            keep.append(fallback)
            continue
        existing = str(value)
        keep.append(fallback if existing in ("", "nan") else existing)
    return keep


def _blank_text(value: object) -> str:
    """``str(value or "")``. NaN is truthy, so it stays the text ``nan``."""
    if value is None:
        return ""
    try:
        if not value:
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _join_stems(stems: set[str]) -> str:
    names = sorted(
        stems,
        key=lambda name: (0, DIRECTION_STEMS.index(name)) if name in DIRECTION_STEMS else (1, name),
    )
    return ",".join(names)


class _FamilyScan:
    """One collapse family, indexed by picture so each merged hit scans its own rows.

    Stems and label keys are computed once. The old loop rebuilt both from the
    whole table for every merged particle.
    """

    def __init__(self, dirs: pd.DataFrame) -> None:
        self.x = dirs["x_global"].to_numpy(dtype=float)
        self.y = dirs["y_global"].to_numpy(dtype=float)
        self.size = dirs["size"].to_numpy(dtype=float)
        tiles = dirs["source_tile"].astype(str).tolist()
        self.stem = [nsew_merge_stem(tile) for tile in tiles]
        if "key" in dirs.columns:
            self.key_text = [_blank_text(value) for value in dirs["key"].tolist()]
        else:
            self.key_text = [""] * len(dirs)
        if "nsew_dirs" in dirs.columns:
            self.dir_text: list[str] | None = [
                _blank_text(value) for value in dirs["nsew_dirs"].tolist()
            ]
        else:
            self.dir_text = None
        buckets: dict[str, list[int]] = {}
        for index, stem in enumerate(self.stem):
            if stem is None:
                continue
            buckets.setdefault(stem, []).append(index)
        self.by_stem = {
            stem: np.asarray(indexes, dtype=np.intp) for stem, indexes in buckets.items()
        }

    def indexes_for(self, stems: set[str]) -> np.ndarray:
        parts = [self.by_stem[stem] for stem in stems if stem in self.by_stem]
        if not parts:
            return np.empty(0, dtype=np.intp)
        if len(parts) == 1:
            return parts[0]
        return np.sort(np.concatenate(parts))


def _row_distance(scan: _FamilyScan, index: int, particle: Particle) -> float:
    return float(
        np.hypot(
            float(scan.x[index]) - particle.x_global,
            float(scan.y[index]) - particle.y_global,
        )
    )


def _within_merge(
    dist: float,
    particle: Particle,
    row_size: float,
    merge_radius_nm: float,
    size_match_fraction: float,
) -> bool:
    limit = float(merge_radius_nm)
    if size_match_fraction > 0:
        limit = max(limit, float(size_match_fraction) * max(particle.size, float(row_size)))
    return not (dist > limit)


def _dirs_kept_on_cluster(
    scan: _FamilyScan,
    particle: Particle,
    merge_radius_nm: float,
    size_match_fraction: float,
) -> str:
    """Pictures this merged hit was found on, including ones already recorded.

    A second pass must not drop ``ALL`` from a row whose saved file is the combi.
    """
    stems: set[str] = set()
    for part in str(particle.nsew_dirs or "").split(","):
        text = part.strip().upper()
        if text and text != "NAN":
            stems.add(text)
    closest: dict[str, tuple[float, str]] = {}
    if scan.dir_text is not None:
        for stem in stems:
            indexes = scan.by_stem.get(stem)
            if indexes is None:
                continue
            for index in indexes:
                dist = _row_distance(scan, int(index), particle)
                if not _within_merge(
                    dist, particle, float(scan.size[index]), merge_radius_nm, size_match_fraction
                ):
                    continue
                previous = closest.get(stem)
                if previous is None or dist < previous[0]:
                    closest[stem] = (dist, scan.dir_text[int(index)])
    for _, text in closest.values():
        for part in text.split(","):
            name = part.strip().upper()
            if name and name != "NAN":
                stems.add(name)
    return _join_stems(stems)


def _labeled_key_on_cluster(
    scan: _FamilyScan,
    particle: Particle,
    merge_radius_nm: float,
    size_match_fraction: float,
    expanded: set[str],
) -> str:
    """Key to keep on a merged combi/uniform hit.

    A label saved on either picture stays with the shared row. The key already
    on the kept picture is used when neither picture was labeled.
    """
    stems = {
        part.strip().upper()
        for part in str(particle.nsew_dirs or "").split(",")
        if part.strip()
    }
    seed_stem = nsew_merge_stem(particle.source_tile)
    fallback = ""
    fallback_dist = float("inf")
    for index in scan.indexes_for(stems):
        row = int(index)
        dist = _row_distance(scan, row, particle)
        if not _within_merge(
            dist, particle, float(scan.size[row]), merge_radius_nm, size_match_fraction
        ):
            continue
        key = scan.key_text[row]
        if key in ("", "nan"):
            continue
        if key in expanded:
            return key
        if scan.stem[row] == seed_stem and dist < fallback_dist:
            fallback = key
            fallback_dist = dist
    return fallback or detection_key(
        {
            "source_tile": particle.source_tile,
            "x_global": particle.x_global,
            "y_global": particle.y_global,
        }
    )


def _particles_from_frame(dirs: pd.DataFrame) -> list[Particle]:
    xs = dirs["x_global"].to_numpy(dtype=float)
    ys = dirs["y_global"].to_numpy(dtype=float)
    sizes = dirs["size"].to_numpy(dtype=float)
    tiles = dirs["source_tile"].astype(str).tolist()
    ids = dirs["id"].tolist() if "id" in dirs.columns else [None] * len(dirs)
    confs = dirs["confidence"].tolist() if "confidence" in dirs.columns else [None] * len(dirs)
    return [
        Particle(
            id=_particle_id(ids[index]),
            x_global=float(xs[index]),
            y_global=float(ys[index]),
            size=float(sizes[index]),
            confidence=float(confs[index] or 0.0),
            source_tile=tiles[index],
        )
        for index in range(len(dirs))
    ]


def _collapse_one_family(
    dirs: pd.DataFrame,
    merge_radius_nm: float,
    size_match_fraction: float,
    labeled_keys: set[str] | None = None,
) -> pd.DataFrame:
    """One row per nearby location. At most one member per direction or lighting role."""
    dirs = dirs.reset_index(drop=True)
    if dirs.empty:
        return dirs
    particles = _particles_from_frame(dirs)
    merged = deduplicate_nsew(
        particles,
        merge_radius=float(merge_radius_nm),
        size_aggregation="max",
        size_match_fraction=float(size_match_fraction),
    )
    scan = _FamilyScan(dirs)
    expanded = expand_labeled_keys(labeled_keys) if labeled_keys else set()
    kept: list[dict[str, Any]] = []
    taken = np.zeros(len(dirs), dtype=bool)
    x_vals = scan.x
    y_vals = scan.y
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
        record["nsew_dirs"] = _dirs_kept_on_cluster(
            scan, particle, merge_radius_nm, size_match_fraction
        )
        record["nsew_count"] = len(
            [part for part in str(record["nsew_dirs"]).split(",") if part]
        )
        record["key"] = _labeled_key_on_cluster(
            scan,
            particle,
            merge_radius_nm,
            size_match_fraction,
            expanded,
        )
        kept.append(record)
    return pd.DataFrame(kept)


def _shares_submap_review(source_tile: str) -> bool:
    """Quads, the uniform shot, and the combi in the same ``vN`` folder."""
    name = str(source_tile)
    if is_submap_lighting_tile(name):
        return True
    if is_combi_tile(name) and submap_id(name):
        return True
    return combi_key_family(name) == "light"


def collapse_submap_review_detections(
    df: pd.DataFrame | None,
    merge_radius_nm: float,
    size_match_fraction: float = DEFAULT_NSEW_SIZE_MATCH_FRACTION,
    labeled_keys: Iterable[str] | None = None,
) -> pd.DataFrame | None:
    """One row when the combi and the uniform shot are the same particle.

    The kept id is shared. ``nsew_dirs`` still lists every picture it was found
    on, so the summary can draw that id on each of those pictures.
    """
    if df is None or df.empty or "source_tile" not in df.columns:
        return df
    tagged = tagged_if_needed(df)
    tiles = tagged["source_tile"].astype(str)
    lighting = tiles.map(_shares_submap_review)
    if not bool(lighting.any()):
        return tagged.reset_index(drop=True)
    frames: list[pd.DataFrame] = [tagged.loc[~lighting].copy()]
    subset = tagged.loc[lighting]
    set_ids = subset["source_tile"].astype(str).map(submap_id)
    known = set() if labeled_keys is None else {str(key) for key in labeled_keys}
    for _, group in subset.groupby(set_ids, sort=False):
        frames.append(
            _collapse_one_family(group, merge_radius_nm, size_match_fraction, known)
        )
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return tagged.iloc[0:0].reset_index(drop=True)
    return pd.concat(nonempty, ignore_index=True)


def collapse_nsew_detections(
    df: pd.DataFrame,
    merge_radius_nm: float,
    size_match_fraction: float = DEFAULT_NSEW_SIZE_MATCH_FRACTION,
    labeled_keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Keep one row per shared location so Tinder does not ask once per lighting shot.

    N/S/E/W and particles-only share globally. Submap quads, the uniform
    ``all`` shot, and the combi beside them share only inside the same ``vN`` folder.
    """
    if df is None or df.empty or "source_tile" not in df.columns:
        return df
    tagged = tagged_if_needed(df)
    tiles = tagged["source_tile"].astype(str)
    lighting = tiles.map(_shares_submap_review)
    direction = tiles.map(is_nsew_family_tile) & ~lighting
    if not direction.any() and not lighting.any():
        return tagged.reset_index(drop=True)
    known = set() if labeled_keys is None else {str(key) for key in labeled_keys}
    frames: list[pd.DataFrame] = [tagged.loc[~direction & ~lighting].copy()]
    if direction.any():
        frames.append(
            _collapse_one_family(
                tagged.loc[direction], merge_radius_nm, size_match_fraction, known
            )
        )
    if lighting.any():
        subset = tagged.loc[lighting]
        set_ids = subset["source_tile"].astype(str).map(submap_id)
        for _, group in subset.groupby(set_ids, sort=False):
            frames.append(
                _collapse_one_family(group, merge_radius_nm, size_match_fraction, known)
            )
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return tagged.iloc[0:0].reset_index(drop=True)
    return pd.concat(nonempty, ignore_index=True)


def _particle_id(value: Any) -> int:
    if value in (None, "") or (isinstance(value, float) and pd.isna(value)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
