"""Full-tile inspection overlays for the recall-audit tab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

from src.config import cfg_get
from src.io.tile_loader import (
    DEFAULT_FILENAME_PATTERN,
    TILE_SUFFIXES,
    parse_tile_filename,
    resolve_input_dir,
)
from src.labeling.crops import (
    CIRCLE_COLOR,
    CIRCLE_THICKNESS,
    DEFAULT_WAFER_FOLDER,
    MIN_CIRCLE_RADIUS,
    MIN_HALF_PX,
    WINDOW_SCALE,
    draw_particle_circle,
    find_tile_path,
    global_nm_to_local_px,
    placement_for_tile,
)
from src.labeling.queue import detection_key, label_map_with_aliases, nsew_key_aliases, tagged_if_needed
from src.measurement.measurer import (
    DIRECTION_STEMS,
    is_direction_tile,
    is_nsew_family_tile,
    is_particles_only_tile,
)
from src.report.report_generator import _to_display_rgb
from src.stitching.stitcher import TilePlacement

LABEL_CIRCLE_COLORS: dict[str, tuple[int, int, int]] = {
    "particle": (40, 180, 70),
    "not_particle": (255, 40, 40),
    "not_sure": (60, 140, 220),
}
UNLABELED_CIRCLE_COLOR = CIRCLE_COLOR
OVERVIEW_MAX_SIDE = 960
ZOOM_MAX_SIDE = 1080
INSPECT_DISPLAY_WIDTH = 960
UNHAPPINESS_TARGET = 0.95


def unhappiness_score(precision: float, recall: float) -> float:
    """``100 × ((P - 0.95)² + (R - 0.95)²)`` with P and R in 0–1. Lower is better."""
    return 100.0 * (
        (float(precision) - UNHAPPINESS_TARGET) ** 2
        + (float(recall) - UNHAPPINESS_TARGET) ** 2
    )


GRID_N = 3
CELL_NAMES = (
    ("NW", "N", "NE"),
    ("W", "C", "E"),
    ("SW", "S", "SE"),
)
MISS_SOURCE = "tile_inspect"
DEFAULT_MISS_SIZE_UM = 15.0
_PARTICLES_ONLY_DIR_NAMES = {"particles only", "particles only output"}


@dataclass(frozen=True)
class OverlayView:
    """Displayed RGB plus the transform back to tile-local pixels."""

    rgb: np.ndarray
    crop_x0: int
    crop_y0: int
    scale: float


def cell_bounds(
    height: int,
    width: int,
    row: int,
    col: int,
    grid: int = GRID_N,
) -> tuple[int, int, int, int]:
    """Return ``(y0, x0, y1, x1)`` for one cell of a ``grid`` × ``grid`` split."""
    if grid < 1:
        raise ValueError("grid must be >= 1")
    y0 = int(height * row / grid)
    x0 = int(width * col / grid)
    y1 = int(height * (row + 1) / grid)
    x1 = int(width * (col + 1) / grid)
    return y0, x0, max(y1, y0 + 1), max(x1, x0 + 1)


def detections_on_tile(table: pd.DataFrame | None, tile_name: str) -> pd.DataFrame:
    """Rows for ``tile_name``. N/S/E/W share one aligned view, so a hit on any
    of those four is shown on all four. Particles-only 2/3/4-of-4 files do not
    share hits: they are different combined images.
    """
    if table is None or table.empty:
        return pd.DataFrame()
    name = _basename(tile_name)
    names = table["source_tile"].astype(str).map(_basename)
    mask = names == name
    if is_direction_tile(name):
        mask = mask | names.map(is_direction_tile)
        if "nsew_dirs" in table.columns:
            stem = Path(name).stem.upper()
            dirs = table["nsew_dirs"].fillna("").astype(str).str.upper()
            mask = mask | dirs.map(
                lambda text: stem
                in {part.strip() for part in str(text).replace(";", ",").split(",") if part.strip()}
            )
    hits = table.loc[mask].copy()
    if hits.empty:
        return hits.reset_index(drop=True)
    if is_direction_tile(name):
        hits = tagged_if_needed(hits)
        hits["source_tile"] = name
        if "key" in hits.columns:
            hits = hits.drop_duplicates(subset=["key"], keep="first")
        else:
            hits = hits.drop_duplicates(
                subset=["x_global", "y_global"], keep="first"
            )
    return hits.reset_index(drop=True)


def labeled_detections(
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    label: str = "particle",
) -> pd.DataFrame:
    """Detections whose keys are stored with ``label`` (default: particle)."""
    if detections is None or detections.empty or labels is None or labels.empty:
        return pd.DataFrame()
    tagged = tagged_if_needed(detections)
    labeled = labels.loc[labels["label"].astype(str) == str(label)].copy()
    if labeled.empty or "key" not in tagged.columns:
        return pd.DataFrame()
    labeled["key"] = labeled["key"].astype(str)
    hits = tagged.loc[tagged["key"].astype(str).isin(set(labeled["key"]))].copy()
    if hits.empty:
        return hits.reset_index(drop=True)
    extra_cols = [col for col in ("key", "crop_path", "particle_id") if col in labeled.columns]
    extra = labeled.loc[:, extra_cols].drop_duplicates("key")
    hits["key"] = hits["key"].astype(str)
    return hits.merge(extra, on="key", how="left", suffixes=("", "_label")).reset_index(drop=True)


def _tiles_for_folder_labels(
    tile_names: Sequence[str] | None,
    folder_tiles: Sequence[str] | None = None,
) -> set[str] | None:
    """Exact tile basenames whose labels count for this folder.

    N/S/E/W and particles-only versions share labels only among files that
    are actually in ``folder_tiles`` (or ``tile_names``). ``N.bmp`` from
    another folder is not mixed with ``N.png``.
    """
    if tile_names is None and folder_tiles is None:
        return None
    wanted = {_basename(name) for name in (tile_names or ())}
    allowed = {
        _basename(name)
        for name in (folder_tiles if folder_tiles is not None else tile_names or ())
    }
    if any(is_nsew_family_tile(name) for name in wanted):
        wanted.update(name for name in allowed if is_nsew_family_tile(name))
    return wanted


def last_run_class_counts(
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    tile_names: Sequence[str] | None = None,
    min_size_nm: float = 0.0,
    folder_tiles: Sequence[str] | None = None,
) -> dict[str, float]:
    """Last-run detection totals split by label, plus missed inspect marks.

    * ``n_detected`` — every last-run hit
    * ``n_real`` — detected and labeled particle (green)
    * ``n_fake`` — detected and not labeled as particle (red, orange, blue)
    * ``n_unlabeled`` — hits with no label (orange); also counted in ``n_fake``
    * ``n_undetected`` — labeled particles on these tiles whose keys are not
      in the last run (inspect marks and other misses), at least
      ``min_size_nm`` when that floor is > 0 (Last Run shows this as
      undetected real ≥ that size, typically 20 µm)
    * ``n_real_total`` — ``n_real`` + ``n_undetected``
    * ``precision`` — ``n_real / n_detected`` (unlabeled hits count as fake)
    * ``recall`` — ``n_real / n_real_total``
    * ``unhappiness`` — ``100 × ((P - 0.95)² + (R - 0.95)²)`` (lower is better)
    """
    tagged = tagged_if_needed(detections) if detections is not None else pd.DataFrame()
    if tile_names is not None and not tagged.empty:
        wanted = {_basename(name) for name in tile_names}
        tile = tagged["source_tile"].astype(str).map(_basename)
        tagged = tagged.loc[tile.isin(wanted)]
    n_detected = int(len(tagged)) if tagged is not None and not tagged.empty else 0
    scoped = labels
    label_tiles = _tiles_for_folder_labels(tile_names, folder_tiles)
    if scoped is not None and not scoped.empty and label_tiles is not None:
        tile = scoped["source_tile"].astype(str).map(_basename)
        scoped = scoped.loc[tile.isin(label_tiles)]
    label_map = label_map_with_aliases(scoped)
    particles = pd.DataFrame()
    if scoped is not None and not scoped.empty:
        particles = scoped.loc[scoped["label"].astype(str) == "particle"].copy()

    n_real = n_fake = n_not_sure = n_unlabeled = 0
    det_keys: set[str] = set()
    if n_detected and "key" in tagged.columns:
        for key in tagged["key"].astype(str):
            det_keys.add(str(key))
            name = label_map.get(str(key), "")
            if name == "particle":
                n_real += 1
            else:
                n_fake += 1
                if name == "not_sure":
                    n_not_sure += 1
                elif name != "not_particle":
                    n_unlabeled += 1
    n_undetected = 0
    n_undetected_below = 0
    if not particles.empty and "key" in particles.columns:
        missing = particles.loc[~particles["key"].astype(str).isin(det_keys)]
        n_all_missing = int(len(missing))
        floor = float(min_size_nm or 0.0)
        if floor > 0 and not missing.empty and "size" in missing.columns:
            sizes = pd.to_numeric(missing["size"], errors="coerce")
            missing = missing.loc[sizes >= floor]
        n_undetected = int(len(missing))
        n_undetected_below = max(0, n_all_missing - n_undetected)
    n_real_total = n_real + n_undetected
    precision = (n_real / n_detected) if n_detected else 0.0
    recall = (n_real / n_real_total) if n_real_total else 0.0
    return {
        "n_detected": n_detected,
        "n_real": n_real,
        "n_fake": n_fake,
        "n_unlabeled": n_unlabeled,
        "n_undetected": n_undetected,
        "n_undetected_below": n_undetected_below,
        "n_not_sure": n_not_sure,
        "n_real_total": n_real_total,
        "precision": precision,
        "recall": recall,
        "unhappiness": unhappiness_score(precision, recall),
    }


def last_run_class_counts_by_tile(
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    tile_names: Sequence[str],
    min_size_nm: float = 0.0,
    folder_tiles: Sequence[str] | None = None,
) -> pd.DataFrame:
    """``last_run_class_counts`` for each tile basename, one row per tile.

    Hits are those whose ``source_tile`` is that file (N/S/E/W still share).
    ``folder_tiles`` is the label scene (e.g. N.png plus particles-only).
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    label_tiles = list(folder_tiles) if folder_tiles is not None else list(tile_names)
    for name in tile_names:
        tile = _basename(name)
        if tile in seen:
            continue
        seen.add(tile)
        counts = last_run_class_counts(
            detections,
            labels,
            tile_names=[tile],
            min_size_nm=min_size_nm,
            folder_tiles=label_tiles,
        )
        rows.append({"tile": tile, **counts})
    return pd.DataFrame(rows)


def undetected_size_floor_nm(
    detections: pd.DataFrame | None,
    config: Mapping[str, Any] | None = None,
) -> float:
    """Size gate for Last Run misses: this run's min size, else config."""
    floor = 0.0
    if config is not None:
        floor = float(cfg_get(dict(config), "detection.min_size_nm", 0.0) or 0.0)
    if detections is not None and not detections.empty and "size" in detections.columns:
        run_min = float(pd.to_numeric(detections["size"], errors="coerce").min())
        if np.isfinite(run_min) and run_min > 0:
            run_floor = float(np.floor(run_min / 1000.0) * 1000.0)
            floor = max(floor, run_floor)
    return floor


def labeled_particle_recovery(
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    tile_names: Sequence[str] | None = None,
    near_px: float = 20.0,
    pixel_size_nm: float = 960.0,
) -> dict[str, float]:
    """Labeled particles among last-run detections (green circles).

    Only keys that appear in ``detections`` are counted, so labels from older
    pipeline CSVs that this run did not emit are ignored.
    """
    _ = (near_px, pixel_size_nm)
    empty = {"n_labeled": 0, "n_found": 0, "pct": 0.0}
    hits = labeled_detections(detections, labels, label="particle")
    if hits.empty:
        return empty
    if tile_names is not None:
        wanted = {_basename(name) for name in tile_names}
        tile = hits["source_tile"].astype(str).map(_basename)
        hits = hits.loc[tile.isin(wanted)]
    if hits.empty:
        return empty
    n = int(hits["key"].astype(str).nunique()) if "key" in hits.columns else int(len(hits))
    return {"n_labeled": n, "n_found": n, "pct": 100.0}


def tile_names_for_hits(hits: pd.DataFrame | None) -> list[str]:
    """Unique ``source_tile`` basenames, first-seen order."""
    if hits is None or hits.empty or "source_tile" not in hits.columns:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for name in hits["source_tile"].astype(str).map(_basename):
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def expand_direction_tile_names(
    names: Sequence[str],
    available: Sequence[str] | None,
) -> list[str]:
    """If hits include N/S/E/W or particles-only, list every family file in the folder."""
    ordered = list(names)
    if available is None:
        return ordered
    if not any(is_nsew_family_tile(name) for name in ordered):
        return ordered
    have = {_basename(name) for name in ordered}
    extras = [
        _basename(name)
        for name in available
        if is_nsew_family_tile(name) and _basename(name) not in have
    ]

    def _family_sort(name: str) -> tuple[int, str]:
        stem = Path(name).stem.upper()
        if stem in DIRECTION_STEMS:
            return (DIRECTION_STEMS.index(stem), name)
        if is_particles_only_tile(name):
            return (10, name.lower())
        return (99, name.lower())

    extras.sort(key=_family_sort)
    return ordered + extras


def _scene_image_names(folder: Path) -> set[str]:
    if not folder.is_dir():
        return set()
    names: set[str] = set()
    try:
        entries = folder.iterdir()
    except OSError:
        return names
    for path in entries:
        if path.is_file() and path.suffix.lower() in TILE_SUFFIXES:
            names.add(path.name)
    return names


def _resolve_scene_folder(folder: str | Path) -> Path | None:
    text = str(folder).strip()
    if not text:
        return None
    try:
        return resolve_input_dir(text)
    except FileNotFoundError:
        path = Path(text).expanduser()
        try:
            if path.is_dir():
                return path.resolve()
        except OSError:
            return None
        return None


def scene_folders(folder: str | Path) -> list[Path]:
    """This folder plus sibling N/S/E/W and Particles-only dirs for the same scene."""
    resolved = _resolve_scene_folder(folder)
    if resolved is None:
        return []
    folder = resolved
    roots: list[Path] = []

    def add(path: Path) -> None:
        try:
            resolved_path = path.resolve()
        except OSError:
            return
        if resolved_path.is_dir() and resolved_path not in roots:
            roots.append(resolved_path)

    add(folder)
    lower = folder.name.lower()
    if lower in _PARTICLES_ONLY_DIR_NAMES:
        add(folder.parent)
        if folder.parent.name.lower().startswith("output "):
            add(folder.parent.parent)
            add(folder.parent / "Particles only")
    elif lower.startswith("output "):
        add(folder.parent)
        add(folder / "Particles only")
    else:
        add(folder / "Particles only")
        add(folder / "Particles only output")
        add(folder / f"Output {folder.name}")
        add(folder / f"Output {folder.name}" / "Particles only")
    return roots


def related_scene_tile_names(
    folder: str | Path | None,
    current_names: Sequence[str] | None = None,
) -> set[str]:
    """N/S/E/W and particles-only basenames for this scene, matching this folder's suffix.

    Groundup v3 (``.png``) is not mixed with Groundup v2 (``.bmp``).
    """
    current = {_basename(name) for name in (current_names or ())}
    if folder in (None, ""):
        return _with_family_suffix_names(current)
    path = _resolve_scene_folder(folder) or Path(str(folder))
    if not current:
        current = _scene_image_names(path)
    suffixes = {Path(name).suffix.lower() for name in current if Path(name).suffix}
    related = set(current)
    for root in scene_folders(folder):
        for name in _scene_image_names(root):
            if not is_nsew_family_tile(name):
                continue
            if suffixes and Path(name).suffix.lower() not in suffixes:
                continue
            related.add(name)
    return _with_family_suffix_names(related)


def _with_family_suffix_names(names: set[str]) -> set[str]:
    """If this scene is particles-only, also keep N/S/E/W with the same suffix."""
    related = set(names)
    suffixes = {Path(name).suffix.lower() for name in related if Path(name).suffix}
    if not suffixes:
        return related
    if not any(is_particles_only_tile(name) for name in related):
        return related
    if any(is_direction_tile(name) for name in related):
        return related
    for suffix in suffixes:
        related.update(f"{stem}{suffix}" for stem in DIRECTION_STEMS)
    return related


def label_color_for_key(key: str, labels: Mapping[str, str]) -> tuple[int, int, int]:
    name = labels.get(str(key))
    if name not in LABEL_CIRCLE_COLORS:
        for alias in nsew_key_aliases(str(key)):
            name = labels.get(alias)
            if name in LABEL_CIRCLE_COLORS:
                break
    if name in LABEL_CIRCLE_COLORS:
        return LABEL_CIRCLE_COLORS[name]
    return UNLABELED_CIRCLE_COLOR


def overlay_view(
    image: np.ndarray,
    circles: Sequence[Mapping[str, Any]],
    pixel_size_nm: float,
    *,
    crop: tuple[int, int, int, int] | None = None,
    max_side: int | None = OVERVIEW_MAX_SIDE,
) -> OverlayView:
    """Contrast-stretch ``image``, optional crop, downsample, draw circles."""
    gray = np.asarray(image)
    if crop is not None:
        y0, x0, y1, x1 = crop
        gray = gray[y0:y1, x0:x1]
        crop_x0, crop_y0 = int(x0), int(y0)
    else:
        crop_x0, crop_y0 = 0, 0
    rgb = _to_display_rgb(gray)
    if rgb.size == 0:
        return OverlayView(rgb=rgb, crop_x0=crop_x0, crop_y0=crop_y0, scale=1.0)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    scale = 1.0
    if max_side is not None and max(height, width) > int(max_side):
        scale = float(max_side) / float(max(height, width))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.ascontiguousarray(rgb).copy()
    for circle in circles:
        color = tuple(int(c) for c in circle.get("color", UNLABELED_CIRCLE_COLOR))
        x_s = (float(circle["x_local"]) - crop_x0) * scale
        y_s = (float(circle["y_local"]) - crop_y0) * scale
        size_nm = float(circle["size"])
        pixel = float(pixel_size_nm) if pixel_size_nm > 0 else 1.0
        radius = max(
            int(round(size_nm / pixel / 2.0 * scale)) + 10,
            MIN_CIRCLE_RADIUS,
        )
        x_d = int(round(x_s))
        y_d = int(round(y_s))
        if out.size == 0:
            continue
        if not (0 <= x_d < out.shape[1] and 0 <= y_d < out.shape[0]):
            continue
        cv2.circle(
            out,
            (x_d, y_d),
            radius,
            color,
            thickness=max(CIRCLE_THICKNESS, 3),
            lineType=cv2.LINE_AA,
        )
    return OverlayView(rgb=out, crop_x0=crop_x0, crop_y0=crop_y0, scale=scale)


def overlay_circles(
    image: np.ndarray,
    circles: Sequence[Mapping[str, Any]],
    pixel_size_nm: float,
    *,
    crop: tuple[int, int, int, int] | None = None,
    max_side: int | None = OVERVIEW_MAX_SIDE,
) -> np.ndarray:
    """RGB overlay used by tests and callers that only need the image."""
    return overlay_view(
        image, circles, pixel_size_nm, crop=crop, max_side=max_side
    ).rgb


def local_xy_on_tile(
    x_global: float,
    y_global: float,
    placements: Sequence[TilePlacement],
    pixel_size_nm: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Map globals to tile pixels using the first placement that lands in-frame.

    Detector CSVs use the full R3 mosaic origin. Click-marks from a 6-tile
    folder use that folder's origin. Trying both keeps circles on the image.
    """
    if not placements:
        raise ValueError("placements must not be empty")
    fallback = global_nm_to_local_px(
        x_global, y_global, placements[0], pixel_size_nm
    )
    for placement in placements:
        x_local, y_local = global_nm_to_local_px(
            x_global, y_global, placement, pixel_size_nm
        )
        if 0.0 <= x_local < float(width) and 0.0 <= y_local < float(height):
            return x_local, y_local
    return fallback


def extra_grid_placements(base: TilePlacement, config: Mapping[str, Any]) -> list[TilePlacement]:
    """Placements for last-run CSVs that used the full R3 grid, not the sidebar folder."""
    pattern = str(cfg_get(dict(config), "filename_pattern", DEFAULT_FILENAME_PATTERN))
    try:
        meta = parse_tile_filename(base.name, pattern)
    except ValueError:
        return []
    if "row" not in meta or "col" not in meta:
        return []
    row = int(meta["row"])
    col = int(meta["col"])
    configured = float(cfg_get(dict(config), "overlap_fraction", 0.0) or 0.0)
    overlaps: list[float] = []
    for overlap in (configured, 0.0, 0.1):
        if overlap not in overlaps:
            overlaps.append(float(overlap))
    placements: list[TilePlacement] = []
    seen: set[tuple[int, int]] = set()
    for overlap in overlaps:
        step_x = float(base.width) * (1.0 - overlap)
        step_y = float(base.height) * (1.0 - overlap)
        for row_origin, col_origin in ((1, 1), (0, 0)):
            x0 = int(round((col - col_origin) * step_x))
            y0 = int(round((row - row_origin) * step_y))
            key = (x0, y0)
            if key in seen:
                continue
            seen.add(key)
            placements.append(
                TilePlacement(
                    path=base.path,
                    name=base.name,
                    y0=y0,
                    x0=x0,
                    height=base.height,
                    width=base.width,
                )
            )
    return placements


def candidate_placements(
    tile_path: Any,
    config: Mapping[str, Any],
    mosaic: Any | None = None,
    origin_cache: dict[str, tuple[int, int]] | None = None,
) -> list[TilePlacement]:
    """Current-folder placement, full-wafer grid, R3 04-08, then origin (0, 0)."""
    path = Path(tile_path)
    primary = placement_for_tile(
        path, dict(config), mosaic=mosaic, origin_cache=origin_cache
    )
    ordered = [primary]
    seen = {(int(primary.x0), int(primary.y0))}
    for extra in extra_grid_placements(primary, config):
        key = (int(extra.x0), int(extra.y0))
        if key not in seen:
            ordered.append(extra)
            seen.add(key)
    try:
        wafer_path = find_tile_path(path.name, input_dir=DEFAULT_WAFER_FOLDER)
        wafer_config = dict(config)
        wafer_config["input_dir"] = DEFAULT_WAFER_FOLDER
        wafer = placement_for_tile(
            wafer_path, wafer_config, mosaic=None, origin_cache=origin_cache
        )
        key = (int(wafer.x0), int(wafer.y0))
        if key not in seen:
            ordered.append(wafer)
            seen.add(key)
    except (FileNotFoundError, ValueError, OSError):
        pass
    if (0, 0) not in seen:
        ordered.append(
            TilePlacement(
                path=primary.path,
                name=primary.name,
                y0=0,
                x0=0,
                height=primary.height,
                width=primary.width,
            )
        )
    return ordered


def display_xy_to_local(
    x_display: float,
    y_display: float,
    view: OverlayView,
) -> tuple[float, float]:
    """Map a click on the displayed overlay back to tile-local pixels."""
    scale = float(view.scale) if float(view.scale) > 0 else 1.0
    x_local = float(view.crop_x0) + float(x_display) / scale
    y_local = float(view.crop_y0) + float(y_display) / scale
    return x_local, y_local


def missed_particle_record(
    tile_name: str,
    x_global: float,
    y_global: float,
    size_nm: float,
) -> dict[str, Any]:
    """CSV row for a click that was not a detector hit."""
    record: dict[str, Any] = {
        "id": "",
        "source_tile": str(tile_name),
        "x_global": float(x_global),
        "y_global": float(y_global),
        "size": float(size_nm),
        "confidence": 0.0,
        "source_csv": MISS_SOURCE,
    }
    record["key"] = detection_key(record)
    return record


def preview_click_crop(
    image: np.ndarray,
    x_local: float,
    y_local: float,
    size_nm: float,
    pixel_size_nm: float,
    color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Full-res crop around a hit, padded so the circle is not clipped."""
    scale = float(pixel_size_nm) if pixel_size_nm > 0 else 1.0
    diameter_px = float(size_nm) / scale
    radius = max(int(round(diameter_px / 2.0)) + 10, MIN_CIRCLE_RADIUS)
    half = max(
        int(MIN_HALF_PX),
        int(round(float(WINDOW_SCALE) * float(diameter_px))),
        int(radius + CIRCLE_THICKNESS + 24),
    )
    height, width = int(image.shape[0]), int(image.shape[1])
    cx = int(round(float(x_local)))
    cy = int(round(float(y_local)))
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - width)
    pad_bottom = max(0, y1 - height)
    y0c, x0c = max(0, y0), max(0, x0)
    y1c, x1c = min(height, y1), min(width, x1)
    patch = np.asarray(image[y0c:y1c, x0c:x1c])
    if pad_top or pad_bottom or pad_left or pad_right:
        patch = cv2.copyMakeBorder(
            patch,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )
    rgb = _to_display_rgb(patch)
    draw_particle_circle(
        rgb,
        x_local,
        y_local,
        size_nm,
        pixel_size_nm,
        crop_x0=x0,
        crop_y0=y0,
        color=LABEL_CIRCLE_COLORS["particle"] if color is None else color,
    )
    return rgb


def circles_from_rows(
    rows: pd.DataFrame,
    x_locals: Sequence[float],
    y_locals: Sequence[float],
    labels: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Build overlay circle dicts, tagging each row with its label colour."""
    circles: list[dict[str, Any]] = []
    for (x_local, y_local), (_, row) in zip(
        zip(x_locals, y_locals, strict=True), rows.iterrows(), strict=True
    ):
        key = str(row["key"]) if "key" in row and pd.notna(row["key"]) else detection_key(row)
        circles.append(
            {
                "x_local": float(x_local),
                "y_local": float(y_local),
                "size": float(row["size"]),
                "color": label_color_for_key(key, labels),
                "key": key,
            }
        )
    return circles


def _basename(path: str) -> str:
    text = str(path).replace("\\", "/")
    return text.rsplit("/", 1)[-1]
