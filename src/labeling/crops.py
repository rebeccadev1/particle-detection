"""Crop a source tile around one detection and draw a single unlabeled (orange) circle."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.config import DEFAULT_INPUT_DIR, cfg_get
from src.io.tile_loader import (
    DEFAULT_FILENAME_PATTERN,
    TILE_SUFFIXES,
    Tile,
    load_tile_image,
    matching_tile_paths,
    peek_tile_hw,
    resolve_input_dir,
    try_parse_tile_filename,
)
from src.measurement.measurer import (
    DISPLAY_DIRECTION_ORDER,
    DIRECTION_STEMS,
    is_direction_tile,
    is_particles_only_tile,
)
from src.report.report_generator import _to_display_rgb
from src.stitching.stitcher import LazyMosaic, TilePlacement, placement_from_tile

CIRCLE_COLOR = (255, 140, 40)  # unlabeled
CIRCLE_THICKNESS = 3
MIN_CIRCLE_RADIUS = 14
MIN_HALF_PX = 96
WINDOW_SCALE = 1.8
DEFAULT_WAFER_FOLDER = "R3 04-08"


@dataclass(frozen=True)
class ParticleCrop:
    """One labeled-review image: tile crop, local coords, and source path."""

    rgb: np.ndarray
    x_local: float
    y_local: float
    crop_x0: int
    crop_y0: int
    tile_path: Path
    direction: str = ""


class TileImageCache:
    """Keep the last decoded TIFF so neighbouring hits on the same tile are cheap."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.image: np.ndarray | None = None

    def load(self, path: Path) -> np.ndarray:
        resolved = Path(path).resolve()
        if self.path == resolved and self.image is not None:
            return self.image
        self.image = load_tile_image(resolved)
        self.path = resolved
        return self.image


def global_nm_to_local_px(
    x_global: float,
    y_global: float,
    placement: TilePlacement,
    pixel_size_nm: float,
) -> tuple[float, float]:
    """Map global nanometres back to tile-local pixels."""
    scale = float(pixel_size_nm) if pixel_size_nm > 0 else 1.0
    x_local = float(x_global) / scale - float(placement.x0)
    y_local = float(y_global) / scale - float(placement.y0)
    return x_local, y_local


def local_px_to_global_nm(
    x_local: float,
    y_local: float,
    placement: TilePlacement,
    pixel_size_nm: float,
) -> tuple[float, float]:
    """Map tile-local pixels to global nanometres."""
    scale = float(pixel_size_nm) if pixel_size_nm > 0 else 1.0
    x_global = (float(x_local) + float(placement.x0)) * scale
    y_global = (float(y_local) + float(placement.y0)) * scale
    return x_global, y_global


def crop_window(
    x_local: float,
    y_local: float,
    diameter_px: float,
    height: int,
    width: int,
    min_half: int = MIN_HALF_PX,
    scale: float = WINDOW_SCALE,
) -> tuple[int, int, int, int]:
    """Return ``(y0, x0, y1, x1)`` clamped to the tile.

    A coordinate outside the tile stays a local window at the nearest edge.
    It does not expand to the full tile width or height.
    """
    half = max(int(min_half), int(round(float(scale) * float(diameter_px))))
    cx = _clamp_index(int(round(float(x_local))), int(width))
    cy = _clamp_index(int(round(float(y_local))), int(height))
    x0, x1 = _local_span(cx, half, int(width))
    y0, y1 = _local_span(cy, half, int(height))
    return y0, x0, y1, x1


def _clamp_index(index: int, limit: int) -> int:
    if limit <= 0:
        return 0
    return min(max(int(index), 0), limit - 1)


def _local_span(center: int, half: int, limit: int) -> tuple[int, int]:
    """Window of about ``2 * half`` pixels, shifted inside ``limit``."""
    if limit <= 0:
        return 0, 0
    start = max(0, int(center) - int(half))
    end = min(int(limit), int(center) + int(half))
    if end <= start:
        return 0, int(limit)
    target = min(int(limit), 2 * int(half))
    if end - start < target:
        if start == 0:
            end = min(int(limit), start + target)
        elif end == int(limit):
            start = max(0, end - target)
    return start, end


def draw_particle_circle(
    rgb: np.ndarray,
    x_local: float,
    y_local: float,
    size_nm: float,
    pixel_size_nm: float,
    crop_x0: int = 0,
    crop_y0: int = 0,
    color: tuple[int, int, int] = CIRCLE_COLOR,
) -> np.ndarray:
    """Draw a marker on the mosaic overlay, in crop coordinates."""
    scale = float(pixel_size_nm) if pixel_size_nm > 0 else 1.0
    x_d = int(round(float(x_local) - crop_x0))
    y_d = int(round(float(y_local) - crop_y0))
    radius = max(int(round(float(size_nm) / scale / 2.0)) + 10, MIN_CIRCLE_RADIUS)
    if rgb.size == 0:
        return rgb
    if not (0 <= x_d < rgb.shape[1] and 0 <= y_d < rgb.shape[0]):
        return rgb
    cv2.circle(
        rgb,
        (x_d, y_d),
        radius,
        tuple(int(c) for c in color),
        thickness=CIRCLE_THICKNESS,
        lineType=cv2.LINE_AA,
    )
    return rgb


def config_for_hit_placement(
    config: dict[str, Any],
    rows: pd.DataFrame | list[dict[str, Any]],
) -> dict[str, Any]:
    """Copy ``config`` with the tile overlap that puts hits on their source tiles.

    A detection table measured at 10% overlap is mis-placed when the sidebar
    overlap is 0, and the orange circle is then drawn off the crop.
    """
    records = _placement_records(rows)
    if not records:
        return config
    nominal = float(cfg_get(config, "overlap_fraction", 0.0) or 0.0)
    geometry = _tile_geometry(records, config)
    if not geometry:
        return config
    pixel_size = float(cfg_get(config, "pixel_size_nm", 960.0) or 960.0)
    nominal_hits = _hits_inside(records, geometry, nominal, pixel_size)
    if nominal_hits >= len(records):
        return config
    best_overlap = nominal
    best_hits = nominal_hits
    for step in range(0, 11):
        overlap = step / 20.0
        hits = _hits_inside(records, geometry, overlap, pixel_size)
        if hits > best_hits:
            best_overlap = overlap
            best_hits = hits
    if best_hits <= nominal_hits:
        return config
    updated = dict(config)
    updated["overlap_fraction"] = best_overlap
    return updated


def _placement_records(rows: pd.DataFrame | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return []
        records = rows.to_dict(orient="records")
    else:
        records = list(rows)
    if len(records) <= 80:
        return records
    step = max(1, len(records) // 80)
    return records[::step][:80]


def _tile_geometry(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, tuple[float, float, float, float]]:
    """Tile name → ``(row, col, height, width)`` for grid-named files."""
    pattern = str(cfg_get(config, "filename_pattern", DEFAULT_FILENAME_PATTERN))
    input_dir = cfg_get(config, "input_dir", None)
    folders: dict[str, tuple[int, int]] = {}
    geometry: dict[str, tuple[float, float, float, float]] = {}
    for record in records:
        name = Path(str(record.get("source_tile", "") or "")).name
        if not name or name in geometry:
            continue
        try:
            path = find_tile_path(name, input_dir=input_dir)
        except (FileNotFoundError, OSError):
            continue
        meta = try_parse_tile_filename(path.name, pattern) or {}
        if "row" not in meta or "col" not in meta:
            continue
        folder = str(path.parent.resolve())
        if folder not in folders:
            paths = matching_tile_paths(path.parent, pattern)
            folders[folder] = _grid_index_origin(paths, pattern)
        row_origin, col_origin = folders[folder]
        height, width = peek_tile_hw(path)
        geometry[name] = (
            float(int(meta["row"]) - row_origin),
            float(int(meta["col"]) - col_origin),
            float(height),
            float(width),
        )
    return geometry


def _hits_inside(
    records: list[dict[str, Any]],
    geometry: dict[str, tuple[float, float, float, float]],
    overlap: float,
    pixel_size_nm: float = 960.0,
) -> int:
    scale = float(pixel_size_nm) if pixel_size_nm > 0 else 1.0
    inside = 0
    for record in records:
        name = Path(str(record.get("source_tile", "") or "")).name
        geom = geometry.get(name)
        if geom is None:
            continue
        row_off, col_off, height, width = geom
        try:
            x_px = float(record["x_global"]) / scale
            y_px = float(record["y_global"]) / scale
        except (KeyError, TypeError, ValueError):
            continue
        x_local = x_px - col_off * width * (1.0 - overlap)
        y_local = y_px - row_off * height * (1.0 - overlap)
        if 0.0 <= x_local < width and 0.0 <= y_local < height:
            inside += 1
    return inside


def find_tile_path(
    tile_name: str,
    input_dir: str | Path | None = None,
    mosaic: LazyMosaic | None = None,
) -> Path:
    """Locate ``tile_name``: mosaic placement, then ``input_dir``, then R3 04-08."""
    name = Path(str(tile_name)).name
    if mosaic is not None:
        for placement in mosaic.placements:
            if placement.name == name and Path(placement.path).is_file():
                return Path(placement.path)
    for folder in _candidate_folders(input_dir):
        found = _existing_tile(folder / name)
        if found is not None:
            return found
    raise FileNotFoundError(f"Tile {name!r} was not found in R3 04-08 or the tile folder.")


def _existing_tile(path: Path) -> Path | None:
    """Return ``path`` if it exists, or the same stem with another image suffix."""
    if path.is_file():
        return path
    seen = {path.suffix.lower()}
    for suffix in TILE_SUFFIXES:
        if suffix in seen:
            continue
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def placement_for_tile(
    tile_path: Path,
    config: dict[str, Any],
    mosaic: LazyMosaic | None = None,
    origin_cache: dict[str, tuple[int, int]] | None = None,
) -> TilePlacement:
    """Placement of ``tile_path`` from the mosaic, or reconstructed from the grid."""
    name = tile_path.name
    if mosaic is not None:
        for placement in mosaic.placements:
            if placement.name == name:
                return placement
    pattern = str(cfg_get(config, "filename_pattern", DEFAULT_FILENAME_PATTERN))
    overlap = float(cfg_get(config, "overlap_fraction", 0.0))
    folder = tile_path.parent
    cache_key = str(folder.resolve())
    if origin_cache is not None and cache_key in origin_cache:
        row_origin, col_origin = origin_cache[cache_key]
    else:
        paths = matching_tile_paths(folder, pattern)
        row_origin, col_origin = _grid_index_origin(paths, pattern)
        if origin_cache is not None:
            origin_cache[cache_key] = (row_origin, col_origin)
    return _placement_from_path(tile_path, pattern, overlap, row_origin, col_origin)


_STAGED_2OF4 = re.compile(r"(v\d+)_2of4$", re.IGNORECASE)
_SET_DIR = re.compile(r"v\d+$", re.IGNORECASE)
_GROUNDUP_SERIES = re.compile(r"Groundup v(\d+)", re.IGNORECASE)


def direction_images_for_combined(tile_path: str | Path) -> dict[str, Path]:
    """N/S/E/W files for a particles-only or staged 2-of-4 detection tile."""
    path = Path(tile_path)
    beside = list_direction_images(path.parent)
    if len(beside) >= 2 and not is_direction_tile(path.name):
        return beside
    set_name = _set_name_for_combined(path)
    if not set_name:
        return {}
    seen: set[Path] = set()
    for folder in _direction_set_folders(path, set_name):
        try:
            key = folder.resolve()
        except OSError:
            key = folder
        if key in seen:
            continue
        seen.add(key)
        found = list_direction_images(folder)
        if len(found) >= 2:
            return found
    return {}


def _direction_set_folders(tile_path: Path, set_name: str) -> list[Path]:
    """Set folders for this staged image, including ``v3 (10^5)`` exposure names.

    A folder named ``Groundup v5 …`` stays on Groundup v5. Older staged files
    with no series in the name still resolve to Groundup v4.
    """
    match = _GROUNDUP_SERIES.search(tile_path.parent.name)
    series_names = [f"Groundup v{match.group(1)}"] if match else ["Groundup v4"]
    folders: list[Path] = []
    for series in series_names:
        for root in (tile_path.parent.parent / series, DEFAULT_INPUT_DIR / series):
            folders.extend(_set_folders_named(root, set_name))
    return folders


def _set_folders_named(root: Path, set_name: str) -> list[Path]:
    if not root.is_dir():
        return []
    exact = root / set_name
    if exact.is_dir():
        return [exact]
    pattern = re.compile(rf"^{re.escape(set_name)}(?:\s|\()", re.IGNORECASE)
    return [
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and pattern.match(child.name)
    ]


def _set_name_for_combined(tile_path: Path) -> str | None:
    match = _STAGED_2OF4.fullmatch(tile_path.stem)
    if match:
        return match.group(1)
    if is_particles_only_tile(tile_path.name):
        for parent in tile_path.parents:
            if _SET_DIR.fullmatch(parent.name):
                return parent.name
    return None


def _spot_brightness(image: np.ndarray, x: float, y: float, diameter_px: float) -> float:
    """Mean intensity inside the particle disk. Higher means a brighter glint."""
    gray = image if image.ndim == 2 else image.mean(axis=2)
    radius = max(2.0, float(diameter_px) / 2.0)
    height, width = int(gray.shape[0]), int(gray.shape[1])
    x0 = max(0, int(np.floor(x - radius)))
    x1 = min(width, int(np.ceil(x + radius)) + 1)
    y0 = max(0, int(np.floor(y - radius)))
    y1 = min(height, int(np.ceil(y + radius)) + 1)
    if x1 <= x0 or y1 <= y0:
        return -1.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    patch = np.asarray(gray[y0:y1, x0:x1])
    if not np.any(mask):
        return -1.0
    return float(patch[mask].mean())


def brightest_direction_crop(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any],
    mosaic: LazyMosaic | None = None,
    cache: TileImageCache | None = None,
    origin_cache: dict[str, tuple[int, int]] | None = None,
) -> ParticleCrop | None:
    """Crop the N/S/E/W frame where this particles-only hit is brightest.

    The circle stays on the same pixel as the combined detection. Returns
    ``None`` when the hit is not a combined tile or the four angles are missing.
    """
    record = dict(row) if not isinstance(row, dict) else dict(row)
    tile_name = str(record.get("source_tile", "") or "")
    if is_direction_tile(tile_name):
        return None
    input_dir = cfg_get(config, "input_dir", None)
    try:
        combined_path = find_tile_path(tile_name, input_dir=input_dir, mosaic=mosaic)
    except (FileNotFoundError, OSError):
        return None
    directions = direction_images_for_combined(combined_path)
    if len(directions) < 2:
        return None
    placement = placement_for_tile(
        combined_path, config, mosaic=mosaic, origin_cache=origin_cache
    )
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    x_local, y_local = global_nm_to_local_px(
        float(record["x_global"]),
        float(record["y_global"]),
        placement,
        pixel_size,
    )
    diameter_px = float(record["size"]) / (pixel_size if pixel_size > 0 else 1.0)
    loader = cache if cache is not None else TileImageCache()
    best_name = ""
    best_score = -1.0
    for direction in DISPLAY_DIRECTION_ORDER:
        path = directions.get(direction)
        if path is None:
            continue
        score = _spot_brightness(loader.load(path), x_local, y_local, diameter_px)
        if score > best_score:
            best_score = score
            best_name = direction
    if not best_name:
        return None
    sibling = dict(record)
    sibling["source_tile"] = directions[best_name].name
    crop = crop_particle(
        sibling,
        config,
        mosaic=mosaic,
        cache=loader,
        origin_cache=origin_cache,
        tile_path=directions[best_name],
    )
    return ParticleCrop(
        rgb=crop.rgb,
        x_local=crop.x_local,
        y_local=crop.y_local,
        crop_x0=crop.crop_x0,
        crop_y0=crop.crop_y0,
        tile_path=crop.tile_path,
        direction=best_name,
    )


def crop_particle(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any],
    mosaic: LazyMosaic | None = None,
    cache: TileImageCache | None = None,
    origin_cache: dict[str, tuple[int, int]] | None = None,
    tile_path: Path | None = None,
) -> ParticleCrop:
    """Load ``source_tile``, crop around the blob, and draw one red circle."""
    record = dict(row) if not isinstance(row, dict) else dict(row)
    tile_name = str(record["source_tile"])
    input_dir = cfg_get(config, "input_dir", None)
    if tile_path is None:
        tile_path = find_tile_path(tile_name, input_dir=input_dir, mosaic=mosaic)
    else:
        tile_path = Path(tile_path)
    placement = placement_for_tile(
        tile_path, config, mosaic=mosaic, origin_cache=origin_cache
    )
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    x_local, y_local = global_nm_to_local_px(
        float(record["x_global"]),
        float(record["y_global"]),
        placement,
        pixel_size,
    )
    loader = cache if cache is not None else TileImageCache()
    image = loader.load(tile_path)
    height, width = int(image.shape[0]), int(image.shape[1])
    diameter_px = float(record["size"]) / (pixel_size if pixel_size > 0 else 1.0)
    y0, x0, y1, x1 = crop_window(x_local, y_local, diameter_px, height, width)
    patch = np.asarray(image[y0:y1, x0:x1])
    rgb = _to_display_rgb(patch)
    draw_particle_circle(
        rgb,
        x_local,
        y_local,
        float(record["size"]),
        pixel_size,
        crop_x0=x0,
        crop_y0=y0,
    )
    return ParticleCrop(
        rgb=rgb,
        x_local=x_local,
        y_local=y_local,
        crop_x0=x0,
        crop_y0=y0,
        tile_path=tile_path,
    )


def list_direction_images(folder: str | Path) -> dict[str, Path]:
    """N/S/E/W image files in ``folder`` (partial sets are allowed)."""
    directory = Path(folder)
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for path in directory.iterdir():
        if not path.is_file():
            continue
        stem = path.stem.upper()
        if stem not in DIRECTION_STEMS or path.suffix.lower() not in TILE_SUFFIXES:
            continue
        found[stem] = path
    return found


def crop_direction_views(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any],
    mosaic: LazyMosaic | None = None,
    cache: TileImageCache | None = None,
    origin_cache: dict[str, tuple[int, int]] | None = None,
) -> dict[str, ParticleCrop]:
    """Crops of the same location on N, W, E, and S when those tiles exist."""
    record = dict(row) if not isinstance(row, dict) else dict(row)
    tile_name = str(record.get("source_tile", "") or "")
    if not is_direction_tile(tile_name):
        return {}
    input_dir = cfg_get(config, "input_dir", None)
    tile_path = find_tile_path(tile_name, input_dir=input_dir, mosaic=mosaic)
    siblings = list_direction_images(tile_path.parent)
    if len(siblings) < 2:
        return {}
    views: dict[str, ParticleCrop] = {}
    loader = cache if cache is not None else TileImageCache()
    for direction in DISPLAY_DIRECTION_ORDER:
        path = siblings.get(direction)
        if path is None:
            continue
        sibling = dict(record)
        sibling["source_tile"] = path.name
        views[direction] = crop_particle(
            sibling,
            config,
            mosaic=mosaic,
            cache=loader,
            origin_cache=origin_cache,
        )
    return views


def _candidate_folders(input_dir: str | Path | None) -> list[Path]:
    folders: list[Path] = []
    for raw in (input_dir, DEFAULT_WAFER_FOLDER):
        if raw in (None, ""):
            continue
        try:
            resolved = resolve_input_dir(raw)
        except FileNotFoundError:
            continue
        if resolved not in folders:
            folders.append(resolved)
    return folders


def _grid_index_origin(paths: list[Path], pattern: str) -> tuple[int, int]:
    rows: list[int] = []
    cols: list[int] = []
    for path in paths:
        meta = try_parse_tile_filename(path.name, pattern)
        if meta is None:
            continue
        if "row" in meta:
            rows.append(int(meta["row"]))
        if "col" in meta:
            cols.append(int(meta["col"]))
    return (min(rows) if rows else 0, min(cols) if cols else 0)


def _placement_from_path(
    path: Path,
    pattern: str,
    overlap: float,
    row_origin: int,
    col_origin: int,
) -> TilePlacement:
    meta = try_parse_tile_filename(path.name, pattern) or {}
    height, width = peek_tile_hw(path)
    tile = Tile(
        path=path,
        name=path.name,
        image=np.empty((0, 0), dtype=np.uint8),
        row=int(meta["row"]) if "row" in meta else None,
        col=int(meta["col"]) if "col" in meta else None,
        run=int(meta["run"]) if "run" in meta else None,
        magnification=float(meta["mag"]) if "mag" in meta else None,
        x_origin=float(meta["x"]) if "x" in meta else None,
        y_origin=float(meta["y"]) if "y" in meta else None,
        height=height,
        width=width,
    )
    if tile.x_origin is not None and tile.y_origin is not None:
        return placement_from_tile(tile, overlap, row_origin, col_origin)
    if tile.row is not None and tile.col is not None:
        return placement_from_tile(tile, overlap, row_origin, col_origin)
    return TilePlacement(
        path=path,
        name=path.name,
        y0=0,
        x0=0,
        height=height,
        width=width,
    )
