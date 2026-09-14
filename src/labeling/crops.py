"""Crop a source tile around one detection and draw a single unlabeled (orange) circle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.config import cfg_get
from src.io.tile_loader import (
    DEFAULT_FILENAME_PATTERN,
    Tile,
    load_tile_image,
    matching_tile_paths,
    parse_tile_filename,
    peek_tile_hw,
    resolve_input_dir,
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
    """Return ``(y0, x0, y1, x1)`` clamped to the tile."""
    half = max(int(min_half), int(round(float(scale) * float(diameter_px))))
    cx = int(round(float(x_local)))
    cy = int(round(float(y_local)))
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(int(width), cx + half)
    y1 = min(int(height), cy + half)
    if x1 <= x0:
        x0, x1 = 0, int(width)
    if y1 <= y0:
        y0, y1 = 0, int(height)
    return y0, x0, y1, x1


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
        candidate = folder / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Tile {name!r} was not found in R3 04-08 or the tile folder.")


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


def crop_particle(
    row: pd.Series | dict[str, Any],
    config: dict[str, Any],
    mosaic: LazyMosaic | None = None,
    cache: TileImageCache | None = None,
    origin_cache: dict[str, tuple[int, int]] | None = None,
) -> ParticleCrop:
    """Load ``source_tile``, crop around the blob, and draw one red circle."""
    record = dict(row) if not isinstance(row, dict) else dict(row)
    tile_name = str(record["source_tile"])
    input_dir = cfg_get(config, "input_dir", None)
    tile_path = find_tile_path(tile_name, input_dir=input_dir, mosaic=mosaic)
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
        meta = parse_tile_filename(path.name, pattern)
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
    meta = parse_tile_filename(path.name, pattern)
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
    return placement_from_tile(tile, overlap, row_origin, col_origin)
