"""Grid placement of tiles from filenames, without decoding pixels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from src.overall.io.tile_loader import Tile, peek_tile_hw, try_parse_tile_filename


@dataclass(frozen=True)
class TilePlacement:
    """Where a tile sits in the mosaic, in pixel coordinates."""

    path: Path
    name: str
    y0: int
    x0: int
    height: int
    width: int


def tile_origin_px(
    tile: Tile,
    overlap_fraction: float,
    row_origin: int = 0,
    col_origin: int = 0,
) -> tuple[float, float]:
    """Return the top-left mosaic origin ``(x, y)`` of ``tile`` in pixels.

    If the filename provided ``x`` / ``y``, those values are used as the origin.
    Otherwise a regular grid is built from ``row`` / ``col`` and overlap.
    ``row_origin`` / ``col_origin`` are the smallest indices in the run, so the
    mosaic does not reserve empty space for unused column 0 (or row 0).
    """
    if tile.x_origin is not None and tile.y_origin is not None:
        return float(tile.x_origin), float(tile.y_origin)
    if tile.row is None or tile.col is None:
        raise ValueError(
            f"Tile {tile.name} needs row/col or x/y in the filename to place it."
        )
    step_x = tile.width * (1.0 - overlap_fraction)
    step_y = tile.height * (1.0 - overlap_fraction)
    return (
        float(tile.col - col_origin) * step_x,
        float(tile.row - row_origin) * step_y,
    )


def placement_from_tile(
    tile: Tile,
    overlap_fraction: float,
    row_origin: int = 0,
    col_origin: int = 0,
) -> TilePlacement:
    """Build a placement record from a loaded tile (uses shape, not a second read)."""
    x0, y0 = tile_origin_px(tile, overlap_fraction, row_origin, col_origin)
    return TilePlacement(
        path=tile.path,
        name=tile.name,
        y0=int(round(y0)),
        x0=int(round(x0)),
        height=tile.height,
        width=tile.width,
    )


def grid_index_origin(paths: list[Path], pattern: str) -> tuple[int, int]:
    """Smallest row/col among tiles so the mosaic starts at the first real column."""
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


def placement_from_path(
    path: Path,
    pattern: str,
    overlap: float,
    row_origin: int,
    col_origin: int,
    extra_meta: Mapping[str, object] | None = None,
    origin_fallback: bool = False,
) -> TilePlacement | None:
    """Grid placement from filename + image size.

    ``extra_meta`` fills missing row/col (used for ``(row,col)_N`` names).
    ``origin_fallback`` places unparseable files at (0, 0) instead of returning None.
    """
    meta = dict(try_parse_tile_filename(path.name, pattern) or {})
    if extra_meta:
        for key, value in extra_meta.items():
            meta.setdefault(key, value)
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
    if origin_fallback:
        return TilePlacement(
            path=path,
            name=path.name,
            y0=0,
            x0=0,
            height=height,
            width=width,
        )
    return None
