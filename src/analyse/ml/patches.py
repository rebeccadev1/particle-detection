"""Unmarked two-channel crops for the FP filter.

Built from the corrected tile and the DoG residual. Never uses circled JPEGs
under ``labels/crops/`` — the red marker would leak into any patch model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.overall.io.tile_loader import DEFAULT_FILENAME_PATTERN, parse_tile_filename

PATCH_SIZE = 96


def extract_centered(
    image: np.ndarray,
    x_local: float,
    y_local: float,
    size: int = PATCH_SIZE,
) -> np.ndarray:
    """Square crop centred on ``(x_local, y_local)``, reflect-padded at edges."""
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D tile, got shape {array.shape}")
    half = int(size) // 2
    cx = int(round(float(x_local)))
    cy = int(round(float(y_local)))
    y0, y1 = cy - half, cy - half + int(size)
    x0, x1 = cx - half, cx - half + int(size)
    pad_y0 = max(0, -y0)
    pad_x0 = max(0, -x0)
    pad_y1 = max(0, y1 - array.shape[0])
    pad_x1 = max(0, x1 - array.shape[1])
    if pad_y0 or pad_x0 or pad_y1 or pad_x1:
        array = np.pad(
            array,
            ((pad_y0, pad_y1), (pad_x0, pad_x1)),
            mode="reflect",
        )
        y0 += pad_y0
        y1 += pad_y0
        x0 += pad_x0
        x1 += pad_x0
    patch = np.asarray(array[y0:y1, x0:x1], dtype=np.float32)
    if patch.shape != (int(size), int(size)):
        patch = np.resize(patch, (int(size), int(size)))
    return patch


def stack_channels(raw: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """``(2, H, W)`` float32 pair. Independent 1–99 percentile stretch."""
    return np.stack([_stretch(raw), _stretch(residual)], axis=0).astype(np.float32)


def extract_unmarked_channels(
    corrected: np.ndarray,
    residual: np.ndarray,
    x_local: float,
    y_local: float,
    size: int = PATCH_SIZE,
) -> np.ndarray:
    """Two-channel unmarked crop from in-memory tile maps."""
    raw = extract_centered(corrected, x_local, y_local, size=size)
    res = extract_centered(residual, x_local, y_local, size=size)
    return stack_channels(raw, res)


def tile_row_col(
    tile_name: str,
    pattern: str = DEFAULT_FILENAME_PATTERN,
) -> tuple[float, float]:
    """Grid indices from the filename, or (0, 0) when parsing fails."""
    try:
        meta = parse_tile_filename(Path(str(tile_name)).name, pattern)
    except (ValueError, TypeError):
        return 0.0, 0.0
    return float(meta.get("row", 0) or 0), float(meta.get("col", 0) or 0)


def _stretch(channel: np.ndarray) -> np.ndarray:
    array = np.asarray(channel, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
