"""Tile-to-global transforms and a lazy/out-of-core mosaic for visualization."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.overall.io.placement import TilePlacement
from src.overall.io.tile_loader import load_tile_image


def mosaic_shape(placements: Sequence[TilePlacement]) -> tuple[int, int]:
    """Full mosaic height, width in pixels."""
    if not placements:
        return 0, 0
    height = max(p.y0 + p.height for p in placements)
    width = max(p.x0 + p.width for p in placements)
    return height, width


def downsample_for_target(
    height: int,
    width: int,
    target_mb: float = 20.0,
    channels: int = 3,
    bytes_per_channel: int = 1,
) -> int:
    """Integer factor so a ``height×width`` image fits in about ``target_mb``.

    Tiles should be downsampled by this factor *before* they are placed, so
    the stitched canvas never materializes at full resolution.
    """
    safe_h = max(int(height), 1)
    safe_w = max(int(width), 1)
    target_bytes = max(float(target_mb), 1e-6) * 1024 * 1024
    raw_bytes = safe_h * safe_w * max(int(channels), 1) * max(int(bytes_per_channel), 1)
    if raw_bytes <= target_bytes:
        return 1
    return int(np.ceil(np.sqrt(raw_bytes / target_bytes)))


def _load_downsampled(path: Path, factor: int) -> np.ndarray:
    """Read one tile and shrink it. The full-resolution array is discarded."""
    image = np.asarray(load_tile_image(path), dtype=np.float32)
    return _downsample(image, factor)


def _paste(canvas: np.ndarray, small: np.ndarray, y: int, x: int) -> None:
    if small.size == 0 or y >= canvas.shape[0] or x >= canvas.shape[1]:
        return
    y0 = max(y, 0)
    x0 = max(x, 0)
    src_y = y0 - y
    src_x = x0 - x
    hh = min(small.shape[0] - src_y, canvas.shape[0] - y0)
    ww = min(small.shape[1] - src_x, canvas.shape[1] - x0)
    if hh <= 0 or ww <= 0:
        return
    canvas[y0 : y0 + hh, x0 : x0 + ww] = small[src_y : src_y + hh, src_x : src_x + ww]


class LazyMosaic:
    """Downsample every tile first, then paste the small images onto one canvas."""

    def __init__(
        self,
        placements: Sequence[TilePlacement],
        thumbnails: dict[str, np.ndarray] | None = None,
        thumbnail_factor: int | None = None,
    ) -> None:
        self.placements = list(placements)
        self.full_height, self.full_width = mosaic_shape(self.placements)
        self.thumbnails = thumbnails or {}
        self.thumbnail_factor = thumbnail_factor

    def preview(
        self,
        downsample: int = 8,
        crop: tuple[int, int, int, int] | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
        workers: int = 8,
    ) -> np.ndarray:
        """Return a small preview array. ``crop`` is ``(y, x, h, w)`` in full-res pixels.

        The full mosaic path shrinks each TIFF first (in parallel), then stitches
        those thumbnails. Crops still read only the overlapping full-res patch.
        """
        factor = max(int(downsample), 1)
        if crop is None:
            return self._preview_full(factor, progress_cb, workers)
        return self._preview_crop(factor, crop, progress_cb)

    def _preview_full(
        self,
        factor: int,
        progress_cb: Callable[[int, int, str], None] | None,
        workers: int,
    ) -> np.ndarray:
        out_h = max(int(np.ceil(self.full_height / factor)), 0)
        out_w = max(int(np.ceil(self.full_width / factor)), 0)
        canvas = np.zeros((out_h, out_w), dtype=np.float32)
        if out_h == 0 or out_w == 0 or not self.placements:
            return canvas

        total = len(self.placements)
        n_workers = max(1, min(int(workers), total, cpu_count() or 1))
        small_tiles: list[np.ndarray | None] = [None] * total
        use_cache = (
            self.thumbnail_factor is not None
            and factor == int(self.thumbnail_factor)
            and len(self.thumbnails) == total
        )

        if use_cache:
            for index, placement in enumerate(self.placements):
                small = self.thumbnails.get(placement.name)
                if small is None:
                    small = _load_downsampled(placement.path, factor)
                small_tiles[index] = small
                if progress_cb is not None:
                    progress_cb(index + 1, total, placement.name)
        elif n_workers == 1:
            for index, placement in enumerate(self.placements):
                small_tiles[index] = _load_downsampled(placement.path, factor)
                if progress_cb is not None:
                    progress_cb(index + 1, total, placement.name)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(_load_downsampled, placement.path, factor): index
                    for index, placement in enumerate(self.placements)
                }
                done = 0
                for future in as_completed(futures):
                    index = futures[future]
                    small_tiles[index] = future.result()
                    done += 1
                    if progress_cb is not None:
                        progress_cb(done, total, self.placements[index].name)

        for placement, small in zip(self.placements, small_tiles):
            if small is None:
                continue
            _paste(canvas, small, placement.y0 // factor, placement.x0 // factor)
        return canvas

    def _preview_crop(
        self,
        factor: int,
        crop: tuple[int, int, int, int],
        progress_cb: Callable[[int, int, str], None] | None,
    ) -> np.ndarray:
        y0, x0, crop_h, crop_w = crop
        y0 = max(y0, 0)
        x0 = max(x0, 0)
        crop_h = min(crop_h, max(self.full_height - y0, 0))
        crop_w = min(crop_w, max(self.full_width - x0, 0))
        out_h = max(int(np.ceil(crop_h / factor)), 0)
        out_w = max(int(np.ceil(crop_w / factor)), 0)
        canvas = np.zeros((out_h, out_w), dtype=np.float32)
        if out_h == 0 or out_w == 0:
            return canvas

        y1, x1 = y0 + crop_h, x0 + crop_w
        total = len(self.placements)
        for index, placement in enumerate(self.placements, start=1):
            ty0, tx0 = placement.y0, placement.x0
            ty1, tx1 = ty0 + placement.height, tx0 + placement.width
            oy0, ox0 = max(ty0, y0), max(tx0, x0)
            oy1, ox1 = min(ty1, y1), min(tx1, x1)
            if oy0 >= oy1 or ox0 >= ox1:
                if progress_cb is not None:
                    progress_cb(index, total, placement.name)
                continue
            image = np.asarray(load_tile_image(placement.path), dtype=np.float32)
            patch = image[oy0 - ty0 : oy1 - ty0, ox0 - tx0 : ox1 - tx0]
            small = _downsample(patch, factor)
            _paste(canvas, small, (oy0 - y0) // factor, (ox0 - x0) // factor)
            if progress_cb is not None:
                progress_cb(index, total, placement.name)
        return canvas


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Box-filter downsample. ``factor`` 1 returns ``image`` unchanged."""
    if factor <= 1:
        return np.asarray(image)
    array = np.ascontiguousarray(image)
    h, w = array.shape[:2]
    nh, nw = max(h // factor, 1), max(w // factor, 1)
    return cv2.resize(array, (nw, nh), interpolation=cv2.INTER_AREA)


def _downsample(image: np.ndarray, factor: int) -> np.ndarray:
    return downsample_image(image, factor)
