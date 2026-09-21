"""Summary statistics and downsampled overlay images for reporting."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.config import cfg_get
from src.stitching.stitcher import LazyMosaic, downsample_for_target


def summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Return count and size-distribution summary for a particle table."""
    if df.empty:
        return {
            "particle_count": 0,
            "size_mean": 0.0,
            "size_std": 0.0,
            "size_min": 0.0,
            "size_max": 0.0,
        }
    sizes = df["size"].astype(float)
    return {
        "particle_count": int(len(df)),
        "size_mean": float(sizes.mean()),
        "size_std": float(sizes.std(ddof=0)),
        "size_min": float(sizes.min()),
        "size_max": float(sizes.max()),
    }


def overlay_downsample(
    mosaic: LazyMosaic,
    config: dict[str, Any],
    crop: tuple[int, int, int, int] | None = None,
) -> int:
    """Pick the integer factor used to downsample tiles before stitching.

    ``report.downsample`` > 0 is an explicit override. Otherwise the factor is
    chosen so the stitched RGB overlay is about ``report.target_mb`` megabytes.
    """
    configured = int(cfg_get(config, "report.downsample", 0) or 0)
    if configured > 0:
        return configured
    if crop is None:
        height, width = mosaic.full_height, mosaic.full_width
    else:
        height, width = int(crop[2]), int(crop[3])
    target_mb = float(cfg_get(config, "report.target_mb", 20.0))
    return downsample_for_target(height, width, target_mb=target_mb, channels=3)


def overlay_markers(
    mosaic: LazyMosaic,
    particles: pd.DataFrame,
    config: dict[str, Any],
    crop: tuple[int, int, int, int] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> np.ndarray:
    """Downsampled mosaic with particle markers. Coordinates are global nm or px.

    Marker positions are converted back to mosaic pixels using ``pixel_size_nm``.
    ``crop`` is ``(y, x, h, w)`` in full-resolution mosaic pixels. ``crop=None``
    stitches every tile into one overview.
    """
    downsample = overlay_downsample(mosaic, config, crop=crop)
    preview = mosaic.preview(downsample=downsample, crop=crop, progress_cb=progress_cb)
    rgb = _to_display_rgb(preview)
    if particles.empty or rgb.size == 0:
        return rgb

    placed_names = {placement.name for placement in mosaic.placements}
    if placed_names and "source_tile" in particles.columns:
        names = particles["source_tile"].astype(str).map(
            lambda path: str(path).replace("\\", "/").rsplit("/", 1)[-1]
        )
        particles = particles.loc[names.isin(placed_names)]
        if particles.empty:
            return rgb

    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    if pixel_size <= 0:
        pixel_size = 1.0
    y_off = 0 if crop is None else crop[0]
    x_off = 0 if crop is None else crop[1]

    for _, row in particles.iterrows():
        x_px = float(row["x_global"]) / pixel_size
        y_px = float(row["y_global"]) / pixel_size
        x_d = int(round((x_px - x_off) / downsample))
        y_d = int(round((y_px - y_off) / downsample))
        if not (0 <= x_d < rgb.shape[1] and 0 <= y_d < rgb.shape[0]):
            continue
        particle_r = float(row["size"]) / pixel_size / downsample / 2.0
        radius = max(int(round(particle_r)) + 10, 14)
        cv2.circle(
            rgb, (x_d, y_d), radius, (255, 140, 40), thickness=3, lineType=cv2.LINE_AA
        )
    return rgb


def _to_display_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    low, high = np.percentile(array, (1.0, 99.5))
    if high <= low:
        high = low + 1e-6
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    gray = (scaled * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def encode_overlay_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    """JPEG bytes for download; ``rgb`` is uint8 RGB."""
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("Failed to encode mosaic JPEG.")
    return buffer.tobytes()


def write_overlay_image(
    mosaic: LazyMosaic,
    particles: pd.DataFrame,
    config: dict[str, Any],
    path: str | Path,
    crop: tuple[int, int, int, int] | None = None,
) -> Path:
    """Stitch a size-capped mosaic overlay and write it to ``path``."""
    rgb = overlay_markers(mosaic, particles, config, crop=crop)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        destination.write_bytes(encode_overlay_jpeg(rgb))
    else:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(destination), bgr):
            raise RuntimeError(f"Failed to write mosaic image to {destination}")
    return destination
