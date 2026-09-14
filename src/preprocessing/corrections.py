"""Per-tile illumination, contrast, and denoise corrections (pure functions)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.config import cfg_get

_FLOAT = np.float32
_FLATTEN_DOWNSAMPLE = 8
_HIST_BINS = 512
_PERCENTILE_EXACT_LIMIT = 50_000
_MEDIAN_SAMPLE = 250_000


def to_float(image: np.ndarray) -> np.ndarray:
    """Convert a tile to float32 in approximately [0, 1] if integer-valued."""
    array = np.asarray(image)
    if array.size == 0:
        return np.asarray(array, dtype=_FLOAT)
    array = np.ascontiguousarray(array, dtype=_FLOAT)
    vmax = float(array.max())
    if vmax > 1.5:
        array = array / _FLOAT(max(vmax, 1.0))
    return array


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur. ``sigma`` is in pixels of ``image``."""
    array = np.ascontiguousarray(image, dtype=_FLOAT)
    if sigma <= 0:
        return array
    return cv2.GaussianBlur(
        array,
        ksize=(0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT_101,
    )


def denoise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian denoise; ``sigma`` is in pixels."""
    if sigma <= 0:
        return np.ascontiguousarray(image, dtype=_FLOAT)
    return _gaussian_blur(image, sigma)


def _approx_percentiles(image: np.ndarray, p_low: float, p_high: float) -> tuple[float, float]:
    """Percentiles via a histogram for large tiles; exact sort for small ones."""
    array = np.asarray(image, dtype=_FLOAT)
    if array.size == 0:
        return 0.0, 1.0
    if array.size < _PERCENTILE_EXACT_LIMIT:
        low, high = np.percentile(array, (p_low, p_high))
        return float(low), float(high)
    lo = float(array.min())
    hi = float(array.max())
    if hi <= lo:
        return lo, hi
    counts, edges = np.histogram(array, bins=_HIST_BINS, range=(lo, hi))
    cdf = np.cumsum(counts, dtype=np.float64)
    total = float(cdf[-1])
    if total <= 0:
        return lo, hi
    cdf /= total
    low_idx = int(np.searchsorted(cdf, p_low / 100.0, side="left"))
    high_idx = int(np.searchsorted(cdf, p_high / 100.0, side="left"))
    low_idx = min(max(low_idx, 0), len(edges) - 1)
    high_idx = min(max(high_idx, 0), len(edges) - 1)
    return float(edges[low_idx]), float(edges[high_idx])


def contrast_stretch(image: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Percentile contrast stretch to [0, 1]."""
    low, high = _approx_percentiles(image, p_low, p_high)
    array = np.asarray(image, dtype=_FLOAT)
    if high <= low:
        return array
    stretched = (array - _FLOAT(low)) / _FLOAT(high - low)
    return np.clip(stretched, 0.0, 1.0, out=stretched)


def _approx_median(image: np.ndarray) -> float:
    """Median via a systematic subsample so large tiles skip a full sort."""
    array = np.asarray(image).reshape(-1)
    if array.size == 0:
        return 0.0
    if array.size > _MEDIAN_SAMPLE:
        step = max(int(array.size // _MEDIAN_SAMPLE), 1)
        array = array[::step]
    return float(np.median(array))


def _flatten_factor(height: int, width: int, sigma: float) -> int:
    """Downsample factor for a slowly varying illumination field."""
    max_factor = min(
        _FLATTEN_DOWNSAMPLE,
        max(height // 16, 1),
        max(width // 16, 1),
        max(int(sigma), 1),
    )
    return max(1, max_factor)


def flatten_illumination(image: np.ndarray, sigma: float) -> np.ndarray:
    """Divide by a large-scale Gaussian estimate of illumination.

    The illumination field varies slowly, so the blur is computed at reduced
    resolution and bilinearly upsampled. That preserves the divide-by-field
    model without a 300-tap full-res kernel.
    """
    array = np.ascontiguousarray(image, dtype=_FLOAT)
    if sigma <= 0:
        return array
    height, width = array.shape
    factor = _flatten_factor(height, width, sigma)
    if factor > 1:
        small_w = max(width // factor, 8)
        small_h = max(height // factor, 8)
        small = cv2.resize(array, (small_w, small_h), interpolation=cv2.INTER_AREA)
        spatial = width / float(small_w)
        background = cv2.resize(
            _gaussian_blur(small, float(sigma) / spatial),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        background = _gaussian_blur(array, sigma)
    background = np.maximum(background, _FLOAT(1e-6))
    flattened = array / background
    median = _approx_median(flattened)
    if median > 0:
        flattened *= _FLOAT(_approx_median(array) / median)
    return flattened


def apply_corrections(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Apply configured preprocessing steps. Array in, array out; no I/O."""
    result = to_float(image)
    if cfg_get(config, "preprocessing.denoise", True):
        result = denoise(result, float(cfg_get(config, "preprocessing.denoise_sigma", 1.0)))
    if cfg_get(config, "preprocessing.flatten_illumination", True):
        result = flatten_illumination(
            result, float(cfg_get(config, "preprocessing.flatten_sigma", 160.0))
        )
    if cfg_get(config, "preprocessing.contrast_stretch", True):
        percentiles = cfg_get(config, "preprocessing.contrast_percentiles", [1.0, 99.0])
        result = contrast_stretch(result, float(percentiles[0]), float(percentiles[1]))
    return result
