"""Per-tile particle candidate detection with structured-background suppression."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np
from scipy import fft as sp_fft
from scipy.spatial import cKDTree
from skimage.feature import blob_log
from skimage.morphology import skeletonize

from src.config import cfg_get, with_recall_profile

_FLOAT = np.float32
_SNR_PERCENTILE = 99.5
_CIRCULARITY_MASS_FRACTION = 0.4
_SUPPORT_AREA_FACTOR = 6.0
_LOCAL_PEAK_SNR = 2.5
# Keep a clustered / in-line blob if it is clearly bigger or brighter than
# the layout nodes around it. 1.30× recovers ~20 µm flakes sitting in a
# 13 µm top-hat lattice; equal-size bead chains still drop.
_STRUCTURE_SIZE_RATIO = 1.30
_STRUCTURE_LINE_MAX_GAP_PX = 48.0
_STRUCTURE_SNR_RATIO = 2.5
_AUTO_SNR_SIGMA_FACTOR = 3.5
_PERCENTILE_EXACT_LIMIT = 50_000
_HIST_BINS = 512
_MEDIAN_SAMPLE = 250_000
_LATTICE_PROBE_MIN = 320
_LATTICE_PROBE_SIZE = 256
# Compact residual islands that sit on a layout ridge. DoG only sees the rim
# after edge softening zeros the interior. ≥ island_large_nm always protected;
# ≥ island_min_nm protected only on a *single* edge (not an L-junction / pad
# corner). Sizes are nm; convert with pixel_size_nm.
_ISLAND_MIN_NM = 20_000.0
_ISLAND_LARGE_NM = 40_000.0
_ISLAND_MAX_ASPECT = 3.0
_ISLAND_MIN_SOLIDITY = 0.35
_ISLAND_OPEN_PX = 4
_ISLAND_JUNCTION_HALF_PX = 32


CANDIDATE_FEATURE_FIELDS = (
    "circularity",
    "support_over_area",
    "edge_distance_px",
    "tile_border_dist_px",
    "n_neighbors_48",
    "local_peak_snr",
    "radial_inner",
    "radial_mid",
    "radial_outer",
)
NEIGHBOR_FEATURE_PX = 48.0


@dataclass(frozen=True)
class ParticleCandidate:
    """A detection in local (tile) pixel coordinates, origin at top-left."""

    y_local: float
    x_local: float
    size: float
    confidence: float
    circularity: float = 0.0
    support_over_area: float = 0.0
    edge_distance_px: float = 0.0
    tile_border_dist_px: float = 0.0
    n_neighbors_48: float = 0.0
    local_peak_snr: float = 0.0
    radial_inner: float = 0.0
    radial_mid: float = 0.0
    radial_outer: float = 0.0


def _notch_mask_from_spectrum(
    spectrum: np.ndarray,
    peak_threshold: float,
    notch_radius: int,
) -> np.ndarray:
    """Boolean notch mask in unshifted ``rfft2`` layout, DC neighborhood protected."""
    magnitude = np.abs(spectrum)
    peak = float(np.max(magnitude))
    height, nfreq = magnitude.shape
    if peak <= 0:
        return np.zeros((height, nfreq), dtype=bool)

    mag_norm = magnitude / _FLOAT(peak)
    yy = np.arange(height, dtype=np.int32)
    dy = np.minimum(yy, height - yy)[:, None]
    dx = np.arange(nfreq, dtype=np.int32)[None, :]
    dc_radius = max(int(notch_radius) * 3, 6)
    dc_mask = dy * dy + dx * dx <= dc_radius * dc_radius
    peaks = (mag_norm >= peak_threshold) & ~dc_mask
    radius = max(int(notch_radius), 0)
    if radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        binary = peaks.astype(np.uint8)
        if height > 2 * radius:
            padded = np.concatenate([binary[-radius:], binary, binary[:radius]], axis=0)
            padded = cv2.dilate(padded, kernel)
            peaks = padded[radius:-radius].astype(bool)
        else:
            peaks = cv2.dilate(binary, kernel).astype(bool)
    return peaks


def build_fft_notch_mask(
    image: np.ndarray,
    peak_threshold: float,
    notch_radius: int,
) -> np.ndarray:
    """Lattice notch mask from one representative tile (``rfft2`` layout)."""
    array = np.ascontiguousarray(image, dtype=_FLOAT)
    spectrum = sp_fft.rfft2(array, workers=1)
    return _notch_mask_from_spectrum(spectrum, peak_threshold, notch_radius)


def suppress_periodic_fft(
    image: np.ndarray,
    peak_threshold: float,
    notch_radius: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Notch-filter dominant lattice peaks in the FFT, leaving DC intact.

    If ``mask`` matches this tile's ``rfft2`` shape it is reused; otherwise peaks
    are detected on this tile's spectrum. An empty mask skips the transform.
    """
    array = np.ascontiguousarray(image, dtype=_FLOAT)
    if mask is not None and mask.shape[0] == array.shape[0] and not mask.any():
        return array
    spectrum = sp_fft.rfft2(array, workers=1)
    if mask is None or mask.shape != spectrum.shape:
        mask = _notch_mask_from_spectrum(spectrum, peak_threshold, notch_radius)
        if not mask.any():
            return array
    spectrum[mask] = 0
    return sp_fft.irfft2(spectrum, s=array.shape, workers=1)


def suppress_periodic_tophat(image: np.ndarray, radius: int) -> np.ndarray:
    """Morphological white top-hat: keep bright objects smaller than ``radius``.

    Uses OpenCV's elliptical kernel. skimage's ``white_tophat`` with ``disk(r)``
    is equivalent in shape but much slower on large tiles (seconds vs tens of ms).
    """
    r = max(int(radius), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    array = np.ascontiguousarray(image, dtype=_FLOAT)
    return cv2.morphologyEx(array, cv2.MORPH_TOPHAT, kernel)


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur. Large ``sigma`` is applied at reduced resolution."""
    array = np.ascontiguousarray(image, dtype=_FLOAT)
    if sigma <= 0:
        return array
    height, width = array.shape[:2]
    factor = 1
    if sigma >= 16.0 and min(height, width) >= 64:
        factor = min(
            max(int(sigma / 4.0), 2),
            max(height // 16, 1),
            max(width // 16, 1),
        )
    if factor > 1:
        small_w = max(width // factor, 8)
        small_h = max(height // factor, 8)
        spatial = width / float(small_w)
        small = cv2.resize(array, (small_w, small_h), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(
            small,
            ksize=(0, 0),
            sigmaX=float(sigma) / spatial,
            sigmaY=float(sigma) / spatial,
            borderType=cv2.BORDER_REFLECT_101,
        )
        return cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(
        array,
        ksize=(0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT_101,
    )


def _gaussian_blur_full(image: np.ndarray, sigma: float) -> np.ndarray:
    """Full-resolution Gaussian. Used by DoG so blob scales match scikit-image."""
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


def _subsample_for_stats(values: np.ndarray, limit: int = _MEDIAN_SAMPLE) -> np.ndarray:
    """Systematic subsample so median / MAD stay O(limit) on multi-megapixel tiles."""
    array = np.asarray(values).reshape(-1)
    if array.size <= limit:
        return array
    step = max(int(array.size // limit), 1)
    return array[::step]


def _mad_floor(values: np.ndarray) -> float:
    """Robust noise floor: median absolute deviation, scaled like a std-dev."""
    array = np.asarray(_subsample_for_stats(values), dtype=np.float64)
    if array.size == 0:
        return 1e-6
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return max(1.4826 * mad, 1e-6)


def _approx_percentile(image: np.ndarray, q: float) -> float:
    """Percentile via a histogram for large tiles; exact sort for small ones."""
    array = np.asarray(image, dtype=_FLOAT)
    if array.size == 0:
        return 0.0
    if array.size < _PERCENTILE_EXACT_LIMIT:
        return float(np.percentile(array, q))
    lo = float(array.min())
    hi = float(array.max())
    if hi <= lo:
        return lo
    counts, edges = np.histogram(array, bins=_HIST_BINS, range=(lo, hi))
    cdf = np.cumsum(counts, dtype=np.float64)
    total = float(cdf[-1])
    if total <= 0:
        return lo
    cdf /= total
    idx = int(np.searchsorted(cdf, float(q) / 100.0, side="left"))
    idx = min(max(idx, 0), len(edges) - 1)
    return float(edges[idx])


def _scale_unit(image: np.ndarray) -> np.ndarray:
    """Shift to non-negative and scale a robust high percentile into ``[0, 1]``."""
    array = np.ascontiguousarray(image, dtype=_FLOAT)
    array = array - _FLOAT(float(np.min(array)))
    hi = float(_approx_percentile(array, _SNR_PERCENTILE))
    if hi <= 1e-12:
        hi = float(np.max(array))
    if hi <= 0:
        return array
    array = array / _FLOAT(hi)
    return np.clip(array, 0.0, 1.0, out=array)


def local_snr(residual: np.ndarray, sigma: float) -> np.ndarray:
    """Divide residual by a slow envelope of local energy.

    Isolated blobs in a smooth field are boosted. Long edges and busy texture
    have high local energy and are suppressed. ``sigma`` should be several times
    the particle scale. ``sigma <= 0`` skips this and only does unit scaling.
    """
    array = np.ascontiguousarray(residual, dtype=_FLOAT)
    array = array - _FLOAT(float(np.min(array)))
    if sigma <= 0:
        return _scale_unit(array)
    energy = array * array
    envelope = np.sqrt(np.maximum(_gaussian_blur(energy, sigma), 0.0))
    floor = _FLOAT(_mad_floor(array))
    snr = array / np.maximum(envelope, floor)
    return _scale_unit(snr)


def _keep_long_edges(mask: np.ndarray, length: int) -> np.ndarray:
    """Keep connected components that are elongated or thinly filled and long enough."""
    if length <= 0 or not np.any(mask):
        return mask
    _nlab, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    if stats.shape[0] <= 1:
        return np.zeros_like(mask)
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    long_side = np.maximum(widths, heights)
    short_side = np.maximum(np.minimum(widths, heights), 1)
    fill = areas / np.maximum(widths * heights, 1).astype(np.float64)
    keep = (long_side >= length) & ((long_side / short_side >= 2.0) | (fill < 0.35))
    keep_ids = np.flatnonzero(keep) + 1
    if keep_ids.size == 0:
        return np.zeros_like(mask)
    return np.isin(labels, keep_ids).astype(np.uint8)


def coarse_edge_distance(
    image: np.ndarray,
    sigma: float,
    min_length: float = 40.0,
) -> np.ndarray:
    """Distance (px) to the nearest *long* coarse region border of ``image``.

    Gradient is taken after a heavy blur so lattice/texture is ignored.
    Hysteresis keeps weaker box borders that connect to stronger corners;
    morphological close rejoins broken segments. Compact particle halos are
    dropped (need an elongated or thin component at least ``min_length`` px).
    The mask is then skeletonized so distance is to the border *line*, not a
    fat gradient band (which would swallow real specks sitting inside a pad).
    The tile frame is included so truncated pads at the crop edge are not
    treated as particles. ``sigma <= 0`` returns a large constant.
    """
    height, width = np.asarray(image).shape[:2]
    far = _FLOAT(float(max(height, width, 1)))
    if sigma <= 0 or height < 3 or width < 3:
        return np.full((height, width), far, dtype=_FLOAT)
    blurred = _gaussian_blur(image, sigma)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)
    peak = float(_approx_percentile(magnitude, 99.5))
    if peak <= 1e-12:
        return np.full((height, width), far, dtype=_FLOAT)
    floor = _mad_floor(magnitude) * 4.0 + float(np.median(_subsample_for_stats(magnitude)))
    high = max(0.12 * peak, floor)
    low = max(0.04 * peak, 0.35 * floor)
    strong = magnitude >= high
    weak = magnitude >= low
    raw = np.zeros(magnitude.shape, dtype=np.uint8)
    if np.any(weak):
        _nlab, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
            weak.astype(np.uint8), 8
        )
        keep_labels = np.unique(labels[strong])
        keep_labels = keep_labels[keep_labels != 0]
        if keep_labels.size:
            raw = np.isin(labels, keep_labels).astype(np.uint8)
        else:
            raw = strong.astype(np.uint8)

    length = max(int(min_length), 1)
    raw = _keep_long_edges(raw, length)
    if np.any(raw):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
        raw = _keep_long_edges(raw, length)
    if not np.any(raw):
        ridge = np.zeros((height, width), dtype=bool)
    else:
        ridge = skeletonize(raw > 0)
        if not np.any(ridge):
            ridge = raw > 0
    ridge[0, :] = True
    ridge[-1, :] = True
    ridge[:, 0] = True
    ridge[:, -1] = True
    seeds = np.where(ridge, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(seeds, cv2.DIST_L2, 5)
    return np.ascontiguousarray(dist, dtype=_FLOAT)


def soften_structure_edges(
    residual: np.ndarray,
    image: np.ndarray,
    sigma: float,
    strength: float,
    exclude_px: float = 0.0,
    distance: np.ndarray | None = None,
    protect: np.ndarray | None = None,
) -> np.ndarray:
    """Attenuate residual in a wide band around coarse region borders.

    A narrow dip only on the gradient ridge just moves DoG peaks *inside* the
    box. This uses distance-to-edge with a smooth ramp of width ``exclude_px``
    so the ringing band is suppressed too. ``strength <= 0`` or ``sigma <= 0``
    leaves the residual unchanged.

    ``protect`` is a boolean mask of compact residual islands that should
    keep most of their residual even on a region border (large flakes, or
    ≥20 µm debris on a *single* edge). Pad-corner L-junctions stay unmasked.
    """
    if strength <= 0 or sigma <= 0:
        return np.ascontiguousarray(residual, dtype=_FLOAT)
    if distance is None:
        distance = coarse_edge_distance(image, sigma)
    margin = max(float(exclude_px), float(sigma), 1.0)
    ramp = np.clip(distance / _FLOAT(margin), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    ramp = np.power(ramp, max(float(strength), 1.0))
    if protect is not None:
        ramp = np.where(protect, np.maximum(ramp, _FLOAT(0.85)), ramp)
    return np.ascontiguousarray(residual, dtype=_FLOAT) * ramp.astype(_FLOAT, copy=False)


def _is_structure_outlier(
    size: float,
    snr: float,
    neighbor_sizes: np.ndarray,
    neighbor_snrs: np.ndarray | None = None,
    size_ratio: float = _STRUCTURE_SIZE_RATIO,
    snr_ratio: float = _STRUCTURE_SNR_RATIO,
) -> bool:
    """True if this blob is much larger (or higher SNR) than its neighbors."""
    if neighbor_sizes.size:
        median_size = float(np.median(neighbor_sizes))
        if median_size > 0 and float(size) >= float(size_ratio) * median_size:
            return True
    if neighbor_snrs is not None and neighbor_snrs.size:
        median_snr = float(np.median(neighbor_snrs))
        if median_snr > 0 and float(snr) >= float(snr_ratio) * median_snr:
            return True
    return False


def reject_clustered_candidates(
    candidates: list[ParticleCandidate],
    neighbor_px: float,
    min_neighbors: int = 2,
    size_ratio: float = _STRUCTURE_SIZE_RATIO,
) -> list[ParticleCandidate]:
    """Drop blobs that sit in a chain or grid of other blobs.

    Isolated particles have no nearby hits. Region-border beads and repeating
    layout cells have several neighbors within ``neighbor_px``. A blob that is
    clearly larger or higher-SNR than those neighbors is kept (real flake in
    a scatter of small layout nodes). ``neighbor_px`` or ``min_neighbors``
    <= 0 disables the filter.
    """
    if neighbor_px <= 0 or min_neighbors <= 0 or len(candidates) < min_neighbors + 1:
        return candidates
    coords = np.array([[c.y_local, c.x_local] for c in candidates], dtype=np.float64)
    sizes = np.array([c.size for c in candidates], dtype=np.float64)
    snrs = np.array([c.local_peak_snr for c in candidates], dtype=np.float64)
    tree = cKDTree(coords)
    keep: list[ParticleCandidate] = []
    ratio = float(size_ratio)
    for index, candidate in enumerate(candidates):
        nearby = tree.query_ball_point(coords[index], float(neighbor_px))
        others = [i for i in nearby if i != index]
        if len(others) < int(min_neighbors):
            keep.append(candidate)
            continue
        if _is_structure_outlier(
            candidate.size,
            candidate.local_peak_snr,
            sizes[others],
            snrs[others],
            size_ratio=ratio,
        ):
            keep.append(candidate)
    return keep


def reject_axis_aligned_chains(
    candidates: list[ParticleCandidate],
    bin_px: float = 10.0,
    min_run: int = 3,
    min_span: float = 48.0,
    max_gap: float = _STRUCTURE_LINE_MAX_GAP_PX,
    size_ratio: float = _STRUCTURE_SIZE_RATIO,
) -> list[ParticleCandidate]:
    """Drop blobs that form a long horizontal or vertical row (box frames).

    Isolated specks do not share a row/column with many others over a long
    span. A blob that is clearly larger than the rest of the run is kept.
    Runs with a hole larger than ``max_gap`` are not frames (two flakes can
    share a 10 px row by chance). ``bin_px`` or ``min_run`` <= 0 disables
    the filter. ``max_gap`` <= 0 skips the hole check.
    """
    if bin_px <= 0 or min_run <= 0 or len(candidates) < min_run:
        return candidates
    ys = np.array([c.y_local for c in candidates], dtype=np.float64)
    xs = np.array([c.x_local for c in candidates], dtype=np.float64)
    sizes = np.array([c.size for c in candidates], dtype=np.float64)
    drop = np.zeros(len(candidates), dtype=bool)
    ratio = float(size_ratio)
    gap_limit = float(max_gap)
    for primary, secondary in ((ys, xs), (xs, ys)):
        bins = np.round(primary / float(bin_px))
        for value in np.unique(bins):
            idx = np.flatnonzero(bins == value)
            if idx.size < int(min_run):
                continue
            order = idx[np.argsort(secondary[idx])]
            along = secondary[order]
            if float(along[-1] - along[0]) < float(min_span):
                continue
            if gap_limit > 0 and order.size >= 2:
                holes = np.diff(along)
                if float(np.max(holes)) > gap_limit:
                    continue
            median_size = float(np.median(sizes[idx]))
            for index in idx:
                if not _is_structure_outlier(
                    sizes[index],
                    0.0,
                    np.array([median_size], dtype=np.float64),
                    size_ratio=ratio,
                ):
                    drop[index] = True
    return [cand for i, cand in enumerate(candidates) if not drop[i]]


def attach_neighbor_counts(
    candidates: list[ParticleCandidate],
    neighbor_px: float = NEIGHBOR_FEATURE_PX,
) -> list[ParticleCandidate]:
    """Set ``n_neighbors_48`` from a spatial radius (does not drop blobs)."""
    if not candidates:
        return candidates
    radius = float(neighbor_px) if neighbor_px > 0 else NEIGHBOR_FEATURE_PX
    coords = np.array([[c.y_local, c.x_local] for c in candidates], dtype=np.float64)
    tree = cKDTree(coords)
    counted: list[ParticleCandidate] = []
    for index, candidate in enumerate(candidates):
        nearby = tree.query_ball_point(coords[index], radius)
        counted.append(replace(candidate, n_neighbors_48=float(len(nearby) - 1)))
    return counted


def _blob_window(
    image: np.ndarray,
    y: float,
    x: float,
    radius: float,
    half: int | None = None,
) -> tuple[np.ndarray, int, int, int, int] | None:
    """Patch around ``(y, x)`` and the peak's integer indices."""
    height, width = image.shape
    span = int(half) if half is not None else max(int(np.ceil(3.0 * radius)), 4)
    span = max(span, 4)
    iy, ix = int(round(y)), int(round(x))
    y0, y1 = max(iy - span, 0), min(iy + span + 1, height)
    x0, x1 = max(ix - span, 0), min(ix + span + 1, width)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    if patch.size == 0:
        return None
    return patch, iy, ix, y0, x0


def blob_support_area(
    residual: np.ndarray,
    y: float,
    x: float,
    radius: float,
    mass_fraction: float = _CIRCULARITY_MASS_FRACTION,
) -> float:
    """Pixel count of the bright support around ``(y, x)``.

    Plateaus and line segments fill most of the window; compact particles do
    not. Used to drop layout pads that DoG reports at a small sigma.
    """
    window = _blob_window(residual, y, x, radius)
    if window is None:
        return 0.0
    patch, iy, ix, _y0, _x0 = window
    height, width = residual.shape
    cy_i = min(max(iy, 0), height - 1)
    cx_i = min(max(ix, 0), width - 1)
    center = float(residual[cy_i, cx_i])
    thresh = float(mass_fraction) * center
    if thresh <= 0:
        return 0.0
    return float(np.count_nonzero(patch >= thresh))


def blob_equivalent_diameter(
    image: np.ndarray,
    y: float,
    x: float,
    radius: float,
    mass_fraction: float = _CIRCULARITY_MASS_FRACTION,
    *,
    bright: bool = True,
    half: int | None = None,
) -> float:
    """Equivalent circular diameter (px) of the peak's connected footprint on the photo.

    DoG ``sigma`` is a discrete search scale, not a measurement. Size is
    ``2 √(A / π)`` where ``A`` is the 8-connected pixels that stand out from
    the local photo background by ``mass_fraction`` of the peak contrast
    (bright particles: ≥ bg + f×(peak−bg); dark: the inverse).
    """
    window = _blob_window(image, y, x, radius, half=half)
    if window is None:
        return 0.0
    patch, iy, ix, y0, x0 = window
    height, width = image.shape
    cy_i = min(max(iy, 0), height - 1)
    cx_i = min(max(ix, 0), width - 1)
    center = float(image[cy_i, cx_i])
    cy = float(y) - y0
    cx = float(x) - x0
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    dist = np.hypot(yy - cy, xx - cx)
    span = 0.5 * float(max(patch.shape[0], patch.shape[1], 1))
    r_bg = max(1.5 * float(radius), 0.65 * span)
    bg_vals = patch[dist >= r_bg]
    if bg_vals.size < 8:
        bg_vals = patch.ravel()
    background = float(np.median(bg_vals))
    if bright:
        contrast = center - background
        if contrast <= 1e-12:
            return 0.0
        thresh = background + float(mass_fraction) * contrast
        mask = (patch >= thresh).astype(np.uint8)
    else:
        contrast = background - center
        if contrast <= 1e-12:
            return 0.0
        thresh = background - float(mass_fraction) * contrast
        mask = (patch <= thresh).astype(np.uint8)
    py = min(max(iy - y0, 0), mask.shape[0] - 1)
    px = min(max(ix - x0, 0), mask.shape[1] - 1)
    if mask[py, px] == 0:
        return 0.0
    _n_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    area = float(np.count_nonzero(labels == labels[py, px]))
    if area <= 0:
        return 0.0
    return float(2.0 * np.sqrt(area / np.pi))


def blob_local_peak_snr(
    image: np.ndarray,
    y: float,
    x: float,
    radius: float,
    bright: bool = True,
) -> float:
    """Local peak SNR of ``(y, x)`` versus the patch median (MAD-scaled)."""
    height, width = image.shape
    half = max(int(np.ceil(3.0 * radius)), 4)
    iy, ix = int(round(y)), int(round(x))
    y0, y1 = max(iy - half, 0), min(iy + half + 1, height)
    x0, x1 = max(ix - half, 0), min(ix + half + 1, width)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    if patch.size == 0:
        return 0.0
    cy_i = min(max(iy, 0), height - 1)
    cx_i = min(max(ix, 0), width - 1)
    center = float(image[cy_i, cx_i])
    med = float(np.median(patch))
    mad = float(np.median(np.abs(patch - med)))
    floor = max(1.4826 * mad, 1e-6)
    excess = (center - med) if bright else (med - center)
    return float(excess / floor)


def blob_is_local_peak(
    image: np.ndarray,
    y: float,
    x: float,
    radius: float,
    bright: bool = True,
    min_snr: float = _LOCAL_PEAK_SNR,
) -> bool:
    """True if ``(y, x)`` is a strong local intensity peak on ``image``.

    Top-hat and flattening can leave residual peaks on empty dark field.
    Those are not particles: the preprocessed intensity is not a high-SNR
    extremum relative to the local median.
    """
    return blob_local_peak_snr(image, y, x, radius, bright=bright) >= float(min_snr)


def blob_radial_means(
    residual: np.ndarray,
    y: float,
    x: float,
    radius: float,
) -> tuple[float, float, float]:
    """Mean residual in rings ``≤0.5R``, ``0.5–1R``, and ``1–1.5R`` around the peak."""
    window = _blob_window(residual, y, x, radius)
    if window is None:
        return 0.0, 0.0, 0.0
    patch, _iy, _ix, y0, x0 = window
    cy = float(y) - y0
    cx = float(x) - x0
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    dist = np.hypot(yy - cy, xx - cx)
    r = max(float(radius), 1.0)

    def _mean(mask: np.ndarray) -> float:
        values = patch[mask]
        return float(values.mean()) if values.size else 0.0

    inner = _mean(dist <= 0.5 * r)
    mid = _mean((dist > 0.5 * r) & (dist <= r))
    outer = _mean((dist > r) & (dist <= 1.5 * r))
    return inner, mid, outer


def blob_circularity(
    residual: np.ndarray,
    y: float,
    x: float,
    radius: float,
) -> float:
    """Inertia-ratio circularity in ``[0, 1]`` (1 = round, 0 = line-like)."""
    height, width = residual.shape
    half = max(int(np.ceil(3.0 * radius)), 4)
    iy, ix = int(round(y)), int(round(x))
    y0, y1 = max(iy - half, 0), min(iy + half + 1, height)
    x0, x1 = max(ix - half, 0), min(ix + half + 1, width)
    patch = np.asarray(residual[y0:y1, x0:x1], dtype=np.float64)
    if patch.size < 9:
        return 1.0
    cy_i = min(max(iy, 0), height - 1)
    cx_i = min(max(ix, 0), width - 1)
    center = float(residual[cy_i, cx_i])
    thresh = _CIRCULARITY_MASS_FRACTION * center
    if thresh <= 0:
        return 1.0
    weights = np.clip(patch - thresh, 0.0, None)
    mass = float(weights.sum())
    if mass <= 1e-8:
        return 1.0
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    inv = 1.0 / mass
    cy = float((weights * yy).sum() * inv)
    cx = float((weights * xx).sum() * inv)
    dy = yy - cy
    dx = xx - cx
    mu20 = float((weights * dx * dx).sum() * inv)
    mu02 = float((weights * dy * dy).sum() * inv)
    mu11 = float((weights * dx * dy).sum() * inv)
    trace = mu20 + mu02
    disc = max(trace * trace - 4.0 * (mu20 * mu02 - mu11 * mu11), 0.0)
    spread = float(np.sqrt(disc))
    lam_max = 0.5 * (trace + spread)
    lam_min = 0.5 * (trace - spread)
    if lam_max <= 1e-12:
        return 1.0
    return float(max(lam_min, 0.0) / lam_max)


def _size_bounds_px(config: dict[str, Any]) -> tuple[float, float]:
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    min_nm = float(cfg_get(config, "detection.min_size_nm", 10.0))
    max_nm = float(cfg_get(config, "detection.max_size_nm", 100.0))
    if pixel_size <= 0:
        pixel_size = 1.0
    return min_nm / pixel_size, max_nm / pixel_size


def _blob_sigma_ratio(config: dict[str, Any], min_sigma: float, max_sigma: float) -> float:
    configured = cfg_get(config, "detection.blob_sigma_ratio", None)
    if configured is not None:
        return max(float(configured), 1.05)
    num_sigma = int(cfg_get(config, "detection.blob_num_sigma", 5))
    if max_sigma > min_sigma and num_sigma > 1:
        return max(float((max_sigma / min_sigma) ** (1.0 / (num_sigma - 1))), 1.05)
    return 1.4


def _local_snr_sigma(config: dict[str, Any], max_sigma: float) -> float:
    configured = cfg_get(config, "detection.local_snr_sigma", None)
    if configured is None or float(configured) <= 0:
        return max(float(max_sigma) * _AUTO_SNR_SIGMA_FACTOR, 1.0)
    return float(configured)


def _snr_sigma_for_residual(
    config: dict[str, Any], max_sigma: float, used_tophat: bool
) -> float:
    """Local SNR sigma for this residual. ``< 0`` disables. Top-hat skips auto SNR.

    Top-hat is already a local-contrast map. Auto SNR on it boosts dark-field
    noise into thousands of DoG hits, which then cluster-kill real particles.
    """
    configured = cfg_get(config, "detection.local_snr_sigma", None)
    if configured is not None and float(configured) < 0:
        return 0.0
    if used_tophat and (configured is None or float(configured) <= 0):
        return 0.0
    return _local_snr_sigma(config, max_sigma)


def _probe_has_lattice(
    image: np.ndarray,
    peak_threshold: float,
    notch_radius: int,
) -> bool:
    """True if a full-resolution FFT notch is worth running.

    Small tiles skip the probe (the real FFT is cheap). Large mixed-layout
    tiles with no thumbnail peaks go straight to top-hat.
    """
    array = np.ascontiguousarray(image, dtype=_FLOAT)
    height, width = array.shape[:2]
    if min(height, width) <= _LATTICE_PROBE_MIN:
        return True
    scale = min(height, width) / float(_LATTICE_PROBE_SIZE)
    small_w = max(int(round(width / scale)), 32)
    small_h = max(int(round(height / scale)), 32)
    small = cv2.resize(array, (small_w, small_h), interpolation=cv2.INTER_AREA)
    mask = build_fft_notch_mask(small, peak_threshold, max(int(notch_radius), 2))
    return bool(np.any(mask))


def _background_residual(
    array: np.ndarray,
    config: dict[str, Any],
    fft_mask: np.ndarray | None,
) -> tuple[np.ndarray, bool]:
    """Periodic-background residual and whether a morphological top-hat was used.

    FFT on mixed-layout tiles often finds no lattice peaks (empty notch mask)
    and would otherwise leave the raw image unchanged. In that case this
    falls back to white top-hat so compact bright debris still stands out.
    A cheap thumbnail FFT decides whether the full-resolution transform is
    worth running.
    """
    method = str(cfg_get(config, "detection.method", "fft")).lower()
    tophat_radius = int(cfg_get(config, "detection.tophat_radius", 8))
    if method == "tophat":
        return suppress_periodic_tophat(array, tophat_radius), True
    peak_threshold = float(cfg_get(config, "detection.fft_peak_threshold", 0.35))
    notch_radius = int(cfg_get(config, "detection.fft_notch_radius", 3))
    mask = fft_mask
    if mask is not None and mask.shape[0] == array.shape[0]:
        if not np.any(mask):
            return suppress_periodic_tophat(array, tophat_radius), True
        return (
            suppress_periodic_fft(array, peak_threshold, notch_radius, mask=mask),
            False,
        )
    if not _probe_has_lattice(array, peak_threshold, notch_radius):
        return suppress_periodic_tophat(array, tophat_radius), True
    spectrum_array = np.ascontiguousarray(array, dtype=_FLOAT)
    spectrum = sp_fft.rfft2(spectrum_array, workers=1)
    mask = _notch_mask_from_spectrum(spectrum, peak_threshold, notch_radius)
    if not np.any(mask):
        return suppress_periodic_tophat(array, tophat_radius), True
    spectrum[mask] = 0
    return sp_fft.irfft2(spectrum, s=spectrum_array.shape, workers=1), False


def _prune_overlapping_blobs(blobs: np.ndarray, overlap: float) -> np.ndarray:
    """Drop the smaller-sigma blob when two circles overlap by more than ``overlap``."""
    if blobs.shape[0] < 2 or overlap >= 1.0:
        return blobs
    radii = blobs[:, 2] * float(np.sqrt(2.0))
    tree = cKDTree(blobs[:, :2])
    pairs = tree.query_pairs(float(2.0 * radii.max()) + 1e-6)
    if not pairs:
        return blobs
    drop = np.zeros(blobs.shape[0], dtype=bool)
    for i, j in pairs:
        if drop[i] or drop[j]:
            continue
        d = float(np.hypot(blobs[i, 0] - blobs[j, 0], blobs[i, 1] - blobs[j, 1]))
        r1 = float(radii[i])
        r2 = float(radii[j])
        if d >= r1 + r2:
            continue
        smaller = max(min(r1, r2), 1e-6)
        contained = d <= abs(r1 - r2)
        overlap_frac = 1.0 if contained else (r1 + r2 - d) / (2.0 * smaller)
        if overlap_frac > overlap:
            if blobs[i, 2] > blobs[j, 2]:
                drop[j] = True
            else:
                drop[i] = True
    return blobs[~drop]


def blob_dog_fast(
    image: np.ndarray,
    min_sigma: float,
    max_sigma: float,
    sigma_ratio: float = 1.4,
    threshold: float = 0.08,
    overlap: float = 0.5,
) -> np.ndarray:
    """Difference-of-Gaussians blobs, same scale convention as ``skimage.feature.blob_dog``.

    Uses OpenCV Gaussians (large sigmas at reduced resolution) instead of
    scikit-image's scipy filters. Returns ``(n, 3)`` array of ``(y, x, sigma)``.
    """
    array = np.ascontiguousarray(image, dtype=_FLOAT)
    if array.size == 0 or float(np.max(array)) < float(threshold):
        return np.empty((0, 3), dtype=np.float64)
    min_sigma = max(float(min_sigma), 0.5)
    max_sigma = max(float(max_sigma), min_sigma)
    sigma_ratio = max(float(sigma_ratio), 1.05)
    k = int(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1)
    sigma_list = np.array(
        [min_sigma * (sigma_ratio**i) for i in range(k + 1)], dtype=np.float64
    )
    gaussians = [_gaussian_blur_full(array, float(sigma)) for sigma in sigma_list]
    scale = _FLOAT(1.0 / (sigma_ratio - 1.0))
    dogs: list[np.ndarray] = []
    for prev, current in zip(gaussians, gaussians[1:]):
        dog = np.subtract(prev, current, dtype=_FLOAT)
        dog *= scale
        dogs.append(dog)
    del gaussians

    kernel = np.ones((3, 3), dtype=np.uint8)
    peaks: list[np.ndarray] = []
    n_scales = len(dogs)
    for index, dog in enumerate(dogs):
        spatial = cv2.dilate(dog, kernel)
        local = (dog >= spatial) & (dog >= _FLOAT(threshold))
        if index > 0:
            local &= dog >= cv2.dilate(dogs[index - 1], kernel)
        if index + 1 < n_scales:
            local &= dog >= cv2.dilate(dogs[index + 1], kernel)
        ys, xs = np.nonzero(local)
        if ys.size == 0:
            continue
        peaks.append(
            np.column_stack(
                (
                    ys.astype(np.float64),
                    xs.astype(np.float64),
                    np.full(ys.size, sigma_list[index], dtype=np.float64),
                )
            )
        )
    if not peaks:
        return np.empty((0, 3), dtype=np.float64)
    return _prune_overlapping_blobs(np.vstack(peaks), float(overlap))


def _orthogonal_image_junction(
    image: np.ndarray,
    y: float,
    x: float,
    half: int = _ISLAND_JUNCTION_HALF_PX,
    min_frac: float = 0.55,
) -> bool:
    """True if a long horizontal AND vertical intensity step meet near ``(y, x)``.

    Pad corners are L-junctions of two box edges. A flake on a single scribe
    or pad side has only one long run. Uses the preprocessed image, not the
    skeletonized ridge (pad corners often sit tens of px inside that ridge).
    """
    iy, ix = int(round(y)), int(round(x))
    height, width = image.shape[:2]
    y0, y1 = max(iy - half, 0), min(iy + half + 1, height)
    x0, x1 = max(ix - half, 0), min(ix + half + 1, width)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    if min(patch.shape) < 8:
        return False
    gx = np.abs(np.diff(patch, axis=1))
    gy = np.abs(np.diff(patch, axis=0))
    if gx.size == 0 or gy.size == 0:
        return False
    thr_x = max(0.12, float(np.percentile(gx, 80)))
    thr_y = max(0.12, float(np.percentile(gy, 80)))
    v_run = int(np.max(np.sum(gx >= thr_x, axis=0)))
    h_run = int(np.max(np.sum(gy >= thr_y, axis=1)))
    min_run = int(min_frac * min(patch.shape))
    return h_run >= min_run and v_run >= min_run


def _compact_island_protect(
    scaled_residual: np.ndarray,
    image: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Boolean mask of compact residual islands to keep on layout ridges."""
    pixel = float(cfg_get(config, "pixel_size_nm", 1.0) or 1.0)
    min_d = float(cfg_get(config, "detection.island_min_nm", _ISLAND_MIN_NM)) / pixel
    large_d = float(cfg_get(config, "detection.island_large_nm", _ISLAND_LARGE_NM)) / pixel
    max_aspect = float(cfg_get(config, "detection.island_max_aspect", _ISLAND_MAX_ASPECT))
    thresh = float(cfg_get(config, "detection.min_prominence", 0.30))
    open_px = int(cfg_get(config, "detection.island_open_px", _ISLAND_OPEN_PX))
    min_solid = float(cfg_get(config, "detection.island_min_solidity", _ISLAND_MIN_SOLIDITY))
    mask = (np.ascontiguousarray(scaled_residual, dtype=_FLOAT) >= _FLOAT(thresh)).astype(
        np.uint8
    )
    if open_px > 0:
        k = 2 * int(open_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    protect = np.zeros(scaled_residual.shape, dtype=bool)
    for index in range(1, n_labels):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if area < 4.0:
            continue
        ecd = float(2.0 * np.sqrt(area / np.pi))
        if ecd < min_d:
            continue
        width = float(stats[index, cv2.CC_STAT_WIDTH])
        height = float(stats[index, cv2.CC_STAT_HEIGHT])
        box = max(width * height, 1.0)
        aspect = max(width, height) / max(min(width, height), 1.0)
        if aspect > max_aspect:
            continue
        if area / box < min_solid:
            continue
        cx = float(centroids[index, 0])
        cy = float(centroids[index, 1])
        if ecd < large_d and _orthogonal_image_junction(image, cy, cx):
            continue
        protect |= labels == index
    return protect


def _island_seed_blobs(protect: np.ndarray) -> np.ndarray:
    """DoG-scale seeds at centroids of protected islands (y, x, sigma)."""
    if protect is None or not np.any(protect):
        return np.empty((0, 3), dtype=np.float64)
    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        protect.astype(np.uint8), 8
    )
    rows: list[list[float]] = []
    for index in range(1, n_labels):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if area < 4.0:
            continue
        ecd = float(2.0 * np.sqrt(area / np.pi))
        sigma = ecd / (2.0 * np.sqrt(2.0))
        rows.append([float(centroids[index, 1]), float(centroids[index, 0]), sigma])
    if not rows:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def compute_tile_residual(
    image: np.ndarray,
    config: dict[str, Any],
    fft_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the corrected tile and the DoG residual, without blob finding.

    Used by the unmarked-crop ML filter so training and inference see the
    same residual the proposer uses. Circled label JPEGs are never read.
    """
    array, residual, _edge, _protect = _tile_residual_maps(image, config, fft_mask)
    return array, residual


def _tile_residual_maps(
    image: np.ndarray,
    config: dict[str, Any],
    fft_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Corrected float image, residual, coarse edge-distance, island-protect mask."""
    config = with_recall_profile(config)
    array = np.ascontiguousarray(image, dtype=_FLOAT)
    residual, used_tophat = _background_residual(array, config, fft_mask)

    particles_bright = bool(cfg_get(config, "detection.particles_bright", True))
    if not particles_bright:
        residual = -residual

    max_sigma = float(cfg_get(config, "detection.blob_max_sigma", 8.0))
    edge_sigma = float(cfg_get(config, "detection.edge_soften_sigma", 12.0))
    edge_strength = float(cfg_get(config, "detection.edge_soften_strength", 2.0))
    exclude_px = float(cfg_get(config, "detection.edge_exclude_px", 12.0))
    min_length = float(cfg_get(config, "detection.edge_min_length_px", 40.0))
    min_prominence = float(cfg_get(config, "detection.min_prominence", 0.30))
    edge_distance = (
        coarse_edge_distance(array, edge_sigma, min_length=min_length)
        if edge_sigma > 0
        else None
    )
    residual = local_snr(residual, _snr_sigma_for_residual(config, max_sigma, used_tophat))
    scaled = _scale_unit(residual)
    protect = _compact_island_protect(scaled, array, config)
    residual = soften_structure_edges(
        residual,
        array,
        edge_sigma,
        edge_strength,
        exclude_px=exclude_px,
        distance=edge_distance,
        protect=protect,
    )
    residual = _scale_unit(residual)
    if min_prominence > 0:
        residual = np.where(residual >= min_prominence, residual, 0).astype(_FLOAT, copy=False)
    return array, residual, edge_distance, protect


def _residual_and_blobs(
    image: np.ndarray,
    config: dict[str, Any],
    fft_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Preprocess residual and run DoG/LoG. ``blobs`` is ``(N, 3)`` y, x, sigma."""
    config = with_recall_profile(config)
    array, residual, edge_distance, protect = _tile_residual_maps(
        image, config, fft_mask
    )
    min_sigma = float(cfg_get(config, "detection.blob_min_sigma", 1.0))
    max_sigma = float(cfg_get(config, "detection.blob_max_sigma", 8.0))
    threshold = float(cfg_get(config, "detection.blob_threshold", 0.025))
    blob_method = str(cfg_get(config, "detection.blob_method", "dog")).lower()
    empty = np.empty((0, 3), dtype=np.float64)
    if float(np.max(residual)) < threshold:
        seeds = _island_seed_blobs(protect)
        return array, residual, edge_distance, seeds, protect
    if blob_method == "log":
        blobs = blob_log(
            residual,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=int(cfg_get(config, "detection.blob_num_sigma", 5)),
            threshold=threshold,
            overlap=0.5,
        )
    else:
        blobs = blob_dog_fast(
            residual,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            sigma_ratio=_blob_sigma_ratio(config, min_sigma, max_sigma),
            threshold=threshold,
            overlap=0.5,
        )
    blobs = np.asarray(blobs, dtype=np.float64)
    seeds = _island_seed_blobs(protect)
    if seeds.size:
        blobs = np.vstack([blobs, seeds]) if blobs.size else seeds
        blobs = _prune_overlapping_blobs(blobs, 0.5)
    if not blobs.size:
        blobs = empty
    return array, residual, edge_distance, blobs, protect


def _gate_blob(
    y: float,
    x: float,
    sigma: float,
    residual: np.ndarray,
    array: np.ndarray,
    edge_distance: np.ndarray | None,
    config: dict[str, Any],
    protect: np.ndarray | None = None,
) -> tuple[ParticleCandidate | None, str]:
    """Return a candidate or the first gate that drops this DoG peak."""
    config = with_recall_profile(config)
    min_d, max_d = _size_bounds_px(config)
    min_area = float(cfg_get(config, "detection.min_area_px", 4))
    min_circ = float(cfg_get(config, "detection.min_circularity", 0.0))
    min_confidence = float(cfg_get(config, "detection.min_confidence", 0.0))
    min_prominence = float(cfg_get(config, "detection.min_prominence", 0.30))
    exclude_px = float(cfg_get(config, "detection.edge_exclude_px", 12.0))
    max_support = float(cfg_get(config, "detection.max_support_area_px", 0.0) or 0.0)
    mass_fraction = float(
        cfg_get(config, "detection.size_mass_fraction", _CIRCULARITY_MASS_FRACTION)
    )
    particles_bright = bool(cfg_get(config, "detection.particles_bright", True))
    height, width = residual.shape
    radius = float(sigma) * float(np.sqrt(2.0))
    dog_diameter = 2.0 * radius
    area = np.pi * radius**2
    if dog_diameter < min_d or dog_diameter > max_d or area < min_area:
        return None, "dog_size"
    iy, ix = int(round(y)), int(round(x))
    iy = min(max(iy, 0), height - 1)
    ix = min(max(ix, 0), width - 1)
    on_island = bool(protect is not None and protect[iy, ix])
    prominence_here = float(residual[iy, ix])
    if min_prominence > 0 and prominence_here < min_prominence:
        return None, "prominence"
    if min_confidence > 0 and prominence_here < min_confidence:
        return None, "confidence"
    edge_d = (
        float(edge_distance[iy, ix])
        if edge_distance is not None
        else float(max(height, width, 1))
    )
    if (
        edge_distance is not None
        and exclude_px > 0
        and edge_d <= exclude_px
        and not on_island
    ):
        return None, "edge_exclude"
    circ = blob_circularity(residual, float(y), float(x), radius)
    if min_circ > 0 and circ < min_circ:
        return None, "circularity"
    snr = blob_local_peak_snr(
        array, float(y), float(x), radius, bright=particles_bright
    )
    if snr < _LOCAL_PEAK_SNR and not on_island:
        return None, "local_peak_snr"
    support = blob_support_area(residual, float(y), float(x), radius)
    support_cap = max_support if max_support > 0 else _SUPPORT_AREA_FACTOR * area
    if support > support_cap:
        return None, "support"
    size_half = max(int(np.ceil(3.0 * radius)), int(np.ceil(0.55 * max_d)), 4)
    measured = blob_equivalent_diameter(
        array,
        float(y),
        float(x),
        radius,
        mass_fraction=mass_fraction,
        bright=particles_bright,
        half=size_half,
    )
    diameter = measured if measured > 0 else dog_diameter
    measured_area = np.pi * (diameter * 0.5) ** 2
    if diameter < min_d or diameter > max_d or measured_area < min_area:
        return None, "measured_size"
    inner, mid, outer = blob_radial_means(residual, float(y), float(x), radius)
    candidate = ParticleCandidate(
        y_local=float(y),
        x_local=float(x),
        size=diameter,
        confidence=prominence_here,
        circularity=float(circ),
        support_over_area=float(support / max(area, 1e-6)),
        edge_distance_px=edge_d,
        tile_border_dist_px=float(min(x, y, width - 1.0 - x, height - 1.0 - y)),
        local_peak_snr=float(snr),
        radial_inner=float(inner),
        radial_mid=float(mid),
        radial_outer=float(outer),
    )
    return candidate, "kept"


def apply_structure_filters(
    candidates: list[ParticleCandidate],
    config: dict[str, Any],
) -> list[ParticleCandidate]:
    """Drop clustered / axis-aligned layout hits. Cluster rejection stays on.

    Default pipeline order is gates → these filters → ML. ``structure_filters``
    on ``detect_particles`` can skip this so a caller scores proposals first.
    """
    config = with_recall_profile(config)
    if not candidates:
        return candidates
    counted = attach_neighbor_counts(candidates, NEIGHBOR_FEATURE_PX)
    size_ratio = float(
        cfg_get(config, "detection.structure_size_ratio", _STRUCTURE_SIZE_RATIO)
    )
    clustered = reject_clustered_candidates(
        counted,
        float(cfg_get(config, "detection.structure_neighbor_px", 48.0)),
        int(cfg_get(config, "detection.structure_min_neighbors", 2)),
        size_ratio=size_ratio,
    )
    return reject_axis_aligned_chains(
        clustered,
        bin_px=float(cfg_get(config, "detection.structure_line_bin_px", 10.0)),
        min_run=int(cfg_get(config, "detection.structure_line_min_run", 3)),
        min_span=float(cfg_get(config, "detection.structure_line_min_span_px", 48.0)),
        max_gap=float(
            cfg_get(config, "detection.structure_line_max_gap_px", _STRUCTURE_LINE_MAX_GAP_PX)
        ),
        size_ratio=size_ratio,
    )


def detect_particles(
    image: np.ndarray,
    config: dict[str, Any],
    fft_mask: np.ndarray | None = None,
    structure_filters: bool = True,
) -> list[ParticleCandidate]:
    """Detect particle candidates on a single preprocessed tile.

    Separates periodic structure (FFT notches, or white top-hat if the FFT
    finds no lattice), keeps only compact high-prominence peaks, attenuates
    coarse region borders (pad corners on the gradient ridge are never
    kept; compact residual islands ≥20 µm on a *single* edge, or ≥40 µm
    anywhere, are protected), then runs DoG blobs (LoG if ``blob_method``
    is ``log``). Elongated and clustered layout hits are dropped when
    ``structure_filters`` is true. Returns local pixel coordinates.
    """
    config = with_recall_profile(config)
    _array, residual, edge_distance, blobs, protect = _residual_and_blobs(
        image, config, fft_mask
    )
    if blobs.size == 0:
        return []
    array = np.ascontiguousarray(image, dtype=_FLOAT)
    candidates: list[ParticleCandidate] = []
    for y, x, sigma in blobs:
        cand, _reason = _gate_blob(
            float(y),
            float(x),
            float(sigma),
            residual,
            array,
            edge_distance,
            config,
            protect=protect,
        )
        if cand is not None:
            candidates.append(cand)
    if structure_filters:
        return apply_structure_filters(candidates, config)
    return attach_neighbor_counts(candidates, NEIGHBOR_FEATURE_PX)


def trace_locations(
    image: np.ndarray,
    config: dict[str, Any],
    locations: list[tuple[float, float]],
    fft_mask: np.ndarray | None = None,
    near_px: float = 20.0,
) -> list[dict[str, Any]]:
    """Explain why each ``(y_local, x_local)`` is not a kept hit."""
    config = with_recall_profile(config)
    array, residual, edge_distance, blobs, protect = _residual_and_blobs(
        image, config, fft_mask
    )
    height, width = residual.shape
    exclude_px = float(cfg_get(config, "detection.edge_exclude_px", 12.0))
    min_prominence = float(cfg_get(config, "detection.min_prominence", 0.30))
    rows: list[dict[str, Any]] = []
    blob_xy = (
        np.column_stack([blobs[:, 1], blobs[:, 0]])
        if blobs.size
        else np.empty((0, 2), dtype=np.float64)
    )
    gated: list[ParticleCandidate] = []
    gated_reasons: list[str] = []
    for y, x, sigma in blobs:
        cand, reason = _gate_blob(
            float(y),
            float(x),
            float(sigma),
            residual,
            array,
            edge_distance,
            config,
            protect=protect,
        )
        if cand is not None:
            gated.append(cand)
        else:
            gated.append(
                ParticleCandidate(
                    y_local=float(y), x_local=float(x), size=0.0, confidence=0.0
                )
            )
        gated_reasons.append(reason)
    pre = attach_neighbor_counts(
        [c for c, r in zip(gated, gated_reasons) if r == "kept"],
        NEIGHBOR_FEATURE_PX,
    )
    size_ratio = float(
        cfg_get(config, "detection.structure_size_ratio", _STRUCTURE_SIZE_RATIO)
    )
    clustered = reject_clustered_candidates(
        pre,
        float(cfg_get(config, "detection.structure_neighbor_px", 48.0)),
        int(cfg_get(config, "detection.structure_min_neighbors", 2)),
        size_ratio=size_ratio,
    )
    chained = reject_axis_aligned_chains(
        clustered,
        bin_px=float(cfg_get(config, "detection.structure_line_bin_px", 10.0)),
        min_run=int(cfg_get(config, "detection.structure_line_min_run", 3)),
        min_span=float(cfg_get(config, "detection.structure_line_min_span_px", 48.0)),
        max_gap=float(
            cfg_get(config, "detection.structure_line_max_gap_px", _STRUCTURE_LINE_MAX_GAP_PX)
        ),
        size_ratio=size_ratio,
    )
    chained_ids = {(c.x_local, c.y_local) for c in chained}
    clustered_ids = {(c.x_local, c.y_local) for c in clustered}
    kept_xy = (
        np.array([[c.x_local, c.y_local] for c in chained], dtype=np.float64)
        if chained
        else np.empty((0, 2), dtype=np.float64)
    )

    for y_local, x_local in locations:
        iy = min(max(int(round(y_local)), 0), height - 1)
        ix = min(max(int(round(x_local)), 0), width - 1)
        residual_here = float(residual[iy, ix])
        edge_d = (
            float(edge_distance[iy, ix]) if edge_distance is not None else float("nan")
        )
        kept_dist = float("inf")
        if kept_xy.size:
            kept_dist = float(np.min(np.hypot(kept_xy[:, 0] - x_local, kept_xy[:, 1] - y_local)))
        blob_dist = float("inf")
        nearest_reason = "no_dog_peak"
        if blob_xy.size:
            d = np.hypot(blob_xy[:, 0] - x_local, blob_xy[:, 1] - y_local)
            nearest = int(np.argmin(d))
            blob_dist = float(d[nearest])
            nearest_reason = gated_reasons[nearest]
            near_cand = gated[nearest]
            if nearest_reason == "kept" and (near_cand.x_local, near_cand.y_local) not in chained_ids:
                if blob_dist <= near_px:
                    nearest_reason = (
                        "cluster"
                        if (near_cand.x_local, near_cand.y_local) not in clustered_ids
                        else "axis_chain"
                    )
        if kept_dist <= near_px:
            reason = "already_kept"
        elif blob_dist > near_px:
            if residual_here < min_prominence:
                reason = "no_dog_peak_low_residual"
            elif edge_distance is not None and exclude_px > 0 and edge_d <= exclude_px:
                reason = "no_dog_peak_in_edge_zone"
            else:
                reason = "no_dog_peak"
        else:
            reason = nearest_reason
        rows.append(
            {
                "y_local": float(y_local),
                "x_local": float(x_local),
                "reason": reason,
                "residual": residual_here,
                "edge_distance_px": edge_d,
                "nearest_blob_px": blob_dist,
                "nearest_kept_px": kept_dist,
            }
        )
    return rows

