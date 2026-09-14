"""HOG / LBP / layout descriptors on unmarked two-channel crops.

These numbers are meant to see letter strokes and pad corners — the residual
float ExtraTrees never had them. Circled JPEGs are not an input.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.morphology import skeletonize

from src.detection.detector import CANDIDATE_FEATURE_FIELDS, ParticleCandidate
from src.io.tile_loader import DEFAULT_FILENAME_PATTERN
from src.ml.features import size_um_from_nm, size_um_from_px
from src.ml.patches import PATCH_SIZE, tile_row_col

HOG_SIZE = 64
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BINS = LBP_POINTS + 2
HOG_ORIENTATIONS = 8
HOG_PIXELS = 16
HOG_CELLS = 2

TABULAR_LAYOUT_FIELDS = ("size_um", "confidence") + CANDIDATE_FEATURE_FIELDS + (
    "tile_row",
    "tile_col",
)


def _hog_length() -> int:
    cells = HOG_SIZE // HOG_PIXELS
    blocks = max(cells - HOG_CELLS + 1, 1)
    return int((blocks**2) * (HOG_CELLS**2) * HOG_ORIENTATIONS)


GEOM_NAMES = (
    "solidity",
    "eccentricity",
    "extent",
    "skeleton_frac",
    "orient_entropy",
    "cc_aspect",
    "cc_holes",
    "ring_contrast",
)

PATCH_FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"hog_raw_{i}" for i in range(_hog_length()))
    + tuple(f"hog_res_{i}" for i in range(_hog_length()))
    + tuple(f"lbp_raw_{i}" for i in range(LBP_BINS))
    + tuple(f"lbp_res_{i}" for i in range(LBP_BINS))
    + tuple(f"hu_raw_{i}" for i in range(7))
    + tuple(f"hu_res_{i}" for i in range(7))
    + GEOM_NAMES
    + TABULAR_LAYOUT_FIELDS
)


def _resize(channel: np.ndarray, size: int = HOG_SIZE) -> np.ndarray:
    array = np.asarray(channel, dtype=np.float32)
    if array.shape == (size, size):
        return array
    return cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA)


def _hog_vec(channel: np.ndarray) -> np.ndarray:
    image = _resize(channel)
    return hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=(HOG_PIXELS, HOG_PIXELS),
        cells_per_block=(HOG_CELLS, HOG_CELLS),
        block_norm="L2-Hys",
        feature_vector=True,
        transform_sqrt=True,
    ).astype(np.float64)


def _lbp_hist(channel: np.ndarray) -> np.ndarray:
    image = _resize(channel)
    scaled = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    codes = local_binary_pattern(scaled, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(codes, bins=LBP_BINS, range=(0, LBP_BINS), density=True)
    return hist.astype(np.float64)


def _hu(channel: np.ndarray) -> np.ndarray:
    image = _resize(channel)
    scaled = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    moments = cv2.moments(scaled)
    hu = cv2.HuMoments(moments).reshape(-1)
    return np.sign(hu) * np.log1p(np.abs(hu))


def _binary(channel: np.ndarray) -> np.ndarray:
    image = _resize(channel)
    thresh = float(np.percentile(image, 70.0))
    return (image >= max(thresh, 0.15)).astype(np.uint8)


def _geometry(raw: np.ndarray, residual: np.ndarray) -> np.ndarray:
    mask = _binary(residual)
    area = float(np.count_nonzero(mask))
    total = float(mask.size)
    solidity = 0.0
    eccentricity = 0.0
    extent = area / total if total else 0.0
    skeleton_frac = 0.0
    cc_aspect = 1.0
    cc_holes = 0.0
    if area >= 4:
        ys, xs = np.nonzero(mask)
        yy = ys.astype(np.float64)
        xx = xs.astype(np.float64)
        hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
        hull_area = float(cv2.contourArea(hull)) if hull is not None and len(hull) >= 3 else area
        solidity = float(area / hull_area) if hull_area > 0 else 0.0
        cov = np.cov(np.vstack([xx, yy]))
        eig = np.linalg.eigvalsh(cov)
        eig = np.clip(eig, 1e-9, None)
        eccentricity = float(np.sqrt(1.0 - eig[0] / eig[-1])) if eig[-1] > 0 else 0.0
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if nlab > 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            w = max(float(stats[idx, cv2.CC_STAT_WIDTH]), 1.0)
            h = max(float(stats[idx, cv2.CC_STAT_HEIGHT]), 1.0)
            cc_aspect = float(max(w, h) / min(w, h))
        inv = 1 - mask
        n_holes, _ = cv2.connectedComponents(inv, 8)
        cc_holes = float(max(n_holes - 1, 0))
        skeleton_frac = float(np.count_nonzero(skeletonize(mask.astype(bool)))) / area
    gy, gx = np.gradient(_resize(residual).astype(np.float64))
    angles = np.arctan2(gy, gx)
    mag = np.hypot(gx, gy)
    weights = mag.reshape(-1)
    if float(weights.sum()) > 0:
        hist, _ = np.histogram(
            angles.reshape(-1), bins=8, range=(-np.pi, np.pi), weights=weights
        )
        hist = hist / hist.sum()
        orient_entropy = float(-(hist[hist > 0] * np.log(hist[hist > 0] + 1e-12)).sum())
    else:
        orient_entropy = 0.0
    small = _resize(residual)
    h, w = small.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(yy - cy, xx - cx)
    inner = small[dist <= 8]
    ring = small[(dist > 8) & (dist <= 20)]
    ring_contrast = float(inner.mean() - ring.mean()) if inner.size and ring.size else 0.0
    return np.array(
        [
            solidity,
            eccentricity,
            extent,
            skeleton_frac,
            orient_entropy,
            cc_aspect,
            cc_holes,
            ring_contrast,
        ],
        dtype=np.float64,
    )


def patch_descriptor(channels: np.ndarray) -> np.ndarray:
    """Image-only features for one ``(2, H, W)`` unmarked crop."""
    raw = np.asarray(channels[0], dtype=np.float32)
    residual = np.asarray(channels[1], dtype=np.float32)
    parts = (
        _hog_vec(raw),
        _hog_vec(residual),
        _lbp_hist(raw),
        _lbp_hist(residual),
        _hu(raw),
        _hu(residual),
        _geometry(raw, residual),
    )
    return np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts])


def tabular_layout_vector(
    size_um: float = 0.0,
    confidence: float = 0.0,
    circularity: float = 0.0,
    support_over_area: float = 0.0,
    edge_distance_px: float = 0.0,
    tile_border_dist_px: float = 0.0,
    n_neighbors_48: float = 0.0,
    local_peak_snr: float = 0.0,
    radial_inner: float = 0.0,
    radial_mid: float = 0.0,
    radial_outer: float = 0.0,
    tile_row: float = 0.0,
    tile_col: float = 0.0,
) -> np.ndarray:
    return np.array(
        [
            float(size_um),
            float(confidence),
            float(circularity),
            float(support_over_area),
            float(edge_distance_px),
            float(tile_border_dist_px),
            float(n_neighbors_48),
            float(local_peak_snr),
            float(radial_inner),
            float(radial_mid),
            float(radial_outer),
            float(tile_row),
            float(tile_col),
        ],
        dtype=np.float64,
    )


def layout_from_record(
    record: dict[str, object] | None,
    pattern: str = DEFAULT_FILENAME_PATTERN,
) -> np.ndarray:
    row = dict(record or {})
    size = float(row.get("size") or 0.0)
    size_um = size_um_from_nm(size) if size > 100 else float(size)
    tile = str(row.get("source_tile") or "")
    tile_row, tile_col = tile_row_col(tile, pattern)
    return tabular_layout_vector(
        size_um=size_um,
        confidence=float(row.get("confidence") or 0.0),
        circularity=float(row.get("circularity") or 0.0),
        support_over_area=float(row.get("support_over_area") or 0.0),
        edge_distance_px=float(row.get("edge_distance_px") or 0.0),
        tile_border_dist_px=float(row.get("tile_border_dist_px") or 0.0),
        n_neighbors_48=float(row.get("n_neighbors_48") or 0.0),
        local_peak_snr=float(row.get("local_peak_snr") or 0.0),
        radial_inner=float(row.get("radial_inner") or 0.0),
        radial_mid=float(row.get("radial_mid") or 0.0),
        radial_outer=float(row.get("radial_outer") or 0.0),
        tile_row=tile_row,
        tile_col=tile_col,
    )


def layout_from_candidate(
    candidate: ParticleCandidate,
    pixel_size_nm: float,
    source_tile: str = "",
    pattern: str = DEFAULT_FILENAME_PATTERN,
) -> np.ndarray:
    tile_row, tile_col = tile_row_col(source_tile, pattern)
    return tabular_layout_vector(
        size_um=size_um_from_px(candidate.size, pixel_size_nm),
        confidence=float(candidate.confidence),
        circularity=float(candidate.circularity),
        support_over_area=float(candidate.support_over_area),
        edge_distance_px=float(candidate.edge_distance_px),
        tile_border_dist_px=float(candidate.tile_border_dist_px),
        n_neighbors_48=float(candidate.n_neighbors_48),
        local_peak_snr=float(candidate.local_peak_snr),
        radial_inner=float(candidate.radial_inner),
        radial_mid=float(candidate.radial_mid),
        radial_outer=float(candidate.radial_outer),
        tile_row=tile_row,
        tile_col=tile_col,
    )


def patch_feature_vector(
    channels: np.ndarray,
    layout: np.ndarray | None = None,
) -> np.ndarray:
    image_part = patch_descriptor(channels)
    extra = (
        np.asarray(layout, dtype=np.float64)
        if layout is not None
        else np.zeros(len(TABULAR_LAYOUT_FIELDS), dtype=np.float64)
    )
    vector = np.concatenate([image_part, extra])
    if vector.size != len(PATCH_FEATURE_NAMES):
        raise ValueError(
            f"Patch feature length {vector.size} != {len(PATCH_FEATURE_NAMES)}"
        )
    return vector


def patches_to_matrix(
    channels: np.ndarray,
    layouts: Sequence[np.ndarray] | np.ndarray | None = None,
) -> np.ndarray:
    """``channels`` is ``(N, 2, H, W)``."""
    n = int(channels.shape[0])
    if n == 0:
        return np.empty((0, len(PATCH_FEATURE_NAMES)), dtype=np.float64)
    rows = []
    for i in range(n):
        layout = None if layouts is None else np.asarray(layouts[i], dtype=np.float64)
        rows.append(patch_feature_vector(channels[i], layout))
    return np.vstack(rows)
