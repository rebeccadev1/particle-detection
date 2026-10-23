"""Synthetic structured-surface tiles with known particle positions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def make_structured_tile(
    shape: tuple[int, int] = (128, 128),
    lattice_period: int = 16,
    lattice_sigma: float = 1.1,
    particles: list[tuple[float, float, float]] | None = None,
) -> np.ndarray:
    """Return a float image: periodic lattice plus optional blobs ``(y, x, sigma)``."""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros(shape, dtype=np.float64)
    for y in range(lattice_period // 2, height, lattice_period):
        for x in range(lattice_period // 2, width, lattice_period):
            image += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * lattice_sigma**2))
    for y, x, sigma in particles or []:
        image += 2.4 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma**2))
    image += 0.02
    return image


def make_two_region_tile(
    shape: tuple[int, int] = (128, 192),
    step_x: int = 96,
    bright: float = 0.9,
    dark: float = 0.2,
    particle: tuple[float, float, float] = (64.0, 48.0, 4.0),
) -> np.ndarray:
    """Bright pad | dark pad with one compact blob in the dark field.

    ``particle`` is ``(y, x, sigma)``. The blob is placed left of ``step_x``.
    There is no lattice — this isolates region-border false positives.
    """
    height, width = shape
    image = np.full(shape, float(dark), dtype=np.float64)
    image[:, int(step_x) :] = float(bright)
    y, x, sigma = particle
    yy, xx = np.mgrid[0:height, 0:width]
    image += 0.75 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma**2))
    return image


def make_pad_tile(
    shape: tuple[int, int] = (256, 320),
    pad: tuple[int, int, int, int] = (40, 40, 180, 200),
    bright: float = 0.9,
    dark: float = 0.15,
    particle: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Bright rectangular pad on a dark field, optional compact blob.

    ``pad`` is ``(y0, x0, y1, x1)``. Top-hat leaves compact residual at the
    pad *corners*; those must not be reported as particles.
    """
    height, width = shape
    image = np.full(shape, float(dark), dtype=np.float64)
    y0, x0, y1, x1 = pad
    image[int(y0) : int(y1), int(x0) : int(x1)] = float(bright)
    if particle is not None:
        y, x, sigma = particle
        yy, xx = np.mgrid[0:height, 0:width]
        image += 0.75 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma**2))
    return image


def make_dark_field_flake(
    shape: tuple[int, int] = (280, 320),
    center: tuple[float, float] = (140.0, 160.0),
    axes: tuple[float, float] = (36.0, 32.0),
    dark: float = 0.12,
    bright: float = 0.95,
) -> np.ndarray:
    """Dark field with one soft elliptical flake ``(semi-axis y, semi-axis x)``.

    Default axes ≈ 70×65 px, a large contaminant on a blank pad.
    """
    height, width = shape
    image = np.full(shape, float(dark), dtype=np.float64)
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = center
    ay, ax = axes
    rr = ((yy - cy) / max(float(ay), 1e-6)) ** 2 + (
        (xx - cx) / max(float(ax), 1e-6)
    ) ** 2
    image += (float(bright) - float(dark)) * np.clip(1.0 - rr, 0.0, 1.0) ** 2
    return image


def make_flake_on_step(
    shape: tuple[int, int] = (220, 280),
    step_x: int = 140,
    center: tuple[float, float] = (110.0, 140.0),
    axes: tuple[float, float] = (28.0, 26.0),
    bright: float = 0.9,
    dark: float = 0.15,
    flake: float = 0.95,
) -> np.ndarray:
    """Vertical region step with one elliptical flake sitting on the step.

    ``center`` is ``(y, x)``; place it on ``step_x`` and away from the tile
    border so it is a single-edge hit, not an L-junction pad corner.
    """
    height, width = shape
    image = np.full(shape, float(dark), dtype=np.float64)
    image[:, int(step_x) :] = float(bright)
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = center
    ay, ax = axes
    rr = ((yy - cy) / max(float(ay), 1e-6)) ** 2 + (
        (xx - cx) / max(float(ax), 1e-6)
    ) ** 2
    image += (float(flake) - float(dark)) * np.clip(1.0 - rr, 0.0, 1.0) ** 2
    return image


def write_tile_tiff(path: Path, image: np.ndarray) -> Path:
    """Write a 32-bit TIFF suitable for ``tifffile.imread``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scaled = np.clip(image / max(float(image.max()), 1e-6) * 65535.0, 0, 65535).astype(
        np.uint16
    )
    tifffile.imwrite(path, scaled)
    return path


def detector_test_config(**overrides: object) -> dict:
    """Config tuned for small synthetic tiles (pixel_size_nm = 1)."""
    config: dict = {
        "pixel_size_nm": 1.0,
        "overlap_fraction": 0.25,
        "filename_pattern": r"R(?P<run>\d+)_(?P<row>\d+)_(?P<col>\d+)_(?P<mag>[\d.]+)X\.(?:tiff?|bmp|png|jpe?g)",
        "preprocessing": {
            "denoise": False,
            "denoise_sigma": 0.5,
            "contrast_stretch": True,
            "contrast_percentiles": [0.5, 99.5],
            "flatten_illumination": False,
            "flatten_sigma": 20.0,
        },
        "detection": {
            "method": "tophat",
            "particles_bright": True,
            "fft_peak_threshold": 0.3,
            "fft_notch_radius": 2,
            "tophat_radius": 7,
            "min_size_nm": 6.0,
            "max_size_nm": 40.0,
            "blob_min_sigma": 1.5,
            "blob_max_sigma": 6.0,
            "blob_num_sigma": 5,
            "blob_sigma_ratio": 1.4,
            "blob_method": "dog",
            "blob_threshold": 0.05,
            "min_area_px": 8,
            "fft_mask": "per_tile",
            "local_snr_sigma": 0.0,
            "edge_soften_sigma": 0.0,
            "edge_soften_strength": 0.0,
            "edge_exclude_px": 0.0,
            "edge_min_length_px": 40.0,
            "min_circularity": 0.35,
            "structure_neighbor_px": 0.0,
            "structure_min_neighbors": 2,
            "structure_line_bin_px": 0.0,
            "min_prominence": 0.15,
            "max_support_area_px": 0.0,
        },
        "measurement": {"merge_radius_px": 6.0, "size_aggregation": "max"},
        "report": {"downsample": 2, "target_mb": 20.0},
        "pipeline": {"workers": 1},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config
