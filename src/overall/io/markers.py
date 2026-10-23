"""Shared particle-circle drawing for overlays, crops, and reports."""

from __future__ import annotations

import cv2
import numpy as np

CIRCLE_COLOR = (255, 140, 40)
CIRCLE_THICKNESS = 3
MIN_CIRCLE_RADIUS = 14


def marker_radius(size_nm: float, pixel_size_nm: float, scale: float = 1.0) -> int:
    """Orange/review circle radius in display pixels."""
    step = float(pixel_size_nm) if pixel_size_nm > 0 else 1.0
    return max(int(round(float(size_nm) / step / 2.0 * scale)) + 10, MIN_CIRCLE_RADIUS)


def draw_marker(
    rgb: np.ndarray,
    x: int,
    y: int,
    radius: int,
    color: tuple[int, int, int] = CIRCLE_COLOR,
    thickness: int = CIRCLE_THICKNESS,
) -> np.ndarray:
    """Draw one antialiased circle if ``(x, y)`` is inside ``rgb``."""
    if rgb.size == 0:
        return rgb
    if not (0 <= x < rgb.shape[1] and 0 <= y < rgb.shape[0]):
        return rgb
    cv2.circle(
        rgb,
        (x, y),
        int(radius),
        tuple(int(c) for c in color),
        thickness=int(thickness),
        lineType=cv2.LINE_AA,
    )
    return rgb
