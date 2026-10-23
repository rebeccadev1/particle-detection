"""Tabular features for the particle vs not-particle classifier.

Features come from the residual at detection time, never from the circled
JPEG crops in ``labels/crops/`` (the red marker would leak into the model).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.analyse.detection.detector import CANDIDATE_FEATURE_FIELDS, ParticleCandidate

NM_PER_UM = 1000.0
FEATURE_COLUMNS = ("size_um", "confidence") + CANDIDATE_FEATURE_FIELDS


def size_um_from_px(size_px: float, pixel_size_nm: float) -> float:
    scale = float(pixel_size_nm) if pixel_size_nm > 0 else 1.0
    return float(size_px) * scale / NM_PER_UM


def candidate_feature_vector(
    candidate: ParticleCandidate,
    pixel_size_nm: float,
) -> np.ndarray:
    """One row in ``FEATURE_COLUMNS`` order. ``candidate.size`` is pixels."""
    return np.array(
        [
            size_um_from_px(candidate.size, pixel_size_nm),
            float(candidate.confidence),
            *(float(getattr(candidate, name)) for name in CANDIDATE_FEATURE_FIELDS),
        ],
        dtype=np.float64,
    )


def candidates_to_matrix(
    candidates: Sequence[ParticleCandidate],
    pixel_size_nm: float,
) -> np.ndarray:
    if not candidates:
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float64)
    return np.vstack(
        [candidate_feature_vector(candidate, pixel_size_nm) for candidate in candidates]
    )
