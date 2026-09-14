"""Global coordinates, size conversion, and overlap de-duplication."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import cfg_get
from src.detection.detector import CANDIDATE_FEATURE_FIELDS, ParticleCandidate


@dataclass
class Particle:
    """One measured particle in the global frame."""

    id: int
    x_global: float
    y_global: float
    size: float
    confidence: float
    source_tile: str
    circularity: float = 0.0
    support_over_area: float = 0.0
    edge_distance_px: float = 0.0
    tile_border_dist_px: float = 0.0
    n_neighbors_48: float = 0.0
    local_peak_snr: float = 0.0
    radial_inner: float = 0.0
    radial_mid: float = 0.0
    radial_outer: float = 0.0


RESULT_COLUMNS = (
    "id",
    "x_global",
    "y_global",
    "size",
    "confidence",
    "source_tile",
) + CANDIDATE_FEATURE_FIELDS


def local_to_global(
    x_local: float,
    y_local: float,
    origin_x: float,
    origin_y: float,
) -> tuple[float, float]:
    """Map tile-local pixels to mosaic pixels (top-left origin)."""
    return origin_x + x_local, origin_y + y_local


def _feature_kwargs(source: ParticleCandidate | Particle) -> dict[str, float]:
    return {name: float(getattr(source, name)) for name in CANDIDATE_FEATURE_FIELDS}


def measure_candidates(
    candidates: Sequence[ParticleCandidate],
    origin_x: float,
    origin_y: float,
    pixel_size_nm: float,
    source_tile: str,
) -> list[Particle]:
    """Convert local candidates to global coordinates and size in nm.

    ``size`` is equivalent diameter in nm. ``id`` is 0 until pipeline numbering.
    """
    particles: list[Particle] = []
    scale = pixel_size_nm if pixel_size_nm > 0 else 1.0
    for candidate in candidates:
        x_g, y_g = local_to_global(
            candidate.x_local, candidate.y_local, origin_x, origin_y
        )
        particles.append(
            Particle(
                id=0,
                x_global=x_g * scale,
                y_global=y_g * scale,
                size=candidate.size * scale,
                confidence=candidate.confidence,
                source_tile=source_tile,
                **_feature_kwargs(candidate),
            )
        )
    return particles


def deduplicate(
    particles: Sequence[Particle],
    merge_radius: float,
    size_aggregation: str = "max",
) -> list[Particle]:
    """Merge candidates whose global distance is below ``merge_radius``.

    Distance is in the same units as ``x_global`` / ``y_global`` (nm if a
    pixel size was applied). The kept record is the highest-confidence member;
    ``size`` is the max or mean of the cluster (see ``size_aggregation``).
    Features come from the seed.
    """
    if not particles:
        return []
    if merge_radius <= 0:
        return _renumber(list(particles))

    remaining = sorted(particles, key=lambda p: p.confidence, reverse=True)
    coords = np.array([[p.x_global, p.y_global] for p in remaining], dtype=np.float64)
    tree = cKDTree(coords)
    neighbor_lists = tree.query_ball_tree(tree, float(merge_radius))
    used = np.zeros(len(remaining), dtype=bool)
    kept: list[Particle] = []

    for i, seed in enumerate(remaining):
        if used[i]:
            continue
        members_idx = [j for j in neighbor_lists[i] if not used[j]]
        sizes = np.array([remaining[j].size for j in members_idx], dtype=np.float64)
        size = float(np.max(sizes) if size_aggregation == "max" else np.mean(sizes))
        kept.append(replace(seed, id=0, size=size))
        used[members_idx] = True

    return _renumber(kept)


def _renumber(particles: list[Particle]) -> list[Particle]:
    return [replace(particle, id=index) for index, particle in enumerate(particles, start=1)]


def particles_to_dataframe(particles: Sequence[Particle]) -> pd.DataFrame:
    """Build the standard results table, including per-blob features."""
    names = [field.name for field in fields(Particle)]
    rows = [{name: getattr(p, name) for name in names} for p in particles]
    if not rows:
        return pd.DataFrame(columns=list(RESULT_COLUMNS))
    return pd.DataFrame(rows).loc[:, list(RESULT_COLUMNS)]


def measure_and_dedupe(
    particles: Sequence[Particle],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply configured clustering and return a results table."""
    merged = deduplicate(
        particles,
        merge_radius=float(cfg_get(config, "measurement.merge_radius_px", 8.0))
        * float(cfg_get(config, "pixel_size_nm", 1.0)),
        size_aggregation=str(cfg_get(config, "measurement.size_aggregation", "max")),
    )
    return particles_to_dataframe(merged)
