"""Global coordinates, size conversion, and overlap de-duplication."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import cfg_get
from src.detection.detector import CANDIDATE_FEATURE_FIELDS, ParticleCandidate

DIRECTION_STEMS = ("N", "S", "E", "W")
DISPLAY_DIRECTION_ORDER = ("N", "W", "E", "S")
DEFAULT_NSEW_MERGE_RADIUS_PX = 24.0
DEFAULT_NSEW_SIZE_MATCH_FRACTION = 0.4
_PARTICLES_ONLY_PREFIX = "particles_only"


@dataclass
class Particle:
    """One measured particle in the global frame."""

    id: int
    x_global: float
    y_global: float
    size: float
    confidence: float
    source_tile: str
    nsew_count: int = 0
    nsew_dirs: str = ""
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
    "nsew_count",
    "nsew_dirs",
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


def _direction_stem(source_tile: str) -> str | None:
    stem = Path(str(source_tile)).stem.upper()
    return stem if stem in DIRECTION_STEMS else None


def is_direction_tile(source_tile: str) -> bool:
    """True for images named N, S, E, or W (any suffix)."""
    return _direction_stem(source_tile) is not None


def is_particles_only_tile(source_tile: str) -> bool:
    """True for combined particles-only images (any 2/3/4-of-4 version)."""
    stem = Path(str(source_tile)).stem.lower()
    return stem == _PARTICLES_ONLY_PREFIX or stem.startswith(f"{_PARTICLES_ONLY_PREFIX}_")


def is_nsew_family_tile(source_tile: str) -> bool:
    """N/S/E/W and particles-only versions of the same scene share identity."""
    return is_direction_tile(source_tile) or is_particles_only_tile(source_tile)


def nsew_merge_stem(source_tile: str) -> str | None:
    """Cluster id: one slot per lighting direction or particles-only version file."""
    stem = _direction_stem(source_tile)
    if stem is not None:
        return stem
    if is_particles_only_tile(source_tile):
        return Path(str(source_tile)).stem.upper()
    return None


def _nsew_fields(members: Sequence[Particle]) -> dict[str, int | str]:
    dirs: list[str] = []
    seen: set[str] = set()
    for particle in members:
        stem = nsew_merge_stem(particle.source_tile)
        if stem is None or stem in seen:
            continue
        seen.add(stem)
        dirs.append(stem)
    dirs.sort(
        key=lambda name: (
            0,
            DIRECTION_STEMS.index(name),
        )
        if name in DIRECTION_STEMS
        else (1, name)
    )
    return {"nsew_count": len(dirs), "nsew_dirs": ",".join(dirs)}


def _pair_merge_limit(
    merge_radius: float,
    size_a: float,
    size_b: float,
    size_match_fraction: float,
) -> float:
    limit = float(merge_radius)
    if size_match_fraction > 0:
        limit = max(limit, float(size_match_fraction) * max(float(size_a), float(size_b)))
    return limit


def deduplicate(
    particles: Sequence[Particle],
    merge_radius: float,
    size_aggregation: str = "max",
    size_match_fraction: float = 0.0,
) -> list[Particle]:
    """Merge candidates whose global distance is below ``merge_radius``.

    Distance is in the same units as ``x_global`` / ``y_global`` (nm if a
    pixel size was applied). The kept record is the highest-confidence member;
    ``size`` is the max or mean of the cluster (see ``size_aggregation``).
    Features come from the seed. Size is not a match criterion.

    When ``size_match_fraction`` is positive, two hits also merge if they
    are closer than that fraction of the larger diameter (N/S/E/W views of
    the same flake can sit farther apart than the overlap radius).
    """
    if not particles:
        return []
    remaining = sorted(particles, key=lambda p: p.confidence, reverse=True)
    if merge_radius <= 0 and size_match_fraction <= 0:
        return _renumber([replace(p, **_nsew_fields([p])) for p in remaining])

    coords = np.array([[p.x_global, p.y_global] for p in remaining], dtype=np.float64)
    sizes = np.array([p.size for p in remaining], dtype=np.float64)
    search_r = float(merge_radius)
    if size_match_fraction > 0 and len(sizes):
        search_r = max(search_r, float(size_match_fraction) * float(np.max(sizes)))
    if search_r <= 0:
        return _renumber([replace(p, **_nsew_fields([p])) for p in remaining])

    tree = cKDTree(coords)
    neighbor_lists = tree.query_ball_tree(tree, search_r)
    used = np.zeros(len(remaining), dtype=bool)
    kept: list[Particle] = []

    for i, seed in enumerate(remaining):
        if used[i]:
            continue
        members_idx: list[int] = []
        for j in neighbor_lists[i]:
            if used[j]:
                continue
            dist = float(
                np.hypot(
                    seed.x_global - remaining[j].x_global,
                    seed.y_global - remaining[j].y_global,
                )
            )
            if dist <= _pair_merge_limit(
                merge_radius, seed.size, remaining[j].size, size_match_fraction
            ):
                members_idx.append(j)
        members = [remaining[j] for j in members_idx]
        cluster_sizes = np.array([p.size for p in members], dtype=np.float64)
        size = float(
            np.max(cluster_sizes) if size_aggregation == "max" else np.mean(cluster_sizes)
        )
        kept.append(replace(seed, id=0, size=size, **_nsew_fields(members)))
        used[members_idx] = True

    return _renumber(kept)


def deduplicate_nsew(
    particles: Sequence[Particle],
    merge_radius: float,
    size_aggregation: str = "max",
    size_match_fraction: float = DEFAULT_NSEW_SIZE_MATCH_FRACTION,
) -> list[Particle]:
    """Merge N/W/E/S views of the same place. At most one hit per direction."""
    if not particles:
        return []
    remaining = sorted(particles, key=lambda p: p.confidence, reverse=True)
    coords = np.array([[p.x_global, p.y_global] for p in remaining], dtype=np.float64)
    sizes = np.array([p.size for p in remaining], dtype=np.float64)
    search_r = float(merge_radius)
    if size_match_fraction > 0 and len(sizes):
        search_r = max(search_r, float(size_match_fraction) * float(np.max(sizes)))
    if search_r <= 0:
        return _renumber([replace(p, **_nsew_fields([p])) for p in remaining])

    tree = cKDTree(coords)
    neighbor_lists = tree.query_ball_tree(tree, search_r)
    used = np.zeros(len(remaining), dtype=bool)
    kept: list[Particle] = []

    for i, seed in enumerate(remaining):
        if used[i]:
            continue
        neighbors: list[tuple[float, int]] = []
        for j in neighbor_lists[i]:
            if used[j] or j == i:
                continue
            dist = float(
                np.hypot(
                    seed.x_global - remaining[j].x_global,
                    seed.y_global - remaining[j].y_global,
                )
            )
            neighbors.append((dist, j))
        neighbors.sort()
        members_idx = [i]
        stems: set[str] = set()
        seed_stem = nsew_merge_stem(seed.source_tile)
        if seed_stem is not None:
            stems.add(seed_stem)
        for dist, j in neighbors:
            other = remaining[j]
            stem = nsew_merge_stem(other.source_tile)
            if stem is None or stem in stems:
                continue
            if dist <= _pair_merge_limit(
                merge_radius, seed.size, other.size, size_match_fraction
            ):
                members_idx.append(j)
                stems.add(stem)
        members = [remaining[j] for j in members_idx]
        cluster_sizes = np.array([p.size for p in members], dtype=np.float64)
        size = float(
            np.max(cluster_sizes) if size_aggregation == "max" else np.mean(cluster_sizes)
        )
        kept.append(replace(seed, id=0, size=size, **_nsew_fields(members)))
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
    size_match_fraction: float = 0.0,
    merge_radius_nm: float | None = None,
    directional: bool = False,
) -> pd.DataFrame:
    """Apply configured clustering and return a results table."""
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    if merge_radius_nm is None:
        merge_px = float(cfg_get(config, "measurement.merge_radius_px", 8.0))
        if directional:
            merge_px = max(
                merge_px,
                float(
                    cfg_get(
                        config,
                        "measurement.nsew_merge_radius_px",
                        DEFAULT_NSEW_MERGE_RADIUS_PX,
                    )
                ),
            )
        merge_radius_nm = merge_px * pixel_size
    aggregation = str(cfg_get(config, "measurement.size_aggregation", "max"))
    cluster = deduplicate_nsew if directional else deduplicate
    merged = cluster(
        particles,
        merge_radius=float(merge_radius_nm),
        size_aggregation=aggregation,
        size_match_fraction=float(size_match_fraction),
    )
    return particles_to_dataframe(merged)
