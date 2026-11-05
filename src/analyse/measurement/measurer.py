from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.overall.config import cfg_get
from src.overall.io.naming import DIRECTION_STEMS, nsew_merge_stem
from src.analyse.detection.detector import CANDIDATE_FEATURE_FIELDS, ParticleCandidate

DEFAULT_NSEW_MERGE_RADIUS_PX = 24.0
DEFAULT_NSEW_SIZE_MATCH_FRACTION = 0.4


@dataclass
class Particle:
    

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


def _merge_neighbors(
    particles: Sequence[Particle],
    merge_radius: float,
    size_match_fraction: float,
) -> tuple[list[Particle], list[list[int]] | None]:
    remaining = sorted(particles, key=lambda p: p.confidence, reverse=True)
    search_r = float(merge_radius)
    if size_match_fraction > 0 and remaining:
        largest = max(float(p.size) for p in remaining)
        search_r = max(search_r, float(size_match_fraction) * largest)
    if search_r <= 0:
        return remaining, None
    coords = np.array([[p.x_global, p.y_global] for p in remaining], dtype=np.float64)
    tree = cKDTree(coords)
    return remaining, tree.query_ball_tree(tree, search_r)


def _cluster_size(members: Sequence[Particle], size_aggregation: str) -> float:
    sizes = np.array([p.size for p in members], dtype=np.float64)
    if size_aggregation == "max":
        return float(np.max(sizes))
    return float(np.mean(sizes))


def _keep_each(particles: Sequence[Particle]) -> list[Particle]:
    return _renumber([replace(p, **_nsew_fields([p])) for p in particles])


def deduplicate(
    particles: Sequence[Particle],
    merge_radius: float,
    size_aggregation: str = "max",
    size_match_fraction: float = 0.0,
) -> list[Particle]:
    if not particles:
        return []
    remaining, neighbor_lists = _merge_neighbors(
        particles, merge_radius, size_match_fraction
    )
    if neighbor_lists is None:
        return _keep_each(remaining)

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
        kept.append(
            replace(
                seed,
                id=0,
                size=_cluster_size(members, size_aggregation),
                **_nsew_fields(members),
            )
        )
        used[members_idx] = True
    return _renumber(kept)


def deduplicate_nsew(
    particles: Sequence[Particle],
    merge_radius: float,
    size_aggregation: str = "max",
    size_match_fraction: float = DEFAULT_NSEW_SIZE_MATCH_FRACTION,
) -> list[Particle]:
    if not particles:
        return []
    remaining, neighbor_lists = _merge_neighbors(
        particles, merge_radius, size_match_fraction
    )
    if neighbor_lists is None:
        return _keep_each(remaining)

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
        kept.append(
            replace(
                seed,
                id=0,
                size=_cluster_size(members, size_aggregation),
                **_nsew_fields(members),
            )
        )
        used[members_idx] = True
    return _renumber(kept)


def _renumber(particles: list[Particle]) -> list[Particle]:
    return [replace(particle, id=index) for index, particle in enumerate(particles, start=1)]


def particles_to_dataframe(particles: Sequence[Particle]) -> pd.DataFrame:
    
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
