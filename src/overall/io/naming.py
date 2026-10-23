"""Filename roles for grid, N/S/E/W, combi, and Groundup lighting tiles."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from src.overall.config import DEFAULT_INPUT_DIR

DIRECTION_STEMS = ("N", "S", "E", "W")
LIGHTING_ROLES = ("quad0", "quad1", "quad2", "quad3", "all")
QUAD_DIRECTIONS = ("N", "E", "S", "W")
_PARTICLES_ONLY_PREFIX = "particles_only"
_SET_DIR = re.compile(r"^v\d+$", re.IGNORECASE)
_GRID_TILE = re.compile(r"^\(\d+,\d+\)$")
_WAFER_LIGHT = re.compile(
    r"^wafer_shot_\d+_(quad[0-3]|all)_exp\d+$",
    re.IGNORECASE,
)


def direction_stem(source_tile: str) -> str | None:
    stem = Path(str(source_tile)).stem.upper()
    return stem if stem in DIRECTION_STEMS else None


def is_direction_tile(source_tile: str) -> bool:
    """True for images named N, S, E, or W (any suffix)."""
    return direction_stem(source_tile) is not None


def is_particles_only_tile(source_tile: str) -> bool:
    """True for combined particles-only images (any 2/3/4-of-4 version)."""
    stem = Path(str(source_tile)).stem.lower()
    return stem == _PARTICLES_ONLY_PREFIX or stem.startswith(f"{_PARTICLES_ONLY_PREFIX}_")


def is_nsew_family_tile(source_tile: str) -> bool:
    """N/S/E/W and particles-only versions of the same scene share identity."""
    return is_direction_tile(source_tile) or is_particles_only_tile(source_tile)


def is_grid_tile(source_tile: str) -> bool:
    """``(row,col).png``. The same filename is used in every scan folder."""
    stem = Path(str(source_tile).replace("\\", "/")).stem
    return _GRID_TILE.fullmatch(stem) is not None


def is_combi_tile(source_tile: str) -> bool:
    """True for a combined image written from N, S, E, and W."""
    stem = Path(str(source_tile).replace("\\", "/")).stem.lower()
    return stem == "combi" or stem.startswith("combi_")


def lighting_role(source_tile: str) -> str | None:
    """``quad0``–``quad3`` or ``all`` for one Groundup submap lighting shot."""
    match = _WAFER_LIGHT.fullmatch(Path(str(source_tile).replace("\\", "/")).stem)
    if match is None:
        return None
    return match.group(1).lower()


def quad_direction(source_tile: str) -> str | None:
    """Compass letter for a ``wafer_shot_*_quadN_*`` file. ``all`` has none."""
    role = lighting_role(source_tile)
    if role is None or not role.startswith("quad"):
        return None
    index = int(role[-1])
    if index >= len(QUAD_DIRECTIONS):
        return None
    return QUAD_DIRECTIONS[index]


def is_submap_lighting_tile(source_tile: str) -> bool:
    """True for the five shots of one submap: four quads plus uniform ``all``."""
    return lighting_role(source_tile) is not None


def submap_id(source_tile: str) -> str:
    """``v7`` from ``v7/wafer_shot_0_quad0_….png``. Empty when the folder is not a submap."""
    parent = Path(str(source_tile).replace("\\", "/")).parent.name
    if _SET_DIR.fullmatch(parent):
        return parent.lower()
    return ""


def lighting_tile_id(source_tile: str) -> str:
    """Stable tile id. Submap lighting keeps ``v7/filename`` so v7 is not v8."""
    text = str(source_tile).replace("\\", "/")
    name = Path(text).name
    if is_submap_lighting_tile(text) or is_combi_tile(text):
        set_id = submap_id(text)
        if set_id:
            return f"{set_id}/{name}"
    return name


def combi_source_name(path: str | Path) -> str:
    """Path stored on combi hits so the source folder can be found again."""
    file_path = Path(path).resolve()
    try:
        return file_path.relative_to(DEFAULT_INPUT_DIR.resolve()).as_posix()
    except ValueError:
        return str(file_path)


def _resolve_source_file(source_tile: str) -> Path | None:
    raw = Path(str(source_tile).replace("\\", "/"))
    if raw.is_file():
        return raw
    candidate = DEFAULT_INPUT_DIR / raw
    if candidate.is_file():
        return candidate
    return None


@lru_cache(maxsize=256)
def combi_key_family(source_tile: str) -> str:
    """``light`` beside Groundup quads, ``nsew`` beside letter directions.

    Empty when this file is not a combi, or its folder has neither family.
    Groundup quads win when both are present, so the combi shares ``vN_LIGHT`` keys.
    """
    if not is_combi_tile(source_tile):
        return ""
    resolved = _resolve_source_file(source_tile)
    if resolved is None:
        return ""
    has_light = False
    has_dir = False
    try:
        children = list(resolved.parent.iterdir())
    except OSError:
        return ""
    for child in children:
        if not child.is_file():
            continue
        if is_submap_lighting_tile(child.name):
            has_light = True
        elif is_direction_tile(child.name):
            has_dir = True
    if has_light:
        return "light"
    if has_dir:
        return "nsew"
    return ""


def input_relative_tile(path: str | Path) -> str:
    """``folder/(1,1).png`` under Inputs. The filename alone is not a tile id."""
    file_path = Path(path)
    try:
        resolved = file_path.resolve()
    except OSError:
        resolved = file_path
    try:
        return resolved.relative_to(DEFAULT_INPUT_DIR.resolve()).as_posix()
    except ValueError:
        return file_path.name


def lighting_source_name(path: str | Path) -> str:
    """``v7/wafer_shot_….png`` for one submap shot. A ``(row,col)`` file keeps its folder.

    Other photos keep the filename. N/S/E/W labels share only inside that set.
    """
    file_path = Path(path)
    if is_submap_lighting_tile(file_path.name) and _SET_DIR.fullmatch(file_path.parent.name):
        return f"{file_path.parent.name}/{file_path.name}"
    if is_grid_tile(file_path.name):
        return input_relative_tile(file_path)
    return file_path.name


def nsew_merge_stem(source_tile: str) -> str | None:
    """Cluster id: one slot per lighting direction, quad, all-lights, or particles-only file."""
    if is_combi_tile(source_tile):
        return "COMBI"
    stem = direction_stem(source_tile)
    if stem is not None:
        return stem
    role = lighting_role(source_tile)
    if role is not None:
        return role.upper()
    if is_particles_only_tile(source_tile):
        return Path(str(source_tile)).stem.upper()
    return None
