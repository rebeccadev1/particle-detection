"""Generator-based TIFF tile loading and filename metadata parsing."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile

from src.config import DEFAULT_INPUT_DIR, PACKAGE_ROOT, WORKSPACE_ROOT

# Example: R3_5_32_5X.tif → run=3, row=5, col=32, mag=5
DEFAULT_FILENAME_PATTERN = r"R(?P<run>\d+)_(?P<row>\d+)_(?P<col>\d+)_(?P<mag>[\d.]+)X\.tiff?"


@dataclass(frozen=True)
class Tile:
    """One wafer tile and the metadata parsed from its filename."""

    path: Path
    name: str
    image: np.ndarray
    row: int | None
    col: int | None
    run: int | None
    magnification: float | None
    x_origin: float | None
    y_origin: float | None
    height: int
    width: int


def resolve_input_dir(folder: str | Path) -> Path:
    """Resolve a tile folder under ``ASML SE/Inputs`` (or an absolute path).

    Bare names such as ``R3 04-08`` are looked up in ``Inputs``. Relative
    ``../…`` paths still work if they already point at that folder.
    """
    text = str(folder).strip().strip('"').strip("'")
    if not text:
        raise FileNotFoundError("Input folder is empty.")
    raw = Path(text)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        first = raw.parts[0]
        if first in (".", ".."):
            candidates.append(PACKAGE_ROOT / raw)
            candidates.append(DEFAULT_INPUT_DIR / raw.name)
        else:
            candidates.append(Path.cwd() / raw)
            candidates.append(PACKAGE_ROOT / raw)
            candidates.append(DEFAULT_INPUT_DIR / raw)
            if first in (DEFAULT_INPUT_DIR.name, "Input"):
                candidates.append(WORKSPACE_ROOT / raw)
            candidates.append(WORKSPACE_ROOT / "Input" / raw)
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = candidate
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    for candidate in unique:
        if candidate.is_dir():
            return candidate.resolve()
    tried = ", ".join(str(c) for c in unique)
    raise FileNotFoundError(
        f"Input folder does not exist: {folder!r}. Tried: {tried}"
    )


def list_tile_paths(folder: str | Path) -> list[Path]:
    """Return sorted TIFF paths in ``folder`` without reading pixel data."""
    directory = resolve_input_dir(folder)
    if not directory.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {directory}")
    paths = sorted(directory.glob("*.tif")) + sorted(directory.glob("*.tiff"))
    # glob("*.tif") also matches ".tiff" on some platforms; unique keep order
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def parse_tile_filename(filename: str, pattern: str) -> dict[str, float | int]:
    """Extract named groups from ``filename`` using ``pattern``.

    Recognised groups: ``run``, ``row``, ``col`` (ints), ``mag`` (float),
    and ``x`` / ``y`` (floats, stage origin). Matching is case-insensitive
    so ``R3_5_32_5X.tif`` and ``r3_5_32_5x.tif`` both work.
    """
    match = re.search(pattern, filename, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Filename {filename!r} does not match filename_pattern {pattern!r}."
        )
    groups = match.groupdict()
    parsed: dict[str, float | int] = {}
    if groups.get("run") is not None:
        parsed["run"] = int(groups["run"])
    if groups.get("row") is not None:
        parsed["row"] = int(groups["row"])
    if groups.get("col") is not None:
        parsed["col"] = int(groups["col"])
    if groups.get("mag") is not None:
        parsed["mag"] = float(groups["mag"])
    if groups.get("x") is not None:
        parsed["x"] = float(groups["x"])
    if groups.get("y") is not None:
        parsed["y"] = float(groups["y"])
    if "row" not in parsed and "col" not in parsed and "x" not in parsed:
        raise ValueError(
            f"filename_pattern must contain named groups row/col (and optionally run/mag) "
            f"or x/y, got {pattern!r}."
        )
    return parsed


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Reduce extra channels/pages to a 2D array immediately after load.

    Microscope tiles are RGB uint8 (H, W, 3). Detection is intensity-based, so
    extra channels are dropped here before any preprocessing.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        rgb = np.ascontiguousarray(array[..., :3])
        if rgb.dtype == np.uint8:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return np.tensordot(
            rgb.astype(np.float32, copy=False), weights, axes=([-1], [0])
        )
    while array.ndim > 2:
        array = (
            array[0]
            if array.shape[0] <= 4
            else array.mean(axis=-1, dtype=np.float32)
        )
    return array


def peek_tile_hw(path: str | Path) -> tuple[int, int]:
    """Return ``(height, width)`` from the TIFF header without decoding pixels."""
    with tifffile.TiffFile(path) as handle:
        shape = handle.pages[0].shape
    if len(shape) < 2:
        raise ValueError(f"Tile {path} has unexpected shape {shape}")
    return int(shape[0]), int(shape[1])


def load_tile_image(path: str | Path) -> np.ndarray:
    """Read a single TIFF into memory as 2D grayscale. Callers should not retain many at once."""
    return _to_grayscale(tifffile.imread(path))


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def matching_tile_paths(
    folder: str | Path,
    filename_pattern: str,
    run: object = None,
    magnification: object = None,
) -> list[Path]:
    """TIFF paths whose names match ``filename_pattern`` and optional run/mag filters.

    Tiles with ``col`` 0 are omitted (they sit outside this wafer grid).
    """
    run_filter = _optional_int(run)
    mag_filter = _optional_float(magnification)
    matched: list[Path] = []
    for path in list_tile_paths(folder):
        try:
            meta = parse_tile_filename(path.name, filename_pattern)
        except ValueError:
            continue
        if run_filter is not None and int(meta.get("run", -1)) != run_filter:
            continue
        if mag_filter is not None and abs(float(meta.get("mag", -1)) - mag_filter) > 1e-6:
            continue
        # Column 0 is outside the wafer grid (e.g. R3_2_0_5X.tif).
        if "col" in meta and int(meta["col"]) == 0:
            continue
        matched.append(path)
    return matched


def iter_tiles(
    folder: str | Path,
    filename_pattern: str,
    run: object = None,
    magnification: object = None,
) -> Iterator[Tile]:
    """Yield one loaded tile at a time. Never returns a list of image arrays."""
    for path in matching_tile_paths(folder, filename_pattern, run, magnification):
        meta = parse_tile_filename(path.name, filename_pattern)
        image = load_tile_image(path)
        height, width = int(image.shape[0]), int(image.shape[1])
        yield Tile(
            path=path,
            name=path.name,
            image=image,
            row=int(meta["row"]) if "row" in meta else None,
            col=int(meta["col"]) if "col" in meta else None,
            run=int(meta["run"]) if "run" in meta else None,
            magnification=float(meta["mag"]) if "mag" in meta else None,
            x_origin=float(meta["x"]) if "x" in meta else None,
            y_origin=float(meta["y"]) if "y" in meta else None,
            height=height,
            width=width,
        )
