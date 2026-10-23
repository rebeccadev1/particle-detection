"""Generator-based tile loading and filename metadata parsing."""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import tifffile

from src.overall.config import DEFAULT_INPUT_DIR, PACKAGE_ROOT, WORKSPACE_ROOT, cfg_get
from src.overall.io.naming import (
    is_combi_tile,
    is_submap_lighting_tile,
    lighting_source_name,
    lighting_tile_id,
)

_IMAGE_SUFFIX = r"(?:tiff?|bmp|png|jpe?g)"
DEFAULT_FILENAME_PATTERN = (
    rf"R(?P<run>\d+)_(?P<row>\d+)_(?P<col>\d+)_(?P<mag>[\d.]+)X\.{_IMAGE_SUFFIX}"
)
TILE_GLOBS = ("*.tif", "*.tiff", "*.bmp", "*.png", "*.jpg", "*.jpeg")
TILE_SUFFIXES = (".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg")
CV_DECODE_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}
_CV_HEADER_KINDS = {"jpeg", "bmp", "png"}


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


def input_folder_labels(root: str | Path | None = None) -> list[str]:
    """Folder names under Inputs that the main pipeline can run.

    Paths are relative to ``root`` (default ``ASML SE/Inputs``), so a typed
    suggestion such as ``Groundup/Groundup v7 (06-10)`` resolves as a tile folder.
    A folder counts when it holds images, or when its ``vN`` submaps hold the
    quad and all-lights shots.
    """
    base = DEFAULT_INPUT_DIR if root is None else Path(root)
    if not base.is_dir():
        return []
    labels: list[str] = []
    for dirpath, dirnames, _filenames in os.walk(base):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        folder = Path(dirpath)
        if folder == base or not _is_pipeline_input(folder):
            continue
        labels.append(folder.relative_to(base).as_posix())
    return labels


def _is_pipeline_input(folder: Path) -> bool:
    try:
        entries = list(folder.iterdir())
    except OSError:
        return False
    for entry in entries:
        if entry.is_file() and entry.suffix.lower() in TILE_SUFFIXES:
            return True
    for entry in entries:
        if not entry.is_dir() or not re.fullmatch(r"v\d+", entry.name, flags=re.IGNORECASE):
            continue
        try:
            children = list(entry.iterdir())
        except OSError:
            continue
        if any(child.is_file() and is_submap_lighting_tile(child.name) for child in children):
            return True
    return False


def list_tile_paths(folder: str | Path) -> list[Path]:
    """Return sorted image paths in ``folder`` without reading pixel data."""
    directory = resolve_input_dir(folder)
    if not directory.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {directory}")
    paths: list[Path] = []
    for pattern in TILE_GLOBS:
        paths.extend(sorted(directory.glob(pattern)))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    _append_submap_lighting(directory, unique, seen)
    return unique


def input_tile_names(config: Mapping[str, Any]) -> set[str]:
    """Basenames of images in ``config.input_dir``, with submap ids when needed."""
    folder = cfg_get(dict(config), "input_dir", "")
    if not folder:
        return set()
    pattern = str(cfg_get(dict(config), "filename_pattern", DEFAULT_FILENAME_PATTERN))
    try:
        paths = matching_tile_paths(folder, pattern, include_unmatched=True)
    except FileNotFoundError:
        return set()
    return {
        lighting_tile_id(path)
        if is_combi_tile(path.name) or is_submap_lighting_tile(path.name)
        else lighting_source_name(path)
        for path in paths
    }


def _append_submap_lighting(directory: Path, unique: list[Path], seen: set[Path]) -> None:
    """Add quad, uniform, and combi images under each Groundup ``vN`` folder.

    Each ``vN`` folder is one field: four quadrant shots, one uniform ``all``
    shot, and ``combi_2of4`` when that file is beside them. Other nested
    folders stay put.
    """
    try:
        children = list(directory.iterdir())
    except OSError:
        return
    for child in children:
        if not child.is_dir() or not re.fullmatch(r"v\d+", child.name, flags=re.IGNORECASE):
            continue
        nested: list[Path] = []
        for pattern in TILE_GLOBS:
            nested.extend(sorted(child.glob(pattern)))
        for path in nested:
            if not is_submap_lighting_tile(path.name) and not is_combi_tile(path.name):
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved not in seen:
                seen.add(resolved)
                unique.append(path)


def try_parse_tile_filename(filename: str, pattern: str) -> dict[str, float | int] | None:
    """Like :func:`parse_tile_filename`, or ``None`` when the name does not match.

    A blank pattern matches every name and then fails the row/col check. Treat
    that as no match so quad and all-lights shots still run.
    """
    if not str(pattern or "").strip():
        return None
    if re.search(pattern, filename, flags=re.IGNORECASE) is None:
        return None
    try:
        return parse_tile_filename(filename, pattern)
    except ValueError:
        return None


def parse_tile_filename(filename: str, pattern: str) -> dict[str, float | int]:
    """Extract named groups from ``filename`` using ``pattern``.

    Recognised groups: ``run``, ``row``, ``col`` (ints), ``mag`` (float),
    and ``x`` / ``y`` (floats, stage origin). Matching is case-insensitive
    so ``R3_5_32_5X.tif`` and ``r3_5_32_5x.bmp`` both work.
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


def _file_header(path: str | Path, size: int = 8) -> bytes:
    with open(path, "rb") as handle:
        return handle.read(size)


def _header_kind(header: bytes) -> str:
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"BM"):
        return "bmp"
    if header.startswith(b"\x89PNG"):
        return "png"
    if header[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return "unknown"


def _use_cv_decode(path: str | Path) -> bool:
    """True for BMP/PNG/JPEG, including those bytes stored with a ``.tif`` name."""
    suffix = Path(path).suffix.lower()
    if suffix in CV_DECODE_EXTS:
        return True
    return _header_kind(_file_header(path)) in _CV_HEADER_KINDS


def _load_cv_image(path: str | Path) -> np.ndarray:
    """Decode BMP/PNG/JPEG with OpenCV. Handles ``.tiff`` files that are actually JPEG."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image tile {path}")
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        rgb = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
        if image.shape[-1] == 4:
            rgb = np.dstack((rgb, image[..., 3]))
        return rgb
    return image


def _shape_hw(image: np.ndarray, path: str | Path) -> tuple[int, int]:
    if image.ndim < 2:
        raise ValueError(f"Tile {path} has unexpected shape {image.shape}")
    return int(image.shape[0]), int(image.shape[1])


def peek_tile_hw(path: str | Path) -> tuple[int, int]:
    """Return ``(height, width)`` from the file header when possible."""
    raw = Path(path)
    try:
        stat = raw.stat()
        key = (str(raw.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        return _peek_tile_hw_uncached(raw)
    return _peek_tile_hw_cached(key)


@lru_cache(maxsize=512)
def _peek_tile_hw_cached(key: tuple[str, int, int]) -> tuple[int, int]:
    return _peek_tile_hw_uncached(Path(key[0]))


def _peek_tile_hw_uncached(path: Path) -> tuple[int, int]:
    header_size = _size_from_header(path)
    if header_size is not None:
        return header_size
    if _use_cv_decode(path):
        return _shape_hw(_load_cv_image(path), path)
    try:
        with tifffile.TiffFile(path) as handle:
            shape = handle.pages[0].shape
    except tifffile.TiffFileError:
        return _shape_hw(_load_cv_image(path), path)
    if len(shape) < 2:
        raise ValueError(f"Tile {path} has unexpected shape {shape}")
    return int(shape[0]), int(shape[1])


def _size_from_header(path: Path) -> tuple[int, int] | None:
    with open(path, "rb") as handle:
        head = handle.read(26)
        kind = _header_kind(head)
        if kind == "png":
            if len(head) >= 24 and head[12:16] == b"IHDR":
                width, height = struct.unpack(">II", head[16:24])
                return int(height), int(width)
            return None
        if kind == "bmp":
            if len(head) >= 26:
                width, height = struct.unpack_from("<ii", head, 18)
                return abs(int(height)), abs(int(width))
            return None
        if kind == "jpeg":
            return _jpeg_hw(path, head, handle)
    return None


def _jpeg_hw(path: Path, head: bytes, handle) -> tuple[int, int] | None:
    data = head
    offset = 2
    while True:
        while offset + 1 < len(data) and data[offset] != 0xFF:
            offset += 1
        if offset + 3 >= len(data):
            extra = handle.read(4096)
            if not extra:
                return None
            data += extra
            continue
        if data[offset] != 0xFF:
            return None
        code = data[offset + 1]
        if code in (0xD8, 0x01) or (0xD0 <= code <= 0xD7):
            offset += 2
            continue
        if code == 0xD9:
            return None
        if offset + 4 > len(data):
            extra = handle.read(4096)
            if not extra:
                return None
            data += extra
            continue
        length = struct.unpack_from(">H", data, offset + 2)[0]
        start = offset + 4
        end = start + length - 2
        if end > len(data):
            extra = handle.read(end - len(data))
            if len(extra) < end - len(data):
                return None
            data += extra
        if code in (
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        ):
            if end - start < 5:
                return None
            height, width = struct.unpack_from(">HH", data, start + 1)
            return int(height), int(width)
        offset = end


def load_tile_image(path: str | Path) -> np.ndarray:
    """Read a tile as 2D grayscale. Callers should not retain many at once.

    Real microscope tiles are TIFF. BMP/PNG/JPEG are decoded with OpenCV.
    Exports that keep a ``.tif`` name but contain JPEG or BMP bytes are decoded
    with OpenCV instead of failing in ``tifffile``.
    """
    if _use_cv_decode(path):
        return _to_grayscale(_load_cv_image(path))
    try:
        return _to_grayscale(tifffile.imread(path))
    except tifffile.TiffFileError:
        return _to_grayscale(_load_cv_image(path))


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
    include_unmatched: bool = False,
) -> list[Path]:
    """Image paths whose names match ``filename_pattern`` and optional run/mag filters.

    Tiles with ``col`` 0 are omitted (they sit outside this wafer grid).
    When ``include_unmatched`` is true, files that do not match the pattern are
    still returned (detection can run; they cannot be placed on the mosaic).
    """
    run_filter = _optional_int(run)
    mag_filter = _optional_float(magnification)
    matched: list[Path] = []
    for path in list_tile_paths(folder):
        meta = try_parse_tile_filename(path.name, filename_pattern)
        if meta is None:
            if include_unmatched:
                matched.append(path)
            continue
        if run_filter is not None and int(meta.get("run", -1)) != run_filter:
            continue
        if mag_filter is not None and abs(float(meta.get("mag", -1)) - mag_filter) > 1e-6:
            continue
        if "col" in meta and int(meta["col"]) == 0:
            continue
        matched.append(path)
    return matched
