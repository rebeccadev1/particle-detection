from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.overall.config import DEFAULT_INPUT_DIR, resolve_output_dir
from src.overall.io.naming import (
    is_combi_tile,
    lighting_role,
    quad_direction,
)

DIRECTIONS = ("N", "S", "E", "W")
_POSITIONAL_DIRECTION = re.compile(
    r"^\((?P<row>\d+),(?P<col>\d+)\)_(?P<direction>[NSEW])$",
    re.IGNORECASE,
)
_POSITIONAL_COMBI = re.compile(r"^\((?P<row>\d+),(?P<col>\d+)\)$")
ORTHOGONAL_PAIRS = (("N", "E"), ("N", "W"), ("S", "E"), ("S", "W"))
IMAGE_EXTS = {".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}
MIN_COUNTS = (2, 3, 4)
COMBINE_CORES = 10
KIND_FOLDERS = {
    "combi": "Combi",
    "symmetry_map": "Symmetry",
}
def _direction_name(path: Path) -> str | None:
    
    stem = path.stem.upper()
    if stem in DIRECTIONS:
        return stem
    return quad_direction(path.name)


def list_direction_images(folder: str | Path) -> dict[str, Path]:
    
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")

    found: dict[str, Path] = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        name = _direction_name(path)
        if name not in DIRECTIONS:
            continue
        if name in found:
            raise FileExistsError(
                f"Multiple images named {name} in {folder}: "
                f"{found[name].name} and {path.name}"
            )
        found[name] = path
    return {d: found[d] for d in DIRECTIONS if d in found}


def find_nsew_images(folder: str | Path) -> dict[str, Path]:
    
    found = list_direction_images(folder)
    missing = [d for d in DIRECTIONS if d not in found]
    if missing:
        raise FileNotFoundError(
            f"Missing directional images in {folder}: {', '.join(missing)}. "
            "Expected files named N, S, E, W (e.g. N.bmp, S.png), "
            "or wafer_shot quad files (quad0=N, quad1=E, quad2=S, quad3=W)."
        )
    return found


def nsew_output_dir(input_folder: str | Path) -> Path:
    
    folder = Path(input_folder).resolve()
    name = f"Output {folder.name}"
    try:
        folder.relative_to(DEFAULT_INPUT_DIR.resolve())
    except ValueError:
        return folder.parent / name
    return resolve_output_dir(name)


def kind_output_dir(output_dir: str | Path, kind: str) -> Path:
    
    name = KIND_FOLDERS.get(kind, kind)
    folder = Path(output_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _usable_dpi(dpi: Any) -> tuple[float, float] | None:
    
    if not dpi or len(dpi) < 2:
        return None
    x_dpi, y_dpi = float(dpi[0]), float(dpi[1])
    if x_dpi <= 0 or y_dpi <= 0:
        return None
    return (x_dpi, y_dpi)


def output_suffix(reference_path: str | Path) -> str:
    
    suffix = Path(reference_path).suffix.lower()
    if suffix in IMAGE_EXTS:
        return suffix
    return ".png"


def save_same_resolution(path: str | Path, array: np.ndarray, reference_path: str | Path) -> None:
    
    path = Path(path)
    h, w = array.shape[:2]
    with Image.open(reference_path) as ref:
        ref_w, ref_h = ref.size
        if (w, h) != (ref_w, ref_h):
            raise ValueError(
                f"{path}: output is {w}x{h} px but input is {ref_w}x{ref_h} px"
            )
        dpi = _usable_dpi(ref.info.get("dpi"))
    image = Image.fromarray(array)
    save_kw: dict[str, Any] = {}
    if dpi is not None:
        save_kw["dpi"] = dpi
    if path.suffix.lower() == ".png":
        save_kw["compress_level"] = 1
    image.save(path, **save_kw)


def symmetry_signal_4way(
    sorted_stack: np.ndarray,
    sum_val: np.ndarray,
    min_count: int,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    
    n_dirs = sorted_stack.shape[0]
    if min_count < 1 or min_count > n_dirs:
        raise ValueError(f"min_count must be between 1 and {n_dirs}, got {min_count}")
    min_val = sorted_stack[n_dirs - min_count]
    sym = (n_dirs * min_val) / (sum_val + eps)
    combined_brightness = sum_val / float(n_dirs)
    isotropic = sym * combined_brightness
    return isotropic, sym


def symmetry_signal_orthogonal_2of4(
    stack_nsew: np.ndarray,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    
    if stack_nsew.shape[0] != 4:
        raise ValueError(f"expected 4 directions, got {stack_nsew.shape[0]}")
    by_dir = {name: stack_nsew[i] for i, name in enumerate(DIRECTIONS)}
    pair_mins = [
        np.minimum(by_dir[a], by_dir[b]) for a, b in ORTHOGONAL_PAIRS
    ]
    min_val = pair_mins[0]
    for other in pair_mins[1:]:
        min_val = np.maximum(min_val, other)
    sum_val = np.sum(stack_nsew, axis=0)
    n_dirs = 4
    sym = (n_dirs * min_val) / (sum_val + eps)
    isotropic = sym * (sum_val / float(n_dirs))
    return isotropic, sym


def scale_combi_uint8(values: np.ndarray, gain: float) -> np.ndarray:
    
    array = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(gain) or gain <= 0:
        gain = 1.0
    scaled = np.clip(np.rint(array * np.float32(gain)), 0, 255)
    return scaled.astype(np.uint8)


def combine_worker_count(n_cells: int) -> int:
    
    cores = cpu_count() or 1
    return max(1, min(COMBINE_CORES, cores, max(int(n_cells), 1)))


def load_flattened_nsew(
    path1: str | Path,
    path2: str | Path,
    path3: str | Path,
    path4: str | Path,
    *,
    use_all_cores: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    
    if use_all_cores:
        cv2.setNumThreads(-1)
    paths = [path1, path2, path3, path4]
    imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    for i, img in enumerate(imgs):
        if img is None:
            raise FileNotFoundError(f"Could not read image: {paths[i]}")
        if img.shape != imgs[0].shape:
            raise ValueError("All images must be the same size and pixel-aligned")
    stack = np.array([img.astype(np.float32) for img in imgs])
    return np.sort(stack, axis=0), np.sum(stack, axis=0), stack, 1.0


def combine_nsew_folder(
    folder: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    
    folder = Path(folder)
    dest = Path(output_dir) if output_dir is not None else nsew_output_dir(folder)
    dest.mkdir(parents=True, exist_ok=True)
    nsew = find_nsew_images(folder)
    paths = [nsew[d] for d in DIRECTIONS]
    sorted_stack, sum_val, flat_stack, gain = load_flattened_nsew(*paths)

    suffix = output_suffix(paths[0])
    variants: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for min_count in MIN_COUNTS:
        isotropic, sym = symmetry_signal_4way(sorted_stack, sum_val, min_count)
        iso_u8 = scale_combi_uint8(isotropic, gain)
        sym_u8 = (sym * 255).astype(np.uint8)
        named: list[tuple[str, str, np.ndarray]] = [
            ("combi", f"combi_{min_count}of4{suffix}", iso_u8),
            ("symmetry_map", f"symmetry_map_{min_count}of4{suffix}", sym_u8),
        ]
        if min_count == 2:
            named.extend(
                (
                    ("combi", f"combi{suffix}", iso_u8),
                    ("symmetry_map", f"symmetry_map{suffix}", sym_u8),
                )
            )
            ortho_iso, ortho_sym = symmetry_signal_orthogonal_2of4(flat_stack)
            ortho_iso_u8 = scale_combi_uint8(ortho_iso, gain)
            ortho_sym_u8 = (ortho_sym * 255).astype(np.uint8)
            named.extend(
                (
                    ("combi", f"combi_2of4_ortho{suffix}", ortho_iso_u8),
                    ("symmetry_map", f"symmetry_map_2of4_ortho{suffix}", ortho_sym_u8),
                )
            )
        for kind, filename, array in named:
            out_path = kind_output_dir(dest, kind) / filename
            save_same_resolution(out_path, array, paths[0])
            outputs.append(out_path)
        variants.append(
            {
                "min_count": min_count,
                "outputs": [
                    str(kind_output_dir(dest, "combi") / f"combi_{min_count}of4{suffix}"),
                    str(
                        kind_output_dir(dest, "symmetry_map")
                        / f"symmetry_map_{min_count}of4{suffix}"
                    ),
                ],
            }
        )

    height, width = sorted_stack.shape[1], sorted_stack.shape[2]
    return {
        "folder": folder,
        "output_dir": dest,
        "nsew": nsew,
        "width": width,
        "height": height,
        "outputs": outputs,
        "variants": variants,
    }


def is_uniform_image(path: str | Path) -> bool:
    
    file_path = Path(path)
    if is_combi_tile(file_path.name):
        return False
    if lighting_role(file_path.name) == "all":
        return True
    return file_path.stem.lower() in {"all", "uniform"}


def _combi_2of4_uint8(paths: list[Path], *, use_all_cores: bool = True) -> np.ndarray:
    sorted_stack, sum_val, _flat, gain = load_flattened_nsew(
        *paths, use_all_cores=use_all_cores
    )
    isotropic, _sym = symmetry_signal_4way(sorted_stack, sum_val, 3)
    return scale_combi_uint8(isotropic, gain)


def _combi_2of4_path(folder: Path, reference: Path) -> Path:
    return folder / f"combi_2of4{output_suffix(reference)}"


def _write_letter_combi(folder: Path, sources: list[Path]) -> Path:
    out = _combi_2of4_path(folder, sources[0])
    save_same_resolution(out, _combi_2of4_uint8(sources), sources[0])
    return out


def write_combi_2of4(folder: str | Path) -> Path:
    folder = Path(folder)
    nsew = find_nsew_images(folder)
    return _write_letter_combi(folder, [nsew[d] for d in DIRECTIONS])


def positional_direction(path: str | Path) -> tuple[int, int, str] | None:
    
    file_path = Path(path)
    if file_path.suffix.lower() not in IMAGE_EXTS:
        return None
    match = _POSITIONAL_DIRECTION.fullmatch(file_path.stem)
    if match is None:
        return None
    return (
        int(match.group("row")),
        int(match.group("col")),
        match.group("direction").upper(),
    )


def positional_cell_sides(tile_path: str | Path) -> dict[str, Path]:
    
    path = Path(tile_path)
    parsed = positional_direction(path)
    if parsed is not None:
        row, col, _direction = parsed
    else:
        match = _POSITIONAL_COMBI.fullmatch(path.stem)
        if match is None:
            return {}
        row, col = int(match.group("row")), int(match.group("col"))
    found: dict[str, Path] = {}
    suffixes = (path.suffix.lower(), *(ext for ext in IMAGE_EXTS if ext != path.suffix.lower()))
    for direction in DIRECTIONS:
        for suffix in suffixes:
            candidate = path.parent / f"({int(row)},{int(col)})_{direction}{suffix}"
            if candidate.is_file():
                found[direction] = candidate
                break
    return found


def positional_combi_path(
    folder: str | Path,
    row: int,
    col: int,
    by_direction: dict[str, Path],
) -> Path:
    
    reference = by_direction[DIRECTIONS[0]]
    return Path(folder) / f"({int(row)},{int(col)}){output_suffix(reference)}"


def combi_is_current(path: str | Path, sources: list[Path]) -> bool:
    
    file_path = Path(path)
    if not file_path.is_file():
        return False
    try:
        built = file_path.stat().st_mtime
        if any(built < source.stat().st_mtime for source in sources):
            return False
        with Image.open(file_path) as image:
            image.verify()
    except (OSError, ValueError, SyntaxError):
        return False
    return True


def write_positional_combi(
    folder: str | Path,
    row: int,
    col: int,
    by_direction: dict[str, Path],
    *,
    use_all_cores: bool = True,
) -> Path:
    
    folder = Path(folder)
    paths = [by_direction[direction] for direction in DIRECTIONS]
    out = positional_combi_path(folder, row, col, by_direction)
    save_same_resolution(
        out, _combi_2of4_uint8(paths, use_all_cores=use_all_cores), paths[0]
    )
    return out


def _combine_positional_cell(
    folder: Path,
    row: int,
    col: int,
    found: dict[str, Path],
    *,
    use_all_cores: bool = False,
) -> tuple[int, int, Path]:
    sources = [found[direction] for direction in DIRECTIONS]
    out = positional_combi_path(folder, row, col, found)
    if combi_is_current(out, sources):
        return row, col, out
    return row, col, write_positional_combi(
        folder, row, col, found, use_all_cores=use_all_cores
    )


def _positional_groups(paths: list[Path]) -> tuple[list[Path], dict[tuple[Path, int, int], dict[str, Path]]]:
    
    groups: dict[tuple[Path, int, int], dict[str, Path]] = {}
    rest: list[Path] = []
    for path in paths:
        parsed = positional_direction(path)
        if parsed is None:
            rest.append(path)
            continue
        row, col, direction = parsed
        slot = groups.setdefault((path.parent.resolve(), row, col), {})
        if direction in slot:
            raise FileExistsError(
                f"Multiple {direction} images for ({row},{col}) in {path.parent}: "
                f"{slot[direction].name} and {path.name}"
            )
        slot[direction] = path
    return rest, groups


def prepare_detection_paths(
    paths: list[Path],
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> list[Path]:
    
    rest, groups = _positional_groups(paths)
    extras: list[Path] = []
    consumed: set[Path] = set()
    pending = [
        (folder, row, col, found)
        for (folder, row, col), found in groups.items()
        if all(direction in found for direction in DIRECTIONS)
    ]
    n_cells = len(pending)
    for _folder, _row, _col, found in pending:
        consumed.update(found[direction].resolve() for direction in DIRECTIONS)
    if n_cells == 1:
        folder, row, col, found = pending[0]
        if progress_cb is not None:
            progress_cb(1, 1, f"Combining ({row},{col})")
        _row, _col, out = _combine_positional_cell(
            folder, row, col, found, use_all_cores=True
        )
        extras.append(out)
    elif n_cells > 1:
        workers = combine_worker_count(n_cells)
        cv2.setNumThreads(0)
        pool = ThreadPoolExecutor(max_workers=workers)
        futures = [
            pool.submit(_combine_positional_cell, folder, row, col, found)
            for folder, row, col, found in pending
        ]
        completed = 0
        try:
            for future in as_completed(futures):
                row, col, out = future.result()
                extras.append(out)
                completed += 1
                if progress_cb is not None:
                    progress_cb(completed, n_cells, f"Combining ({row},{col})")
        finally:
            pool.shutdown(wait=True, cancel_futures=False)
            cv2.setNumThreads(-1)

    folders = {path.parent.resolve() for path in rest}
    for folder in sorted(folders, key=lambda path: str(path)):
        found = list_direction_images(folder)
        if len(found) < 4:
            continue
        if progress_cb is not None:
            progress_cb(0, 1, f"Combining {folder.name}")
        sources = [found[direction] for direction in DIRECTIONS]
        consumed.update(source.resolve() for source in sources)
        for path in rest:
            if path.parent.resolve() == folder and is_uniform_image(path):
                consumed.add(path.resolve())
        out = _combi_2of4_path(folder, sources[0])
        if combi_is_current(out, sources):
            extras.append(out)
        else:
            extras.append(_write_letter_combi(folder, sources))
        if progress_cb is not None:
            progress_cb(1, 1, f"Combining {folder.name}")
    kept: list[Path] = []
    seen: set[Path] = set()
    for path in list(rest) + extras:
        resolved = path.resolve()
        if resolved in seen or resolved in consumed:
            continue
        seen.add(resolved)
        kept.append(path)
    return kept
