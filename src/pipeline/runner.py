"""Orchestrate per-tile load → preprocess → detect → measure. Accumulate rows only."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.config import cfg_get, resolve_output_dir, with_recall_profile
from src.detection.detector import (
    apply_structure_filters,
    build_fft_notch_mask,
    detect_particles,
)
from src.io.tile_loader import (
    Tile,
    load_tile_image,
    matching_tile_paths,
    parse_tile_filename,
    peek_tile_hw,
)
from src.measurement.measurer import Particle, measure_and_dedupe, measure_candidates
from src.ml.infer import apply_ml_filter
from src.preprocessing.corrections import apply_corrections
from src.stitching.stitcher import (
    LazyMosaic,
    TilePlacement,
    downsample_for_target,
    downsample_image,
    mosaic_shape,
    placement_from_tile,
)


ProgressCallback = Callable[[int, int, str], None]

_WORKER_STATE: dict[str, Any] = {
    "fft_mask": None,
    "thumbnail_factor": 0,
}


@dataclass(frozen=True)
class _TileWorkResult:
    """Lightweight per-tile output returned from worker processes."""

    particles: list[Particle]
    placement: TilePlacement
    name: str
    thumbnail: np.ndarray | None


def _set_worker_state(
    fft_mask: np.ndarray | None = None,
    thumbnail_factor: int = 0,
    limit_threads: bool = False,
) -> None:
    if limit_threads:
        # One OpenCV thread per process so N workers do not oversubscribe.
        cv2.setNumThreads(1)
    _WORKER_STATE["fft_mask"] = fft_mask
    _WORKER_STATE["thumbnail_factor"] = int(thumbnail_factor)


def _tile_from_path(path: Path, pattern: str) -> Tile:
    meta = parse_tile_filename(path.name, pattern)
    image = load_tile_image(path)
    height, width = int(image.shape[0]), int(image.shape[1])
    return Tile(
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


def _placement_from_path(
    path: Path,
    pattern: str,
    overlap: float,
    row_origin: int,
    col_origin: int,
) -> TilePlacement:
    """Grid placement from filename + TIFF header (no pixel decode)."""
    meta = parse_tile_filename(path.name, pattern)
    height, width = peek_tile_hw(path)
    tile = Tile(
        path=path,
        name=path.name,
        image=np.empty((0, 0), dtype=np.uint8),
        row=int(meta["row"]) if "row" in meta else None,
        col=int(meta["col"]) if "col" in meta else None,
        run=int(meta["run"]) if "run" in meta else None,
        magnification=float(meta["mag"]) if "mag" in meta else None,
        x_origin=float(meta["x"]) if "x" in meta else None,
        y_origin=float(meta["y"]) if "y" in meta else None,
        height=height,
        width=width,
    )
    return placement_from_tile(tile, overlap, row_origin, col_origin)


def _grid_index_origin(paths: list[Path], pattern: str) -> tuple[int, int]:
    """Smallest row/col among tiles so the mosaic starts at the first real column."""
    rows: list[int] = []
    cols: list[int] = []
    for path in paths:
        meta = parse_tile_filename(path.name, pattern)
        if "row" in meta:
            rows.append(int(meta["row"]))
        if "col" in meta:
            cols.append(int(meta["col"]))
    return (min(rows) if rows else 0, min(cols) if cols else 0)


def _prepare_fft_mask(paths: list[Path], config: dict[str, Any]) -> np.ndarray | None:
    """Shared lattice mask from the first tile, or None for per-tile notches."""
    if str(cfg_get(config, "detection.method", "fft")).lower() != "fft":
        return None
    mode = str(cfg_get(config, "detection.fft_mask", "per_tile")).lower()
    if mode != "shared":
        return None
    corrected = apply_corrections(load_tile_image(paths[0]), config)
    return build_fft_notch_mask(
        corrected,
        float(cfg_get(config, "detection.fft_peak_threshold", 0.35)),
        int(cfg_get(config, "detection.fft_notch_radius", 3)),
    )


def _thumbnail_factor(placements: list[TilePlacement], config: dict[str, Any]) -> int:
    configured = int(cfg_get(config, "report.downsample", 0) or 0)
    if configured > 0:
        return configured
    height, width = mosaic_shape(placements)
    return downsample_for_target(
        height,
        width,
        target_mb=float(cfg_get(config, "report.target_mb", 20.0)),
        channels=3,
    )


def _process_tile(
    path_str: str,
    pattern: str,
    overlap: float,
    pixel_size: float,
    config: dict[str, Any],
    row_origin: int = 0,
    col_origin: int = 0,
) -> _TileWorkResult:
    """Load and process one tile. Top-level for ProcessPoolExecutor pickling."""
    tile = _tile_from_path(Path(path_str), pattern)
    placement = placement_from_tile(tile, overlap, row_origin, col_origin)
    corrected = apply_corrections(tile.image, config)
    ml_on = bool(cfg_get(config, "ml.enabled", False))
    score_before_structure = ml_on and bool(
        cfg_get(config, "ml.score_before_structure", True)
    )
    candidates = detect_particles(
        corrected,
        config,
        fft_mask=_WORKER_STATE.get("fft_mask"),
        structure_filters=not score_before_structure,
    )
    candidates = apply_ml_filter(
        candidates, config, image=corrected, source_tile=tile.name
    )
    if score_before_structure:
        candidates = apply_structure_filters(candidates, config)
    particles = measure_candidates(
        candidates,
        origin_x=float(placement.x0),
        origin_y=float(placement.y0),
        pixel_size_nm=pixel_size,
        source_tile=tile.name,
    )
    factor = int(_WORKER_STATE.get("thumbnail_factor") or 0)
    thumbnail = None
    if factor > 1:
        thumbnail = downsample_image(
            np.asarray(tile.image, dtype=np.float32), factor
        )
    return _TileWorkResult(
        particles=particles,
        placement=placement,
        name=tile.name,
        thumbnail=thumbnail,
    )


def _worker_count(config: dict[str, Any], total_tiles: int) -> int:
    requested = int(cfg_get(config, "pipeline.workers", 0) or 0)
    if requested <= 0:
        requested = int(cpu_count() or 1)
    return max(1, min(requested, total_tiles))


def _collect_tile_result(
    result: _TileWorkResult,
    gathered: list[Particle],
    placements: list[TilePlacement],
    thumbnails: dict[str, np.ndarray],
) -> None:
    placements.append(result.placement)
    gathered.extend(result.particles)
    if result.thumbnail is not None:
        thumbnails[result.name] = result.thumbnail


def run_pipeline(
    config: dict[str, Any],
    progress_cb: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, LazyMosaic]:
    """Run detection on each tile without retaining full-resolution images.

    Returns the de-duplicated particle table and a lazy mosaic for reporting.
    Mosaic thumbnails are generated during the detection pass when the overlay
    downsample factor is greater than 1.
    """
    config = with_recall_profile(config)
    folder = cfg_get(config, "input_dir", "")
    if not folder:
        raise ValueError("config.input_dir is required.")
    pattern = str(cfg_get(config, "filename_pattern"))
    overlap = float(cfg_get(config, "overlap_fraction", 0.0))
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    run_filter = cfg_get(config, "run", None)
    mag_filter = cfg_get(config, "magnification", None)

    paths = matching_tile_paths(folder, pattern, run_filter, mag_filter)
    total = len(paths)
    if total == 0:
        raise FileNotFoundError(
            f"No TIFF tiles matching {pattern!r} in {folder}"
            + (
                f" (run={run_filter}, mag={mag_filter})"
                if run_filter not in (None, "") or mag_filter not in (None, "")
                else ""
            )
        )

    if progress_cb is not None:
        progress_cb(0, total, f"found {total} tiles…")

    workers = _worker_count(config, total)
    row_origin, col_origin = _grid_index_origin(paths, pattern)
    placements_plan = [
        _placement_from_path(path, pattern, overlap, row_origin, col_origin)
        for path in paths
    ]
    thumb_factor = _thumbnail_factor(placements_plan, config)
    fft_mask = _prepare_fft_mask(paths, config)
    _set_worker_state(fft_mask, thumb_factor)

    gathered: list[Particle] = []
    placements: list[TilePlacement] = []
    thumbnails: dict[str, np.ndarray] = {}
    # Threads, not processes: ProcessPoolExecutor spawned from Streamlit hangs
    # on macOS (workers re-import the server) and the UI stays on "Idle".
    if progress_cb is not None:
        kind = "thread" if workers > 1 else "serial"
        progress_cb(0, total, f"starting {workers} {kind} worker(s)…")

    if workers == 1:
        for index, path in enumerate(paths, start=1):
            result = _process_tile(
                str(path), pattern, overlap, pixel_size, config, row_origin, col_origin
            )
            _collect_tile_result(result, gathered, placements, thumbnails)
            if progress_cb is not None:
                progress_cb(index, total, result.name)
    else:
        cv2.setNumThreads(1)
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_tile,
                    str(path),
                    pattern,
                    overlap,
                    pixel_size,
                    config,
                    row_origin,
                    col_origin,
                ): path
                for path in paths
            }
            for future in as_completed(futures):
                result = future.result()
                _collect_tile_result(result, gathered, placements, thumbnails)
                completed += 1
                if progress_cb is not None:
                    progress_cb(completed, total, result.name)

    if progress_cb is not None:
        progress_cb(total, total, f"merging {len(gathered)} detections…")
    table = measure_and_dedupe(gathered, config)
    mosaic = LazyMosaic(
        placements,
        thumbnails=thumbnails or None,
        thumbnail_factor=thumb_factor if thumbnails else None,
    )
    return table, mosaic


def default_output_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    """CSV, JSON, and Excel destinations under ``output_dir``."""
    out = resolve_output_dir(cfg_get(config, "output_dir"))
    return out / "particles.csv", out / "particles.json", out / "particles.xlsx"
