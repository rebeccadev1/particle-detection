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

from src.overall.config import (
    apply_beta_ml,
    cfg_get,
    reserve_fresh_output_dir,
    resolve_output_dir,
)
from src.analyse.detection.detector import (
    apply_structure_filters,
    build_fft_notch_mask,
    detect_particles,
)
from src.overall.io.tile_loader import (
    Tile,
    load_tile_image,
    matching_tile_paths,
    try_parse_tile_filename,
)
from src.overall.io.naming import (
    combi_source_name,
    is_combi_tile,
    is_nsew_family_tile,
    is_submap_lighting_tile,
    lighting_source_name,
    submap_id,
)
from src.analyse.measurement.measurer import (
    Particle,
    measure_and_dedupe,
    measure_candidates,
    particles_to_dataframe,
)
from src.analyse.nsew import positional_direction, prepare_detection_paths
from src.analyse.ml.infer import apply_ml_filter
from src.analyse.preprocessing.corrections import apply_corrections
from src.overall.io.placement import (
    TilePlacement,
    grid_index_origin,
    placement_from_path,
    placement_from_tile,
)
from src.analyse.stitching.stitcher import (
    LazyMosaic,
    downsample_for_target,
    downsample_image,
    mosaic_shape,
)


ProgressCallback = Callable[[int, int, str], None]
StopCheck = Callable[[], bool]


class PipelineStopped(Exception):
    pass

_WORKER_STATE: dict[str, Any] = {
    "fft_mask": None,
    "thumbnail_factor": 0,
}


@dataclass(frozen=True)
class _TileWorkResult:
    

    particles: list[Particle]
    placement: TilePlacement | None
    name: str
    thumbnail: np.ndarray | None


def _set_worker_state(
    fft_mask: np.ndarray | None = None,
    thumbnail_factor: int = 0,
    limit_threads: bool = False,
) -> None:
    if limit_threads:
        cv2.setNumThreads(1)
    _WORKER_STATE["fft_mask"] = fft_mask
    _WORKER_STATE["thumbnail_factor"] = int(thumbnail_factor)


def _tile_from_path(path: Path, pattern: str) -> Tile:
    meta = try_parse_tile_filename(path.name, pattern) or {}
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


def _tile_is_placeable(tile: Tile) -> bool:
    if tile.x_origin is not None and tile.y_origin is not None:
        return True
    return tile.row is not None and tile.col is not None


def _placements_for_paths(
    paths: list[Path],
    pattern: str,
    overlap: float,
    row_origin: int,
    col_origin: int,
) -> list[TilePlacement]:
    return [
        placement
        for path in paths
        if (placement := placement_from_path(path, pattern, overlap, row_origin, col_origin))
        is not None
    ]


def load_report_mosaic(config: dict[str, Any]) -> LazyMosaic:
    
    folder = cfg_get(config, "input_dir", "")
    if not folder:
        return LazyMosaic([])
    pattern = str(cfg_get(config, "filename_pattern"))
    overlap = float(cfg_get(config, "overlap_fraction", 0.0))
    run_filter = cfg_get(config, "run", None)
    mag_filter = cfg_get(config, "magnification", None)
    paths = matching_tile_paths(
        folder, pattern, run_filter, mag_filter, include_unmatched=True
    )
    paths = prepare_detection_paths(paths)
    row_origin, col_origin = grid_index_origin(paths, pattern)
    return LazyMosaic(_placements_for_paths(paths, pattern, overlap, row_origin, col_origin))


def _prepare_fft_mask(paths: list[Path], config: dict[str, Any]) -> np.ndarray | None:
    
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
    
    tile = _tile_from_path(Path(path_str), pattern)
    if _tile_is_placeable(tile):
        placement = placement_from_tile(tile, overlap, row_origin, col_origin)
        origin_x = float(placement.x0)
        origin_y = float(placement.y0)
    else:
        placement = None
        origin_x = 0.0
        origin_y = 0.0
        parsed = positional_direction(tile.path)
        if parsed is not None:
            row, col, _direction = parsed
            combi = tile.path.with_name(f"({row},{col}){tile.path.suffix}")
            combi_placement = placement_from_path(
                combi, pattern, overlap, row_origin, col_origin
            )
            if combi_placement is not None:
                origin_x = float(combi_placement.x0)
                origin_y = float(combi_placement.y0)
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
        photo=tile.image,
    )
    source_name = (
        combi_source_name(tile.path)
        if is_combi_tile(tile.path.name)
        else lighting_source_name(tile.path)
    )
    candidates = apply_ml_filter(candidates, config)
    if score_before_structure:
        candidates = apply_structure_filters(candidates, config)
    particles = measure_candidates(
        candidates,
        origin_x=origin_x,
        origin_y=origin_y,
        pixel_size_nm=pixel_size,
        source_tile=source_name,
    )
    factor = int(_WORKER_STATE.get("thumbnail_factor") or 0)
    thumbnail = None
    if placement is not None and factor > 1:
        thumbnail = downsample_image(
            np.asarray(tile.image, dtype=np.float32), factor
        )
    return _TileWorkResult(
        particles=particles,
        placement=placement,
        name=source_name,
        thumbnail=thumbnail,
    )


def _worker_count(config: dict[str, Any], total_tiles: int) -> int:
    requested = int(cfg_get(config, "pipeline.workers", 0) or 0)
    if requested <= 0:
        requested = int(cpu_count() or 1)
    return max(1, min(requested, total_tiles))


def _collect_tile_result(
    result: _TileWorkResult,
    placed_particles: list[Particle],
    unplaced_particles: dict[str, list[Particle]],
    placements: list[TilePlacement],
    thumbnails: dict[str, np.ndarray],
) -> None:
    if result.placement is not None:
        placements.append(result.placement)
        placed_particles.extend(result.particles)
        if result.thumbnail is not None:
            thumbnails[result.placement.name] = result.thumbnail
        return
    unplaced_particles.setdefault(result.name, []).extend(result.particles)


def _combine_particle_tables(
    placed_particles: list[Particle],
    unplaced_particles: dict[str, list[Particle]],
    config: dict[str, Any],
) -> pd.DataFrame:
    frames = [measure_and_dedupe(placed_particles, config)]
    directional: list[Particle] = []
    lighting: dict[str, list[Particle]] = {}
    for name, particles in unplaced_particles.items():
        if is_submap_lighting_tile(name) or (is_combi_tile(name) and submap_id(name)):
            lighting.setdefault(submap_id(name) or name, []).extend(particles)
        elif is_nsew_family_tile(name):
            directional.extend(particles)
        else:
            frames.append(measure_and_dedupe(particles, config))
    fraction = float(cfg_get(config, "measurement.nsew_size_match_fraction", 0.4))
    groups = list(lighting.values())
    if directional:
        groups.append(directional)
    for particles in groups:
        frames.append(
            measure_and_dedupe(
                particles,
                config,
                size_match_fraction=fraction,
                directional=True,
            )
        )
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return frames[0] if frames else particles_to_dataframe([])
    table = pd.concat(nonempty, ignore_index=True)
    table["id"] = np.arange(1, len(table) + 1)
    return table


def _raise_if_stopped(should_stop: StopCheck | None) -> None:
    if should_stop is not None and should_stop():
        raise PipelineStopped("Pipeline stopped.")


def run_pipeline(
    config: dict[str, Any],
    progress_cb: ProgressCallback | None = None,
    *,
    write_outputs: bool = True,
    should_stop: StopCheck | None = None,
) -> tuple[pd.DataFrame, LazyMosaic]:
    
    apply_beta_ml(config)
    folder = cfg_get(config, "input_dir", "")
    if not folder:
        raise ValueError("config.input_dir is required.")
    pattern = str(cfg_get(config, "filename_pattern"))
    overlap = float(cfg_get(config, "overlap_fraction", 0.0))
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    run_filter = cfg_get(config, "run", None)
    mag_filter = cfg_get(config, "magnification", None)

    paths = matching_tile_paths(
        folder, pattern, run_filter, mag_filter, include_unmatched=True
    )

    def _prepare_progress(current: int, total_cells: int, name: str) -> None:
        _raise_if_stopped(should_stop)
        if progress_cb is not None:
            progress_cb(current, total_cells, name)

    _raise_if_stopped(should_stop)
    paths = prepare_detection_paths(
        paths, _prepare_progress if progress_cb is not None or should_stop is not None else None
    )
    total = len(paths)
    if total == 0:
        raise FileNotFoundError(
            f"No image tiles in {folder}"
            + (
                f" (run={run_filter}, mag={mag_filter})"
                if run_filter not in (None, "") or mag_filter not in (None, "")
                else ""
            )
        )

    if progress_cb is not None:
        progress_cb(0, total, f"found {total} tiles…")
    _raise_if_stopped(should_stop)

    workers = _worker_count(config, total)
    row_origin, col_origin = grid_index_origin(paths, pattern)
    placements_plan = _placements_for_paths(paths, pattern, overlap, row_origin, col_origin)
    thumb_factor = _thumbnail_factor(placements_plan, config) if placements_plan else 0
    fft_mask = _prepare_fft_mask(paths, config)
    _set_worker_state(fft_mask, thumb_factor)

    placed_particles: list[Particle] = []
    unplaced_particles: dict[str, list[Particle]] = {}
    placements: list[TilePlacement] = []
    thumbnails: dict[str, np.ndarray] = {}
    if progress_cb is not None:
        kind = "thread" if workers > 1 else "serial"
        progress_cb(0, total, f"starting {workers} {kind} worker(s)…")

    if workers == 1:
        for index, path in enumerate(paths, start=1):
            _raise_if_stopped(should_stop)
            result = _process_tile(
                str(path), pattern, overlap, pixel_size, config, row_origin, col_origin
            )
            _collect_tile_result(
                result, placed_particles, unplaced_particles, placements, thumbnails
            )
            if progress_cb is not None:
                progress_cb(index, total, result.name)
    else:
        cv2.setNumThreads(1)
        completed = 0
        pool = ThreadPoolExecutor(max_workers=workers)
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
        try:
            _raise_if_stopped(should_stop)
            for future in as_completed(futures):
                _raise_if_stopped(should_stop)
                result = future.result()
                _collect_tile_result(
                    result, placed_particles, unplaced_particles, placements, thumbnails
                )
                completed += 1
                if progress_cb is not None:
                    progress_cb(completed, total, result.name)
        except PipelineStopped:
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=False)
            cv2.setNumThreads(-1)

    _raise_if_stopped(should_stop)
    n_gathered = len(placed_particles) + sum(len(v) for v in unplaced_particles.values())
    if progress_cb is not None:
        progress_cb(total, total, f"merging {n_gathered} detections…")
    table = _combine_particle_tables(placed_particles, unplaced_particles, config)
    mosaic = LazyMosaic(
        placements,
        thumbnails=thumbnails or None,
        thumbnail_factor=thumb_factor if thumbnails else None,
    )
    if write_outputs:
        if progress_cb is not None:
            progress_cb(total, total, "writing report…")
        _write_run_outputs(table, mosaic, config)
    return table, mosaic


def _write_run_outputs(
    table: pd.DataFrame,
    mosaic: LazyMosaic,
    config: dict[str, Any],
) -> None:
    
    if not str(cfg_get(config, "output_dir", "") or "").strip():
        return
    from src.overall.io.results_writer import write_csv, write_run_settings, write_xlsx
    from src.analyse.report.report_generator import (
        draw_overlay_markers,
        overlay_rgb,
        summary_pdf_name,
        unstitched_tile_pair,
        unstitched_tile_paths,
        write_last_run_pdf,
    )

    out = reserve_fresh_output_dir(config)
    csv_path, xlsx_path = default_output_paths(config)
    write_csv(table, csv_path)
    write_xlsx(table, xlsx_path, config)
    write_run_settings(config, out / "last_pipeline.json", csv_path=csv_path)

    mosaic_png = out / "mosaic_overlay.png"
    mosaic_plain_png = out / "mosaic_plain.png"
    mosaic_jpg = out / "mosaic_overlay.jpg"
    mosaic_rgb = None
    plain_rgb = None
    if mosaic.placements:
        plain_rgb = overlay_rgb(mosaic, config, crop=None)
        mosaic_rgb = draw_overlay_markers(plain_rgb, mosaic, table, config, crop=None)
        _write_overlay_png(mosaic_rgb, mosaic_png)
        _write_overlay_png(plain_rgb, mosaic_plain_png)
        if mosaic_jpg.is_file():
            mosaic_jpg.unlink()
    else:
        for stale in (mosaic_png, mosaic_plain_png, mosaic_jpg):
            if stale.is_file():
                stale.unlink()
    separate: list[tuple[str, np.ndarray, np.ndarray]] = []
    for path in unstitched_tile_paths(config):
        try:
            labeled, plain = unstitched_tile_pair(path, table, config)
        except (OSError, ValueError, RuntimeError):
            continue
        separate.append((path.name, labeled, plain))
    pdf_path = out / summary_pdf_name(config)
    stale_pdf = out / "last_run.pdf"
    if stale_pdf.is_file() and stale_pdf.resolve() != pdf_path.resolve():
        stale_pdf.unlink()
    write_last_run_pdf(
        table,
        config,
        pdf_path,
        mosaic_rgb=mosaic_rgb,
        mosaic_plain=plain_rgb,
        unstitched=separate,
    )


def _write_overlay_png(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write mosaic image to {path}")


def default_output_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    out = resolve_output_dir(cfg_get(config, "output_dir"))
    return out / "particles.csv", out / "particles.xlsx"
