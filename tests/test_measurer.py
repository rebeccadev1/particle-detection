"""Tests for global mapping and boundary de-duplication."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from src.detection.detector import ParticleCandidate
from src.io.tile_loader import parse_tile_filename
from src.measurement.measurer import (
    Particle,
    deduplicate,
    local_to_global,
    measure_and_dedupe,
    measure_candidates,
)
from src.pipeline.runner import run_pipeline
from src.stitching.stitcher import tile_origin_px
from src.io.tile_loader import Tile
from tests.fixtures.synthetic import detector_test_config, make_structured_tile, write_tile_tiff


def test_parse_run_row_col_mag_filename() -> None:
    meta = parse_tile_filename(
        "R3_5_32_5X.tif",
        r"R(?P<run>\d+)_(?P<row>\d+)_(?P<col>\d+)_(?P<mag>[\d.]+)X\.tiff?",
    )
    assert meta == {"run": 3, "row": 5, "col": 32, "mag": 5.0}


def test_parse_filename_is_case_insensitive() -> None:
    meta = parse_tile_filename(
        "r3_5_32_5x.tif",
        r"R(?P<run>\d+)_(?P<row>\d+)_(?P<col>\d+)_(?P<mag>[\d.]+)X\.tiff?",
    )
    assert meta["run"] == 3 and meta["row"] == 5 and meta["col"] == 32 and meta["mag"] == 5.0


def test_resolve_input_dir_finds_workspace_tile_folder() -> None:
    from src.config import DEFAULT_INPUT_DIR
    from src.io.tile_loader import resolve_input_dir

    found = resolve_input_dir("R3 04-08")
    assert found.is_dir()
    assert found.name == "R3 04-08"
    assert found.parent == DEFAULT_INPUT_DIR.resolve()

    via_dotdot = resolve_input_dir("../Single image test big boy")
    assert via_dotdot.name == "Single image test big boy"
    assert via_dotdot.parent == DEFAULT_INPUT_DIR.resolve()


def test_resolve_output_dir_uses_workspace_outputs(tmp_path: Path) -> None:
    from src.config import DEFAULT_OUTPUT_DIR, WORKSPACE_ROOT, resolve_output_dir

    assert resolve_output_dir("") == DEFAULT_OUTPUT_DIR
    assert resolve_output_dir("../Outputs") == DEFAULT_OUTPUT_DIR.resolve()
    assert resolve_output_dir("Single v7") == (DEFAULT_OUTPUT_DIR / "Single v7").resolve()
    assert resolve_output_dir(tmp_path / "out") == tmp_path / "out"
    assert DEFAULT_OUTPUT_DIR.parent == WORKSPACE_ROOT
    assert DEFAULT_OUTPUT_DIR.name == "Outputs"


def test_matching_skips_column_zero(tmp_path: Path) -> None:
    from src.io.tile_loader import matching_tile_paths

    pattern = r"R(?P<run>\d+)_(?P<row>\d+)_(?P<col>\d+)_(?P<mag>[\d.]+)X\.tiff?"
    write_tile_tiff(tmp_path / "R3_2_0_5X.tif", np.zeros((8, 8)))
    write_tile_tiff(tmp_path / "R3_2_1_5X.tif", np.zeros((8, 8)))
    names = [p.name for p in matching_tile_paths(tmp_path, pattern)]
    assert names == ["R3_2_1_5X.tif"]


def test_local_to_global_and_grid_origin() -> None:
    tile = Tile(
        path=Path("R3_1_2_5X.tif"),
        name="R3_1_2_5X.tif",
        image=np.zeros((100, 80)),
        row=1,
        col=2,
        run=3,
        magnification=5.0,
        x_origin=None,
        y_origin=None,
        height=100,
        width=80,
    )
    x0, y0 = tile_origin_px(tile, overlap_fraction=0.1)
    assert x0 == 2 * 80 * 0.9
    assert y0 == 1 * 100 * 0.9
    x_shifted, y_shifted = tile_origin_px(
        tile, overlap_fraction=0.1, row_origin=1, col_origin=1
    )
    assert x_shifted == 1 * 80 * 0.9
    assert y_shifted == 0.0
    gx, gy = local_to_global(10.0, 5.0, x0, y0)
    assert gx == x0 + 10.0
    assert gy == y0 + 5.0


def test_deduplicate_scales_to_many_points() -> None:
    rng = np.random.default_rng(0)
    coords = rng.random((4000, 2)) * 2000.0
    conf = rng.random(4000)
    particles = [
        Particle(
            id=0,
            x_global=float(x),
            y_global=float(y),
            size=5.0,
            confidence=float(c),
            source_tile="t0",
        )
        for (x, y), c in zip(coords, conf)
    ]
    merged = deduplicate(particles, merge_radius=8.0, size_aggregation="max")
    assert 0 < len(merged) <= len(particles)
    assert merged[0].id == 1


def test_deduplicate_merges_nearby_candidates() -> None:
    a = ParticleCandidate(y_local=10, x_local=10, size=8, confidence=0.9, circularity=0.8)
    b = ParticleCandidate(y_local=12, x_local=11, size=6, confidence=0.4, circularity=0.2)
    first = measure_candidates([a], 0, 0, 1.0, "t0")
    second = measure_candidates([b], 0, 0, 1.0, "t1")
    merged = deduplicate(first + second, merge_radius=5.0, size_aggregation="max")
    assert len(merged) == 1
    assert merged[0].confidence == 0.9
    assert merged[0].size == 8
    assert merged[0].source_tile == "t0"
    assert merged[0].id == 1
    assert merged[0].circularity == 0.8


def test_seam_particle_is_not_double_counted() -> None:
    """Particle centered on the shared edge of two overlapping tiles → one row."""
    overlap = 0.25
    width = 64
    step = width * (1.0 - overlap)
    # Global seam at x = step; particle at (x=step, y=32)
    left = ParticleCandidate(y_local=32.0, x_local=step, size=10.0, confidence=0.8)
    right = ParticleCandidate(y_local=32.0, x_local=0.0, size=9.0, confidence=0.7)
    from_left = measure_candidates([left], origin_x=0.0, origin_y=0.0, pixel_size_nm=1.0, source_tile="tile_r0_c0.tif")
    from_right = measure_candidates(
        [right], origin_x=step, origin_y=0.0, pixel_size_nm=1.0, source_tile="tile_r0_c1.tif"
    )
    config = detector_test_config(measurement={"merge_radius_px": 4.0})
    table = measure_and_dedupe(from_left + from_right, config)
    assert len(table) == 1
    assert abs(table.iloc[0]["x_global"] - step) < 1e-6


def test_load_rgb_tile_as_uint8_gray(tmp_path: Path) -> None:
    from src.io.tile_loader import load_tile_image, peek_tile_hw

    rgb = np.zeros((12, 10, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    path = tmp_path / "R3_0_1_5X.tif"
    tifffile.imwrite(path, rgb)
    assert peek_tile_hw(path) == (12, 10)
    gray = load_tile_image(path)
    assert gray.ndim == 2
    assert gray.shape == (12, 10)
    assert gray.dtype == np.uint8
    assert int(gray.max()) > 0


def test_pipeline_progress_reports_before_first_tile(tmp_path: Path) -> None:
    write_tile_tiff(tmp_path / "R3_0_1_5X.tif", make_structured_tile((32, 32)))
    config = detector_test_config()
    config["input_dir"] = str(tmp_path)
    config["output_dir"] = str(tmp_path / "out")
    events: list[tuple[int, int, str]] = []
    run_pipeline(config, progress_cb=lambda cur, tot, name: events.append((cur, tot, name)))
    assert events, "expected progress callbacks"
    assert events[0][0] == 0
    assert events[0][1] == 1
    assert "tile" in events[0][2]


def test_pipeline_thread_pool_processes_two_tiles(tmp_path: Path) -> None:
    write_tile_tiff(tmp_path / "R3_0_1_5X.tif", make_structured_tile((32, 32)))
    write_tile_tiff(tmp_path / "R3_0_2_5X.tif", make_structured_tile((32, 32)))
    config = detector_test_config(pipeline={"workers": 2})
    config["input_dir"] = str(tmp_path)
    config["output_dir"] = str(tmp_path / "out")
    events: list[tuple[int, int, str]] = []
    table, mosaic = run_pipeline(
        config, progress_cb=lambda cur, tot, name: events.append((cur, tot, name))
    )
    assert mosaic.placements
    assert any("thread" in name for _c, _t, name in events)
    assert isinstance(table, pd.DataFrame)


def test_pipeline_caches_mosaic_thumbnails(tmp_path: Path) -> None:
    height, width = 64, 64
    left = make_structured_tile((height, width), particles=[(20.0, 22.0, 3.5)])
    right = make_structured_tile((height, width), particles=[(20.0, 10.0, 3.5)])
    write_tile_tiff(tmp_path / "R3_0_1_5X.tif", left)
    write_tile_tiff(tmp_path / "R3_0_2_5X.tif", right)
    config = detector_test_config(report={"downsample": 4, "target_mb": 20.0})
    config["input_dir"] = str(tmp_path)
    config["output_dir"] = str(tmp_path / "out")
    _table, mosaic = run_pipeline(config)
    assert mosaic.thumbnail_factor == 4
    assert set(mosaic.thumbnails) == {"R3_0_1_5X.tif", "R3_0_2_5X.tif"}
    preview = mosaic.preview(downsample=4, workers=1)
    step = int(width * (1.0 - 0.25))
    assert preview.shape == (height // 4, (step + width) // 4)
    for thumb in mosaic.thumbnails.values():
        assert thumb.shape == (height // 4, width // 4)


def test_pipeline_seam_on_synthetic_tiffs(tmp_path: Path) -> None:
    overlap = 0.25
    height, width = 96, 96
    step = int(width * (1.0 - overlap))
    # Place the blob inside the overlap so both tiles see a full particle, not a clip.
    y_p, x_global = 48.0, float(step + 12)
    sigma = 4.0

    left = make_structured_tile(
        (height, width), particles=[(y_p, x_global, sigma)]
    )
    right = make_structured_tile(
        (height, width), particles=[(y_p, x_global - step, sigma)]
    )
    write_tile_tiff(tmp_path / "R3_0_1_5X.tif", left)
    write_tile_tiff(tmp_path / "R3_0_2_5X.tif", right)

    config = detector_test_config()
    config["input_dir"] = str(tmp_path)
    config["output_dir"] = str(tmp_path / "out")
    config["overlap_fraction"] = overlap
    table, mosaic = run_pipeline(config)

    near_seam = table[
        np.hypot(table["x_global"] - x_global, table["y_global"] - y_p) < 8.0
    ]
    assert len(near_seam) == 1
    assert mosaic.full_width == step + width
    preview = mosaic.preview(downsample=4)
    assert preview.shape[0] > 0 and preview.shape[1] > 0


def test_downsample_for_target_caps_mosaic_bytes() -> None:
    from src.stitching.stitcher import downsample_for_target

    height, width = 9549, 86464
    target_mb = 20.0
    factor = downsample_for_target(height, width, target_mb=target_mb, channels=3)
    assert factor > 1
    out_bytes = (height / factor) * (width / factor) * 3
    assert out_bytes <= target_mb * 1024 * 1024
    assert downsample_for_target(64, 64, target_mb=20.0, channels=3) == 1


def test_preview_downsizes_tiles_then_pastes_on_canvas(tmp_path: Path) -> None:
    from src.stitching.stitcher import LazyMosaic, TilePlacement

    height, width, factor = 16, 16, 4
    left_path = tmp_path / "R3_0_0_5X.tif"
    right_path = tmp_path / "R3_0_1_5X.tif"
    tifffile.imwrite(left_path, np.full((height, width), 1000, dtype=np.uint16))
    tifffile.imwrite(right_path, np.full((height, width), 4000, dtype=np.uint16))
    mosaic = LazyMosaic(
        [
            TilePlacement(left_path, left_path.name, 0, 0, height, width),
            TilePlacement(right_path, right_path.name, 0, width, height, width),
        ]
    )
    preview = mosaic.preview(downsample=factor, workers=1)
    assert preview.shape == (height // factor, (2 * width) // factor)
    assert float(preview[0, 0]) == 1000.0
    assert float(preview[0, -1]) == 4000.0


def test_full_mosaic_preview_downsamples_tiles_before_stitch(tmp_path: Path) -> None:
    from src.report.report_generator import overlay_downsample, overlay_markers, write_overlay_image
    from src.stitching.stitcher import LazyMosaic, TilePlacement

    height, width = 80, 120
    left = make_structured_tile((height, width), particles=[(20.0, 30.0, 3.0)])
    right = make_structured_tile((height, width), particles=[(20.0, 10.0, 3.0)])
    left_path = write_tile_tiff(tmp_path / "R3_0_0_5X.tif", left)
    right_path = write_tile_tiff(tmp_path / "R3_0_1_5X.tif", right)
    placements = [
        TilePlacement(left_path, left_path.name, y0=0, x0=0, height=height, width=width),
        TilePlacement(right_path, right_path.name, y0=0, x0=width, height=height, width=width),
    ]
    mosaic = LazyMosaic(placements)
    config = detector_test_config(report={"downsample": 0, "target_mb": 0.01})
    factor = overlay_downsample(mosaic, config, crop=None)
    assert factor > 1
    overlay = overlay_markers(mosaic, pd.DataFrame(), config, crop=None)
    assert overlay.ndim == 3
    assert overlay.shape[0] == mosaic.preview(downsample=factor).shape[0]
    assert overlay.shape[1] == mosaic.preview(downsample=factor).shape[1]
    assert overlay.nbytes <= 0.05 * 1024 * 1024
    written = write_overlay_image(
        mosaic, pd.DataFrame(), config, tmp_path / "mosaic_overlay.jpg"
    )
    assert written.exists() and written.stat().st_size > 0


def test_pipeline_table_includes_feature_columns(tmp_path: Path) -> None:
    from src.detection.detector import CANDIDATE_FEATURE_FIELDS
    from src.io.results_writer import write_csv

    image = make_structured_tile((64, 64), particles=[(32.0, 30.0, 3.5)])
    write_tile_tiff(tmp_path / "R3_0_1_5X.tif", image)
    config = detector_test_config()
    config["input_dir"] = str(tmp_path)
    config["output_dir"] = str(tmp_path / "out")
    table, _mosaic = run_pipeline(config)
    for column in CANDIDATE_FEATURE_FIELDS:
        assert column in table.columns
    written = write_csv(table, tmp_path / "out" / "particles.csv")
    loaded = pd.read_csv(written)
    for column in CANDIDATE_FEATURE_FIELDS:
        assert column in loaded.columns
