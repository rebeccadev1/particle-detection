from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from src.analyse.detection.detector import ParticleCandidate
from src.overall.io.tile_loader import parse_tile_filename, try_parse_tile_filename
from src.analyse.measurement.measurer import (
    Particle,
    deduplicate,
    local_to_global,
    measure_and_dedupe,
    measure_candidates,
)
from src.analyse.runner import PipelineStopped, run_pipeline
from src.overall.io.placement import tile_origin_px
from src.overall.io.tile_loader import Tile
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


def test_blank_filename_pattern_does_not_match_quad_shots() -> None:
    name = "wafer_shot_0_quad0_exp200000.png"
    assert try_parse_tile_filename(name, "") is None
    assert try_parse_tile_filename(name, r"R(?P<run>\d+)") is None


def test_resolve_input_dir_finds_workspace_tile_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.overall import config as config_mod
    from src.overall.io import tile_loader

    inputs = tmp_path / "Inputs"
    folder = inputs / "R3 04-08"
    folder.mkdir(parents=True)
    (folder / "R3_1_26_5X.tif").write_bytes(b"placeholder")

    monkeypatch.setattr(config_mod, "DEFAULT_INPUT_DIR", inputs)
    monkeypatch.setattr(tile_loader, "DEFAULT_INPUT_DIR", inputs)

    found = tile_loader.resolve_input_dir("R3 04-08")
    assert found == folder.resolve()
    assert (found / "R3_1_26_5X.tif").is_file()

    via_dotdot = tile_loader.resolve_input_dir("../R3 04-08")
    assert via_dotdot == folder.resolve()


def test_resolve_output_dir_uses_workspace_outputs(tmp_path: Path) -> None:
    from src.overall.config import DEFAULT_OUTPUT_DIR, WORKSPACE_ROOT, resolve_output_dir

    assert resolve_output_dir("") == DEFAULT_OUTPUT_DIR
    assert resolve_output_dir("../Outputs") == DEFAULT_OUTPUT_DIR.resolve()
    assert resolve_output_dir("Single v7") == (DEFAULT_OUTPUT_DIR / "Single v7").resolve()
    assert resolve_output_dir(tmp_path / "out") == tmp_path / "out"
    assert DEFAULT_OUTPUT_DIR.parent == WORKSPACE_ROOT
    assert DEFAULT_OUTPUT_DIR.name == "Outputs"


def test_next_versioned_output_name_uses_input_folder(tmp_path: Path) -> None:
    from src.overall.config import next_versioned_output_name

    outputs = tmp_path / "Outputs"
    outputs.mkdir()
    (outputs / "Groundup v7 (06-10) v1").mkdir()
    (outputs / "Groundup v7 (06-10) v3").mkdir()
    (outputs / "Groundup v7 (06-10) notes").mkdir()

    assert next_versioned_output_name("") == ""
    assert (
        next_versioned_output_name("Groundup/Groundup v7 (06-10)", outputs)
        == "Groundup v7 (06-10) v4"
    )
    assert next_versioned_output_name("R3 04-08", outputs) == "R3 04-08 v1"


def test_advance_versioned_output_name_bumps_each_run(tmp_path: Path) -> None:
    from src.overall.config import advance_versioned_output_name

    outputs = tmp_path / "Outputs"
    outputs.mkdir()
    (outputs / "v7 2026-10-16 v1").mkdir()

    assert (
        advance_versioned_output_name(
            "Groundup/v7 2026-10-16", "v7 2026-10-16 v1", outputs
        )
        == "v7 2026-10-16 v2"
    )
    assert (
        advance_versioned_output_name("R3 04-08", "R3 04-08 v1", outputs)
        == "R3 04-08 v2"
    )
    assert (
        advance_versioned_output_name("R3 04-08", "kept-name", outputs)
        == "kept-name"
    )


def test_reserve_fresh_output_dir_skips_an_existing_folder(tmp_path: Path, monkeypatch) -> None:
    import src.overall.config as config_mod
    from src.overall.config import reserve_fresh_output_dir

    outputs = tmp_path / "Outputs"
    monkeypatch.setattr(config_mod, "DEFAULT_OUTPUT_DIR", outputs)
    run = "Tiles"
    taken = outputs / f"{run} v1"
    taken.mkdir(parents=True)
    (taken / "particles.csv").write_text("old", encoding="utf-8")
    (outputs / f"{run} v2").mkdir()
    config = {
        "input_dir": str(tmp_path / "Inputs" / run),
        "output_dir": f"{run} v1",
    }

    dest = reserve_fresh_output_dir(config)

    assert dest == (outputs / f"{run} v3").resolve()
    assert config["output_dir"] == f"{run} v3"
    assert (taken / "particles.csv").read_text(encoding="utf-8") == "old"

    absolute = {
        "input_dir": str(tmp_path / "Inputs" / run),
        "output_dir": str(taken),
    }
    assert reserve_fresh_output_dir(absolute) == (outputs / f"{run} v3").resolve()
    assert absolute["output_dir"] == str((outputs / f"{run} v3").resolve())

    empty = outputs / f"{run} v4"
    fresh = {"input_dir": config["input_dir"], "output_dir": str(empty)}
    assert reserve_fresh_output_dir(fresh) == empty
    assert fresh["output_dir"] == str(empty)


def test_output_folder_choices_keep_current_and_standard_name(tmp_path: Path) -> None:
    from src.overall.config import output_choice_label, output_folder_choices, output_folder_labels

    outputs = tmp_path / "Outputs"
    (outputs / "Single v7").mkdir(parents=True)
    (outputs / ".hidden").mkdir()
    (outputs / "notes.txt").write_text("skip", encoding="utf-8")

    assert output_folder_labels(outputs) == ["Single v7"]
    assert output_choice_label("../Outputs/Single v7") == "Single v7"
    assert output_choice_label("") == ""
    assert output_folder_choices("Single v7", "Single v8", outputs) == ["Single v8", "Single v7"]
    assert output_folder_choices("", "R3 04-08 v1", outputs) == ["R3 04-08 v1", "Single v7"]


def test_matching_skips_column_zero(tmp_path: Path) -> None:
    from src.overall.io.tile_loader import matching_tile_paths

    pattern = r"R(?P<run>\d+)_(?P<row>\d+)_(?P<col>\d+)_(?P<mag>[\d.]+)X\.tiff?"
    write_tile_tiff(tmp_path / "R3_2_0_5X.tif", np.zeros((8, 8)))
    write_tile_tiff(tmp_path / "R3_2_1_5X.tif", np.zeros((8, 8)))
    names = [p.name for p in matching_tile_paths(tmp_path, pattern)]
    assert names == ["R3_2_1_5X.tif"]


def test_matching_can_include_unmatched_names(tmp_path: Path) -> None:
    from src.overall.io.tile_loader import matching_tile_paths

    pattern = r"R(?P<run>\d+)_(?P<row>\d+)_(?P<col>\d+)_(?P<mag>[\d.]+)X\.tiff?"
    write_tile_tiff(tmp_path / "R3_2_1_5X.tif", np.zeros((8, 8)))
    write_tile_tiff(tmp_path / "loose_photo.tif", np.zeros((8, 8)))
    skipped = [p.name for p in matching_tile_paths(tmp_path, pattern)]
    included = [
        p.name
        for p in matching_tile_paths(tmp_path, pattern, include_unmatched=True)
    ]
    assert skipped == ["R3_2_1_5X.tif"]
    assert included == ["R3_2_1_5X.tif", "loose_photo.tif"]


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
    assert merged[0].nsew_count == 0
    assert merged[0].nsew_dirs == ""


def test_nsew_hits_at_same_place_are_one_particle() -> None:
    
    north = Particle(
        id=0,
        x_global=100.0,
        y_global=200.0,
        size=12.0,
        confidence=0.9,
        source_tile="N.bmp",
    )
    south = Particle(
        id=0,
        x_global=112.0,
        y_global=203.0,
        size=40.0,
        confidence=0.5,
        source_tile="S.bmp",
    )
    east = Particle(
        id=0,
        x_global=108.0,
        y_global=198.0,
        size=22.0,
        confidence=0.7,
        source_tile="E.png",
    )
    west = Particle(
        id=0,
        x_global=400.0,
        y_global=200.0,
        size=18.0,
        confidence=0.8,
        source_tile="W.bmp",
    )
    merged = deduplicate(
        [north, south, east, west],
        merge_radius=6.0,
        size_aggregation="max",
        size_match_fraction=0.4,
    )
    assert len(merged) == 2
    same = next(p for p in merged if p.source_tile == "N.bmp")
    other = next(p for p in merged if p.source_tile == "W.bmp")
    assert same.size == 40.0
    assert same.nsew_count == 3
    assert same.nsew_dirs == "N,S,E"
    assert other.nsew_count == 1
    assert other.nsew_dirs == "W"


def test_nsew_does_not_merge_two_hits_on_the_same_tile() -> None:
    first = Particle(
        id=0, x_global=100.0, y_global=200.0, size=12.0, confidence=0.9, source_tile="N.bmp"
    )
    second = Particle(
        id=0, x_global=108.0, y_global=200.0, size=11.0, confidence=0.8, source_tile="N.bmp"
    )
    west = Particle(
        id=0, x_global=104.0, y_global=201.0, size=13.0, confidence=0.7, source_tile="W.bmp"
    )
    from src.analyse.measurement.measurer import deduplicate_nsew

    merged = deduplicate_nsew(
        [first, second, west],
        merge_radius=20.0,
        size_aggregation="max",
        size_match_fraction=0.4,
    )
    north = [p for p in merged if p.source_tile == "N.bmp"]
    assert len(north) == 2
    assert any(p.nsew_dirs == "N,W" for p in north)
    assert any(p.nsew_count == 1 and p.source_tile == "N.bmp" for p in north)


def test_combine_tables_merges_directional_unplaced_tiles() -> None:
    from src.analyse.runner import _combine_particle_tables

    north = Particle(id=0, x_global=10.0, y_global=12.0, size=8.0, confidence=0.9, source_tile="N.tif")
    south = Particle(id=0, x_global=11.0, y_global=12.0, size=30.0, confidence=0.4, source_tile="S.tif")
    loose = Particle(id=0, x_global=10.0, y_global=12.0, size=9.0, confidence=0.8, source_tile="loose.tif")
    table = _combine_particle_tables(
        [],
        {"N.tif": [north], "S.tif": [south], "loose.tif": [loose]},
        detector_test_config(),
    )
    dirs = table.loc[table["source_tile"].isin(["N.tif", "S.tif"])]
    assert len(dirs) == 1
    assert int(dirs.iloc[0]["nsew_count"]) == 2
    assert str(dirs.iloc[0]["nsew_dirs"]) == "N,S"
    assert float(dirs.iloc[0]["size"]) == 30.0
    assert "loose.tif" in set(table["source_tile"].astype(str))
    assert len(table) == 2


def test_combine_tables_merges_particles_only_versions() -> None:
    from src.analyse.runner import _combine_particle_tables

    two = Particle(
        id=0,
        x_global=10.0,
        y_global=12.0,
        size=8.0,
        confidence=0.5,
        source_tile="particles_only_2of4.png",
    )
    four = Particle(
        id=0,
        x_global=11.0,
        y_global=12.0,
        size=30.0,
        confidence=0.9,
        source_tile="particles_only_4of4.png",
    )
    table = _combine_particle_tables(
        [],
        {
            "particles_only_2of4.png": [two],
            "particles_only_4of4.png": [four],
        },
        detector_test_config(),
    )
    assert len(table) == 1
    assert int(table.iloc[0]["nsew_count"]) == 2
    dirs = set(str(table.iloc[0]["nsew_dirs"]).split(","))
    assert dirs == {"PARTICLES_ONLY_2OF4", "PARTICLES_ONLY_4OF4"}
    assert float(table.iloc[0]["size"]) == 30.0


def test_seam_particle_is_not_double_counted() -> None:
    
    overlap = 0.25
    width = 64
    step = width * (1.0 - overlap)
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
    from src.overall.io.tile_loader import load_tile_image, peek_tile_hw

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


def test_load_bmp_bytes_named_as_tiff(tmp_path: Path) -> None:
    import cv2
    from src.overall.io.tile_loader import load_tile_image, peek_tile_hw

    gray = np.arange(12 * 10, dtype=np.uint8).reshape(12, 10)
    bmp = tmp_path / "scratch.bmp"
    path = tmp_path / "R3_0_1_5X.tif"
    assert cv2.imwrite(str(bmp), gray)
    path.write_bytes(bmp.read_bytes())
    assert peek_tile_hw(path) == (12, 10)
    loaded = load_tile_image(path)
    assert loaded.shape == (12, 10)
    assert int(loaded.max()) > 0


def test_load_jpeg_bytes_named_as_tiff(tmp_path: Path) -> None:
    import cv2
    from src.overall.io.tile_loader import load_tile_image, peek_tile_hw

    bgr = np.zeros((16, 14, 3), dtype=np.uint8)
    bgr[..., 2] = 255
    path = tmp_path / "R0_1_1.tiff"
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok
    path.write_bytes(encoded.tobytes())
    assert peek_tile_hw(path) == (16, 14)
    gray = load_tile_image(path)
    assert gray.ndim == 2
    assert gray.shape == (16, 14)
    assert int(gray.max()) > 0


def test_load_and_match_bmp_tiles(tmp_path: Path) -> None:
    import cv2
    from src.overall.io.tile_loader import (
        DEFAULT_FILENAME_PATTERN,
        list_tile_paths,
        load_tile_image,
        matching_tile_paths,
        peek_tile_hw,
    )

    gray = np.arange(12 * 10, dtype=np.uint8).reshape(12, 10)
    path = tmp_path / "R3_2_1_5X.bmp"
    assert cv2.imwrite(str(path), gray)
    names = [p.name for p in list_tile_paths(tmp_path)]
    assert names == ["R3_2_1_5X.bmp"]
    assert matching_tile_paths(tmp_path, DEFAULT_FILENAME_PATTERN) == [path]
    assert peek_tile_hw(path) == (12, 10)
    loaded = load_tile_image(path)
    assert loaded.ndim == 2
    assert loaded.shape == (12, 10)
    assert int(loaded.max()) > 0


def test_pipeline_runs_on_bmp_tiles(tmp_path: Path) -> None:
    import cv2

    image = make_structured_tile((64, 64), particles=[(20.0, 22.0, 3.5)])
    scaled = np.clip(image / image.max() * 255.0, 0, 255).astype(np.uint8)
    path = tmp_path / "R3_0_1_5X.bmp"
    assert cv2.imwrite(str(path), scaled)
    config = detector_test_config()
    config["input_dir"] = str(tmp_path)
    config["output_dir"] = str(tmp_path / "out")
    table, mosaic = run_pipeline(config)
    assert [p.name for p in mosaic.placements] == ["R3_0_1_5X.bmp"]
    assert set(table["source_tile"].astype(str)) == {"R3_0_1_5X.bmp"}
    book = pd.read_excel(tmp_path / "out" / "particles.xlsx", sheet_name=None)
    assert list(book) == ["particles", "parameters"]
    settings = json.loads((tmp_path / "out" / "last_pipeline.json").read_text(encoding="utf-8"))
    assert settings["detection"]["method"] == "tophat"
    saved_table = pd.read_csv(tmp_path / "out" / "particles.csv")
    assert list(saved_table.columns) == [
        "id",
        "x_global_um",
        "y_global_um",
        "size_um",
        "source_tile",
        "confidence",
    ]
    from src.analyse.report.report_generator import summary_pdf_name

    assert (tmp_path / "out" / summary_pdf_name(config)).read_bytes().startswith(b"%PDF")
    assert (tmp_path / "out" / "mosaic_overlay.png").is_file()
    assert (tmp_path / "out" / "mosaic_plain.png").is_file()
    saved = dict(zip(book["parameters"]["parameter"], book["parameters"]["value"]))
    assert saved["preprocessing.denoise"] is False
    assert saved["detection.method"] == "tophat"


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


def test_pipeline_stop_skips_later_tiles_and_skips_outputs(tmp_path: Path) -> None:
    write_tile_tiff(tmp_path / "R3_0_1_5X.tif", make_structured_tile((32, 32)))
    write_tile_tiff(tmp_path / "R3_0_2_5X.tif", make_structured_tile((32, 32)))
    config = detector_test_config()
    config["input_dir"] = str(tmp_path)
    config["output_dir"] = str(tmp_path / "out")
    seen: list[str] = []

    def progress(_current: int, _total: int, name: str) -> None:
        seen.append(name)

    def should_stop() -> bool:
        return any(name.endswith(".tif") for name in seen)

    with pytest.raises(PipelineStopped):
        run_pipeline(config, progress_cb=progress, should_stop=should_stop)

    assert [name for name in seen if name.endswith(".tif")] == ["R3_0_1_5X.tif"]
    assert not (tmp_path / "out" / "particles.csv").exists()
    assert not (tmp_path / "out" / "particles.xlsx").exists()
    assert not (tmp_path / "out" / "last_pipeline.json").exists()
    from src.analyse.report.report_generator import summary_pdf_name

    assert not (tmp_path / "out" / summary_pdf_name(config)).exists()


def test_pipeline_stop_cancels_queued_tiles(tmp_path: Path) -> None:
    write_tile_tiff(tmp_path / "R3_0_1_5X.tif", make_structured_tile((32, 32)))
    write_tile_tiff(tmp_path / "R3_0_2_5X.tif", make_structured_tile((32, 32)))
    config = detector_test_config(pipeline={"workers": 2})
    config["input_dir"] = str(tmp_path)
    config["output_dir"] = str(tmp_path / "out")
    seen: list[str] = []

    def progress(_current: int, _total: int, name: str) -> None:
        seen.append(name)

    def should_stop() -> bool:
        return any("worker" in name for name in seen)

    with pytest.raises(PipelineStopped):
        run_pipeline(config, progress_cb=progress, should_stop=should_stop)

    assert not any(name.endswith(".tif") for name in seen)
    assert not (tmp_path / "out" / "particles.csv").exists()


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
    config = detector_test_config(report={"target_mb": 0.0015})
    config["input_dir"] = str(tmp_path)
    config["output_dir"] = str(tmp_path / "out")
    _table, mosaic = run_pipeline(config)
    factor = mosaic.thumbnail_factor
    assert factor == 4
    assert set(mosaic.thumbnails) == {"R3_0_1_5X.tif", "R3_0_2_5X.tif"}
    preview = mosaic.preview(downsample=factor, workers=1)
    step = int(width * (1.0 - 0.25))
    assert preview.shape == (height // factor, (step + width) // factor)
    for thumb in mosaic.thumbnails.values():
        assert thumb.shape == (height // factor, width // factor)


def test_photo_mosaic_keeps_saved_brightness() -> None:
    from src.analyse.report.report_generator import photo_display_rgb

    dark = np.full((8, 8), 20, np.float32)
    assert int(photo_display_rgb(dark).mean()) == 20


def test_grid_tile_mosaic_keeps_the_photo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    
    monkeypatch.setattr(
        "src.overall.io.naming.DEFAULT_INPUT_DIR",
        tmp_path,
    )
    image = make_structured_tile((64, 64), particles=[(20.0, 22.0, 3.5)])
    write_tile_tiff(tmp_path / "(1,1).tif", image)
    config = detector_test_config(report={"target_mb": 0.01})
    config["filename_pattern"] = r"\((?P<row>\d+),(?P<col>\d+)\)\.(?:tiff?|png)"
    config["input_dir"] = str(tmp_path)
    config["output_dir"] = str(tmp_path / "out")
    _table, mosaic = run_pipeline(config)
    assert set(mosaic.thumbnails) == {"(1,1).tif"}
    preview = mosaic.preview(downsample=mosaic.thumbnail_factor, workers=1)
    assert float(preview.max()) > 0.0


def test_pipeline_seam_on_synthetic_tiffs(tmp_path: Path) -> None:
    overlap = 0.25
    height, width = 96, 96
    step = int(width * (1.0 - overlap))
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
    from src.analyse.stitching.stitcher import downsample_for_target

    height, width = 9549, 86464
    target_mb = 20.0
    factor = downsample_for_target(height, width, target_mb=target_mb, channels=3)
    assert factor > 1
    out_bytes = (height / factor) * (width / factor) * 3
    assert out_bytes <= target_mb * 1024 * 1024
    assert downsample_for_target(64, 64, target_mb=20.0, channels=3) == 1


def test_preview_downsizes_tiles_then_pastes_on_canvas(tmp_path: Path) -> None:
    from src.overall.io.placement import TilePlacement
    from src.analyse.stitching.stitcher import LazyMosaic

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
    from src.analyse.report.report_generator import draw_overlay_markers, overlay_downsample, overlay_rgb
    from src.overall.io.placement import TilePlacement
    from src.analyse.stitching.stitcher import LazyMosaic

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
    config = detector_test_config(report={"target_mb": 0.01})
    factor = overlay_downsample(mosaic, config, crop=None)
    assert factor > 1
    overlay = draw_overlay_markers(
        overlay_rgb(mosaic, config, crop=None), mosaic, pd.DataFrame(), config, crop=None
    )
    assert overlay.ndim == 3
    assert overlay.shape[0] == mosaic.preview(downsample=factor).shape[0]
    assert overlay.shape[1] == mosaic.preview(downsample=factor).shape[1]
    assert overlay.nbytes <= 0.05 * 1024 * 1024


def test_pipeline_table_includes_feature_columns(tmp_path: Path) -> None:
    from src.analyse.detection.detector import CANDIDATE_FEATURE_FIELDS
    from src.overall.io.results_writer import write_csv

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
    assert list(loaded.columns) == [
        "id",
        "x_global_um",
        "y_global_um",
        "size_um",
        "source_tile",
        "confidence",
    ]
    assert "size" not in loaded.columns


def test_pipeline_detects_only_the_combi(tmp_path: Path) -> None:
    image = make_structured_tile((64, 64), particles=[(20.0, 22.0, 3.5)])
    write_tile_tiff(tmp_path / "N.tif", image)
    write_tile_tiff(tmp_path / "S.tif", image)
    write_tile_tiff(tmp_path / "E.tif", image)
    write_tile_tiff(tmp_path / "W.tif", image)
    config = detector_test_config()
    config["input_dir"] = str(tmp_path)
    config["output_dir"] = str(tmp_path / "out")
    table, mosaic = run_pipeline(config)
    assert (tmp_path / "combi_2of4.tif").is_file()
    assert mosaic.placements == []
    assert not table.empty
    sources = {Path(name).name for name in table["source_tile"].astype(str)}
    assert sources == {"combi_2of4.tif"}
    from src.analyse.report.report_generator import unstitched_tile_paths

    assert [path.name for path in unstitched_tile_paths(config)] == ["combi_2of4.tif"]
    dirs = {
        part.strip()
        for text in table["nsew_dirs"].fillna("").astype(str)
        for part in text.split(",")
        if part.strip()
    }
    assert dirs == {"COMBI"}


def test_pipeline_detects_unmatched_names_without_stitching(tmp_path: Path) -> None:
    image = make_structured_tile((64, 64), particles=[(20.0, 22.0, 3.5)])
    write_tile_tiff(tmp_path / "random_photo.tif", image)
    config = detector_test_config()
    config["input_dir"] = str(tmp_path)
    config["output_dir"] = str(tmp_path / "out")
    table, mosaic = run_pipeline(config)
    assert mosaic.placements == []
    assert mosaic.full_height == 0
    assert not table.empty
    assert set(table["source_tile"].astype(str)) == {"random_photo.tif"}


def test_pipeline_stitches_only_patterned_tiles(tmp_path: Path) -> None:
    from src.analyse.report.report_generator import draw_overlay_markers, overlay_rgb

    patterned = make_structured_tile((64, 64), particles=[(20.0, 22.0, 3.5)])
    loose = make_structured_tile((64, 64), particles=[(20.0, 22.0, 3.5)])
    write_tile_tiff(tmp_path / "R3_0_1_5X.tif", patterned)
    write_tile_tiff(tmp_path / "loose_photo.tif", loose)
    config = detector_test_config()
    config["input_dir"] = str(tmp_path)
    config["output_dir"] = str(tmp_path / "out")
    table, mosaic = run_pipeline(config)
    assert [p.name for p in mosaic.placements] == ["R3_0_1_5X.tif"]
    assert "loose_photo.tif" in set(table["source_tile"].astype(str))
    assert "R3_0_1_5X.tif" in set(table["source_tile"].astype(str))
    overlay = draw_overlay_markers(overlay_rgb(mosaic, config, crop=None), mosaic, table, config, crop=None)
    assert overlay.shape[0] > 0
    assert overlay.shape[1] > 0
    from src.analyse.runner import load_report_mosaic

    report = load_report_mosaic(config)
    assert [p.name for p in report.placements] == [p.name for p in mosaic.placements]
    plain = overlay_rgb(report, config, crop=None)
    orange = (overlay[:, :, 0].astype(int) - overlay[:, :, 2].astype(int)) > 80
    plain_orange = (plain[:, :, 0].astype(int) - plain[:, :, 2].astype(int)) > 80
    assert int(orange.sum()) > int(plain_orange.sum())
