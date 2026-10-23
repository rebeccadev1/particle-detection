"""Capture folders are named by grid location, inside a daily run folder."""

from datetime import date

import json

from src.capture.camera_automation import (
    CAPTURES_DIR,
    allocate_run_folder,
    latest_run_folder,
    next_run_folder,
    image_filename,
    write_manifest,
)


def test_picture_names_are_the_grid_position() -> None:
    assert image_filename(1, 1) == "(1,1).png"
    assert image_filename(2, 7, "N") == "(2,7)_N.png"


def test_manifest_lives_in_the_run_folder(tmp_path) -> None:
    write_manifest(
        tmp_path,
        {"mode": "uniform", "row": 1, "col": 1, "files": ["(1,1).png"], "shutter_ms": 25.0},
    )
    write_manifest(
        tmp_path,
        {
            "mode": "nsew",
            "row": 1,
            "col": 2,
            "files": ["(1,2)_N.png"],
            "shutter_ms": 100.0,
        },
    )
    write_manifest(
        tmp_path,
        {"mode": "nsew", "row": 1, "col": 2, "files": ["(1,2)_E.png"], "shutter_ms": 100.0},
    )
    data = json.loads((tmp_path / "capture.json").read_text())
    assert data["stops"] == [
        {"row": 1, "col": 1, "files": ["(1,1).png"]},
        {"row": 1, "col": 2, "files": ["(1,2)_N.png", "(1,2)_E.png"]},
    ]
    assert data["shutter_ms"] == 100.0


def test_manifest_keeps_stage_settings(tmp_path) -> None:
    stage = {
        "fov_mm": 7.4,
        "overlap": 0.10,
        "step_mm": 6.66,
        "columns": 10,
        "rows": 7,
        "feed_mm_per_min": 1000,
    }
    write_manifest(
        tmp_path,
        {"mode": "uniform", "row": 1, "col": 1, "files": ["(1,1).png"], "stage": stage},
    )
    write_manifest(
        tmp_path,
        {"mode": "uniform", "row": 1, "col": 2, "files": ["(1,2).png"]},
    )
    data = json.loads((tmp_path / "capture.json").read_text())
    assert data["stage"] == stage
    assert [stop["col"] for stop in data["stops"]] == [1, 2]


def test_captures_go_to_inputs_groundup() -> None:
    assert CAPTURES_DIR.name == "Groundup"
    assert CAPTURES_DIR.parent.name == "Inputs"
    assert (CAPTURES_DIR.parent.parent / "Outputs").is_dir()


def test_latest_run_folder_is_the_newest_date_then_version(tmp_path) -> None:
    assert latest_run_folder(tmp_path) is None
    older = tmp_path / "v9 2026-10-08"
    newer_low = tmp_path / "v1 2026-10-20"
    newer_high = tmp_path / "v2 2026-10-20"
    for folder in (older, newer_low, newer_high):
        folder.mkdir()
    (tmp_path / "notes").mkdir()
    assert latest_run_folder(tmp_path) == newer_high


def test_next_run_folder_does_not_create_the_directory(tmp_path) -> None:
    today = date(2026, 10, 9)
    (tmp_path / "v1 2026-10-09").mkdir()
    folder = next_run_folder(tmp_path, on=today)
    assert folder.name == "v2 2026-10-09"
    assert not folder.exists()
    missing = tmp_path / "missing"
    assert next_run_folder(missing, on=today).name == "v1 2026-10-09"
    assert not missing.exists()


def test_run_folders_count_only_that_day(tmp_path) -> None:
    today = date(2026, 10, 9)
    yesterday = date(2026, 10, 8)
    (tmp_path / "v3 2026-10-08").mkdir()
    (tmp_path / "(1,1)").mkdir()
    first = allocate_run_folder(tmp_path, on=today)
    second = allocate_run_folder(tmp_path, on=today)
    earlier = allocate_run_folder(tmp_path, on=yesterday)
    assert first.name == "v1 2026-10-09"
    assert second.name == "v2 2026-10-09"
    assert earlier.name == "v4 2026-10-08"
