from pathlib import Path

import pytest

from src.capture.camera_automation import MOTION_PAUSE_S, NSEW_GAP_S, SETTLE_S
from src.capture.stage import (
    PAUSE_MARGIN_MS,
    build_home_gcode,
    build_relative_move_gcode,
    build_scan_gcode,
    move_axis,
    move_z,
    pause_plan,
    return_from_stop_moves,
    return_to_start_moves,
    scan_stops,
    scan_summary,
)


def test_old_scan_40x60_still_builds() -> None:
    path = Path("/Users/rebeccajekel/Desktop/scan_40x60.gcode")
    if not path.is_file():
        pytest.skip("reference gcode is not on this machine")
    built = build_scan_gcode(step_mm=6.3, columns=7, rows=10, dwell_ms=1000)
    without_return = built.replace("; Return to stop (1, 1)\nG1 Y56.70 F1000\n", "")
    assert without_return.replace("6.30", "6.3") == path.read_text()


def test_step_is_the_field_left_after_overlap() -> None:
    from src.capture.stage import (
        DEFAULT_COLUMNS,
        DEFAULT_ROWS,
        DEFAULT_STEP_MM,
        FOV_MM,
        OVERLAP,
        SCAN_HEIGHT_MM,
        SCAN_WIDTH_MM,
        grid_stops,
        step_from_fov,
    )

    assert step_from_fov(7.4, 0.10) == pytest.approx(6.66)
    assert DEFAULT_STEP_MM == pytest.approx(step_from_fov(FOV_MM, OVERLAP))
    assert DEFAULT_COLUMNS == grid_stops(SCAN_WIDTH_MM)
    assert DEFAULT_ROWS == grid_stops(SCAN_HEIGHT_MM)
    covered_x = FOV_MM + (DEFAULT_COLUMNS - 1) * DEFAULT_STEP_MM
    covered_y = FOV_MM + (DEFAULT_ROWS - 1) * DEFAULT_STEP_MM
    assert covered_x >= SCAN_WIDTH_MM
    assert covered_y >= SCAN_HEIGHT_MM


def test_one_row_has_no_y_move() -> None:
    text = build_scan_gcode(step_mm=1, columns=3, rows=1, feed=100, dwell_ms=50)
    assert "Y" not in text
    assert text.splitlines() == [
        "G21",
        "G91",
        "",
        "; Start at top-left position",
        "G4 P50",
        "G1 X1.00 F100",
        "G4 P50",
        "G1 X1.00 F100",
        "G4 P50",
        "; Return to stop (1, 1)",
        "G1 X-2.00 F100",
        "",
        "G90",
    ]


def test_second_row_reverses_x() -> None:
    text = build_scan_gcode(step_mm=2.5, columns=2, rows=2, feed=800, dwell_ms=10)
    assert "G1 X2.50 F800" in text
    assert "G1 Y-2.50 F800" in text
    assert "G1 X-2.50 F800" in text


def test_summary_counts_stops() -> None:
    summary = scan_summary(6.3, 7, 10)
    assert summary["columns"] == 7
    assert summary["rows"] == 10
    assert summary["stops"] == 70
    assert summary["x_mm"] == pytest.approx(37.8)
    assert summary["y_mm"] == pytest.approx(56.7)
    assert summary["covered_x_mm"] == pytest.approx(7.4 + 37.8)
    assert summary["covered_y_mm"] == pytest.approx(7.4 + 56.7)


def test_uniform_pause_fits_one_frame() -> None:
    plan = pause_plan(100.0, "Uniform")
    assert plan["frames"] == 1
    assert plan["shutter_ms"] == pytest.approx(100.0)
    assert plan["motion_ms"] == pytest.approx(MOTION_PAUSE_S * 1000)
    assert plan["dwell_ms"] == int(
        MOTION_PAUSE_S * 1000 + SETTLE_S * 1000 + 100 + PAUSE_MARGIN_MS
    )


def test_nsew_pause_fits_four_frames() -> None:
    plan = pause_plan(800.0, "NSEW")
    assert plan["frames"] == 4
    assert plan["shutter_ms"] == pytest.approx(800.0)
    assert plan["gaps"] == 3
    assert plan["gap_ms"] == pytest.approx(NSEW_GAP_S * 1000)
    assert plan["dwell_ms"] == int(
        MOTION_PAUSE_S * 1000
        + 4 * (SETTLE_S * 1000 + 800 + PAUSE_MARGIN_MS)
        + 3 * NSEW_GAP_S * 1000
    )
    assert plan["dwell_ms"] > pause_plan(100.0, "Uniform")["dwell_ms"]


def test_nsew_standard_exposure_is_eight_times_uniform() -> None:
    from src.capture.camera_automation import (
        NSEW_TOTAL_EXPOSURE_MS,
        TOTAL_EXPOSURE_MS,
        standard_total_exposure_ms,
    )

    assert NSEW_TOTAL_EXPOSURE_MS == pytest.approx(8 * TOTAL_EXPOSURE_MS)
    assert standard_total_exposure_ms("Uniform") == pytest.approx(100.0)
    assert standard_total_exposure_ms("NSEW") == pytest.approx(800.0)


def test_rejects_a_zero_step() -> None:
    with pytest.raises(ValueError):
        build_scan_gcode(step_mm=0)


def test_jog_needs_a_printer() -> None:
    with pytest.raises(RuntimeError, match="Connect the printer"):
        move_z(0.1, 300)
    with pytest.raises(RuntimeError, match="Connect the printer"):
        move_axis("X", 1, 1000)


def test_xy_jog_is_one_relative_millimetre() -> None:
    assert build_relative_move_gcode("X", -1, 1000).splitlines() == [
        "G21",
        "G91",
        "G1 X-1.00 F1000",
        "M400",
        "G90",
    ]
    assert build_relative_move_gcode("Y", 1, 1000).splitlines() == [
        "G21",
        "G91",
        "G1 Y1.00 F1000",
        "M400",
        "G90",
    ]


def test_stopped_scan_returns_from_the_current_stop() -> None:
    step = 1.5
    stops = scan_stops(3, 3, step)
    x = 0.0
    y = 0.0
    for row, col, axis, delta in stops:
        if axis == "X":
            x += delta
        elif axis == "Y":
            y += delta
        back = return_from_stop_moves(row, col, step)
        back_x = sum(distance for moved, distance in back if moved == "X")
        back_y = sum(distance for moved, distance in back if moved == "Y")
        assert back_x == pytest.approx(-x)
        assert back_y == pytest.approx(-y)
    assert return_from_stop_moves(1, 1, step) == []


def test_full_scan_returns_to_the_first_stop() -> None:
    assert return_to_start_moves(3, 1, 1.5) == [("X", -3.0)]
    assert return_to_start_moves(3, 2, 1.5) == [("Y", 1.5)]
    assert return_to_start_moves(1, 1, 1.5) == []
    text = build_scan_gcode(step_mm=2.5, columns=2, rows=2, feed=800, dwell_ms=10)
    assert "G1 Y2.50 F800" in text
    assert text.index("G1 Y-2.50 F800") < text.index("; Return to stop (1, 1)")


def test_home_parks_on_the_ender_endstops() -> None:
    assert build_home_gcode().splitlines() == ["G21", "G28 X Y", "M400"]


def test_scan_stops_follow_the_serpentine() -> None:
    stops = scan_stops(3, 2, 1.5)
    assert [(row, col) for row, col, _axis, _delta in stops] == [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 2),
        (2, 1),
    ]
    assert stops[0][2] is None
    assert stops[1][2:] == ("X", 1.5)
    assert stops[3][2:] == ("Y", -1.5)
    assert stops[4][2:] == ("X", -1.5)


@pytest.mark.parametrize(
    "site_parts",
    [
        ("lib", "python3.13", "site-packages"),
        ("Lib", "site-packages"),
    ],
)
def test_pipx_printrun_site_finds_a_venv(
    tmp_path: Path, monkeypatch, site_parts: tuple[str, ...]
) -> None:
    from src.capture import stage

    root = tmp_path / "venvs" / "printrun"
    package = root.joinpath(*site_parts) / "printrun"
    package.mkdir(parents=True)
    (package / "printcore.py").write_text("# test\n", encoding="utf-8")
    monkeypatch.setattr(stage, "_pipx_printrun_roots", lambda: [root])
    found = stage._pipx_printrun_site()
    assert found is not None
    assert (found / "printrun" / "printcore.py").is_file()


def test_missing_printrun_asks_for_an_install_without_deps(monkeypatch) -> None:
    import sys

    from src.capture import stage

    monkeypatch.setattr(stage, "_pipx_printrun_site", lambda: None)
    monkeypatch.setitem(sys.modules, "printrun", None)
    monkeypatch.setitem(sys.modules, "printrun.gcoder", None)
    monkeypatch.setitem(sys.modules, "printrun.printcore", None)
    with pytest.raises(ImportError, match="--no-deps"):
        stage.load_printcore()
