import json

import cv2
import numpy as np

from src.capture.focus import claim_focus_capture
from src.capture.scan_capture import (
    FOCUS_EVERY_STEPS,
    FOCUS_MIN_RATIO,
    autofocus_log,
    needs_another_focus,
    refocus_due,
    score_saved_frame,
    sharpness_stop,
    write_sharpness_log,
)
from src.capture.stage import scan_stops


def test_refocus_lands_on_every_fifth_step() -> None:
    assert FOCUS_EVERY_STEPS == 5
    assert [step for step in range(0, 16) if refocus_due(step)] == [5, 10, 15]
    assert [step for step in range(0, 16) if refocus_due(step, enabled=False)] == []


def test_serpentine_refocuses_after_five_moves() -> None:
    stops = scan_stops(12, 1, 1.0)
    taken = 0
    focused = []
    for _row, _col, axis, _delta in stops:
        if axis is None:
            continue
        taken += 1
        if refocus_due(taken):
            focused.append(taken)
    assert focused == [5, 10]


def test_sharpness_log_keeps_each_stop_and_the_autofocus_move(tmp_path) -> None:
    rng = np.random.default_rng(2)
    sharp = rng.integers(0, 256, (80, 80), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
    sharp_path = tmp_path / "(1,1).png"
    blur_path = tmp_path / "(1,2)_N.png"
    cv2.imwrite(str(sharp_path), sharp)
    cv2.imwrite(str(blur_path), blurred)
    assert score_saved_frame(sharp_path) > score_saved_frame(blur_path)

    focus = autofocus_log(
        [
            {
                "attempt": 1,
                "z_change_mm": 0.05,
                "sharpness": 80.0,
                "best_sharpness": 80.0,
                "start_sharpness": 100.0,
                "stopped": False,
            },
            {
                "attempt": 2,
                "z_change_mm": -0.02,
                "sharpness": 140.0,
                "best_sharpness": 140.0,
                "start_sharpness": 80.0,
                "stopped": False,
            },
        ],
        after_step=7,
    )
    assert focus["after_step"] == 7
    assert abs(focus["z_change_mm"] - 0.03) < 1e-9
    assert focus["attempts"] == 2
    assert focus["start_sharpness"] == 100.0
    assert focus["best_sharpness"] == 140.0

    stops = [
        sharpness_stop(1, 1, [{"name": "(1,1).png", "sharpness": 10.0}]),
        sharpness_stop(
            1,
            2,
            [
                {"name": "(1,2)_N.png", "sharpness": 20.0},
                {"name": "(1,2)_E.png", "sharpness": 40.0},
            ],
            focus,
        ),
    ]
    assert stops[0]["sharpness"] == 10.0
    assert "autofocus" not in stops[0]
    assert stops[1]["sharpness"] == 30.0
    assert abs(stops[1]["autofocus"]["z_change_mm"] - 0.03) < 1e-9
    write_sharpness_log(tmp_path, stops)
    saved = json.loads((tmp_path / "sharpness.json").read_text())
    assert saved["stops"] == stops


def test_idle_autofocus_does_not_claim_a_capture() -> None:
    assert claim_focus_capture() is False


def test_soft_focus_runs_again() -> None:
    assert FOCUS_MIN_RATIO == 0.8
    assert needs_another_focus(800, starting=1000)
    assert needs_another_focus(0, starting=1000)
    assert not needs_another_focus(800.1, starting=1000)
    assert not needs_another_focus(1200.4, starting=1000)
    assert not needs_another_focus(400, starting=400)
    assert not needs_another_focus(100, stopped=True, starting=1000)
