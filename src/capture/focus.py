"""Autofocus by stepping printer Z until the live image is sharpest.

The search is pure: it asks a ``measure`` function for a sharpness score and a
``move`` function for a relative Z step. The Capture tab supplies the camera
and the printer. Nothing here is homed. Offsets are millimetres from the
height where the search started.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_MAX_TRAVEL_MM = 1.0
DEFAULT_COARSE_MM = 0.05
DEFAULT_FINE_MM = 0.01
DEFAULT_MIN_GAIN = 0.02
DEFAULT_SETTLE_S = 0.3
DEFAULT_FRAME_TIMEOUT_S = 12.0

_LIMIT_EPS_MM = 1e-6

_JOB_LOCK = threading.Lock()
_JOB: dict = {
    "thread": None,
    "stop": None,
    "running": False,
    "stopping": False,
    "phase": "idle",
    "offset_mm": 0.0,
    "sharpness": 0.0,
    "best_offset_mm": 0.0,
    "best_sharpness": 0.0,
    "error": "",
    "stopped": False,
    "steps": [],
    "run_id": 0,
    "still_saved": False,
}


@dataclass(frozen=True)
class FocusResult:
    """Where the search finished, relative to the height it started at."""

    offset_mm: float
    sharpness: float
    best_offset_mm: float
    best_sharpness: float
    stopped: bool
    start_sharpness: float


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian on the center of the frame.

    A focused pattern has stronger edges, so the variance is higher. The center
    half is used so the rim and vignetting do not dominate the score.
    """
    frame = np.asarray(image)
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("expected a gray or RGB image")
    height, width = gray.shape[:2]
    y0 = height // 4
    x0 = width // 4
    crop = gray[y0: height - y0, x0: width - x0]
    if crop.size == 0 or min(crop.shape[:2]) < 3:
        return 0.0
    lap = cv2.Laplacian(crop, cv2.CV_64F)
    return float(lap.var())

def _better(score: float, best: float, min_gain: float) -> bool:
    return score > best * (1.0 + min_gain)

def focus_search(
    measure,
    move,
    *,
    max_travel_mm: float = DEFAULT_MAX_TRAVEL_MM,
    coarse_mm: float = DEFAULT_COARSE_MM,
    fine_mm: float = DEFAULT_FINE_MM,
    min_gain: float = DEFAULT_MIN_GAIN,
    should_stop=None,
    on_sample=None,
) -> FocusResult:
    """Hill-climb Z from the current height and finish on the sharpest sample.

    ``measure()`` returns a sharpness score at the current height. ``move(delta_mm)``
    steps Z and returns when that step has finished. Positive delta is printer +Z.
    The first probe is upward. A step counts as sharper only when the score rises by
    ``min_gain``. A miss is scored again on the next frame at the same height. If
    that still misses, one more step in the same direction is taken before the
    search turns around. The same climb then repeats with ``fine_mm``. No step is
    commanded past ``max_travel_mm`` from the start. On the way out, including
    when ``should_stop`` is set, the stage returns to the best height measured.
    """
    max_travel_mm = float(max_travel_mm)
    coarse_mm = float(coarse_mm)
    fine_mm = float(fine_mm)
    min_gain = float(min_gain)
    if max_travel_mm < 0:
        raise ValueError("max travel cannot be negative")
    if coarse_mm <= 0 or fine_mm <= 0:
        raise ValueError("steps must be greater than 0")
    if min_gain < 0:
        raise ValueError("min gain cannot be negative")
    if should_stop is None:
        def should_stop() -> bool:
            return False

    offset = 0.0
    stopped = False

    def within(target: float) -> bool:
        return abs(target) <= max_travel_mm + _LIMIT_EPS_MM

    def go(target: float) -> None:
        nonlocal offset
        delta = target - offset
        if abs(delta) <= _LIMIT_EPS_MM:
            return
        if not within(target):
            raise RuntimeError("Z move would pass the travel limit")
        move(delta)
        offset = target

    def halt() -> bool:
        nonlocal stopped
        if should_stop():
            stopped = True
            return True
        return False

    def report(
        score: float,
        best_offset: float,
        best_score: float,
        *,
        phase: str,
        decision: str,
    ) -> None:
        if on_sample is not None:
            on_sample(
                offset,
                score,
                best_offset,
                best_score,
                phase=phase,
                decision=decision,
            )

    score = float(measure())
    start_score = score
    best_offset = 0.0
    best_score = score
    report(score, best_offset, best_score, phase="start", decision="start")

    def sample(phase: str) -> bool | None:
        """Score this height. A miss gets a second frame before it counts."""
        nonlocal score, best_offset, best_score
        score = float(measure())
        if _better(score, best_score, min_gain):
            best_score = score
            best_offset = offset
            report(score, best_offset, best_score, phase=phase, decision="sharper")
            return True
        report(score, best_offset, best_score, phase=phase, decision="not sharper")
        if halt():
            return None
        score = float(measure())
        if _better(score, best_score, min_gain):
            best_score = score
            best_offset = offset
            report(score, best_offset, best_score, phase=phase, decision="sharper")
            return True
        report(score, best_offset, best_score, phase=phase, decision="again")
        return False

    def retreat(phase: str) -> None:
        nonlocal score
        go(best_offset)
        score = best_score
        report(score, best_offset, best_score, phase=phase, decision="back")

    def climb(step: float, phase: str) -> None:
        for direction in (1.0, -1.0):
            if halt():
                return
            improved = False
            extra_used = False
            while True:
                if halt():
                    return
                target = offset + direction * step
                if not within(target):
                    if extra_used:
                        retreat(phase)
                        if improved:
                            return
                        break
                    if improved:
                        return
                    break
                go(target)
                found = sample(phase)
                if found is None:
                    return
                if found:
                    improved = True
                    extra_used = False
                    continue
                if not extra_used:
                    extra_used = True
                    continue
                retreat(phase)
                if improved:
                    return
                break

    climb(coarse_mm, "coarse")
    if not stopped:
        climb(fine_mm, "fine")
    if abs(offset - best_offset) > _LIMIT_EPS_MM:
        go(best_offset)
        report(best_score, best_offset, best_score, phase="return", decision="back")
    return FocusResult(
        offset_mm=best_offset,
        sharpness=best_score,
        best_offset_mm=best_offset,
        best_sharpness=best_score,
        stopped=stopped,
        start_sharpness=start_score,
    )


def score_settled_frame(
    settle_s: float = DEFAULT_SETTLE_S,
    timeout_s: float = DEFAULT_FRAME_TIMEOUT_S,
) -> float:
    """Wait out vibration, then score the next live frame after that."""
    from src.capture.preview import preview_snapshot

    settle_s = float(settle_s)
    timeout_s = float(timeout_s)
    if settle_s > 0:
        time.sleep(settle_s)
    before = preview_snapshot()["seq"]
    deadline = time.monotonic() + timeout_s
    while True:
        snap = preview_snapshot()
        frame = snap["frame"]
        if frame is not None and snap["seq"] > before and not snap["error"]:
            return sharpness(frame)
        if snap["error"] and frame is None:
            raise RuntimeError(snap["error"])
        if time.monotonic() >= deadline:
            if snap["error"]:
                raise RuntimeError(snap["error"])
            raise TimeoutError("No new camera frame")
        time.sleep(0.05)


def autofocus_status() -> dict:
    """Snapshot of the background search. Safe to call from the page rerun."""
    with _JOB_LOCK:
        return {
            "running": bool(_JOB["running"]),
            "stopping": bool(_JOB["stopping"]),
            "phase": _JOB["phase"],
            "offset_mm": float(_JOB["offset_mm"]),
            "sharpness": float(_JOB["sharpness"]),
            "best_offset_mm": float(_JOB["best_offset_mm"]),
            "best_sharpness": float(_JOB["best_sharpness"]),
            "error": _JOB["error"],
            "stopped": bool(_JOB["stopped"]),
            "steps": [dict(step) for step in _JOB["steps"]],
            "run_id": int(_JOB["run_id"]),
        }


def claim_focus_capture() -> bool:
    """True once, after a finished autofocus, so the page can save one still."""
    with _JOB_LOCK:
        if _JOB["running"] or _JOB["phase"] != "done" or _JOB["error"]:
            return False
        if _JOB["still_saved"]:
            return False
        _JOB["still_saved"] = True
        return True


def stop_autofocus() -> None:
    """Ask the search to finish the current move, then return to the best Z."""
    with _JOB_LOCK:
        stop = _JOB.get("stop")
        if stop is not None:
            stop.set()
            _JOB["stopping"] = True


def _check_focus_settings(
    max_travel_mm: float,
    coarse_mm: float,
    fine_mm: float,
    feed: float,
) -> None:
    if max_travel_mm <= 0:
        raise ValueError("max travel must be greater than 0")
    if coarse_mm <= 0 or fine_mm <= 0:
        raise ValueError("steps must be greater than 0")
    if coarse_mm > max_travel_mm + _LIMIT_EPS_MM or fine_mm > max_travel_mm + _LIMIT_EPS_MM:
        raise ValueError("steps must fit inside the max travel")
    if feed <= 0:
        raise ValueError("feed must be greater than 0")


def start_autofocus(
    *,
    max_travel_mm: float = DEFAULT_MAX_TRAVEL_MM,
    coarse_mm: float = DEFAULT_COARSE_MM,
    fine_mm: float = DEFAULT_FINE_MM,
    feed: float,
    settle_s: float = DEFAULT_SETTLE_S,
    min_gain: float = DEFAULT_MIN_GAIN,
    frame_timeout_s: float = DEFAULT_FRAME_TIMEOUT_S,
) -> None:
    """Start the hill climb on a background thread. Returns once the thread is running."""
    from src.capture.preview import preview_snapshot
    from src.capture.stage import printer_status

    max_travel_mm = float(max_travel_mm)
    coarse_mm = float(coarse_mm)
    fine_mm = float(fine_mm)
    feed = float(feed)
    _check_focus_settings(max_travel_mm, coarse_mm, fine_mm, feed)

    with _JOB_LOCK:
        thread = _JOB.get("thread")
        if thread is not None and thread.is_alive():
            raise RuntimeError("Autofocus is already running.")

    status = printer_status()
    if not status["connected"]:
        raise RuntimeError("Connect the printer first.")
    if not status["online"]:
        raise RuntimeError("The printer has not answered yet. Wait a moment and try again.")
    if status["printing"]:
        raise RuntimeError("A scan is already running.")
    snap = preview_snapshot()
    if snap["frame"] is None:
        raise RuntimeError(snap["error"] or "The live view has no frame yet.")

    stop = threading.Event()
    worker = threading.Thread(
        target=_autofocus_thread,
        kwargs={
            "max_travel_mm": max_travel_mm,
            "coarse_mm": coarse_mm,
            "fine_mm": fine_mm,
            "feed": feed,
            "settle_s": float(settle_s),
            "min_gain": float(min_gain),
            "frame_timeout_s": float(frame_timeout_s),
            "stop": stop,
        },
        name="autofocus",
        daemon=True,
    )
    with _JOB_LOCK:
        _JOB["thread"] = worker
        _JOB["stop"] = stop
        _JOB["running"] = True
        _JOB["stopping"] = False
        _JOB["phase"] = "running"
        _JOB["offset_mm"] = 0.0
        _JOB["sharpness"] = 0.0
        _JOB["best_offset_mm"] = 0.0
        _JOB["best_sharpness"] = 0.0
        _JOB["error"] = ""
        _JOB["stopped"] = False
        _JOB["steps"] = []
        _JOB["run_id"] = int(_JOB["run_id"]) + 1
        _JOB["still_saved"] = False
    worker.start()


def _autofocus_thread(
    *,
    max_travel_mm: float,
    coarse_mm: float,
    fine_mm: float,
    feed: float,
    settle_s: float,
    min_gain: float,
    frame_timeout_s: float,
    stop: threading.Event,
) -> None:
    from src.capture.stage import move_z

    def measure() -> float:
        return score_settled_frame(settle_s, frame_timeout_s)

    def move(delta_mm: float) -> None:
        move_z(delta_mm, feed)

    def on_sample(
        offset_mm: float,
        score: float,
        best_offset_mm: float,
        best_score: float,
        *,
        phase: str,
        decision: str,
    ) -> None:
        with _JOB_LOCK:
            _JOB["offset_mm"] = offset_mm
            _JOB["sharpness"] = score
            _JOB["best_offset_mm"] = best_offset_mm
            _JOB["best_sharpness"] = best_score
            _JOB["steps"].append(
                {
                    "phase": phase,
                    "offset_mm": float(offset_mm),
                    "sharpness": float(score),
                    "decision": decision,
                }
            )

    try:
        result = focus_search(
            measure,
            move,
            max_travel_mm=max_travel_mm,
            coarse_mm=coarse_mm,
            fine_mm=fine_mm,
            min_gain=min_gain,
            should_stop=stop.is_set,
            on_sample=on_sample,
        )
    except Exception as exc:
        with _JOB_LOCK:
            _JOB["phase"] = "error"
            _JOB["error"] = str(exc)
            _JOB["running"] = False
            _JOB["stopping"] = False
        return
    with _JOB_LOCK:
        _JOB["phase"] = "done"
        _JOB["stopped"] = result.stopped
        _JOB["offset_mm"] = result.offset_mm
        _JOB["sharpness"] = result.sharpness
        _JOB["best_offset_mm"] = result.best_offset_mm
        _JOB["best_sharpness"] = result.best_sharpness
        _JOB["error"] = ""
        _JOB["running"] = False
        _JOB["stopping"] = False
