
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from src.capture.camera_automation import (
    CAPTURES_DIR,
    MOTION_PAUSE_S,
    RING_SIDES,
    allocate_run_folder,
    capture_nsew,
    capture_uniform,
    grab_frame,
    grab_timeout_ms,
    open_camera,
    shutter_ms,
    write_manifest,
)
from src.capture.focus import (
    DEFAULT_COARSE_MM,
    DEFAULT_FINE_MM,
    DEFAULT_MAX_TRAVEL_MM,
    DEFAULT_MIN_GAIN,
    DEFAULT_SETTLE_S,
    _check_focus_settings,
    focus_search,
    sharpness,
)
from src.capture.stage import move_axis, move_z, printer_status, return_from_stop_moves, scan_stops

FOCUS_EVERY_STEPS = 7
FOCUS_MIN_RATIO = 0.8

_JOB_LOCK = threading.Lock()
_JOB: dict = {
    "thread": None,
    "stop": None,
    "running": False,
    "stopping": False,
    "index": 0,
    "total": 0,
    "row": 0,
    "col": 0,
    "saved": 0,
    "error": "",
    "mode": "",
    "folder": "",
    "steps": 0,
    "focusing": False,
    "sharpness": 0.0,
    "best_sharpness": 0.0,
    "focus_attempt": 0,
    "returning": False,
}


def refocus_due(
    steps_taken: int,
    every: int = FOCUS_EVERY_STEPS,
    *,
    enabled: bool = True,
) -> bool:
    """True after the 7th, 14th, and later stage steps when focus during the scan is on."""
    if not enabled:
        return False
    steps_taken = int(steps_taken)
    every = int(every)
    return every > 0 and steps_taken > 0 and steps_taken % every == 0


def needs_another_focus(
    score: float,
    stopped: bool = False,
    starting: float = 0.0,
    ratio: float = FOCUS_MIN_RATIO,
) -> bool:
    """Run autofocus again when the search is not above 80% of its start."""
    if stopped:
        return False
    return float(score) <= float(ratio) * float(starting)


def score_saved_frame(path: Path) -> float:
    """Sharpness of a frame already written by the scan."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read {path} to score sharpness.")
    if getattr(image, "ndim", 0) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif getattr(image, "ndim", 0) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return sharpness(image)


def sharpness_stop(
    row: int,
    col: int,
    files: list[dict],
    autofocus: dict | None = None,
) -> dict:
    """One stop: the sharpness of each saved frame, and autofocus when it ran there."""
    scores = [float(item["sharpness"]) for item in files]
    record = {
        "row": int(row),
        "col": int(col),
        "sharpness": float(sum(scores) / len(scores)) if scores else 0.0,
        "files": [
            {"name": str(item["name"]), "sharpness": float(item["sharpness"])}
            for item in files
        ],
    }
    if autofocus is not None:
        record["autofocus"] = autofocus
    return record


def autofocus_log(searches: list[dict], *, after_step: int | None = None) -> dict:
    """Net Z change from the height where this scan autofocus began, and each attempt."""
    net = sum(float(search["z_change_mm"]) for search in searches)
    first = searches[0]
    last = searches[-1]
    record: dict = {}
    if after_step is not None:
        record["after_step"] = int(after_step)
    record.update(
        {
            "z_change_mm": net,
            "attempts": len(searches),
            "stopped": bool(last["stopped"]),
            "start_sharpness": float(first["start_sharpness"]),
            "sharpness": float(last["sharpness"]),
            "best_sharpness": float(last["best_sharpness"]),
            "searches": list(searches),
        }
    )
    return record


def write_sharpness_log(folder: Path, stops: list[dict]) -> None:
    """Rewrite ``sharpness.json`` so a stopped scan still keeps the stops already taken."""
    path = Path(folder) / "sharpness.json"
    path.write_text(json.dumps({"stops": stops}, indent=2) + "\n")


def scan_capture_status() -> dict:
    with _JOB_LOCK:
        thread = _JOB.get("thread")
        if _JOB["running"] and thread is not None and not thread.is_alive():
            _JOB["running"] = False
            _JOB["stopping"] = False
            if not _JOB["error"]:
                _JOB["error"] = "The scan stopped before the first picture."
        return {
            "running": bool(_JOB["running"]),
            "stopping": bool(_JOB["stopping"]),
            "index": int(_JOB["index"]),
            "total": int(_JOB["total"]),
            "row": int(_JOB["row"]),
            "col": int(_JOB["col"]),
            "saved": int(_JOB["saved"]),
            "error": _JOB["error"],
            "mode": _JOB["mode"],
            "folder": _JOB["folder"],
            "steps": int(_JOB["steps"]),
            "focusing": bool(_JOB["focusing"]),
            "sharpness": float(_JOB["sharpness"]),
            "best_sharpness": float(_JOB["best_sharpness"]),
            "focus_attempt": int(_JOB["focus_attempt"]),
            "returning": bool(_JOB["returning"]),
        }


def stop_scan_capture() -> None:
    """Finish the stop in progress, then go back to (1, 1)."""
    with _JOB_LOCK:
        stop = _JOB.get("stop")
        if stop is not None:
            stop.set()
            _JOB["stopping"] = True


def start_scan_capture(
    *,
    ser,
    mode: str,
    total_exposure_ms: float,
    step_mm: float,
    columns: int,
    rows: int,
    feed: float,
    captures_dir: Path = CAPTURES_DIR,
    focus_max_travel_mm: float = DEFAULT_MAX_TRAVEL_MM,
    focus_coarse_mm: float = DEFAULT_COARSE_MM,
    focus_fine_mm: float = DEFAULT_FINE_MM,
    focus_feed: float = 300,
    focus_during_scan: bool = True,
    stage: dict | None = None,
) -> None:
    """Start the combined scan on a background thread."""
    if mode not in ("Uniform", "NSEW"):
        raise ValueError("mode must be Uniform or NSEW")
    ring_open = ser is not None and getattr(ser, "is_open", False)
    if mode == "NSEW" and not ring_open:
        raise RuntimeError("NSEW capture needs the Arduino to switch sides.")
    if ser is not None and not ring_open:
        raise RuntimeError("The Arduino port is closed.")
    status = printer_status()
    if not status["connected"]:
        raise RuntimeError("Connect the printer first.")
    if not status["online"]:
        raise RuntimeError("The printer has not answered yet. Wait a moment and try again.")
    if status["printing"]:
        raise RuntimeError("A scan is already running.")
    step_mm = round(float(step_mm), 2)
    stops = scan_stops(columns, rows, step_mm)
    focus_max_travel_mm = float(focus_max_travel_mm)
    focus_coarse_mm = float(focus_coarse_mm)
    focus_fine_mm = float(focus_fine_mm)
    focus_feed = float(focus_feed)
    focus_during_scan = bool(focus_during_scan)
    if focus_during_scan:
        _check_focus_settings(focus_max_travel_mm, focus_coarse_mm, focus_fine_mm, focus_feed)
    with _JOB_LOCK:
        thread = _JOB.get("thread")
        if thread is not None and thread.is_alive():
            raise RuntimeError("A capture scan is already running.")
        stop = threading.Event()
        _JOB["stop"] = stop
        _JOB["running"] = True
        _JOB["stopping"] = False
        _JOB["index"] = 0
        _JOB["total"] = len(stops)
        _JOB["row"] = 0
        _JOB["col"] = 0
        _JOB["saved"] = 0
        _JOB["error"] = ""
        _JOB["mode"] = mode
        _JOB["folder"] = ""
        _JOB["steps"] = 0
        _JOB["focusing"] = False
        _JOB["sharpness"] = 0.0
        _JOB["best_sharpness"] = 0.0
        _JOB["focus_attempt"] = 0
        _JOB["returning"] = False
    worker = threading.Thread(
        target=_scan_thread,
        kwargs={
            "ser": ser,
            "mode": mode,
            "total_exposure_ms": float(total_exposure_ms),
            "feed": float(feed),
            "step_mm": step_mm,
            "stops": stops,
            "captures_dir": Path(captures_dir),
            "stop": stop,
            "focus_max_travel_mm": focus_max_travel_mm,
            "focus_coarse_mm": focus_coarse_mm,
            "focus_fine_mm": focus_fine_mm,
            "focus_feed": focus_feed,
            "focus_during_scan": focus_during_scan,
            "stage": None if stage is None else dict(stage),
        },
        name="scan-capture",
        daemon=True,
    )
    with _JOB_LOCK:
        _JOB["thread"] = worker
    worker.start()


def _return_to_first_stop(
    row: int,
    col: int,
    step_mm: float,
    feed: float,
    stop: threading.Event,
) -> None:
    """Move from the stop just visited back to (1, 1).

    The click that ended the scan is cleared first, so that stop still goes
    home. A second click during this trip aborts it.
    """
    moves = return_from_stop_moves(row, col, step_mm)
    if not moves:
        return
    stop.clear()
    with _JOB_LOCK:
        _JOB["stopping"] = False
        _JOB["returning"] = True
    for axis, delta in moves:
        if stop.is_set():
            break
        move_axis(axis, delta, feed)


def _scan_thread(
    *,
    ser,
    mode: str,
    total_exposure_ms: float,
    feed: float,
    step_mm: float,
    stops: list[tuple[int, int, str | None, float]],
    captures_dir: Path,
    stop: threading.Event,
    focus_max_travel_mm: float,
    focus_coarse_mm: float,
    focus_fine_mm: float,
    focus_feed: float,
    focus_during_scan: bool,
    stage: dict | None,
) -> None:
    from src.capture.camera_automation import shutdown
    from src.capture.preview import stop_preview

    camera = None
    saved = 0
    steps_taken = 0
    reached: tuple[int, int] | None = None
    sharpness_stops: list[dict] = []
    try:
        sides_on = RING_SIDES if mode == "Uniform" else 1
        shutter = shutter_ms(total_exposure_ms)
        exposure_us = shutter * 1000.0
        timeout_ms = grab_timeout_ms(exposure_us)
        stop_preview()
        camera, converter = open_camera(exposure_us)
        run_dir = allocate_run_folder(captures_dir)
        with _JOB_LOCK:
            _JOB["folder"] = str(run_dir)
        for index, (row, col, axis, delta) in enumerate(stops, start=1):
            if stop.is_set():
                break
            with _JOB_LOCK:
                _JOB["index"] = index
                _JOB["row"] = row
                _JOB["col"] = col
            focus_record = None
            if axis is not None:
                move_axis(axis, delta, feed)
                time.sleep(MOTION_PAUSE_S)
                steps_taken += 1
                with _JOB_LOCK:
                    _JOB["steps"] = steps_taken
                if refocus_due(steps_taken, enabled=focus_during_scan):
                    focus_record = _focus_here(
                        camera,
                        converter,
                        timeout_ms,
                        stop,
                        max_travel_mm=focus_max_travel_mm,
                        coarse_mm=focus_coarse_mm,
                        fine_mm=focus_fine_mm,
                        feed=focus_feed,
                        after_step=steps_taken,
                    )
            reached = (row, col)
            if mode == "Uniform":
                files = [capture_uniform(camera, converter, ser, run_dir, timeout_ms, row, col)]
            else:
                files = capture_nsew(camera, converter, ser, run_dir, timeout_ms, row, col)
            scored = [
                {"name": name, "sharpness": score_saved_frame(run_dir / name)}
                for name in files
            ]
            sharpness_stops.append(sharpness_stop(row, col, scored, focus_record))
            write_sharpness_log(run_dir, sharpness_stops)
            manifest = {
                "mode": "uniform" if mode == "Uniform" else "nsew",
                "row": row,
                "col": col,
                "total_exposure_ms": total_exposure_ms,
                "shutter_ms": shutter,
                "exposure_us": exposure_us,
                "sides_on": sides_on,
                "files": files,
            }
            if stage is not None:
                manifest["stage"] = stage
            write_manifest(run_dir, manifest)
            saved += 1
            with _JOB_LOCK:
                _JOB["saved"] = saved
        if reached is not None:
            _return_to_first_stop(reached[0], reached[1], step_mm, feed, stop)
    except Exception as exc:
        with _JOB_LOCK:
            _JOB["error"] = str(exc)
    finally:
        try:
            shutdown(camera, None)
        except Exception as exc:
            with _JOB_LOCK:
                if not _JOB["error"]:
                    _JOB["error"] = str(exc)
        with _JOB_LOCK:
            _JOB["running"] = False
            _JOB["stopping"] = False
            _JOB["focusing"] = False
            _JOB["returning"] = False
            _JOB["saved"] = saved


def _score_triggered_frame(camera, converter, timeout_ms: int, settle_s: float) -> float:
    """Wait out the Z step, then score one new software-triggered frame."""
    import cv2
    from src.capture.camera_automation import pylon_module

    pylon = pylon_module()

    if settle_s > 0:
        time.sleep(settle_s)
    if not camera.IsGrabbing():
        camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    image = grab_frame(camera, converter, timeout_ms)
    if getattr(image, "ndim", 0) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return sharpness(image)


def _focus_here(
    camera,
    converter,
    timeout_ms: int,
    stop: threading.Event,
    *,
    max_travel_mm: float,
    coarse_mm: float,
    fine_mm: float,
    feed: float,
    after_step: int,
) -> dict | None:
    """Hill-climb Z, and start again while sharpness is not above 80% of the opening score."""

    def measure() -> float:
        return _score_triggered_frame(camera, converter, timeout_ms, DEFAULT_SETTLE_S)

    def move(delta_mm: float) -> None:
        move_z(delta_mm, feed)

    def on_sample(
        _offset_mm: float,
        score: float,
        _best_offset_mm: float,
        best_score: float,
        *,
        phase: str,
        decision: str,
    ) -> None:
        del phase, decision
        with _JOB_LOCK:
            _JOB["sharpness"] = float(score)
            _JOB["best_sharpness"] = float(best_score)

    attempt = 0
    opening = None
    searches: list[dict] = []
    while True:
        if stop.is_set():
            break
        attempt += 1
        with _JOB_LOCK:
            _JOB["focusing"] = True
            _JOB["focus_attempt"] = attempt
            _JOB["sharpness"] = 0.0
            _JOB["best_sharpness"] = 0.0
        try:
            result = focus_search(
                measure,
                move,
                max_travel_mm=max_travel_mm,
                coarse_mm=coarse_mm,
                fine_mm=fine_mm,
                min_gain=DEFAULT_MIN_GAIN,
                should_stop=stop.is_set,
                on_sample=on_sample,
            )
        finally:
            with _JOB_LOCK:
                _JOB["focusing"] = False
        if opening is None:
            opening = float(result.start_sharpness)
        searches.append(
            {
                "attempt": attempt,
                "z_change_mm": float(result.best_offset_mm),
                "sharpness": float(result.sharpness),
                "best_sharpness": float(result.best_sharpness),
                "start_sharpness": float(result.start_sharpness),
                "stopped": bool(result.stopped),
            }
        )
        with _JOB_LOCK:
            _JOB["sharpness"] = float(result.sharpness)
            _JOB["best_sharpness"] = float(result.best_sharpness)
        if not needs_another_focus(result.best_sharpness, result.stopped, opening):
            break
    if not searches:
        return None
    return autofocus_log(searches, after_step=after_step)
