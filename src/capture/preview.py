"""Live Basler view for the Capture tab.

The camera stays open in a background thread. A real capture has to close it
first, because the Basler device allows one connection.
"""

from __future__ import annotations

import builtins
import threading

import cv2
import numpy as np

PREVIEW_MAX_SIDE = 960


def _load_state() -> dict:
    """One camera record for this process, including across a script reload.

    A reload otherwise forgets the open Basler and tries to open it again.
    The camera allows one client, so that second open fails and the live
    view is replaced by the exclusive-device error.
    """
    shared = getattr(builtins, "_asml_camera_preview", None)
    if isinstance(shared, dict) and "lock" in shared:
        return shared
    shared = {
        "camera": None,
        "converter": None,
        "thread": None,
        "stop": None,
        "lock": threading.Lock(),
        "frame": None,
        "error": "",
        "exposure_us": None,
        "applied_us": None,
        "running": False,
        "seq": 0,
    }
    builtins._asml_camera_preview = shared
    return shared


_STATE = _load_state()


def display_frame(image: np.ndarray, max_side: int = PREVIEW_MAX_SIDE) -> np.ndarray:
    """RGB image sized for the page. Accepts a BGR or gray frame from the camera."""
    frame = image
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest > max_side and longest > 0:
        scale = max_side / longest
        frame = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(frame)


def preview_snapshot() -> dict:
    with _STATE["lock"]:
        frame = _STATE["frame"]
        return {
            "running": bool(_STATE["running"]),
            "error": _STATE["error"],
            "frame": None if frame is None else frame.copy(),
            "seq": int(_STATE["seq"]),
        }


def _grab_loop(cap, camera, converter, stop: threading.Event) -> None:
    try:
        while not stop.is_set():
            with _STATE["lock"]:
                wanted = _STATE["exposure_us"]
                applied = _STATE["applied_us"]
            if wanted is not None and wanted != applied:
                cap.set_exposure(camera, wanted)
                with _STATE["lock"]:
                    _STATE["applied_us"] = wanted
            timeout_ms = 2000 if wanted is None else max(500, int(wanted / 1000.0) + 1000)
            raw = cap.read_live_frame(camera, converter, timeout_ms)
            frame = display_frame(raw)
            with _STATE["lock"]:
                _STATE["frame"] = frame
                _STATE["error"] = ""
                _STATE["seq"] = int(_STATE["seq"]) + 1
            if stop.wait(0.1):
                break
    except Exception as exc:
        if not stop.is_set():
            with _STATE["lock"]:
                _STATE["error"] = str(exc)
    finally:
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        except Exception:
            pass
        try:
            camera.Close()
        except Exception:
            pass
        with _STATE["lock"]:
            if _STATE["camera"] is camera:
                _STATE["camera"] = None
                _STATE["converter"] = None
                _STATE["running"] = False
                _STATE["applied_us"] = None


def start_preview(cap, exposure_us: float) -> None:
    """Open the live view, or change the shutter if it is already open."""
    exposure_us = float(exposure_us)
    with _STATE["lock"]:
        _STATE["exposure_us"] = exposure_us
        running = bool(_STATE["running"])
        thread = _STATE["thread"]
    if running or (thread is not None and thread.is_alive()):
        return
    try:
        camera, converter = cap.open_live_camera(exposure_us)
    except Exception as exc:
        with _STATE["lock"]:
            _STATE["error"] = str(exc)
            _STATE["running"] = False
        return
    stop = threading.Event()
    thread = threading.Thread(
        target=_grab_loop,
        args=(cap, camera, converter, stop),
        name="camera-preview",
        daemon=True,
    )
    with _STATE["lock"]:
        _STATE["camera"] = camera
        _STATE["converter"] = converter
        _STATE["stop"] = stop
        _STATE["thread"] = thread
        _STATE["running"] = True
        _STATE["applied_us"] = exposure_us
        _STATE["error"] = ""
    thread.start()


def stop_preview() -> None:
    """Close the live view so a capture can open the camera."""
    with _STATE["lock"]:
        stop = _STATE["stop"]
        camera = _STATE["camera"]
        thread = _STATE["thread"]
        _STATE["running"] = False
    if stop is not None:
        stop.set()
    if camera is not None:
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        except Exception:
            pass
    if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
        thread.join(timeout=5)
