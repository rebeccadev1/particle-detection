from __future__ import annotations

import builtins
import math
import os
import sys
import time
import warnings
from pathlib import Path

from src.capture.camera_automation import (
    MOTION_PAUSE_S,
    NSEW_GAP_S,
    RING_SIDES,
    SETTLE_S,
    shutter_ms,
)

def _load_printer_link() -> dict:
    
    shared = getattr(builtins, "_asml_printer_link", None)
    if isinstance(shared, dict) and "core" in shared:
        return shared
    shared = {"core": None, "port": None, "baud": None, "error": ""}
    builtins._asml_printer_link = shared
    return shared


_LINK = _load_printer_link()

FOV_MM = 7.4
OVERLAP = 0.10
SCAN_WIDTH_MM = 60.0
SCAN_HEIGHT_MM = 40.0
DEFAULT_FEED = 1000
DEFAULT_DWELL_MS = 1000
DEFAULT_BAUD = 115200
HOME_X_MM = -13.0
HOME_Y_MM = -15.0
PAUSE_MARGIN_MS = 500


def step_from_fov(fov_mm: float, overlap: float = OVERLAP) -> float:
    
    fov_mm = float(fov_mm)
    overlap = float(overlap)
    if fov_mm <= 0:
        raise ValueError("field of view must be greater than 0")
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be from 0 up to, but not including, 1")
    step_mm = round(fov_mm * (1.0 - overlap), 2)
    if step_mm <= 0:
        raise ValueError("step must be greater than 0")
    return step_mm


DEFAULT_STEP_MM = step_from_fov(FOV_MM, OVERLAP)


def grid_stops(span_mm: float, fov_mm: float = FOV_MM, step_mm: float = DEFAULT_STEP_MM) -> int:
    
    span_mm = float(span_mm)
    fov_mm = float(fov_mm)
    step_mm = float(step_mm)
    if step_mm <= 0:
        raise ValueError("step must be greater than 0")
    if span_mm <= fov_mm:
        return 1
    moves = math.ceil((span_mm - fov_mm) / step_mm - 1e-9)
    return int(moves) + 1


DEFAULT_COLUMNS = grid_stops(SCAN_WIDTH_MM)
DEFAULT_ROWS = grid_stops(SCAN_HEIGHT_MM)


def _num(value: float) -> str:
    rounded = round(float(value), 4)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.4f}".rstrip("0").rstrip(".")


def _mm(value: float) -> str:
    
    return f"{round(float(value), 2):.2f}"


def build_scan_gcode(
    step_mm: float = DEFAULT_STEP_MM,
    columns: int = DEFAULT_COLUMNS,
    rows: int = DEFAULT_ROWS,
    feed: float = DEFAULT_FEED,
    dwell_ms: float = DEFAULT_DWELL_MS,
) -> str:
    
    step_mm = round(float(step_mm), 2)
    columns = int(columns)
    rows = int(rows)
    feed = float(feed)
    dwell_ms = int(round(float(dwell_ms)))
    if step_mm <= 0:
        raise ValueError("step must be greater than 0")
    if columns < 1 or rows < 1:
        raise ValueError("the scan needs at least one column and one row")
    if feed <= 0:
        raise ValueError("feed must be greater than 0")
    if dwell_ms < 0:
        raise ValueError("pause cannot be negative")

    step = _mm(step_mm)
    feed_text = _num(feed)
    lines = ["G21", "G91", "", "; Start at top-left position"]

    def pause() -> None:
        lines.append(f"G4 P{dwell_ms}")

    def move(token: str) -> None:
        pause()
        lines.append(f"G1 {token} F{feed_text}")

    for row in range(rows):
        sign = "" if row % 2 == 0 else "-"
        for _ in range(columns - 1):
            move(f"X{sign}{step}")
        if row < rows - 1:
            move(f"Y-{step}")
    pause()
    back = return_to_start_moves(columns, rows, step_mm)
    if back:
        lines.append("; Return to stop (1, 1)")
        for axis, delta in back:
            lines.append(f"G1 {axis}{_mm(delta)} F{feed_text}")
    lines.extend(["", "G90", ""])
    return "\n".join(lines)


def pause_plan(total_exposure_ms: float, mode: str) -> dict:
    
    uniform = mode != "NSEW"
    frames = 1 if uniform else RING_SIDES
    gaps = 0 if uniform else frames - 1
    shutter = shutter_ms(total_exposure_ms)
    settle_ms = SETTLE_S * 1000.0
    gap_ms = 0.0 if uniform else NSEW_GAP_S * 1000.0
    motion_ms = MOTION_PAUSE_S * 1000.0
    per_frame = settle_ms + shutter + PAUSE_MARGIN_MS
    return {
        "dwell_ms": int(math.ceil(motion_ms + frames * per_frame + gaps * gap_ms)),
        "frames": frames,
        "gaps": gaps,
        "gap_ms": gap_ms,
        "motion_ms": motion_ms,
        "shutter_ms": shutter,
        "settle_ms": settle_ms,
        "margin_ms": PAUSE_MARGIN_MS,
        "mode": "Uniform" if uniform else "NSEW",
    }


def build_relative_move_gcode(axis: str, delta_mm: float, feed: float) -> str:
    
    axis = str(axis).upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError("axis must be X, Y, or Z")
    delta_mm = round(float(delta_mm), 2)
    feed = float(feed)
    if not math.isfinite(delta_mm):
        raise ValueError(f"{axis} move must be a finite distance")
    if feed <= 0:
        raise ValueError("feed must be greater than 0")
    return "\n".join(
        [
            "G21",
            "G91",
            f"G1 {axis}{_mm(delta_mm)} F{_num(feed)}",
            "M400",
            "G90",
            "",
        ]
    )


def _require_printer_ready() -> None:
    status = printer_status()
    if not status["connected"]:
        raise RuntimeError("Connect the printer first.")
    if not status["online"]:
        raise RuntimeError("The printer has not answered yet. Wait a moment and try again.")
    if status["printing"]:
        raise RuntimeError("A scan is already running.")


def _wait_until_idle(timeout_s: float, failure: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = printer_status()
        if not info["printing"]:
            if not info["connected"]:
                raise RuntimeError(info["error"] or "The printer disconnected.")
            return
        time.sleep(0.05)
    stop_scan()
    raise TimeoutError(failure)


def move_axis(axis: str, delta_mm: float, feed: float) -> None:
    text = build_relative_move_gcode(axis, delta_mm, feed)
    axis = str(axis).upper()
    _require_printer_ready()
    start_gcode(text)
    timeout_s = abs(float(delta_mm)) / float(feed) * 60.0 + 5.0
    _wait_until_idle(timeout_s, f"The {axis} move did not finish.")


def move_z(delta_mm: float, feed: float) -> None:
    
    move_axis("Z", delta_mm, feed)


def return_from_stop_moves(row: int, col: int, step_mm: float) -> list[tuple[str, float]]:
    
    row = int(row)
    col = int(col)
    step_mm = float(step_mm)
    if row < 1 or col < 1:
        raise ValueError("row and column start at 1")
    if step_mm <= 0:
        raise ValueError("step must be greater than 0")
    moves: list[tuple[str, float]] = []
    if col > 1:
        moves.append(("X", -(col - 1) * step_mm))
    if row > 1:
        moves.append(("Y", (row - 1) * step_mm))
    return moves


def return_to_start_moves(columns: int, rows: int, step_mm: float) -> list[tuple[str, float]]:
    
    columns = int(columns)
    rows = int(rows)
    step_mm = float(step_mm)
    if columns < 1 or rows < 1:
        raise ValueError("the scan needs at least one column and one row")
    if step_mm <= 0:
        raise ValueError("step must be greater than 0")
    end_col = columns if rows % 2 == 1 else 1
    return return_from_stop_moves(rows, end_col, step_mm)


def build_home_gcode() -> str:
    
    return "\n".join(
        [
            "G21",
            "G28 X Y",
            "M400",
            "",
        ]
    )


def home_xy() -> None:
    text = build_home_gcode()
    _require_printer_ready()
    start_gcode(text)
    _wait_until_idle(90.0, "Homing did not finish.")


def scan_stops(columns: int, rows: int, step_mm: float) -> list[tuple[int, int, str | None, float]]:
    
    columns = int(columns)
    rows = int(rows)
    step_mm = float(step_mm)
    if columns < 1 or rows < 1:
        raise ValueError("the scan needs at least one column and one row")
    if step_mm <= 0:
        raise ValueError("step must be greater than 0")
    stops: list[tuple[int, int, str | None, float]] = []
    for row_index in range(rows):
        going_right = row_index % 2 == 0
        col_indexes = range(columns) if going_right else range(columns - 1, -1, -1)
        for col_index in col_indexes:
            if not stops:
                stops.append((row_index + 1, col_index + 1, None, 0.0))
                continue
            previous_col = stops[-1][1]
            col = col_index + 1
            if col == previous_col:
                stops.append((row_index + 1, col, "Y", -step_mm))
            else:
                direction = 1.0 if going_right else -1.0
                stops.append((row_index + 1, col, "X", direction * step_mm))
    return stops


def scan_summary(step_mm: float, columns: int, rows: int, fov_mm: float = FOV_MM) -> dict:
    
    step_mm = float(step_mm)
    columns = int(columns)
    rows = int(rows)
    fov_mm = float(fov_mm)
    return {
        "x_mm": (columns - 1) * step_mm,
        "y_mm": (rows - 1) * step_mm,
        "covered_x_mm": fov_mm + (columns - 1) * step_mm,
        "covered_y_mm": fov_mm + (rows - 1) * step_mm,
        "stops": columns * rows,
        "columns": columns,
        "rows": rows,
    }


def _pipx_printrun_roots() -> list[Path]:
    
    home = Path.home()
    roots = [
        home / "Library/Application Support/pipx/venvs/printrun",
        home / ".local/pipx/venvs/printrun",
        home / ".local/share/pipx/venvs/printrun",
        home / "pipx/venvs/printrun",
    ]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(Path(local) / "pipx/venvs/printrun")
    pipx_home = os.environ.get("PIPX_HOME")
    if pipx_home:
        roots.append(Path(pipx_home) / "venvs/printrun")
    return roots


def _pipx_printrun_site() -> Path | None:
    
    patterns = (
        "lib/python*/site-packages/printrun/printcore.py",
        "Lib/site-packages/printrun/printcore.py",
        "lib/site-packages/printrun/printcore.py",
    )
    found: list[Path] = []
    for root in _pipx_printrun_roots():
        for pattern in patterns:
            found.extend(root.glob(pattern))
    if not found:
        return None
    return sorted(found)[-1].parents[1]


def load_printcore():
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            from printrun.gcoder import GCode
            from printrun.printcore import printcore
        except ImportError:
            site = _pipx_printrun_site()
            if site is None:
                raise ImportError(
                    "Printrun is not available. With the virtual environment "
                    "active, run: python -m pip install --no-deps "
                    "\"Printrun==2.2.0\""
                ) from None
            site_text = str(site)
            if site_text not in sys.path:
                sys.path.insert(0, site_text)
            from printrun.gcoder import GCode
            from printrun.printcore import printcore
        return printcore, GCode


def _remember_errors(core) -> None:
    original = core.logError

    def log_error(error) -> None:
        _LINK["error"] = str(error)
        original(error)

    core.logError = log_error


def connect_printer(port: str, baud: int = DEFAULT_BAUD):
    
    printcore, _gcode = load_printcore()
    disconnect_printer()
    _LINK["error"] = ""
    core = printcore()
    _remember_errors(core)
    core.connect(port, int(baud))
    if core.printer is None:
        message = _LINK["error"] or f"Could not open {port}."
        raise RuntimeError(message)
    _LINK["core"] = core
    _LINK["port"] = port
    _LINK["baud"] = int(baud)
    return core


def disconnect_printer() -> None:
    core = _LINK.get("core")
    _LINK["core"] = None
    _LINK["port"] = None
    _LINK["baud"] = None
    if core is None:
        return
    try:
        if getattr(core, "printing", False) or getattr(core, "paused", False):
            core.cancelprint()
        core.disconnect()
    except Exception as exc:
        _LINK["error"] = str(exc)


def _read_thread_alive(core) -> bool:
    thread = getattr(core, "read_thread", None)
    return thread is not None and thread.is_alive()


def printer_status() -> dict:
    core = _LINK.get("core")
    if core is None or getattr(core, "printer", None) is None:
        return {
            "connected": False,
            "online": False,
            "printing": False,
            "port": None,
            "index": 0,
            "total": 0,
            "error": _LINK.get("error") or "",
        }
    queue = getattr(core, "mainqueue", None)
    lines = getattr(queue, "lines", None) or []
    return {
        "connected": True,
        "online": bool(getattr(core, "online", False)) and _read_thread_alive(core),
        "printing": bool(getattr(core, "printing", False)),
        "port": _LINK.get("port"),
        "index": int(getattr(core, "queueindex", 0) or 0),
        "total": len(lines),
        "error": _LINK.get("error") or "",
    }


def start_gcode(text: str) -> None:
    core = _LINK.get("core")
    if core is None or getattr(core, "printer", None) is None:
        raise RuntimeError("Connect the printer first.")
    if not getattr(core, "online", False):
        raise RuntimeError("The printer has not answered yet. Wait a moment and try again.")
    if getattr(core, "printing", False):
        raise RuntimeError("A scan is already running.")
    _printcore, GCode = load_printcore()
    gcode = GCode(text.splitlines())
    if not gcode.lines:
        raise RuntimeError("The scan has no moves.")
    if not core.startprint(gcode):
        raise RuntimeError("The printer did not start the scan.")
    _LINK["error"] = ""


def stop_scan() -> None:
    
    core = _LINK.get("core")
    if core is None:
        return
    if getattr(core, "printing", False) or getattr(core, "paused", False):
        core.cancelprint()
