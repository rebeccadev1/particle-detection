

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

DIRECTIONS = ("N", "E", "S", "W")
RING_SIDES = len(DIRECTIONS)

TOTAL_EXPOSURE_MS = 100.0
NSEW_TOTAL_EXPOSURE_MS = 8 * TOTAL_EXPOSURE_MS
SETTLE_S = 0.05
NSEW_GAP_S = 0.05
MOTION_PAUSE_S = 0.5
BAUD = 9600
GRAB_TIMEOUT_MS = 5000
READY_TIMEOUT_S = 5.0
ACK_TIMEOUT_S = 2.0
BOOT_LISTEN_S = 2.0

SCRIPT_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = SCRIPT_DIR.parents[2] / "Inputs" / "Groundup"
DEFAULT_COLUMNS = 10
RUN_DIR_RE = re.compile(r"^v(\d+) (\d{4}-\d{2}-\d{2})$")
ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86}


def pylon_module():
    """Load Basler's pypylon SDK. Callers import through here so the load is in one place."""
    from pypylon import pylon

    return pylon


def image_filename(row: int, col: int, label: str | None = None) -> str:
    """Picture name inside a run folder. A side label is N, E, S, or W."""
    stem = f"({int(row)},{int(col)})"
    if label:
        return f"{stem}_{label}.png"
    return f"{stem}.png"


def latest_run_folder(base: Path | None = None) -> Path | None:
    """Newest captures/vN YYYY-MM-DD directory, or None when none exist.

    Newest is the latest date, then the highest N on that date.
    """
    root = CAPTURES_DIR if base is None else Path(base)
    if not root.is_dir():
        return None
    found: list[tuple[str, int, Path]] = []
    for path in root.iterdir():
        match = RUN_DIR_RE.match(path.name)
        if match and path.is_dir():
            found.append((match.group(2), int(match.group(1)), path))
    if not found:
        return None
    found.sort()
    return found[-1][2]


def next_run_folder(base: Path, on: date | None = None) -> Path:
    """Path of the next captures/vN YYYY-MM-DD directory. Does not create it.

    N restarts at 1 each day.
    """
    day = (date.today() if on is None else on).isoformat()
    taken: list[int] = []
    if base.is_dir():
        for path in base.iterdir():
            match = RUN_DIR_RE.match(path.name)
            if match and path.is_dir() and match.group(2) == day:
                taken.append(int(match.group(1)))
    return base / f"v{max(taken, default=0) + 1} {day}"


def allocate_run_folder(base: Path, on: date | None = None) -> Path:
    """Create the next captures/vN YYYY-MM-DD directory. N restarts at 1 each day."""
    folder = next_run_folder(base, on=on)
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()
    return folder


def _dedupe_cu_tty(ports):
    """macOS lists each Uno twice. Keep /dev/cu.*, which does not wait for a carrier."""
    devices = {port.device for port in ports}
    return [
        port
        for port in ports
        if not (
            port.device.startswith("/dev/tty.")
            and ("/dev/cu." + port.device.removeprefix("/dev/tty.")) in devices
        )
    ]


def list_serial_ports() -> list[tuple[str, str]]:
    """(device, label) for every serial port, with the macOS tty/cu duplicate removed."""
    from serial.tools import list_ports

    ports = _dedupe_cu_tty(list(list_ports.comports()))
    return [
        (port.device, f"{port.device} — {port.description}")
        for port in ports
    ]


def find_arduino_port() -> str:
    from serial.tools import list_ports

    ports = _dedupe_cu_tty(list(list_ports.comports()))

    def rank(port) -> int:
        text = f"{port.description} {port.manufacturer or ''} {port.device}".lower()
        if "arduino" in text:
            return 2
        if port.vid in ARDUINO_VIDS:
            return 1
        device = port.device.lower()
        if "usbmodem" in device or "usbserial" in device:
            return 1
        return 0

    ranked = [port for port in ports if rank(port) > 0]
    best = max((rank(port) for port in ranked), default=0)
    chosen = [port for port in ranked if rank(port) == best]
    if len(chosen) == 1:
        return chosen[0].device

    listing = [f"  {port.device}: {port.description}" for port in ports]
    found = "\n".join(listing) if listing else "  (no serial ports)"
    raise RuntimeError(f"pass --port to choose a serial port. Found:\n{found}")


def _read_line(ser, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        text = raw.decode(errors="replace").strip()
        if text:
            return text
    raise TimeoutError(f"no serial reply within {timeout_s:.1f}s")


def request_line(ser, command: str, timeout_s: float = ACK_TIMEOUT_S) -> str:
    """Send one line and return the reply. A timeout or an ERR line raises."""
    ser.reset_input_buffer()
    ser.write(f"{command}\n".encode())
    ser.flush()
    line = _read_line(ser, timeout_s)
    if line.startswith("ERR"):
        raise RuntimeError(f"command {command!r} got {line!r}")
    return line


def _handshake(ser) -> str:
    """Return ``READY`` or ``OK OFF``.

    ``READY`` comes from ``setup()`` on the transistor-ring sketch. A CH340
    that is already running does not reset when the port opens, so that line
    is gone. ``x`` is answered from ``loop()`` and switches every quadrant off.
    """
    deadline = time.monotonic() + BOOT_LISTEN_S
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            line = _read_line(ser, min(0.5, remaining))
        except TimeoutError:
            continue
        except Exception as exc:
            if "multiple access" in str(exc) or "returned no data" in str(exc):
                raise RuntimeError(
                    f"{ser.port} is already open. Disconnect the other device "
                    "and choose a different port."
                ) from exc
            raise
        if line.startswith("READY"):
            return line
        if line.startswith("ERR"):
            raise RuntimeError(f"Arduino replied {line!r} during connect")

    reply = request_line(ser, "x", timeout_s=READY_TIMEOUT_S)
    if reply != "OK OFF":
        raise RuntimeError(f"expected 'OK OFF', got {reply!r}")
    return reply


def open_arduino(port: str):
    """Open the transistor-ring sketch. It always has four sides, N E S W."""
    import serial

    ser = serial.Serial(port, BAUD, timeout=0.2)
    try:
        _handshake(ser)
    except Exception:
        ser.close()
        raise
    return ser


def lights_off(ser) -> None:
    reply = request_line(ser, "x")
    if reply != "OK OFF":
        raise RuntimeError(f"expected 'OK OFF', got {reply!r}")


def standard_total_exposure_ms(mode: str) -> float:
    """Standard shutter. NSEW is eight times the uniform standard."""
    if str(mode).lower() == "nsew":
        return float(NSEW_TOTAL_EXPOSURE_MS)
    return float(TOTAL_EXPOSURE_MS)


def shutter_ms(total_exposure_ms: float) -> float:
    """Camera shutter. Total exposure is that shutter time, in milliseconds."""
    total = float(total_exposure_ms)
    if total <= 0:
        raise ValueError("total exposure must be positive")
    return total


def grab_timeout_ms(exposure_us: float) -> int:
    return int(max(GRAB_TIMEOUT_MS, exposure_us / 1000.0 + 2000))


def set_segment(ser, index: int) -> str:
    reply = request_line(ser, str(index))
    expected = f"OK {index}"
    if reply != expected:
        raise RuntimeError(f"expected {expected!r}, got {reply!r}")
    return reply


def light_all(ser) -> str:
    reply = request_line(ser, "f")
    if reply != "OK ALL":
        raise RuntimeError(f"expected 'OK ALL', got {reply!r}")
    return reply


def _set_node(camera, name: str, value: str) -> None:
    node = getattr(camera, name)
    if not node.IsWritable():
        raise RuntimeError(f"camera node {name} is not writable; cannot set {value}")
    node.Value = value


_NO_CAMERA_UNTIL = 0.0
_NO_CAMERA_BACKOFF_S = 5.0


def _open_device():
    global _NO_CAMERA_UNTIL
    pylon = pylon_module()

    now = time.monotonic()
    if now < _NO_CAMERA_UNTIL:
        raise RuntimeError("No Basler camera found")
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if len(devices) == 0:
        _NO_CAMERA_UNTIL = now + _NO_CAMERA_BACKOFF_S
        raise RuntimeError("No Basler camera found")
    _NO_CAMERA_UNTIL = 0.0
    camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    camera.Open()
    _disable_test_image(camera)
    return camera


def _disable_test_image(camera) -> None:
    """Use the sensor. A saved camera setting can otherwise show a test pattern."""
    for name in ("TestImageSelector", "TestPattern"):
        try:
            node = getattr(camera, name)
        except Exception:
            continue
        if not node.IsWritable():
            continue
        try:
            node.Value = "Off"
            return
        except Exception:
            pass
        try:
            choices = list(node.Symbolics)
        except Exception:
            return
        for choice in choices:
            if str(choice).lower() == "off":
                node.Value = choice
                return


def _disable_exposure_auto(camera) -> None:
    try:
        exposure_auto = camera.ExposureAuto
    except Exception:
        exposure_auto = None
    if exposure_auto is not None and exposure_auto.IsWritable():
        exposure_auto.Value = "Off"


def set_exposure(camera, exposure_us: float) -> None:
    """Set the shutter. Raises if the camera keeps a different value."""
    exposure = camera.ExposureTime
    if not exposure.IsWritable():
        raise RuntimeError("ExposureTime is not writable")
    exposure.Value = float(exposure_us)
    actual = float(exposure.Value)
    if abs(actual - float(exposure_us)) > max(1.0, float(exposure_us) * 0.01):
        raise RuntimeError(
            f"camera kept exposure at {actual:.0f} us "
            f"after a request for {float(exposure_us):.0f} us"
        )


def _make_converter():
    pylon = pylon_module()

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def open_camera(exposure_us: float):
    """Camera armed for one software-triggered frame."""
    camera = _open_device()
    try:
        _disable_exposure_auto(camera)
        _set_node(camera, "TriggerSelector", "FrameStart")
        _set_node(camera, "TriggerMode", "Off")
        _set_node(camera, "TriggerSource", "Software")
        _set_node(camera, "TriggerMode", "On")
        set_exposure(camera, exposure_us)
    except Exception:
        camera.Close()
        raise
    return camera, _make_converter()


def open_live_camera(exposure_us: float):
    """Free-running camera for the on-screen feed. Caller must close it."""
    pylon = pylon_module()

    camera = _open_device()
    try:
        _disable_exposure_auto(camera)
        _set_node(camera, "TriggerSelector", "FrameStart")
        _set_node(camera, "TriggerMode", "Off")
        set_exposure(camera, exposure_us)
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    except Exception:
        camera.Close()
        raise
    return camera, _make_converter()


def orient_frame(image):
    """Turn the sensor image 90 degrees left so saved frames match the live view."""
    import numpy as np

    return np.ascontiguousarray(np.rot90(image, k=1))


def read_live_frame(camera, converter, timeout_ms: int):
    """Latest free-running frame, BGR or gray. The camera must already be grabbing."""
    pylon = pylon_module()

    grab = camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
    try:
        if not grab.GrabSucceeded():
            detail = grab.ErrorDescription
            raise RuntimeError(f"grab failed: {detail}")
        return orient_frame(converter.Convert(grab).GetArray())
    finally:
        grab.Release()


def grab_frame(camera, converter, timeout_ms: int):
    pylon = pylon_module()

    camera.ExecuteSoftwareTrigger()
    grab = camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
    try:
        if not grab.GrabSucceeded():
            detail = grab.ErrorDescription
            raise RuntimeError(f"grab failed: {detail}")
        return orient_frame(converter.Convert(grab).GetArray())
    finally:
        grab.Release()


def save_frame(image, path: Path) -> None:
    import cv2

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")


def capture_uniform(
    camera, converter, ser, folder: Path, timeout_ms: int, row: int = 1, col: int = 1
) -> str:
    """Save ``(row,col).png``. With an open Arduino, light all four sides first."""
    pylon = pylon_module()

    if ser is not None:
        ack = light_all(ser)
        print(f"ring: {ack}")
        time.sleep(SETTLE_S)
    else:
        print("ring left as it is")
    if not camera.IsGrabbing():
        camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    image = grab_frame(camera, converter, timeout_ms)
    filename = image_filename(row, col)
    save_frame(image, folder / filename)
    print(f"saved {folder / filename}")
    return filename


def capture_side(
    camera,
    converter,
    ser,
    folder: Path,
    index: int,
    timeout_ms: int,
    row: int = 1,
    col: int = 1,
) -> str:
    """Light one side and save ``(row,col)_N.png`` and the other sides the same way."""
    pylon = pylon_module()

    if index < 0 or index >= len(DIRECTIONS):
        raise ValueError(f"side index {index} is outside 0..{len(DIRECTIONS) - 1}")
    if ser is None:
        raise RuntimeError("NSEW capture needs the Arduino to switch sides.")
    direction = DIRECTIONS[index]
    ack = set_segment(ser, index)
    print(f"segment {index} ({direction}): {ack}")
    time.sleep(SETTLE_S)
    if not camera.IsGrabbing():
        camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    image = grab_frame(camera, converter, timeout_ms)
    filename = image_filename(row, col, direction)
    save_frame(image, folder / filename)
    print(f"saved {folder / filename}")
    return filename


def capture_nsew(
    camera, converter, ser, folder: Path, timeout_ms: int, row: int = 1, col: int = 1
) -> list[str]:
    """Light one side at a time and save ``(row,col)_N.png`` through ``_W.png``."""
    pylon = pylon_module()

    if ser is None:
        raise RuntimeError("NSEW capture needs the Arduino to switch sides.")
    if not camera.IsGrabbing():
        camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    saved: list[str] = []
    for index, direction in enumerate(DIRECTIONS):
        if index > 0:
            time.sleep(NSEW_GAP_S)
        ack = set_segment(ser, index)
        print(f"segment {index} ({direction}): {ack}")
        time.sleep(SETTLE_S)
        image = grab_frame(camera, converter, timeout_ms)
        filename = image_filename(row, col, direction)
        save_frame(image, folder / filename)
        saved.append(filename)
        print(f"saved {folder / filename}")
    return saved


def write_manifest(folder: Path, payload: dict) -> None:
    """Write or extend ``capture.json`` in the run folder, one entry per stop."""
    path = folder / "capture.json"
    stop = {
        "row": payload["row"],
        "col": payload["col"],
        "files": list(payload["files"]),
    }
    if path.is_file():
        data = json.loads(path.read_text())
    else:
        data = {}
    for key in ("mode", "total_exposure_ms", "shutter_ms", "exposure_us", "sides_on", "port"):
        if key in payload:
            data[key] = payload[key]
    if payload.get("stage") is not None:
        stage = payload["stage"]
        data["stage"] = {
            "fov_mm": float(stage["fov_mm"]),
            "overlap": float(stage["overlap"]),
            "step_mm": float(stage["step_mm"]),
            "columns": int(stage["columns"]),
            "rows": int(stage["rows"]),
            "feed_mm_per_min": float(stage["feed_mm_per_min"]),
        }
    stops = data.setdefault("stops", [])
    for existing in stops:
        if existing.get("row") == stop["row"] and existing.get("col") == stop["col"]:
            names = list(existing.get("files") or [])
            for name in stop["files"]:
                if name not in names:
                    names.append(name)
            existing["files"] = names
            break
    else:
        stops.append(stop)
    path.write_text(json.dumps(data, indent=2) + "\n")


def shutdown(camera, ser) -> None:
    if camera is not None:
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        except Exception as exc:
            print(f"StopGrabbing failed: {exc}", file=sys.stderr)
        try:
            if camera.IsOpen():
                trigger_mode = camera.TriggerMode
                if trigger_mode.IsWritable():
                    trigger_mode.Value = "Off"
                camera.Close()
        except Exception as exc:
            print(f"camera close failed: {exc}", file=sys.stderr)

    if ser is not None:
        try:
            if ser.is_open:
                lights_off(ser)
        except Exception as exc:
            print(f"failed to turn LEDs off: {exc}", file=sys.stderr)
        try:
            if ser.is_open:
                ser.close()
        except Exception as exc:
            print(f"serial close failed: {exc}", file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture with uniform or N/S/E/W ring light.")
    parser.add_argument("--port", help="serial port, for example /dev/cu.usbmodem1101")
    parser.add_argument(
        "--no-ring",
        action="store_true",
        help="Do not open the Arduino. Leave the ring as it is. Uniform only.",
    )
    parser.add_argument("--mode", choices=("uniform", "nsew"), default="uniform")
    parser.add_argument(
        "--total-exposure-ms",
        type=float,
        default=None,
        help="Shutter in milliseconds. Standard is 100 for uniform and 800 for nsew.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=DEFAULT_COLUMNS,
        help="stops along a row; even rows reverse, so the next folder is (row+1, columns)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sides_on = RING_SIDES if args.mode == "uniform" else 1
    total_exposure_ms = (
        standard_total_exposure_ms(args.mode)
        if args.total_exposure_ms is None
        else args.total_exposure_ms
    )
    exposure_ms = shutter_ms(total_exposure_ms)
    exposure_us = exposure_ms * 1000.0

    camera = None
    ser = None
    try:
        if args.no_ring and args.mode != "uniform":
            raise RuntimeError("NSEW capture needs the Arduino to switch sides.")
        port = None
        if args.no_ring:
            print("ring left as it is")
        else:
            port = args.port or find_arduino_port()
            print(f"Arduino port: {port}")
        print(f"{args.mode}: total exposure {exposure_ms:.1f} ms")
        if port is not None:
            ser = open_arduino(port)
        camera, converter = open_camera(exposure_us)
        folder = allocate_run_folder(CAPTURES_DIR)
        row, col = 1, 1
        print(f"saving to {folder}")
        timeout_ms = grab_timeout_ms(exposure_us)
        if args.mode == "uniform":
            saved = [capture_uniform(camera, converter, ser, folder, timeout_ms, row, col)]
        else:
            saved = capture_nsew(camera, converter, ser, folder, timeout_ms, row, col)
        write_manifest(
            folder,
            {
                "mode": args.mode,
                "row": row,
                "col": col,
                "total_exposure_ms": total_exposure_ms,
                "shutter_ms": exposure_ms,
                "exposure_us": exposure_us,
                "sides_on": sides_on,
                "port": port,
                "files": saved,
            },
        )
        print(f"saved {', '.join(saved)} to {folder}")
        return 0
    except Exception as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 1
    finally:
        shutdown(camera, ser)


if __name__ == "__main__":
    sys.exit(main())
