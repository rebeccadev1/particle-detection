
from __future__ import annotations

import builtins
from pathlib import Path

import streamlit as st

from src.capture.focus import autofocus_status, claim_focus_capture
from src.capture.preview import preview_snapshot, start_preview, stop_preview
from src.capture import camera_automation as _camera
from src.capture.scan_capture import scan_capture_status
from src.capture.stage import printer_status

CAPTURES_DIR = _camera.CAPTURES_DIR
next_run_folder = _camera.next_run_folder


def _load_arduino_link() -> dict:
    """Keep the open Arduino across a script reload so the port is not opened twice."""
    shared = getattr(builtins, "_asml_arduino_link", None)
    if isinstance(shared, dict) and "ser" in shared:
        return shared
    shared = {"ser": None, "port": None, "ack": ""}
    builtins._asml_arduino_link = shared
    return shared


_LINK = _load_arduino_link()

CAPTURE_WIDGET_DEFAULTS: dict = {
    "capture_mode": "Uniform",
    "capture_nsew_style": "Automatic",
    "capture_total_uniform": _camera.TOTAL_EXPOSURE_MS,
    "capture_total_nsew": _camera.NSEW_TOTAL_EXPOSURE_MS,
}


def keep_capture_inputs() -> None:
    """Call on every run. Streamlit drops widget values while this tab is hidden."""
    for key in CAPTURE_WIDGET_DEFAULTS:
        held = f"_held_{key}"
        if key in st.session_state:
            st.session_state[held] = st.session_state[key]
        elif held in st.session_state:
            st.session_state[key] = st.session_state[held]


def _capture_module():
    return _camera


def _serial():
    ser = _LINK.get("ser")
    if ser is None:
        return None
    if not getattr(ser, "is_open", False):
        _LINK["ser"] = None
        _LINK["port"] = None
        return None
    return ser


def _forget_port() -> None:
    ser = _LINK.get("ser")
    if ser is not None:
        try:
            if getattr(ser, "is_open", False):
                ser.close()
        except Exception:
            pass
    _LINK["ser"] = None
    _LINK["port"] = None


def _run_serial(action):
    """Run a command on the open port. A dead port is dropped so Connect works again."""
    ser = _serial()
    if ser is None:
        st.error("Connect the Arduino first.")
        return None
    try:
        reply = action(ser)
    except Exception as exc:
        _forget_port()
        st.error(f"{exc} The serial port was closed.")
        return None
    _LINK["ack"] = reply
    return reply


def _selected_mode() -> str:
    mode = st.session_state.get("capture_mode", "Uniform")
    return mode if mode in ("Uniform", "NSEW") else "Uniform"


def _total_exposure_key(mode: str) -> str:
    return "capture_total_nsew" if mode == "NSEW" else "capture_total_uniform"


def total_exposure_ms(mode: str | None = None) -> float:
    """Shutter for this lighting mode. NSEW's standard is eight times Uniform."""
    mode = _selected_mode() if mode is None else mode
    key = _total_exposure_key(mode)
    default = CAPTURE_WIDGET_DEFAULTS[key]
    return float(st.session_state.get(key, default))


def _apply_mode_light(cap, mode: str):
    if mode == "Uniform":
        return _run_serial(cap.light_all)
    return _run_serial(cap.lights_off)


def _nsew_style() -> str:
    style = st.session_state.get("capture_nsew_style", "Automatic")
    return style if style in ("Automatic", "Manual") else "Automatic"


def _failure(action: str, next_step: str, exc: BaseException) -> None:
    """Operator-facing failure. The exception text stays behind a disclosure."""
    detail = str(exc).strip() or type(exc).__name__
    first = detail.splitlines()[0]
    if len(first) > 160 or "Traceback" in detail:
        st.error(f"{action} {next_step}")
    else:
        st.error(f"{action} {first} {next_step}")
    with st.expander("Error detail"):
        st.code(detail)


def _port_roles() -> dict[str, str]:
    roles: dict[str, str] = {}
    printer = printer_status()
    if printer["connected"] and printer.get("port"):
        roles[str(printer["port"])] = "the printer"
    if _serial() is not None and _LINK.get("port"):
        roles[str(_LINK["port"])] = "the Arduino"
    return roles


def _marked_port(device: str, labels: dict[str, str], roles: dict[str, str]) -> str:
    name = labels.get(device, device)
    role = roles.get(device)
    if role:
        return f"{name} — in use by {role}"
    return name


def _render_port_select(cap, *, label: str, state_key: str, disabled: bool) -> bool:
    """Port list with taken ports marked. False when pyserial is missing."""
    try:
        ports = cap.list_serial_ports()
    except ImportError:
        st.error("pyserial is not installed. From the project folder: pip install pyserial")
        return False
    devices = [device for device, _name in ports]
    labels = {device: name for device, name in ports}
    roles = _port_roles()
    if devices:
        if st.session_state.get(state_key) not in devices:
            st.session_state.pop(state_key, None)
        st.selectbox(
            label,
            devices,
            index=None,
            placeholder="Select a port",
            key=state_key,
            format_func=lambda device: _marked_port(device, labels, roles),
            disabled=disabled,
        )
    else:
        st.caption("No serial ports found. Plug in the device, then refresh.")
    return True


def _hardware_strip() -> None:
    """One line for Arduino, printer, and camera. Not a banner per device."""
    printer = printer_status()
    snap = preview_snapshot()
    capture = scan_capture_status()
    if _serial() is not None:
        arduino = f"connected ({_LINK['port']})"
    else:
        arduino = "not connected"
    if not printer["connected"]:
        printer_state = "not connected"
    elif not printer["online"]:
        printer_state = "waiting for an answer"
    elif printer["printing"] or capture["running"]:
        printer_state = "moving"
    elif autofocus_status()["running"]:
        printer_state = "autofocus"
    else:
        printer_state = f"ready ({printer['port']})"
    if capture["running"]:
        camera = "in a scan"
    elif snap["frame"] is not None:
        camera = "live"
    elif snap["error"]:
        camera = "not connected"
    else:
        camera = "opening"
    st.caption(f"Arduino {arduino} · Printer {printer_state} · Camera {camera}")


def render_capture_tab() -> None:
    """Uniform lighting, or one side at a time. One total-exposure control."""
    cap = _capture_module()
    _hardware_strip()
    _capture_after_autofocus(cap)
    with st.container(border=True, key="block-capture"):
        st.header("Lighting")
        mode = st.radio(
            "Select mode",
            ("Uniform", "NSEW"),
            horizontal=True,
            key="capture_mode",
        )
        nsew_style = "Automatic"
        if mode == "NSEW":
            nsew_style = st.radio(
                "NSEW capture",
                ("Automatic", "Manual"),
                horizontal=True,
                key="capture_nsew_style",
            )

        ser = _serial()
        ports_ready = _render_port_select(
            cap,
            label="Arduino port",
            state_key="arduino_port",
            disabled=ser is not None,
        )
        if ports_ready:
            _render_lighting_controls(cap, mode, nsew_style, ser)

    live_col, last_col = st.columns(2)
    with live_col:
        _render_live(cap, _shutter_ms(cap))
    with last_col:
        _show_last_run()
    if not ports_ready:
        return
    render_stage(cap)


def _render_lighting_controls(cap, mode: str, nsew_style: str, ser) -> None:
    refresh_col, connect_col, disconnect_col = st.columns(3)
    if refresh_col.button("Refresh ports", key="refresh_arduino_port", width="stretch"):
        st.rerun()
    arduino_port = st.session_state.get("arduino_port")
    if connect_col.button(
        "Connect Arduino",
        disabled=not arduino_port or ser is not None,
        width="stretch",
    ):
        port = arduino_port
        printer = printer_status()
        if printer["connected"] and port == printer["port"]:
            st.error("That port is already the printer. Choose the other port for the Arduino.")
        else:
            opened = None
            try:
                opened = cap.open_arduino(port)
                if _selected_mode() == "Uniform":
                    cap.light_all(opened)
                    ack = "OK ALL"
                else:
                    cap.lights_off(opened)
                    ack = "OK OFF"
            except Exception as exc:
                if opened is not None:
                    try:
                        opened.close()
                    except Exception:
                        pass
                _failure(
                    "Could not open the Arduino.",
                    "Pick another port, or unplug it and refresh ports.",
                    exc,
                )
            else:
                _LINK["ser"] = opened
                _LINK["port"] = port
                _LINK["ack"] = ack
                st.session_state["capture_lit_mode"] = _selected_mode()
                st.rerun()

    if disconnect_col.button("Disconnect Arduino", disabled=ser is None, width="stretch"):
        live = _serial()
        if live is not None:
            try:
                cap.lights_off(live)
            except Exception as exc:
                _failure(
                    "The Arduino closed, and the lights may still be on.",
                    "Disconnect power if they stay lit.",
                    exc,
                )
        _forget_port()
        _LINK["ack"] = ""
        st.session_state.pop("capture_lit_mode", None)
        st.rerun()

    if ser is not None and _LINK.get("ack"):
        st.caption(f"Last reply: {_LINK['ack']}")

    if ser is not None and st.session_state.get("capture_lit_mode") != mode:
        if _apply_mode_light(cap, mode):
            st.session_state["capture_lit_mode"] = mode

    total_key = _total_exposure_key(mode)
    total_ms = st.number_input(
        "Total exposure",
        min_value=1.0,
        max_value=4000.0,
        value=float(CAPTURE_WIDGET_DEFAULTS[total_key]),
        step=1.0,
        key=total_key,
        help="Shutter in milliseconds. Standard is 100 for Uniform and 800 for NSEW.",
    )
    shutter = cap.shutter_ms(float(total_ms))

    if mode == "Uniform":
        on_col, off_col = st.columns(2)
        if on_col.button("All lights on", disabled=ser is None, width="stretch"):
            _run_serial(cap.light_all)
        if off_col.button("All lights off", disabled=ser is None, width="stretch"):
            _run_serial(cap.lights_off)
        blocked = _still_capture_block_reason(needs_arduino=False)
        if st.button("Capture", type="primary", disabled=blocked is not None):
            _capture_run(cap, mode, shutter)
        _caption_when_blocked("Capture", blocked)
    elif nsew_style == "Automatic":
        st.caption(
            "One click lights N, then E, then S, then W, and saves each frame. "
            f"It waits {cap.NSEW_GAP_S * 1000:.0f} ms between sides."
        )
        blocked = _still_capture_block_reason(needs_arduino=True)
        if st.button("Capture N, E, S, W", type="primary", disabled=blocked is not None):
            _capture_run(cap, mode, shutter)
        _caption_when_blocked("Capture", blocked)
    else:
        _render_manual_sides(cap, shutter)



def _manual_target(cap) -> tuple[Path, int, int]:
    current = st.session_state.get("capture_manual_folder")
    row = st.session_state.get("capture_manual_row")
    col = st.session_state.get("capture_manual_col")
    if current and Path(current).is_dir() and row and col:
        return Path(current), int(row), int(col)
    run = cap.allocate_run_folder(cap.CAPTURES_DIR)
    st.session_state["capture_manual_folder"] = str(run)
    st.session_state["capture_manual_row"] = 1
    st.session_state["capture_manual_col"] = 1
    return run, 1, 1


def _capture_one_side(cap, index: int, shutter_ms: float) -> None:
    ser = _serial()
    if ser is None:
        st.error("Connect the Arduino first.")
        return
    exposure_us = shutter_ms * 1000.0
    camera = None
    try:
        direction = cap.DIRECTIONS[index]
        with st.spinner(f"Capturing {direction}…"):
            stop_preview()
            camera, converter = cap.open_camera(exposure_us)
            folder, row, col = _manual_target(cap)
            filename = cap.capture_side(
                camera,
                converter,
                ser,
                folder,
                index,
                cap.grab_timeout_ms(exposure_us),
                row,
                col,
            )
            cap.write_manifest(
                folder,
                {
                    "mode": "nsew",
                    "row": row,
                    "col": col,
                    "total_exposure_ms": total_exposure_ms(),
                    "shutter_ms": shutter_ms,
                    "exposure_us": exposure_us,
                    "sides_on": 1,
                    "port": _LINK.get("port"),
                    "files": [filename],
                    "stage": _stage_record(),
                },
            )
    except Exception as exc:
        _failure("The picture was not saved.", "The camera is free again. Take the shot again.", exc)
        return
    finally:
        cap.shutdown(camera, None)
        start_preview(cap, exposure_us)
    st.session_state["capture_last_run"] = {
        "folder": str(folder),
        "files": [cap.image_filename(row, col, name) for name in cap.DIRECTIONS],
        "mode": "NSEW manual",
    }
    _LINK["ack"] = f"OK {index}"
    st.caption(f"Saved {filename} to {folder}")


def _capture_run(cap, mode: str, shutter_ms: float, *, label: str | None = None) -> None:
    ser = _serial()
    if mode != "Uniform" and ser is None:
        st.error("NSEW capture needs the Arduino to switch sides.")
        return
    exposure_us = shutter_ms * 1000.0
    timeout_ms = cap.grab_timeout_ms(exposure_us)
    camera = None
    folder = None
    saved: list[str] = []
    failed = False
    try:
        with st.spinner("Capturing…" if mode == "Uniform" else "Capturing N, E, S, W…"):
            stop_preview()
            camera, converter = cap.open_camera(exposure_us)
            folder = cap.allocate_run_folder(cap.CAPTURES_DIR)
            row, col = 1, 1
            if mode == "Uniform":
                saved = [cap.capture_uniform(camera, converter, ser, folder, timeout_ms, row, col)]
            else:
                saved = cap.capture_nsew(camera, converter, ser, folder, timeout_ms, row, col)
            cap.write_manifest(
                folder,
                {
                    "mode": "uniform" if mode == "Uniform" else "nsew",
                    "row": row,
                    "col": col,
                    "total_exposure_ms": total_exposure_ms(),
                    "shutter_ms": shutter_ms,
                    "exposure_us": exposure_us,
                    "sides_on": cap.RING_SIDES if mode == "Uniform" else 1,
                    "port": _LINK.get("port"),
                    "files": saved,
                    "stage": _stage_record(),
                },
            )
    except Exception as exc:
        failed = True
        _failure("The picture was not saved.", "The camera is free again. Take the shot again.", exc)
    finally:
        cap.shutdown(camera, None)
        start_preview(cap, exposure_us)
    if folder is not None and saved:
        st.session_state["capture_last_run"] = {
            "folder": str(folder),
            "files": saved,
            "mode": label or mode,
        }
        _LINK["ack"] = "OK ALL" if mode == "Uniform" else f"OK {len(saved) - 1}"
    if not failed and saved:
        st.caption(f"Saved {', '.join(saved)} to {folder}")


def _camera_busy_reason() -> str | None:
    if scan_capture_status()["running"]:
        return "Scan and capture has the camera."
    if autofocus_status()["running"]:
        return "Autofocus is running."
    return None


def _still_capture_block_reason(*, needs_arduino: bool) -> str | None:
    busy = _camera_busy_reason()
    if busy:
        return busy
    if needs_arduino and _serial() is None:
        return "NSEW capture needs the Arduino."
    return None


def _caption_when_blocked(action: str, reason: str | None) -> None:
    if reason:
        st.caption(f"{action} is off: {reason}")


def _manual_set(cap) -> tuple[Path | None, int, int, list[str]]:
    """Folder and sides already saved for the open manual set. Does not create one."""
    current = st.session_state.get("capture_manual_folder")
    row = st.session_state.get("capture_manual_row")
    col = st.session_state.get("capture_manual_col")
    if not current or not row or not col:
        return None, 0, 0, []
    folder = Path(current)
    if not folder.is_dir():
        return None, 0, 0, []
    row_i, col_i = int(row), int(col)
    saved = [
        direction
        for direction in cap.DIRECTIONS
        if (folder / cap.image_filename(row_i, col_i, direction)).is_file()
    ]
    return folder, row_i, col_i, saved


def _render_manual_sides(cap, shutter_ms: float) -> None:
    """One button per side, with the open set listed beside them."""
    blocked = _still_capture_block_reason(needs_arduino=True)
    folder, row, col, saved = _manual_set(cap)
    st.caption("Each button lights that side and adds it to this set.")
    side_cols = st.columns(len(cap.DIRECTIONS))
    for index, direction in enumerate(cap.DIRECTIONS):
        if side_cols[index].button(
            direction,
            key=f"capture_side_{direction}",
            disabled=blocked is not None,
            width="stretch",
        ):
            _capture_one_side(cap, index, shutter_ms)
    if folder is None:
        st.caption("No set yet. The first side starts a folder.")
    else:
        marks = " · ".join(
            f"{direction} saved" if direction in saved else f"{direction} not taken"
            for direction in cap.DIRECTIONS
        )
        st.caption(f"This set is `{folder.name}`, stop ({row},{col}). {marks}.")
    if st.button(
        "New set",
        disabled=folder is None or blocked is not None,
        help="Pictures already saved stay on disk. The next side starts a new folder.",
    ):
        st.session_state.pop("capture_manual_folder", None)
        st.session_state.pop("capture_manual_row", None)
        st.session_state.pop("capture_manual_col", None)
        st.rerun()
    if folder is not None:
        st.caption("New set keeps these files and starts the next side in a new folder.")
    _caption_when_blocked("Side capture", blocked)


def _shutter_ms(cap) -> float:
    return float(cap.shutter_ms(total_exposure_ms()))


def _capture_after_autofocus(cap) -> None:
    """Save a still once a standalone autofocus has finished on the best height."""
    if scan_capture_status()["running"]:
        return
    if not claim_focus_capture():
        return
    _capture_run(cap, _selected_mode(), _shutter_ms(cap), label="Autofocus")


def _show_last_run(*, key: str = "block-last-capture") -> None:
    last = st.session_state.get("capture_last_run")
    with st.container(border=True, key=key):
        st.header("Last capture")
        if not last:
            st.caption("No capture yet.")
            return
        folder = Path(last["folder"])
        files = last.get("files") or ([last["file"]] if last.get("file") else [])
        manual = last.get("mode") == "NSEW manual"
        label = last.get("mode") or "Capture"
        st.write(f"{label} — `{folder}`")
        existing = [name for name in files if (folder / name).is_file()]
        missing = [name for name in files if name not in existing]
        if existing:
            cols = st.columns(len(existing))
            for column, name in zip(cols, existing):
                column.image(str(folder / name), caption=Path(name).stem, width="stretch")
        if missing and not manual:
            for name in missing:
                st.warning(f"Missing {name}")


def suspend_live_view() -> None:
    """Close the camera when the Capture tab is not on screen."""
    stop_preview()


@st.fragment(run_every=0.25)
def _live_feed() -> None:
    snap = preview_snapshot()
    frame = snap["frame"]
    if frame is None:
        if snap["error"] == "No Basler camera found":
            st.caption(snap["error"])
        elif snap["error"]:
            st.error(snap["error"])
        else:
            st.info("Opening the camera…")
        return
    shutter_ms = float(st.session_state.get("_capture_shutter_ms", 0.0))
    st.image(frame, caption=f"{shutter_ms:.1f} ms", width="stretch")
    if snap["error"]:
        st.caption(snap["error"])


def _render_live(cap, shutter_ms: float) -> None:
    with st.container(border=True, key="block-live"):
        st.header("Live view")
        st.session_state["_capture_shutter_ms"] = float(shutter_ms)
        if scan_capture_status()["running"]:
            st.caption("The scan has the camera. The live view returns when it finishes.")
            return
        start_preview(cap, shutter_ms * 1000.0)
        _live_feed()


from src.capture.capture_stage import keep_stage_inputs, render_stage
