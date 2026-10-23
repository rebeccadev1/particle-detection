"""Stage, focus, and scan controls for the Capture tab."""

from __future__ import annotations

import streamlit as st

from src.capture.capture_ui import (
    CAPTURES_DIR,
    next_run_folder,
    _LINK,
    _caption_when_blocked,
    _capture_module,
    _failure,
    _nsew_style,
    _render_port_select,
    _run_serial,
    _selected_mode,
    _serial,
    total_exposure_ms,
)
from src.capture.focus import autofocus_status, start_autofocus, stop_autofocus
from src.capture.preview import preview_snapshot
from src.capture.scan_capture import scan_capture_status, start_scan_capture, stop_scan_capture
from src.capture.stage import (
    DEFAULT_COLUMNS,
    DEFAULT_FEED,
    DEFAULT_ROWS,
    FOV_MM,
    HOME_X_MM,
    HOME_Y_MM,
    OVERLAP,
    build_scan_gcode,
    connect_printer,
    disconnect_printer,
    home_xy,
    move_axis,
    pause_plan,
    printer_status,
    scan_summary,
    step_from_fov,
    stop_scan,
)

STAGE_WIDGET_DEFAULTS: dict = {
    "stage_fov_mm": FOV_MM,
    "stage_overlap": OVERLAP,
    "stage_columns": DEFAULT_COLUMNS,
    "stage_rows": DEFAULT_ROWS,
    "stage_feed": DEFAULT_FEED,
    "jog_minor_xy_mm": 1.0,
    "jog_major_xy_mm": 10.0,
    "jog_minor_z_mm": 0.1,
    "jog_major_z_mm": 1.0,
}

FOCUS_WIDGET_DEFAULTS: dict = {
    "focus_during_scan": True,
    "focus_max_travel_mm": 1.0,
    "focus_coarse_mm": 0.05,
    "focus_fine_mm": 0.01,
    "focus_feed": 300,
}


def keep_stage_inputs() -> None:
    """Call on every run so stage factors survive leaving the Capture tab."""
    for key in (*STAGE_WIDGET_DEFAULTS, *FOCUS_WIDGET_DEFAULTS):
        held = f"_held_{key}"
        if key in st.session_state:
            st.session_state[held] = st.session_state[key]
        elif held in st.session_state:
            st.session_state[key] = st.session_state[held]


def _light_for_manual_autofocus(cap) -> bool:
    """Set the ring before a standalone autofocus. Scan autofocus does not call this.

    Uniform turns every side on. NSEW manual turns North on. Returns False when
    the lights could not be set, so the search does not start in the dark.
    """
    mode = _selected_mode()
    if mode == "NSEW" and _nsew_style() != "Manual":
        return True
    ser = _serial()
    if ser is None:
        if mode == "Uniform":
            return True
        st.error("Connect the Arduino first so North can turn on.")
        return False
    if mode == "Uniform":
        return _run_serial(cap.light_all) is not None
    return _run_serial(lambda port: cap.set_segment(port, 0)) is not None


def _stage_pause() -> dict:
    mode = _selected_mode()
    return pause_plan(total_exposure_ms(mode), mode)


def _stage_step_mm() -> float:
    return step_from_fov(
        float(st.session_state.get("stage_fov_mm", FOV_MM)),
        float(st.session_state.get("stage_overlap", OVERLAP)),
    )


def _stage_record() -> dict:
    """Stage settings stored with every capture."""
    return {
        "fov_mm": float(st.session_state.get("stage_fov_mm", FOV_MM)),
        "overlap": float(st.session_state.get("stage_overlap", OVERLAP)),
        "step_mm": _stage_step_mm(),
        "columns": int(st.session_state.get("stage_columns", DEFAULT_COLUMNS)),
        "rows": int(st.session_state.get("stage_rows", DEFAULT_ROWS)),
        "feed_mm_per_min": float(st.session_state.get("stage_feed", DEFAULT_FEED)),
    }


def _stage_program() -> str:
    return build_scan_gcode(
        step_mm=_stage_step_mm(),
        columns=int(st.session_state["stage_columns"]),
        rows=int(st.session_state["stage_rows"]),
        feed=float(st.session_state["stage_feed"]),
        dwell_ms=_stage_pause()["dwell_ms"],
    )


def _focus_during_scan() -> bool:
    return bool(st.session_state.get("focus_during_scan", True))


def _focus_limits() -> str | None:
    travel = float(st.session_state["focus_max_travel_mm"])
    coarse = float(st.session_state["focus_coarse_mm"])
    fine = float(st.session_state["focus_fine_mm"])
    if coarse > travel or fine > travel:
        return "Coarse and fine steps have to fit inside the max travel."
    return None


def _render_focus(locked: bool) -> None:
    """Hill-climb Z from the current height using the live image."""
    with st.container(border=True, key="block-focus"):
        st.header("Focus")
        st.checkbox(
            "Check and correct focus during the scan",
            value=bool(FOCUS_WIDGET_DEFAULTS["focus_during_scan"]),
            key="focus_during_scan",
            disabled=locked,
            help=(
                "After every seventh stage step, score the frame and run autofocus. "
                "If the best score stays at or below 80% of the sharpness at the start of that search, "
                "autofocus runs again. Leave this off to keep the current Z for the whole scan."
            ),
        )
        with st.expander("Focus settings"):
            _focus_settings(locked)
        limit_error = _focus_limits()
        if limit_error:
            st.error(limit_error)
        _focus_progress()


def _focus_settings(locked: bool) -> None:
    travel_col, coarse_col, fine_col, feed_col = st.columns(4)
    travel_col.number_input(
        "Max travel (mm)",
        min_value=0.01,
        max_value=20.0,
        value=float(FOCUS_WIDGET_DEFAULTS["focus_max_travel_mm"]),
        step=0.01,
        format="%.2f",
        key="focus_max_travel_mm",
        disabled=locked,
        help="How far Z may move up or down from the height where autofocus starts.",
    )
    coarse_col.number_input(
        "Coarse step (mm)",
        min_value=0.01,
        max_value=5.0,
        value=float(FOCUS_WIDGET_DEFAULTS["focus_coarse_mm"]),
        step=0.01,
        format="%.2f",
        key="focus_coarse_mm",
        disabled=locked,
        help="First pass. A soft frame is scored again, and one more step is taken before reversing.",
    )
    fine_col.number_input(
        "Fine step (mm)",
        min_value=0.01,
        max_value=5.0,
        value=float(FOCUS_WIDGET_DEFAULTS["focus_fine_mm"]),
        step=0.01,
        format="%.2f",
        key="focus_fine_mm",
        disabled=locked,
        help="Second pass around the sharpest height from the coarse pass.",
    )
    feed_col.number_input(
        "Z speed (mm/min)",
        min_value=1,
        max_value=6000,
        value=int(FOCUS_WIDGET_DEFAULTS["focus_feed"]),
        step=50,
        key="focus_feed",
        disabled=locked,
    )


def _format_score(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _format_focus_step(index: int, step: dict) -> str:
    place = f"{step['offset_mm']:+.3f} mm"
    score = _format_score(float(step["sharpness"]))
    decision = step["decision"]
    if decision == "start":
        return f"{index:>2}. Start      {place}   {score}"
    if decision == "back":
        return f"{index:>2}. Back to    {place}   {score}"
    phase = {"coarse": "Coarse", "fine": "Fine"}.get(step["phase"], str(step["phase"]))
    verdict = {"sharper": "sharper", "again": "again"}.get(decision, "not sharper")
    return f"{index:>2}. {phase:<10} {place}   {score}   {verdict}"


def _focus_steps(info: dict) -> None:
    """Expandable log of each height autofocus has scored."""
    steps = info["steps"]
    run_id = info["run_id"]
    if run_id and st.session_state.get("_focus_steps_seen") != run_id:
        st.session_state["focus_steps_open"] = True
        st.session_state["_focus_steps_seen"] = run_id
    with st.expander("Focus steps", expanded=False, key="focus_steps_open", on_change="rerun"):
        if not steps:
            st.caption("Each height that autofocus tries shows up here.")
            return
        st.code("\n".join(_format_focus_step(index, step) for index, step in enumerate(steps, start=1)))


@st.fragment(run_every=0.5)
def _focus_progress() -> None:
    info = autofocus_status()
    printer = printer_status()
    snap = preview_snapshot()
    if info["running"]:
        st.session_state["_focus_was_running"] = True
        place = f"{info['offset_mm']:+.3f} mm from the start"
        if info["stopping"]:
            st.caption(f"Stopping. Camera is at {place}.")
        else:
            st.caption(
                f"Z {place}. Sharpness {_format_score(info['sharpness'])}. "
                f"Best {info['best_offset_mm']:+.3f} mm "
                f"({_format_score(info['best_sharpness'])})."
            )
    elif st.session_state.pop("_focus_was_running", False):
        st.rerun(scope="app")
    elif info["phase"] == "error" and info["error"]:
        st.error(info["error"])
    elif info["phase"] == "done":
        height = f"{info['best_offset_mm']:+.3f} mm from the start"
        if info["stopped"]:
            st.warning(f"Stopped. Camera is at the sharpest height tried, {height}.")
        else:
            st.success(
                f"Focused at {height}. Sharpness {_format_score(info['best_sharpness'])}."
            )
    _focus_steps(info)
    if info["running"]:
        if st.button("Stop focus"):
            stop_autofocus()
        return
    blocked = _autofocus_block_reason(printer, snap)
    if st.button("Autofocus", type="primary", disabled=blocked is not None):
        if not _light_for_manual_autofocus(_capture_module()):
            return
        try:
            start_autofocus(
                max_travel_mm=float(st.session_state["focus_max_travel_mm"]),
                coarse_mm=float(st.session_state["focus_coarse_mm"]),
                fine_mm=float(st.session_state["focus_fine_mm"]),
                feed=float(st.session_state["focus_feed"]),
            )
        except Exception as exc:
            _failure("Autofocus did not start.", "Check the printer and the live view, then try again.", exc)
        else:
            st.rerun(scope="app")
    _caption_when_blocked("Autofocus", blocked)


def _autofocus_block_reason(printer: dict, snap: dict) -> str | None:
    if not printer["connected"]:
        return "Printer is not connected."
    if not printer["online"]:
        return "Waiting for the printer to answer."
    if printer["printing"]:
        return "The printer is moving."
    if snap["frame"] is None:
        return "The live view is not ready."
    if _focus_limits() is not None:
        return "Coarse and fine steps have to fit inside the max travel."
    if _selected_mode() == "NSEW" and _nsew_style() == "Manual" and _serial() is None:
        return "NSEW manual focus needs the Arduino to light North."
    return None


def _scan_capture_block_note(info: dict, focusing: bool) -> str | None:
    """Why Scan and capture cannot run."""
    if not info["connected"]:
        shared = "Printer is not connected."
    elif not info["online"]:
        shared = "Waiting for the printer to answer."
    elif focusing:
        shared = "Autofocus is running."
    else:
        shared = None
    if shared:
        return f"Scan and capture is off: {shared}"
    if _focus_during_scan() and _focus_limits() is not None:
        return "Scan and capture is off: coarse and fine steps have to fit inside the max travel."
    if _selected_mode() == "NSEW" and _serial() is None:
        return "Scan and capture is off: NSEW capture needs the Arduino."
    return None


@st.fragment(run_every=1)
def _stage_progress() -> None:
    info = printer_status()
    focusing = autofocus_status()["running"]
    capture = scan_capture_status()
    if capture["error"] and not capture["running"]:
        st.error(capture["error"])
    if capture["running"]:
        st.session_state["_scan_capture_was_running"] = True
        total = max(capture["total"], 1)
        label = f"({capture['row']},{capture['col']})" if capture["row"] else "starting"
        if capture["returning"]:
            if capture["stopping"]:
                st.caption("Stopping the return to (1, 1).")
            else:
                st.progress(1.0, text="Returning to stop (1, 1).")
        elif capture["stopping"]:
            where = f" after {label}" if capture["row"] else ""
            st.caption(f"Stopping{where}, then returning to (1, 1).")
        elif capture["focusing"]:
            st.progress(
                min(capture["index"], total) / total,
                text=(
                    f"Autofocus after step {capture['steps']}"
                    + (
                        f", attempt {capture['focus_attempt']}"
                        if capture.get("focus_attempt", 1) > 1
                        else ""
                    )
                    + f". Sharpness {_format_score(capture['sharpness'])}."
                ),
            )
        else:
            st.progress(
                min(capture["index"], total) / total,
                text=f"Stop {capture['index']} of {capture['total']} — {label}",
            )
        if capture.get("folder"):
            st.caption(f"Saving to `{capture['folder']}`.")
        if st.button("Stop scan and capture"):
            stop_scan_capture()
        return
    if st.session_state.pop("_scan_capture_was_running", False):
        st.rerun(scope="app")
    if capture["saved"] and not capture["error"]:
        where = f" to {capture['folder']}" if capture.get("folder") else ""
        st.success(f"Saved {capture['saved']} stops{where}.")
        st.caption("Sharpness at each stop, and each autofocus result and Z change, are in sharpness.json.")
    if info["printing"]:
        total = max(info["total"], 1)
        index = min(info["index"], total)
        st.progress(index / total, text=f"Moving — command {index} of {info['total']}")
        if st.button("Stop scan"):
            stop_scan()
            st.rerun(scope="app")
        return
    if info["error"]:
        st.caption(info["error"])
    st.caption(f"Files will be saved to `{next_run_folder(CAPTURES_DIR)}`.")
    scan_note = _scan_capture_block_note(info, focusing)
    if st.button(
        "Scan and capture",
        type="primary",
        disabled=scan_note is not None,
    ):
        try:
            start_scan_capture(
                ser=_serial(),
                mode=_selected_mode(),
                total_exposure_ms=total_exposure_ms(),
                step_mm=_stage_step_mm(),
                stage=_stage_record(),
                columns=int(st.session_state["stage_columns"]),
                rows=int(st.session_state["stage_rows"]),
                feed=float(st.session_state["stage_feed"]),
                focus_max_travel_mm=float(st.session_state.get("focus_max_travel_mm", 1.0)),
                focus_coarse_mm=float(st.session_state.get("focus_coarse_mm", 0.05)),
                focus_fine_mm=float(st.session_state.get("focus_fine_mm", 0.01)),
                focus_feed=float(st.session_state.get("focus_feed", 300)),
                focus_during_scan=_focus_during_scan(),
            )
        except Exception as exc:
            _failure(
                "Scan and capture did not start.",
                "Check the printer, the camera, and the lighting mode, then try again.",
                exc,
            )
        else:
            st.rerun(scope="app")
    if scan_note:
        st.caption(scan_note)


def _format_jog_mm(delta_mm: float) -> str:
    return f"{delta_mm:+.2f}"


def _jog_stage(axis: str, delta_mm: float) -> None:
    feed = float(st.session_state.get("stage_feed", DEFAULT_FEED))
    try:
        move_axis(axis, delta_mm, feed)
    except Exception as exc:
        _failure("The stage did not move.", "Check that the printer is still connected.", exc)
    else:
        st.session_state["stage_last_jog"] = f"Last move: {axis} {_format_jog_mm(delta_mm)} mm."


def _jog_distance(key: str) -> float:
    return abs(float(st.session_state.get(key, STAGE_WIDGET_DEFAULTS[key])))


def _pad_button(column, label: str, key: str, ready: bool, axis: str, delta_mm: float, help_text: str) -> None:
    if column.button(label, key=key, disabled=not ready, width="stretch", help=help_text):
        _jog_stage(axis, delta_mm)


def _stage_cross(ready: bool, minor_xy: float, major_xy: float, minor_z: float, major_z: float) -> None:
    xy_col, z_col = st.columns([4.2, 1.15], vertical_alignment="center")
    with xy_col:
        st.caption("XY")
        major_up = st.columns(5)
        _pad_button(major_up[2], "▲▲", "jog_y_major_up", ready, "Y", major_xy, "Major step, +Y")
        minor_up = st.columns(5)
        _pad_button(minor_up[2], "▲", "jog_y_minor_up", ready, "Y", minor_xy, "Minor step, +Y")
        west, minor_west, home, minor_east, east = st.columns(5)
        _pad_button(west, "◀◀", "jog_x_major_left", ready, "X", -major_xy, "Major step, −X")
        _pad_button(minor_west, "◀", "jog_x_minor_left", ready, "X", -minor_xy, "Minor step, −X")
        if home.button(
            "⌂",
            key="jog_home",
            disabled=not ready,
            width="stretch",
            help="Home X and Y to the Ender 3 V3 SE endstops. Z stays where it is.",
        ):
            _home_stage()
        _pad_button(minor_east, "▶", "jog_x_minor_right", ready, "X", minor_xy, "Minor step, +X")
        _pad_button(east, "▶▶", "jog_x_major_right", ready, "X", major_xy, "Major step, +X")
        minor_down = st.columns(5)
        _pad_button(minor_down[2], "▼", "jog_y_minor_down", ready, "Y", -minor_xy, "Minor step, −Y")
        major_down = st.columns(5)
        _pad_button(major_down[2], "▼▼", "jog_y_major_down", ready, "Y", -major_xy, "Major step, −Y")
    with z_col:
        st.caption("Z")
        _pad_button(st, "▲▲", "jog_z_major_up", ready, "Z", major_z, "Major step, +Z")
        _pad_button(st, "▲", "jog_z_minor_up", ready, "Z", minor_z, "Minor step, +Z")
        _pad_button(st, "▼", "jog_z_minor_down", ready, "Z", -minor_z, "Minor step, −Z")
        _pad_button(st, "▼▼", "jog_z_major_down", ready, "Z", -major_z, "Major step, −Z")


def _render_stage_pad(locked: bool) -> None:
    """Minor and major steps around a home that centers X and Y."""
    info = printer_status()
    ready = info["online"] and not locked
    minor_xy = _jog_distance("jog_minor_xy_mm")
    major_xy = _jog_distance("jog_major_xy_mm")
    minor_z = _jog_distance("jog_minor_z_mm")
    major_z = _jog_distance("jog_major_z_mm")

    st.subheader("Manual movement")
    pad_col, field_col = st.columns([1.6, 1], vertical_alignment="center")
    with pad_col:
        _stage_cross(ready, minor_xy, major_xy, minor_z, major_z)
    with field_col:
        _jog_step_inputs(locked)
    last_jog = st.session_state.get("stage_last_jog")
    if last_jog:
        st.caption(last_jog)


def _home_stage() -> None:
    try:
        home_xy()
    except Exception as exc:
        _failure("The stage did not home.", "Check that the printer is still connected.", exc)
    else:
        st.session_state["stage_last_jog"] = (
            f"Homed X and Y to the endstops: X {HOME_X_MM:.2f} mm, Y {HOME_Y_MM:.2f} mm."
        )


def _jog_step_inputs(locked: bool) -> None:
    """Distances for the single-arrow and double-arrow buttons."""
    _jog_number("Minor XY (mm)", "jog_minor_xy_mm", locked)
    _jog_number("Major XY (mm)", "jog_major_xy_mm", locked)
    _jog_number("Minor Z (mm)", "jog_minor_z_mm", locked)
    _jog_number("Major Z (mm)", "jog_major_z_mm", locked)


def _jog_number(label: str, key: str, locked: bool) -> None:
    st.number_input(
        label,
        min_value=0.01,
        max_value=200.0,
        value=float(STAGE_WIDGET_DEFAULTS[key]),
        step=0.01,
        format="%.2f",
        key=key,
        disabled=locked,
    )


@st.fragment(run_every=0.5)
def _printer_reply() -> None:
    """Same confirmation as the Arduino: the device answered, so it is well connected."""
    if printer_status()["online"]:
        st.caption("Last reply: OK ALL")


def render_stage(cap) -> None:
    """Move the 3D-printer stage with the same serpentine as scan_40x60.gcode."""
    with st.container(border=True, key="block-stage"):
        st.header("Stage")

        info = printer_status()
        connected = info["connected"]
        if not _render_port_select(
            cap,
            label="Printer port",
            state_key="printer_port",
            disabled=connected,
        ):
            return

        refresh_col, connect_col, disconnect_col = st.columns(3)
        if refresh_col.button("Refresh ports", key="refresh_printer_port", width="stretch"):
            st.rerun()
        printer_port = st.session_state.get("printer_port")
        if connect_col.button(
            "Connect Printer",
            disabled=not printer_port or connected,
            width="stretch",
        ):
            if _LINK.get("port") and printer_port == _LINK.get("port") and _serial() is not None:
                st.error("That port is already the Arduino. Choose the other port for the printer.")
            else:
                try:
                    connect_printer(printer_port)
                except Exception as exc:
                    _failure(
                        "Could not open the printer.",
                        "Pick another port, or unplug it and refresh ports.",
                        exc,
                    )
                else:
                    st.rerun()
        if disconnect_col.button("Disconnect Printer", disabled=not connected, width="stretch"):
            disconnect_printer()
            st.rerun()
        _printer_reply()

        st.divider()
        scanning = (
            printer_status()["printing"]
            or autofocus_status()["running"]
            or scan_capture_status()["running"]
        )
        _render_stage_pad(scanning)
        st.divider()
        fov_col, overlap_col, step_col = st.columns(3)
        fov_col.number_input(
            "FOV (mm)",
            min_value=0.10,
            max_value=50.0,
            value=float(STAGE_WIDGET_DEFAULTS["stage_fov_mm"]),
            step=0.01,
            format="%.2f",
            key="stage_fov_mm",
            disabled=scanning,
            help="Width and height of one picture, in millimetres.",
        )
        overlap_col.number_input(
            "Overlap",
            min_value=0.0,
            max_value=0.90,
            value=float(STAGE_WIDGET_DEFAULTS["stage_overlap"]),
            step=0.01,
            format="%.2f",
            key="stage_overlap",
            disabled=scanning,
            help="Fraction of the field shared with the next picture. 0.10 overlaps 10%.",
        )
        step_col.metric("Step size", f"{_stage_step_mm():.2f} mm")
        x_col, row_col = st.columns(2)
        x_col.number_input(
            "Columns",
            min_value=1,
            max_value=80,
            value=int(STAGE_WIDGET_DEFAULTS["stage_columns"]),
            step=1,
            key="stage_columns",
            disabled=scanning,
            help="Stops along X. The stage steps one fewer time than this.",
        )
        row_col.number_input(
            "Rows",
            min_value=1,
            max_value=80,
            value=int(STAGE_WIDGET_DEFAULTS["stage_rows"]),
            step=1,
            key="stage_rows",
            disabled=scanning,
            help="Stops along Y. The stage steps one fewer time than this.",
        )
        st.number_input(
            "Moving speed (mm/min)",
            min_value=1,
            max_value=6000,
            value=int(STAGE_WIDGET_DEFAULTS["stage_feed"]),
            step=100,
            key="stage_feed",
            disabled=scanning,
        )

        summary = scan_summary(
            _stage_step_mm(),
            int(st.session_state["stage_columns"]),
            int(st.session_state["stage_rows"]),
            fov_mm=float(st.session_state["stage_fov_mm"]),
        )
        pause = _stage_pause()
        stops_col, area_col, dwell_col = st.columns(3)
        stops_col.metric("Stops", f"{summary['columns']} × {summary['rows']}")
        area_col.metric(
            "Scanned area",
            f"{summary['covered_x_mm']:.2f} × {summary['covered_y_mm']:.2f} mm",
        )
        dwell_col.metric("Pauze at each stop", f"{pause['dwell_ms']} ms")

        try:
            program = _stage_program()
        except ValueError as exc:
            _failure(
                "These scan settings cannot build a path.",
                "Change the field of view, overlap, columns, or rows.",
                exc,
            )
            return

        frame_word = "frame" if pause["frames"] == 1 else "frames"
        timing = (
            f"{pause['frames']} {frame_word} at {pause['shutter_ms']:.1f} ms shutter, "
            f"{pause['motion_ms']:.0f} ms after each move, "
            f"{pause['settle_ms']:.0f} ms light settle, "
            f"and {pause['margin_ms']} ms margin on each frame"
            + (
                f", plus {pause['gap_ms']:.0f} ms between sides."
                if pause["gaps"]
                else "."
            )
        )
        with st.expander("G-code based on current settings"):
            st.caption(timing)
            st.code(program, language="text")
            st.download_button(
                "Download gcode",
                data=program,
                file_name="scanwafer.gcode",
                mime="text/plain",
                disabled=scanning,
                key="stage_download_gcode",
            )

    _render_focus(scanning)
    with st.container(border=True, key="block-scan"):
        st.header("Scan")
        _stage_progress()
