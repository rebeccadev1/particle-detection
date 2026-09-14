"""Streamlit tabs for labeling crops and browsing collections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import cfg_get, resolve_output_dir
from src.io.tile_loader import DEFAULT_FILENAME_PATTERN, matching_tile_paths
from src.labeling.crops import (
    TileImageCache,
    crop_particle,
    find_tile_path,
    local_px_to_global_nm,
    placement_for_tile,
)
from src.labeling.audit_misses import format_trace_caption, input_tile_names, trace_click
from src.labeling.inspect import (
    CELL_NAMES,
    DEFAULT_MISS_SIZE_UM,
    GRID_N,
    INSPECT_DISPLAY_WIDTH,
    OVERVIEW_MAX_SIDE,
    ZOOM_MAX_SIDE,
    candidate_placements,
    cell_bounds,
    circles_from_rows,
    detections_on_tile,
    display_xy_to_local,
    label_color_for_key,
    labeled_particle_recovery,
    last_run_class_counts,
    local_xy_on_tile,
    missed_particle_record,
    overlay_view,
    preview_click_crop,
    tile_names_for_hits,
)
from src.labeling.queue import (
    last_run_csv,
    load_last_pipeline,
    load_particles_csv,
    merge_detection_tables,
    tag_table,
    unlabeled_queue,
)
from src.labeling.store import LabelStore
from src.stitching.stitcher import LazyMosaic

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # pragma: no cover - UI-only optional
    streamlit_image_coordinates = None

NM_PER_UM = 1000.0
GALLERY_PAGE_SIZE = 24
LABEL_IMAGE_WIDTH = 420
GALLERY_IMAGE_WIDTH = 180


def labels_store(project_root: str | Path) -> LabelStore:
    return LabelStore(Path(project_root) / "labels")


def render_label_tab(
    config: dict[str, Any],
    project_root: str | Path,
    table: pd.DataFrame | None,
    mosaic: LazyMosaic | None,
) -> None:
    """One unlabeled crop at a time, labeled with arrow-key buttons."""
    store = labels_store(project_root)
    cache = st.session_state.setdefault("label_tile_cache", TileImageCache())
    origin_cache = st.session_state.setdefault("label_origin_cache", {})

    st.subheader("Label detections")
    st.caption(
        "Right = particle · Down = not sure · Left = not a particle · Up = undo. "
        "Load last run queues Detection → Run pipeline output "
        "(the latest output folder), skipping already-labeled keys."
    )

    pipeline_csv, pipeline_config = load_last_pipeline(config)
    st.button(
        "Load last run",
        key="label_load_last_run",
        disabled=pipeline_csv is None or not Path(pipeline_csv).is_file(),
        on_click=_on_load_last_pipeline,
        help=(
            f"Queue {pipeline_csv.parent.name}/particles.csv"
            if pipeline_csv is not None and Path(pipeline_csv).is_file()
            else "Run Detection first so last_pipeline.json points at an output folder."
        ),
    )

    use_last_pipeline = st.session_state.get("label_source") == "last_pipeline"
    if use_last_pipeline:
        last_csv = pipeline_csv
        crop_config = pipeline_config
        tables = _last_pipeline_tables(pipeline_csv)
    else:
        last_csv = last_run_csv(project_root, config)
        crop_config = config
        tables = _detection_tables(project_root, table, config)
    n_hits = sum(len(frame) for frame in tables)
    queue = unlabeled_queue(tables, store.labeled_keys())
    front = st.session_state.get("label_front_key")
    if front is not None and not queue.empty:
        match = queue["key"].astype(str) == str(front)
        if match.any():
            queue = pd.concat([queue.loc[match], queue.loc[~match]], ignore_index=True)

    counts = store.counts()
    labeled_n = sum(counts.values())
    remaining = len(queue)
    source_label = _source_caption(last_csv, last_pipeline=use_last_pipeline)
    st.caption(
        f"{remaining} remaining · labeled {labeled_n} "
        f"(particles {counts['particle']} · not sure {counts['not_sure']} · "
        f"not particles {counts['not_particle']}) · source {source_label}"
    )
    if st.session_state.pop("label_source_notice", False):
        folder = last_csv.parent.name if last_csv is not None else "none"
        st.info(
            f"Loaded **{n_hits}** detections from **{folder}**. "
            f"**{remaining}** unlabeled at 10 µm and above "
            f"(already-labeled keys are skipped)."
        )

    if queue.empty:
        st.success("Nothing left to label at 10 µm and above.")
        _undo_row(store, counts)
        return

    current = queue.iloc[0]
    try:
        crop = crop_particle(
            current,
            crop_config,
            mosaic=mosaic,
            cache=cache,
            origin_cache=origin_cache,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(str(exc))
        _undo_row(store, counts)
        return

    size_um = float(current["size"]) / NM_PER_UM
    image_col, _ = st.columns([LABEL_IMAGE_WIDTH, 800])
    image_col.image(
        crop.rgb,
        caption=(
            f"id {current.get('id', '')} · {current['source_tile']} · "
            f"{size_um:.2f} µm · confidence {float(current.get('confidence', 0.0)):.3f}"
        ),
        width=LABEL_IMAGE_WIDTH,
    )
    st.caption(f"Reviewing 1 of {remaining} remaining in the unlabeled queue.")

    cols = st.columns(4)
    not_particle = cols[0].button(
        "Not a particle", shortcut="Left", width="stretch", key="label_left"
    )
    not_sure = cols[1].button(
        "Not sure", shortcut="Down", width="stretch", key="label_down"
    )
    is_particle = cols[2].button(
        "Particle", shortcut="Right", type="primary", width="stretch", key="label_right"
    )
    undo = cols[3].button(
        "Undo",
        shortcut="Up",
        width="stretch",
        key="label_undo",
        disabled=labeled_n == 0,
    )

    if is_particle:
        store.apply_label(current, "particle", crop.rgb)
        st.session_state.pop("label_front_key", None)
        st.rerun()
    if not_sure:
        store.apply_label(current, "not_sure", crop.rgb)
        st.session_state.pop("label_front_key", None)
        st.rerun()
    if not_particle:
        store.apply_label(current, "not_particle", crop.rgb)
        st.session_state.pop("label_front_key", None)
        st.rerun()
    if undo:
        restored = store.undo_last()
        if restored is not None:
            st.session_state["label_front_key"] = str(restored["key"])
        st.rerun()


def _on_load_last_pipeline() -> None:
    """Switch Tinder to Detection's last output folder (runs before widgets)."""
    st.session_state["label_source"] = "last_pipeline"
    st.session_state["label_source_notice"] = True
    st.session_state.pop("label_front_key", None)


def _advance_inspect_tile(delta: int) -> None:
    """Move to the previous/next TIFF. Must run in a button callback (before widgets)."""
    names = list(st.session_state.get("_inspect_tile_names") or [])
    if not names:
        return
    current = st.session_state.get("inspect_tile")
    index = names.index(current) if current in names else 0
    st.session_state["inspect_tile"] = names[(index + delta) % len(names)]
    st.session_state["inspect_cell"] = None
    st.session_state.pop("inspect_pending", None)
    st.session_state.pop("inspect_click_sig", None)


def _on_inspect_tile_selected() -> None:
    st.session_state["inspect_cell"] = None
    st.session_state.pop("inspect_pending", None)
    st.session_state.pop("inspect_click_sig", None)


def _set_inspect_cell(row: int, col: int) -> None:
    current = st.session_state.get("inspect_cell")
    st.session_state["inspect_cell"] = None if current == (row, col) else (row, col)
    st.session_state.pop("inspect_pending", None)
    st.session_state.pop("inspect_click_sig", None)


def _clear_inspect_cell() -> None:
    st.session_state["inspect_cell"] = None
    st.session_state.pop("inspect_pending", None)
    st.session_state.pop("inspect_click_sig", None)


def render_tile_inspect_tab(
    config: dict[str, Any],
    project_root: str | Path,
    table: pd.DataFrame | None,
    mosaic: LazyMosaic | None,
) -> None:
    """Walk the input folder one TIFF at a time, with detection circles overlaid."""
    st.subheader("Inspect tiles")
    st.caption(
        "Green = labeled particle · red = labeled not-particle · blue = not sure · "
        "orange = unlabeled detection. Zoom a 3×3 cell, click an unmarked speck, then "
        "Mark missed particle. Left/Right change tile."
    )

    folder = str(cfg_get(config, "input_dir", "") or "")
    if not folder:
        st.info("Set a tile folder in the sidebar.")
        return
    pattern = str(cfg_get(config, "filename_pattern", DEFAULT_FILENAME_PATTERN))
    try:
        paths = matching_tile_paths(
            folder,
            pattern,
            run=cfg_get(config, "run", None),
            magnification=cfg_get(config, "magnification", None),
        )
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return
    if not paths:
        st.warning(f"No matching TIFFs in {folder}.")
        return

    names = [path.name for path in paths]
    st.session_state["_inspect_tile_names"] = names
    if st.session_state.get("inspect_tile") not in names:
        st.session_state["inspect_tile"] = names[0]
        st.session_state["inspect_cell"] = None
        st.session_state.pop("inspect_pending", None)
        st.session_state.pop("inspect_click_sig", None)
    nav = st.columns([1, 4, 1])
    nav[0].button(
        "Previous",
        shortcut="Left",
        width="stretch",
        key="inspect_prev",
        on_click=_advance_inspect_tile,
        args=(-1,),
    )
    nav[1].selectbox(
        "Tile",
        names,
        key="inspect_tile",
        on_change=_on_inspect_tile_selected,
    )
    nav[2].button(
        "Next",
        shortcut="Right",
        width="stretch",
        key="inspect_next",
        on_click=_advance_inspect_tile,
        args=(1,),
    )

    tile_name = str(st.session_state["inspect_tile"])
    tile_path = next(path for path in paths if path.name == tile_name)
    index = names.index(tile_name)
    st.caption(f"Tile {index + 1} of {len(names)} · {tile_path}")

    store = labels_store(project_root)
    labels_df = store.load()
    label_map: dict[str, str] = {}
    if not labels_df.empty:
        for record in labels_df.to_dict(orient="records"):
            label_map[str(record["key"])] = str(record["label"])

    detections = merge_detection_tables(_detection_tables(project_root, table, config))
    tile_hits = detections_on_tile(detections, tile_name)
    extra = _label_rows_missing_from_hits(labels_df, tile_name, tile_hits)
    if extra is not None and not extra.empty:
        tile_hits = pd.concat([tile_hits, extra], ignore_index=True)

    keys = (
        tile_hits["key"].astype(str).tolist()
        if not tile_hits.empty and "key" in tile_hits.columns
        else []
    )
    n_particle = sum(label_map.get(key) == "particle" for key in keys)
    n_fp = sum(label_map.get(key) == "not_particle" for key in keys)
    n_unsure = sum(label_map.get(key) == "not_sure" for key in keys)
    n_open = len(keys) - n_particle - n_fp - n_unsure
    st.caption(
        f"{len(tile_hits)} circles on this tile · labeled particles {n_particle} · "
        f"not particles {n_fp} · unlabeled {n_open}"
    )

    cache = st.session_state.setdefault("label_tile_cache", TileImageCache())
    origin_cache = st.session_state.setdefault("label_origin_cache", {})
    try:
        with st.spinner("Loading tile…"):
            image = cache.load(tile_path)
            placement = placement_for_tile(
                tile_path, config, mosaic=mosaic, origin_cache=origin_cache
            )
            placements = candidate_placements(
                tile_path, config, mosaic=mosaic, origin_cache=origin_cache
            )
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(str(exc))
        return

    pixel_size = float(cfg_get(config, "pixel_size_nm", 960.0))
    height, width = int(image.shape[0]), int(image.shape[1])
    x_locals: list[float] = []
    y_locals: list[float] = []
    if not tile_hits.empty:
        for _, row in tile_hits.iterrows():
            x_local, y_local = local_xy_on_tile(
                float(row["x_global"]),
                float(row["y_global"]),
                placements,
                pixel_size,
                width,
                height,
            )
            x_locals.append(x_local)
            y_locals.append(y_local)
    circles = circles_from_rows(tile_hits, x_locals, y_locals, label_map)

    st.write("Zoom cell")
    selected = st.session_state.get("inspect_cell")
    for row_i in range(GRID_N):
        grid_cols = st.columns(GRID_N)
        for col_i in range(GRID_N):
            name = CELL_NAMES[row_i][col_i]
            grid_cols[col_i].button(
                name,
                width="stretch",
                type="primary" if selected == (row_i, col_i) else "secondary",
                key=f"inspect_cell_{row_i}_{col_i}",
                on_click=_set_inspect_cell,
                args=(row_i, col_i),
            )
    st.button(
        "Whole tile",
        key="inspect_cell_clear",
        on_click=_clear_inspect_cell,
    )

    crop = None
    max_side = OVERVIEW_MAX_SIDE
    caption = f"{tile_name} overview"
    if selected is not None:
        crop = cell_bounds(int(image.shape[0]), int(image.shape[1]), selected[0], selected[1])
        max_side = ZOOM_MAX_SIDE
        caption = (
            f"{tile_name} · {CELL_NAMES[selected[0]][selected[1]]} "
            f"({crop[3] - crop[1]}×{crop[2] - crop[0]} px)"
        )
    view = overlay_view(
        image, circles, pixel_size, crop=crop, max_side=max_side
    )
    display_w = min(
        INSPECT_DISPLAY_WIDTH,
        int(view.rgb.shape[1]) if view.rgb.size else INSPECT_DISPLAY_WIDTH,
    )
    if view.rgb.size == 0:
        st.error("No pixels to show for this view. Try Whole tile, then a cell again.")
        click = None
    else:
        st.caption(caption + " · click an unmarked speck")
        st.image(view.rgb, width=display_w)
        click = None
        if streamlit_image_coordinates is not None:
            with st.expander("Click to mark a miss", expanded=False):
                click = streamlit_image_coordinates(
                    view.rgb,
                    key="inspect_click_overlay",
                    width=display_w,
                    image_format="JPEG",
                    jpeg_quality=85,
                    cursor="crosshair",
                )
    if click:
        disp_scale = display_w / max(int(view.rgb.shape[1]), 1)
        sig = (int(click["x"]), int(click["y"]), tile_name, selected)
        if st.session_state.get("inspect_click_sig") != sig:
            x_local, y_local = display_xy_to_local(
                float(click["x"]) / disp_scale,
                float(click["y"]) / disp_scale,
                view,
            )
            st.session_state["inspect_click_sig"] = sig
            st.session_state["inspect_pending"] = {
                "x_local": x_local,
                "y_local": y_local,
                "tile": tile_name,
            }
            st.rerun()

    pending = st.session_state.get("inspect_pending")
    if pending and pending.get("tile") == tile_name:
        size_um = float(
            st.number_input(
                "Assumed size (µm)",
                min_value=10.0,
                max_value=100.0,
                value=DEFAULT_MISS_SIZE_UM,
                step=1.0,
                key="inspect_miss_size_um",
                help="Missed flakes have no DoG size. 15 µm is the audit target.",
            )
        )
        size_nm = size_um * NM_PER_UM
        preview = preview_click_crop(
            image,
            float(pending["x_local"]),
            float(pending["y_local"]),
            size_nm,
            pixel_size,
        )
        st.image(preview, caption="Pending mark", width=240)
        trace = _pending_mark_trace(
            image,
            config,
            tile_name,
            float(pending["y_local"]),
            float(pending["x_local"]),
        )
        st.caption(format_trace_caption(trace))
        mark_cols = st.columns(2)
        if mark_cols[0].button(
            "Mark missed particle",
            type="primary",
            width="stretch",
            key="inspect_mark",
        ):
            x_global, y_global = local_px_to_global_nm(
                float(pending["x_local"]),
                float(pending["y_local"]),
                placement,
                pixel_size,
            )
            record = missed_particle_record(tile_name, x_global, y_global, size_nm)
            store.apply_label(record, "particle", preview)
            st.session_state.pop("inspect_pending", None)
            st.session_state.pop("inspect_click_sig", None)
            st.rerun()
        if mark_cols[1].button("Cancel click", width="stretch", key="inspect_cancel"):
            st.session_state.pop("inspect_pending", None)
            st.session_state.pop("inspect_click_sig", None)
            st.rerun()

    if not tile_hits.empty:
        display = tile_hits.copy()
        display["size_um"] = display["size"].astype(float) / NM_PER_UM
        display["label"] = display["key"].astype(str).map(lambda key: label_map.get(key, ""))
        keep = [col for col in ("id", "size_um", "confidence", "label", "key") if col in display.columns]
        st.dataframe(display[keep], width="stretch", hide_index=True)


def _pending_mark_trace(
    image: Any,
    config: dict[str, Any],
    tile_name: str,
    y_local: float,
    x_local: float,
) -> dict[str, Any]:
    """Cached ``trace_click`` for the Tiles pending-mark panel."""
    det = config.get("detection") or {}
    fingerprint = (
        tile_name,
        bool(det.get("recall_mode", False)),
        float(det.get("min_confidence", 0.0) or 0.0),
        float(det.get("min_circularity", 0.0) or 0.0),
        float(det.get("edge_exclude_px", 0.0) or 0.0),
        float(det.get("min_prominence", 0.0) or 0.0),
        int(det.get("structure_min_neighbors", 2) or 0),
        float(cfg_get(config, "detection.min_size_nm", 0.0) or 0.0),
    )
    store = st.session_state.setdefault("_inspect_trace_cache", {})
    cached = store.get("fingerprint")
    traces = store.get("traces")
    if cached != fingerprint or not isinstance(traces, dict):
        traces = {}
        store["fingerprint"] = fingerprint
        store["traces"] = traces
    key = (round(float(x_local), 1), round(float(y_local), 1))
    if key not in traces:
        traces[key] = trace_click(image, config, y_local, x_local)
    return dict(traces[key])


def _advance_labeled_review_tile(delta: int) -> None:
    names = list(st.session_state.get("_labeled_review_tile_names") or [])
    if not names:
        return
    current = st.session_state.get("labeled_review_tile")
    index = names.index(current) if current in names else 0
    st.session_state["labeled_review_tile"] = names[(index + delta) % len(names)]


def _last_run_stat_card(label: str, value: int, color: str, wash: str, hint: str) -> str:
    return (
        "<div style='flex:1;min-width:9.5rem;padding:0.9rem 1rem;border-radius:12px;"
        f"background:{wash};border:1px solid {color}66;'>"
        f"<div style='font-size:0.78rem;font-weight:650;letter-spacing:0.04em;"
        f"text-transform:uppercase;color:{color};'>{label}</div>"
        f"<div style='font-size:1.9rem;font-weight:750;line-height:1.15;"
        f"font-variant-numeric:tabular-nums;color:{color};'>{value:,}</div>"
        f"<div style='font-size:0.78rem;opacity:0.78;margin-top:0.15rem;'>{hint}</div>"
        "</div>"
    )


def _render_last_run_summary(counts: dict[str, int]) -> None:
    cards = [
        _last_run_stat_card(
            "Detected", counts["n_detected"], "#5b6472", "#5b647214", "all last-run hits"
        ),
        _last_run_stat_card(
            "Real of detected",
            counts["n_real"],
            "#1f9d3a",
            "#28b44622",
            "green circles",
        ),
        _last_run_stat_card(
            "Fake of detected",
            counts["n_fake"],
            "#d61f1f",
            "#e3262622",
            "red circles",
        ),
        _last_run_stat_card(
            "Total real",
            counts["n_real_total"],
            "#187a2e",
            "#28b44618",
            "green circles only",
        ),
        _last_run_stat_card(
            "Undetected",
            counts["n_undetected"],
            "#c56a00",
            "#d9770618",
            "inspect marks missed",
        ),
    ]
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;gap:0.7rem;margin:0.35rem 0 0.85rem 0;'>"
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )
    extra = []
    if counts["n_unlabeled"]:
        extra.append(f"{counts['n_unlabeled']:,} unlabeled (orange)")
    if counts["n_not_sure"]:
        extra.append(f"{counts['n_not_sure']:,} not sure (blue)")
    if extra:
        st.caption(" · ".join(extra) + ".")


def render_labeled_tiles_tab(
    config: dict[str, Any],
    project_root: str | Path,
    mosaic: LazyMosaic | None,
    table: pd.DataFrame | None = None,
) -> None:
    """Latest pipeline run: tiles with detector circles and zoomed crops."""
    st.subheader("Last pipeline run")
    st.caption(
        "Always the latest Detection → Run pipeline result "
        "(not the labeling recall table). "
        "Orange = unlabeled detector hit · green = labeled particle · "
        "red = not-particle · blue = not sure."
    )
    run_config = dict(config)
    session_config = st.session_state.get("config")
    if isinstance(session_config, dict):
        run_config.update(session_config)
    last_csv: Path | None = None
    detections: pd.DataFrame | None = None
    if table is not None:
        last_csv_text = st.session_state.get("last_pipeline_csv")
        last_csv = Path(str(last_csv_text)) if last_csv_text else None
        detections = tag_table(table, last_csv)
    if detections is None:
        last_csv, run_config = load_last_pipeline(run_config)
        if last_csv is not None and last_csv.is_file():
            try:
                detections = load_particles_csv(last_csv)
            except (ValueError, OSError) as exc:
                st.error(str(exc))
                return
    if detections is None:
        st.info("Run the pipeline on the Detection tab. This tab follows that run.")
        return

    store = labels_store(project_root)
    labels_df = store.load()
    run_tiles = input_tile_names(run_config)
    recovery_tiles = sorted(run_tiles) if run_tiles else tile_names_for_hits(detections)
    counts = last_run_class_counts(
        detections, labels_df, tile_names=recovery_tiles or None
    )
    _render_last_run_summary(counts)

    names = tile_names_for_hits(detections)
    if not names:
        st.info("The last pipeline run has no detections.")
        return

    label_map: dict[str, str] = {}
    if not labels_df.empty:
        for record in labels_df.to_dict(orient="records"):
            label_map[str(record["key"])] = str(record["label"])

    st.session_state["_labeled_review_tile_names"] = names
    if st.session_state.get("labeled_review_tile") not in names:
        st.session_state["labeled_review_tile"] = names[0]
    nav = st.columns([1, 4, 1])
    nav[0].button(
        "Previous",
        width="stretch",
        key="labeled_review_prev",
        on_click=_advance_labeled_review_tile,
        args=(-1,),
    )
    nav[1].selectbox("Tile", names, key="labeled_review_tile")
    nav[2].button(
        "Next",
        width="stretch",
        key="labeled_review_next",
        on_click=_advance_labeled_review_tile,
        args=(1,),
    )

    tile_name = str(st.session_state["labeled_review_tile"])
    tile_hits = detections_on_tile(detections, tile_name)
    index = names.index(tile_name)
    n_labeled = sum(str(key) in label_map for key in tile_hits["key"].astype(str)) if not tile_hits.empty else 0
    source = last_csv if last_csv is not None else "current session"
    st.caption(
        f"Tile {index + 1} of {len(names)} · {len(tile_hits)} detector hits "
        f"({n_labeled} already labeled) · source {source}"
    )

    cache = st.session_state.setdefault("label_tile_cache", TileImageCache())
    origin_cache = st.session_state.setdefault("label_origin_cache", {})
    try:
        tile_path = find_tile_path(
            tile_name, input_dir=cfg_get(run_config, "input_dir", None), mosaic=mosaic
        )
        image = cache.load(tile_path)
        placements = candidate_placements(
            tile_path, run_config, mosaic=mosaic, origin_cache=origin_cache
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(str(exc))
        return

    pixel_size = float(cfg_get(run_config, "pixel_size_nm", 960.0))
    height, width = int(image.shape[0]), int(image.shape[1])
    x_locals: list[float] = []
    y_locals: list[float] = []
    if not tile_hits.empty:
        for _, row in tile_hits.iterrows():
            x_local, y_local = local_xy_on_tile(
                float(row["x_global"]),
                float(row["y_global"]),
                placements,
                pixel_size,
                width,
                height,
            )
            x_locals.append(x_local)
            y_locals.append(y_local)
    circles = circles_from_rows(tile_hits, x_locals, y_locals, label_map)
    view = overlay_view(image, circles, pixel_size, crop=None, max_side=OVERVIEW_MAX_SIDE)
    if view.rgb.size == 0:
        st.error("No pixels to show for this tile.")
    else:
        st.image(view.rgb, caption=f"{tile_name} · detector hits", width="stretch")

    st.subheader("Zoomed detector hits")
    if tile_hits.empty:
        st.info("No detector hits on this tile.")
        return
    n_pages = max(1, (len(tile_hits) + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
    page = int(
        st.number_input(
            "Page",
            min_value=1,
            max_value=n_pages,
            key=f"labeled_review_gallery_{tile_name}",
        )
    )
    start = (page - 1) * GALLERY_PAGE_SIZE
    end = start + GALLERY_PAGE_SIZE
    columns = st.columns(3)
    records = tile_hits.to_dict(orient="records")
    for offset, record in enumerate(records[start:end]):
        with columns[offset % 3]:
            x_local = x_locals[start + offset]
            y_local = y_locals[start + offset]
            key = str(record.get("key", ""))
            color = label_color_for_key(key, label_map)
            crop = preview_click_crop(
                image,
                x_local,
                y_local,
                float(record["size"]),
                pixel_size,
                color=color,
            )
            size_um = float(record["size"]) / NM_PER_UM
            status = label_map.get(key, "unlabeled")
            caption = (
                f"id {record.get('id', '')} · {size_um:.1f} µm · {status}"
            )
            st.image(crop, caption=caption, width="stretch")


def _label_rows_missing_from_hits(
    labels_df: pd.DataFrame,
    tile_name: str,
    hits: pd.DataFrame,
) -> pd.DataFrame | None:
    """Labeled marks on this tile that are not in the current detection CSV."""
    if labels_df is None or labels_df.empty:
        return None
    names = labels_df["source_tile"].astype(str).map(lambda path: str(path).replace("\\", "/").rsplit("/", 1)[-1])
    on_tile = labels_df.loc[names == str(tile_name)].copy()
    if on_tile.empty:
        return None
    known = set()
    if hits is not None and not hits.empty and "key" in hits.columns:
        known = set(hits["key"].astype(str))
    missing = on_tile.loc[~on_tile["key"].astype(str).isin(known)]
    if missing.empty:
        return None
    return missing.reset_index(drop=True)


def render_collection_tab(
    project_root: str | Path,
    label: str,
    title: str,
    n_columns: int = 4,
) -> None:
    """Thumbnail gallery for one label bucket."""
    store = labels_store(project_root)
    df = store.by_label(label)
    st.subheader(title)
    st.caption(f"{len(df)} crops")
    if df.empty:
        st.info("No crops in this collection yet.")
        return

    display = df.copy()
    display.insert(
        display.columns.get_loc("size") + 1,
        "size_um",
        display["size"].astype(float) / NM_PER_UM,
    )
    st.dataframe(
        display.drop(columns=["crop_path"], errors="ignore"),
        width="stretch",
        hide_index=True,
    )

    n_pages = max(1, (len(df) + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
    page = int(
        st.number_input(
            "Page",
            min_value=1,
            max_value=n_pages,
            key=f"gallery_page_{label}",
        )
    )
    start = (page - 1) * GALLERY_PAGE_SIZE
    chunk = df.iloc[start : start + GALLERY_PAGE_SIZE]
    n_columns = max(1, int(n_columns))
    columns = st.columns(n_columns)
    for index, record in enumerate(chunk.to_dict(orient="records")):
        with columns[index % n_columns]:
            path = Path(str(record["crop_path"]))
            caption = (
                f"id {record.get('particle_id', '')} · "
                f"{float(record['size']) / NM_PER_UM:.1f} µm"
            )
            if path.is_file():
                st.image(str(path), caption=caption, width=GALLERY_IMAGE_WIDTH)
            else:
                st.caption(f"Missing file: {path.name}")
            if st.button(
                "Send back to queue",
                key=f"unlabel_{label}_{record['key']}",
                width="stretch",
            ):
                store.unlabel(str(record["key"]))
                st.session_state["label_front_key"] = str(record["key"])
                st.rerun()


def _source_caption(path: Path | None, *, last_pipeline: bool) -> str:
    if path is None:
        return "none"
    folder = path.parent.name or str(path.parent)
    name = path.name
    if last_pipeline:
        return f"{folder}/{name} (last output)"
    return str(path.name if path.is_file() else path)


def _last_pipeline_tables(csv_path: Path | None) -> list[pd.DataFrame]:
    if csv_path is None or not Path(csv_path).is_file():
        return []
    return [load_particles_csv(csv_path)]


def _detection_tables(
    project_root: str | Path,
    table: pd.DataFrame | None,
    config: dict[str, Any],
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    last_csv = last_run_csv(project_root, config)
    if last_csv.is_file():
        frames.append(load_particles_csv(last_csv))
    out = resolve_output_dir(config.get("output_dir")) / "particles.csv"
    if out.is_file() and Path(out).resolve() != Path(last_csv).resolve():
        frames.append(load_particles_csv(out))
    if table is not None and not table.empty:
        frames.append(tag_table(table, out))
    return frames


def _undo_row(store: LabelStore, counts: dict[str, int]) -> None:
    labeled_n = sum(counts.values())
    if st.button(
        "Undo",
        shortcut="Up",
        key="label_undo_empty",
        disabled=labeled_n == 0,
    ):
        restored = store.undo_last()
        if restored is not None:
            st.session_state["label_front_key"] = str(restored["key"])
        st.rerun()
