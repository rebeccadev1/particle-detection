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
    crop_direction_views,
    crop_particle,
    find_tile_path,
    local_px_to_global_nm,
    placement_for_tile,
)
from src.labeling.audit_misses import input_tile_names
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
    expand_direction_tile_names,
    unhappiness_score,
    label_color_for_key,
    labeled_particle_recovery,
    last_run_class_counts,
    last_run_class_counts_by_tile,
    local_xy_on_tile,
    missed_particle_record,
    overlay_view,
    preview_click_crop,
    related_scene_tile_names,
    tile_names_for_hits,
    undetected_size_floor_nm,
)
from src.labeling.queue import (
    expand_labeled_keys,
    last_run_csv,
    load_last_pipeline,
    load_particles_csv,
    label_map_with_aliases,
    merge_detection_tables,
    tag_table,
    unlabeled_queue,
    snap_detection_keys_to_labels,
)
from src.measurement.measurer import (
    DEFAULT_NSEW_MERGE_RADIUS_PX,
    DEFAULT_NSEW_SIZE_MATCH_FRACTION,
    is_nsew_family_tile,
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
LAST_RUN_OVERVIEW_WIDTH = 720
LAST_RUN_CROP_COLUMNS = 4


def labels_store(project_root: str | Path) -> LabelStore:
    return LabelStore(Path(project_root) / "labels")


def _nsew_merge_radius_nm(config: dict[str, Any]) -> float:
    pixel_size = float(cfg_get(config, "pixel_size_nm", 960.0))
    merge_px = float(
        cfg_get(
            config,
            "measurement.nsew_merge_radius_px",
            DEFAULT_NSEW_MERGE_RADIUS_PX,
        )
    )
    return merge_px * pixel_size


def _nsew_size_match_fraction(config: dict[str, Any]) -> float:
    return float(
        cfg_get(
            config,
            "measurement.nsew_size_match_fraction",
            DEFAULT_NSEW_SIZE_MATCH_FRACTION,
        )
    )


def _labels_for_scene(
    labels_df: pd.DataFrame,
    config: dict[str, Any],
    extra_tiles: list[str] | None = None,
) -> pd.DataFrame:
    if labels_df is None or labels_df.empty:
        return labels_df
    scene = related_scene_tile_names(
        cfg_get(config, "input_dir", None), extra_tiles
    )
    if not scene:
        return labels_df
    tile = labels_df["source_tile"].astype(str).map(
        lambda path: str(path).replace("\\", "/").rsplit("/", 1)[-1]
    )
    return labels_df.loc[tile.isin(scene)].copy()


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
        "(the latest output folder), skipping already-labeled keys. "
        "Labeling N, W, E, S, or a particles-only version labels the same location on all of them."
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
    radius_nm = _nsew_merge_radius_nm(crop_config)
    size_frac = _nsew_size_match_fraction(crop_config)
    scene_labels = _labels_for_scene(store.load(), crop_config)
    queue = unlabeled_queue(
        tables,
        scene_labels["key"] if scene_labels is not None and not scene_labels.empty else store.labeled_keys(),
        nsew_merge_radius_nm=radius_nm,
        nsew_size_match_fraction=size_frac,
        labels=scene_labels,
    )
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
        views = crop_direction_views(
            current,
            crop_config,
            mosaic=mosaic,
            cache=cache,
            origin_cache=origin_cache,
        )
        if views:
            crop = views.get(
                Path(str(current["source_tile"])).stem.upper(),
                next(iter(views.values())),
            )
        else:
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
    if views:
        cols = st.columns(len(views))
        for column, (direction, view) in zip(cols, views.items()):
            column.image(
                view.rgb,
                caption=f"{direction} · {view.tile_path.name}",
                width="stretch",
            )
        dirs = ", ".join(views)
        st.caption(
            f"id {current.get('id', '')} · {size_um:.2f} µm · "
            f"confidence {float(current.get('confidence', 0.0)):.3f} · "
            f"same location on {dirs} (one label for all)"
        )
    else:
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
    """Move to the previous/next tile. Must run in a button callback (before widgets)."""
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


def _last_pipeline_view(
    config: dict[str, Any],
    table: pd.DataFrame | None,
) -> tuple[dict[str, Any], Path | None, pd.DataFrame | None]:
    """Folder, CSV, and detections from last_pipeline.json (not the sidebar)."""
    last_csv, run_config = load_last_pipeline(dict(config))
    detections: pd.DataFrame | None = None
    if last_csv is not None and last_csv.is_file():
        detections = load_particles_csv(last_csv)
    elif table is not None:
        detections = tag_table(table, last_csv)
    return run_config, last_csv, detections


def render_tile_inspect_tab(
    config: dict[str, Any],
    project_root: str | Path,
    table: pd.DataFrame | None,
    mosaic: LazyMosaic | None,
) -> None:
    """Walk the last-pipeline tile folder, with detection and label circles."""
    st.subheader("Inspect tiles")
    st.caption(
        "Green = labeled particle · red = labeled not-particle · blue = not sure · "
        "orange = unlabeled detection. Zoom a 3×3 cell, click an unmarked speck, then "
        "Mark missed particle. Left/Right change tile. "
        "Same map as Last Run (not the sidebar Tile folder). "
        "A label on N, S, E, or W is applied to all four."
    )

    try:
        run_config, last_csv, detections = _last_pipeline_view(config, table)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    folder = str(cfg_get(run_config, "input_dir", "") or "")
    if not folder:
        st.info("Run the pipeline, or set a tile folder in the sidebar.")
        return
    pattern = str(cfg_get(run_config, "filename_pattern", DEFAULT_FILENAME_PATTERN))
    source = last_csv if last_csv is not None else "sidebar"
    st.caption(f"Map {folder} · detections {source}")
    inspect_mosaic = mosaic
    if str(cfg_get(config, "input_dir", "") or "") != folder:
        inspect_mosaic = None
    try:
        paths = matching_tile_paths(
            folder,
            pattern,
            run=cfg_get(run_config, "run", None),
            magnification=cfg_get(run_config, "magnification", None),
            include_unmatched=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return
    if not paths:
        st.warning(f"No matching tiles in {folder}.")
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
    scene_labels = _labels_for_scene(labels_df, config, names)
    label_map = label_map_with_aliases(scene_labels)

    if detections is None:
        detections = pd.DataFrame()
    else:
        detections = snap_detection_keys_to_labels(
            detections,
            scene_labels,
            _nsew_merge_radius_nm(config),
            _nsew_size_match_fraction(config),
        )
    tile_hits = detections_on_tile(detections, tile_name)
    extra = _label_rows_missing_from_hits(
        scene_labels, tile_name, tile_hits, folder_tiles=list(
            related_scene_tile_names(cfg_get(config, "input_dir", None), names) or names
        )
    )
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
                tile_path, run_config, mosaic=inspect_mosaic, origin_cache=origin_cache
            )
            placements = candidate_placements(
                tile_path, run_config, mosaic=inspect_mosaic, origin_cache=origin_cache
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


def _advance_labeled_review_tile(delta: int) -> None:
    names = list(st.session_state.get("_labeled_review_tile_names") or [])
    if not names:
        return
    current = st.session_state.get("labeled_review_tile")
    index = names.index(current) if current in names else 0
    st.session_state["labeled_review_tile"] = names[(index + delta) % len(names)]


def _last_run_stat_card(label: str, value: int | str, color: str, wash: str, hint: str) -> str:
    shown = f"{value:,}" if isinstance(value, int) else value
    return (
        "<div style='flex:1;min-width:9.5rem;padding:0.9rem 1rem;border-radius:12px;"
        f"background:{wash};border:1px solid {color}66;'>"
        f"<div style='font-size:0.78rem;font-weight:650;letter-spacing:0.04em;"
        f"text-transform:uppercase;color:{color};'>{label}</div>"
        f"<div style='font-size:1.9rem;font-weight:750;line-height:1.15;"
        f"font-variant-numeric:tabular-nums;color:{color};'>{shown}</div>"
        f"<div style='font-size:0.78rem;opacity:0.78;margin-top:0.15rem;'>{hint}</div>"
        "</div>"
    )


def _last_run_rate_text(value: float, *, defined: bool) -> str:
    if not defined:
        return "—"
    return f"{100.0 * float(value):.1f}%"


def _last_run_unhappiness_text(counts: dict[str, float], *, defined: bool) -> str:
    if not defined:
        return "—"
    score = counts.get("unhappiness")
    if score is None:
        score = unhappiness_score(
            float(counts.get("precision", 0.0)),
            float(counts.get("recall", 0.0)),
        )
    return f"{float(score):.2f}%"


def _render_last_run_summary(
    counts: dict[str, float],
    *,
    min_size_nm: float = 0.0,
) -> None:
    floor_um = float(min_size_nm) / NM_PER_UM if min_size_nm else 20.0
    miss_title = f"Undetected real ≥ {floor_um:.0f} µm"
    miss_hint = "labeled particles missed at this size floor"
    counts_row = [
        _last_run_stat_card(
            "Detected", int(counts["n_detected"]), "#5b6472", "#5b647214", "all last-run hits"
        ),
        _last_run_stat_card(
            "Real of detected",
            int(counts["n_real"]),
            "#1f9d3a",
            "#28b44622",
            "detected and labeled particle",
        ),
        _last_run_stat_card(
            "Fake of detected",
            int(counts["n_fake"]),
            "#d61f1f",
            "#e3262622",
            "detected, not labeled as particle",
        ),
        _last_run_stat_card(
            "Total real",
            int(counts["n_real_total"]),
            "#187a2e",
            "#28b44618",
            f"real of detected + undetected ≥ {floor_um:.0f} µm",
        ),
        _last_run_stat_card(
            miss_title,
            int(counts["n_undetected"]),
            "#c56a00",
            "#d9770618",
            miss_hint,
        ),
    ]
    rates_row = [
        _last_run_stat_card(
            "Precision",
            _last_run_rate_text(
                float(counts.get("precision", 0.0)),
                defined=int(counts["n_detected"]) > 0,
            ),
            "#4b5d8a",
            "#4b5d8a18",
            "real of detected / detected",
        ),
        _last_run_stat_card(
            "Recall",
            _last_run_rate_text(
                float(counts.get("recall", 0.0)),
                defined=int(counts["n_real_total"]) > 0,
            ),
            "#6b3fa0",
            "#6b3fa018",
            "real of detected / total real",
        ),
        _last_run_stat_card(
            "Unhappiness",
            _last_run_unhappiness_text(
                counts,
                defined=int(counts["n_detected"]) > 0 and int(counts["n_real_total"]) > 0,
            ),
            "#8a3b2b",
            "#8a3b2b18",
            "100 × ((P − 0.95)² + (R − 0.95)²) · lower is better",
        ),
    ]
    row_style = (
        "display:flex;flex-wrap:wrap;gap:0.7rem;margin:0.35rem 0 0.35rem 0;"
    )
    st.markdown(
        f"<div style='{row_style}'>{''.join(counts_row)}</div>"
        f"<div style='{row_style}margin-bottom:0.85rem;'>{''.join(rates_row)}</div>",
        unsafe_allow_html=True,
    )
    extra = []
    if counts["n_unlabeled"]:
        extra.append(f"{int(counts['n_unlabeled']):,} unlabeled (orange)")
    if counts["n_not_sure"]:
        extra.append(f"{int(counts['n_not_sure']):,} not sure (blue)")
    if counts.get("n_undetected_below"):
        extra.append(
            f"{int(counts['n_undetected_below']):,} inspect marks below "
            f"{float(min_size_nm) / NM_PER_UM:.0f} µm not counted"
        )
    if extra:
        st.caption(" · ".join(extra) + ".")


def _render_last_run_tile_table(table: pd.DataFrame) -> None:
    if table is None or table.empty:
        st.info("No tiles to score in this folder.")
        return
    display = table.copy()
    rename = {
        "tile": "Tile",
        "n_detected": "Detected",
        "n_real": "Real of detected",
        "n_fake": "Fake of detected",
        "n_real_total": "Total real",
        "n_undetected": "Undetected",
        "precision": "Precision",
        "recall": "Recall",
        "unhappiness": "Unhappiness (%)",
    }
    keep = [col for col in rename if col in display.columns]
    display = display[keep].rename(columns=rename)
    for col in ("Detected", "Real of detected", "Fake of detected", "Total real", "Undetected"):
        if col in display.columns:
            display[col] = display[col].astype(int)
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "Precision": st.column_config.NumberColumn("Precision", format="%.1%"),
            "Recall": st.column_config.NumberColumn("Recall", format="%.1%"),
            "Unhappiness (%)": st.column_config.NumberColumn(
                "Unhappiness (%)", format="%.2f"
            ),
        },
    )


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
        "red = not-particle · blue = not sure. "
        "A label on N, S, E, W, or particles-only in this scene is applied to all of them "
        "(Groundup v3 is not mixed with v2)."
    )
    run_config = dict(config)
    session_config = st.session_state.get("config")
    if isinstance(session_config, dict):
        run_config.update(session_config)
    last_csv, pipeline_config = load_last_pipeline(run_config)
    for key in (
        "input_dir",
        "output_dir",
        "pixel_size_nm",
        "overlap_fraction",
        "filename_pattern",
    ):
        value = pipeline_config.get(key)
        if value not in (None, ""):
            run_config[key] = value
    detections: pd.DataFrame | None = None
    if table is not None:
        detections = tag_table(table, last_csv)
    elif last_csv is not None and last_csv.is_file():
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
    scene_tiles = related_scene_tile_names(
        cfg_get(run_config, "input_dir", None), run_tiles
    )
    if labels_df is not None and not labels_df.empty and scene_tiles:
        tile = labels_df["source_tile"].astype(str).map(
            lambda path: str(path).replace("\\", "/").rsplit("/", 1)[-1]
        )
        labels_df = labels_df.loc[tile.isin(scene_tiles)].copy()
    detections = snap_detection_keys_to_labels(
        detections,
        labels_df,
        _nsew_merge_radius_nm(run_config),
        _nsew_size_match_fraction(run_config),
    )
    folder_name = Path(str(cfg_get(run_config, "input_dir", "") or "")).name
    if folder_name:
        st.caption(
            f"Scoring labels from **{folder_name}** and the same-scene N/S/E/W "
            "and particles-only files"
            + (f" ({', '.join(sorted(scene_tiles))})" if scene_tiles else "")
            + "."
        )
    recovery_tiles = sorted(run_tiles) if run_tiles else tile_names_for_hits(detections)
    size_floor_nm = undetected_size_floor_nm(detections, run_config)
    names = expand_direction_tile_names(
        tile_names_for_hits(detections), recovery_tiles or run_tiles
    )
    metric_tiles = list(recovery_tiles or names)
    label_tiles = list(scene_tiles) if scene_tiles else metric_tiles
    per_tile_table = last_run_class_counts_by_tile(
        detections,
        labels_df,
        metric_tiles,
        min_size_nm=size_floor_nm,
        folder_tiles=label_tiles,
    )
    folder_counts = last_run_class_counts(
        detections,
        labels_df,
        tile_names=metric_tiles or None,
        min_size_nm=size_floor_nm,
        folder_tiles=label_tiles,
    )

    scope = st.radio(
        "Metrics",
        options=["all_tiles", "per_tile"],
        format_func=lambda value: (
            "All tiles in folder" if value == "all_tiles" else "Tile by tile"
        ),
        horizontal=True,
        key="last_run_metrics_scope",
        help="Folder totals match the previous Last Run summary. "
        "Tile by tile scores each file in that folder on its own.",
    )

    if names:
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

    if scope == "per_tile":
        tile_name = str(st.session_state.get("labeled_review_tile") or "")
        if tile_name:
            counts = last_run_class_counts(
                detections,
                labels_df,
                tile_names=[tile_name],
                min_size_nm=size_floor_nm,
                folder_tiles=label_tiles,
            )
            st.caption(
                f"Counts for **{tile_name}** only "
                "(hits on this file; NSEW labels still count as real / missed)."
            )
        else:
            counts = folder_counts
        _render_last_run_summary(counts, min_size_nm=size_floor_nm)
        _render_last_run_tile_table(per_tile_table)
    else:
        st.caption("Counts for every tile in the last-run folder.")
        _render_last_run_summary(folder_counts, min_size_nm=size_floor_nm)

    if not names:
        st.info("The last pipeline run has no detections.")
        return

    label_map = label_map_with_aliases(labels_df)

    tile_name = str(st.session_state["labeled_review_tile"])
    tile_hits = detections_on_tile(detections, tile_name)
    extra = _label_rows_missing_from_hits(
        labels_df, tile_name, tile_hits, folder_tiles=label_tiles
    )
    if extra is not None and not extra.empty:
        tile_hits = pd.concat([tile_hits, extra], ignore_index=True)
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
        st.image(
            view.rgb,
            caption=f"{tile_name} · detector hits",
            width=LAST_RUN_OVERVIEW_WIDTH,
        )

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
    columns = st.columns(LAST_RUN_CROP_COLUMNS)
    records = tile_hits.to_dict(orient="records")
    for offset, record in enumerate(records[start:end]):
        with columns[offset % LAST_RUN_CROP_COLUMNS]:
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
    folder_tiles: list[str] | None = None,
) -> pd.DataFrame | None:
    """Labeled marks on this tile that are not in the current detection CSV."""
    if labels_df is None or labels_df.empty:
        return None
    names = labels_df["source_tile"].astype(str).map(
        lambda path: str(path).replace("\\", "/").rsplit("/", 1)[-1]
    )
    on_tile = labels_df.loc[names == str(tile_name)].copy()
    if is_nsew_family_tile(tile_name):
        allowed = {
            Path(name).name
            for name in (folder_tiles or [tile_name])
            if is_nsew_family_tile(str(name))
        }
        allowed.add(str(tile_name))
        on_tile = labels_df.loc[names.isin(allowed)].copy()
        if "key" in on_tile.columns:
            on_tile = on_tile.drop_duplicates(subset=["key"], keep="last")
    if on_tile.empty:
        return None
    known = set()
    if hits is not None and not hits.empty and "key" in hits.columns:
        known = expand_labeled_keys(hits["key"].astype(str))
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
