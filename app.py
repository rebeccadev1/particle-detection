"""Streamlit UI for per-tile particle detection on structured wafer surfaces."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import apply_nsew_settings, load_config, resolve_output_dir  # noqa: E402
from src.io.results_writer import excel_bytes, write_csv, write_xlsx  # noqa: E402
from src.labeling.ui import (  # noqa: E402
    render_collection_tab,
    render_label_tab,
    render_labeled_tiles_tab,
    render_tile_inspect_tab,
)
from src.labeling.queue import write_last_pipeline  # noqa: E402
from src.nsew import render_nsew_tab  # noqa: E402
from src.pipeline.runner import run_pipeline  # noqa: E402
from src.report.report_generator import (  # noqa: E402
    encode_overlay_jpeg,
    overlay_downsample,
    overlay_markers,
    summary_stats,
)

st.set_page_config(page_title="Particle detection", layout="wide")
st.title("Particle detection on structured surfaces")

NM_PER_UM = 1000.0
UI_SCHEMA = "mixed_layout_v17"


def _nm_to_um(nm: float) -> float:
    return float(nm) / NM_PER_UM


def _um_to_nm(um: float) -> float:
    return float(um) * NM_PER_UM


def _base_config() -> dict:
    return load_config(ROOT / "config.yaml")


def _widget_defaults(base: dict) -> dict:
    """Map config.yaml fields onto sidebar widget session-state keys."""
    pipe = base.get("pipeline") or {}
    pre = base.get("preprocessing") or {}
    det = base.get("detection") or {}
    meas = base.get("measurement") or {}
    report = base.get("report") or {}
    method = str(det.get("method", "fft"))
    if method not in ("fft", "tophat"):
        method = "fft"
    fft_mask = str(det.get("fft_mask", "per_tile"))
    if fft_mask not in ("per_tile", "shared"):
        fft_mask = "per_tile"
    size_agg = str(meas.get("size_aggregation", "max"))
    if size_agg not in ("max", "mean"):
        size_agg = "max"
    return {
        "ui_input_dir": str(base.get("input_dir") or ""),
        "ui_output_dir": str(base.get("output_dir") or "../Outputs"),
        "ui_filename_pattern": str(base.get("filename_pattern")),
        "ui_overlap_fraction": float(base.get("overlap_fraction", 0.0)),
        "ui_pixel_size_um": _nm_to_um(float(base.get("pixel_size_nm", 960.0))),
        "ui_workers": int(pipe.get("workers", 0) or 0),
        "ui_denoise": bool(pre.get("denoise", True)),
        "ui_denoise_sigma": float(pre.get("denoise_sigma", 1.0)),
        "ui_flatten_illumination": bool(pre.get("flatten_illumination", True)),
        "ui_flatten_sigma": float(pre.get("flatten_sigma", 160.0)),
        "ui_contrast_stretch": bool(pre.get("contrast_stretch", True)),
        "ui_method": method,
        "ui_particles_bright": bool(det.get("particles_bright", True)),
        "ui_fft_peak_threshold": float(det.get("fft_peak_threshold", 0.35)),
        "ui_fft_mask": fft_mask,
        "ui_tophat_radius": int(det.get("tophat_radius", 50)),
        "ui_min_size_um": _nm_to_um(float(det.get("min_size_nm", 10000.0))),
        "ui_max_size_um": _nm_to_um(float(det.get("max_size_nm", 100000.0))),
        "ui_blob_threshold": float(det.get("blob_threshold", 0.08)),
        "ui_blob_min_sigma": float(det.get("blob_min_sigma", 3.7)),
        "ui_min_prominence": float(det.get("min_prominence", 0.30)),
        "ui_min_confidence_pct": int(round(float(det.get("min_confidence", 0.50)) * 100.0)),
        "ui_blob_max_sigma": float(det.get("blob_max_sigma", 36.0)),
        "ui_local_snr_sigma": float(det.get("local_snr_sigma", 0.0)),
        "ui_edge_soften_sigma": float(det.get("edge_soften_sigma", 12.0)),
        "ui_edge_soften_strength": float(det.get("edge_soften_strength", 2.0)),
        "ui_edge_exclude_px": float(det.get("edge_exclude_px", 12.0)),
        "ui_min_circularity": float(det.get("min_circularity", 0.0)),
        "ui_structure_neighbor_px": float(det.get("structure_neighbor_px", 48.0)),
        "ui_ml_enabled": bool((base.get("ml") or {}).get("enabled", False)),
        "ui_ml_threshold": float((base.get("ml") or {}).get("threshold", 0.25)),
        "ui_merge_radius_px": float(meas.get("merge_radius_px", 8.0)),
        "ui_size_aggregation": size_agg,
        "ui_target_mb": float(report.get("target_mb", 20.0)),
        "ui_downsample": int(report.get("downsample", 0) or 0),
    }


def _apply_widget_defaults(base: dict) -> None:
    st.session_state.update(_widget_defaults(base))


def _nsew_config(base: dict) -> dict:
    return apply_nsew_settings(base)


def _on_reset_standard(base: dict) -> None:
    st.session_state["ui_apply_nsew_settings"] = False
    _apply_widget_defaults(base)


def _on_toggle_nsew_settings(base: dict) -> None:
    if st.session_state.get("ui_apply_nsew_settings"):
        _apply_widget_defaults(_nsew_config(base))
    else:
        _apply_widget_defaults(base)


def _sidebar(base: dict) -> dict:
    config = deepcopy(base)
    if st.session_state.get("ui_schema") != UI_SCHEMA:
        _apply_widget_defaults(base)
        st.session_state["ui_schema"] = UI_SCHEMA
        st.session_state["ui_initialized"] = True

    st.sidebar.button(
        "Reset to standard values",
        on_click=_on_reset_standard,
        args=(base,),
        help="Restore every sidebar setting from config.yaml.",
        width="stretch",
    )
    st.sidebar.checkbox(
        "Apply NSEW settings",
        key="ui_apply_nsew_settings",
        on_change=_on_toggle_nsew_settings,
        args=(base,),
        help="Use Groundup / N/S/E/W standard values (pixel size, preprocess, "
        "detection, and ML threshold) from nsew_config.yaml.",
    )

    st.sidebar.header("Data")
    config["input_dir"] = st.sidebar.text_input(
        "Tile folder",
        key="ui_input_dir",
        help="Folder of TIFF/BMP tiles under ASML SE/Inputs, e.g. R3 04-08.",
    )
    config["output_dir"] = st.sidebar.text_input(
        "Output folder",
        key="ui_output_dir",
        help="Default is ASML SE/Outputs. A name like Single v7 goes in Outputs/Single v7.",
    )
    config["filename_pattern"] = st.sidebar.text_input(
        "Filename pattern (regex)",
        key="ui_filename_pattern",
        help="Default: R{run}_{row}_{col}_{mag}X.tif or .bmp  e.g. R3_5_32_5X.tif. "
        "Names that do not match still run detection; they are left out of the mosaic.",
    )
    config["overlap_fraction"] = st.sidebar.slider(
        "Tile overlap fraction",
        min_value=0.0,
        max_value=0.5,
        step=0.01,
        key="ui_overlap_fraction",
    )
    config["pixel_size_nm"] = _um_to_nm(
        st.sidebar.number_input(
            "Pixel size (µm)",
            min_value=0.001,
            step=0.1,
            format="%.3f",
            key="ui_pixel_size_um",
            help="These 5X tiles are 1 pixel = 0.96 µm (960 nm).",
        )
    )
    pipe = config.setdefault("pipeline", {})
    pipe["workers"] = st.sidebar.number_input(
        "Parallel workers",
        min_value=0,
        max_value=32,
        key="ui_workers",
        help="Tiles processed in parallel with processes (not threads). 0 uses all CPU cores.",
    )

    st.sidebar.header("Preprocessing")
    pre = config.setdefault("preprocessing", {})
    pre["denoise"] = st.sidebar.checkbox("Denoise", key="ui_denoise")
    pre["denoise_sigma"] = st.sidebar.number_input(
        "Denoise sigma", min_value=0.0, key="ui_denoise_sigma"
    )
    pre["flatten_illumination"] = st.sidebar.checkbox(
        "Flatten illumination", key="ui_flatten_illumination"
    )
    pre["flatten_sigma"] = st.sidebar.number_input(
        "Flatten sigma",
        min_value=1.0,
        key="ui_flatten_sigma",
        help="Large values flatten only illumination. Small values halo region edges.",
    )
    pre["contrast_stretch"] = st.sidebar.checkbox(
        "Contrast stretch", key="ui_contrast_stretch"
    )

    st.sidebar.header("Detection")
    det = config.setdefault("detection", {})
    det["method"] = st.sidebar.selectbox(
        "Background suppression",
        options=["fft", "tophat"],
        key="ui_method",
    )
    det["particles_bright"] = st.sidebar.checkbox(
        "Particles are bright", key="ui_particles_bright"
    )
    det["fft_peak_threshold"] = st.sidebar.slider(
        "FFT peak threshold", 0.05, 0.95, key="ui_fft_peak_threshold"
    )
    det["fft_mask"] = st.sidebar.selectbox(
        "FFT notch source",
        options=["per_tile", "shared"],
        key="ui_fft_mask",
        help="per_tile: notch each tile from its own spectrum. "
        "shared: reuse the first tile (uniform lattice only).",
    )
    det["tophat_radius"] = st.sidebar.number_input(
        "Top-hat radius (px)",
        min_value=1,
        key="ui_tophat_radius",
        help="Must be larger than the biggest particle. Default 50 (≈100 µm flakes). "
        "A small radius erases large debris.",
    )
    det["min_size_nm"] = _um_to_nm(
        st.sidebar.number_input(
            "Min size (µm)",
            min_value=0.1,
            step=0.5,
            format="%.2f",
            key="ui_min_size_um",
            help="Keep particles larger than this. Default 10 µm (≈10.4 px at 0.96 µm/px). "
            "Raise Blob min sigma with this so the detector does not hunt for smaller specks.",
        )
    )
    det["max_size_nm"] = _um_to_nm(
        st.sidebar.number_input(
            "Max size (µm)",
            min_value=0.1,
            step=0.5,
            format="%.2f",
            key="ui_max_size_um",
            help="Upper size cap. Default 100 µm. Blob max sigma 36 is about 100 µm "
            "at 0.96 µm/px. Raise Top-hat radius with this so large flakes are not erased.",
        )
    )
    det["blob_threshold"] = st.sidebar.slider(
        "Blob threshold", 0.001, 0.5, key="ui_blob_threshold"
    )
    det["min_prominence"] = st.sidebar.slider(
        "Min prominence",
        min_value=0.0,
        max_value=0.8,
        key="ui_min_prominence",
        help="Ignore residual weaker than this (0–1). Raises this to drop grid/texture; "
        "lowers it to pick up dimmer specks. 0 disables.",
    )
    det["min_confidence"] = (
        st.sidebar.slider(
            "Confidence score (%)",
            min_value=0,
            max_value=100,
            step=1,
            key="ui_min_confidence_pct",
            help="Drop blobs whose residual at the peak is below this score. 0 disables.",
        )
        / 100.0
    )
    det["blob_min_sigma"] = st.sidebar.number_input(
        "Blob min sigma", min_value=0.5, key="ui_blob_min_sigma"
    )
    det["blob_max_sigma"] = st.sidebar.number_input(
        "Blob max sigma", min_value=1.0, key="ui_blob_max_sigma"
    )
    det["local_snr_sigma"] = st.sidebar.number_input(
        "Local SNR sigma (0 = auto)",
        min_value=0.0,
        key="ui_local_snr_sigma",
        help="Envelope for local energy. 0 auto-enables on FFT residuals "
        "(3.5 × blob max sigma) and stays off for top-hat.",
    )
    det["edge_soften_sigma"] = st.sidebar.number_input(
        "Edge soften sigma",
        min_value=0.0,
        key="ui_edge_soften_sigma",
        help="Coarse blur before the region-border gradient. 0 disables.",
    )
    det["edge_soften_strength"] = st.sidebar.number_input(
        "Edge soften strength",
        min_value=0.0,
        key="ui_edge_soften_strength",
        help="How hard to attenuate residual on area borders. 0 disables.",
    )
    det["edge_exclude_px"] = st.sidebar.number_input(
        "Edge exclude (px)",
        min_value=0.0,
        key="ui_edge_exclude_px",
        help="Ignore blobs this close to a region border. "
        "Pad corners and box rims are always dropped.",
    )
    det["min_circularity"] = st.sidebar.slider(
        "Min circularity",
        min_value=0.0,
        max_value=1.0,
        key="ui_min_circularity",
        help="Drop very elongated hits. Keep low: real debris is irregular, not round. 0 keeps every blob.",
    )
    det["structure_neighbor_px"] = st.sidebar.number_input(
        "Structure neighbor radius (px)",
        min_value=0.0,
        key="ui_structure_neighbor_px",
        help="Drop blobs that have 2+ other blobs this close (chains and grids). 0 disables.",
    )

    st.sidebar.header("ML filter")
    ml = config.setdefault("ml", {})
    ml["enabled"] = st.sidebar.checkbox(
        "Apply ML filter",
        key="ui_ml_enabled",
        help="Score each DoG proposal with the trained patch/cascade model. "
        "Leave off until you have trained (`python -m src.ml.train`). "
        "Detection stays classical; the model only drops false positives.",
    )
    ml_min, ml_max = 0.0, 1.0
    stored = float(st.session_state.get("ui_ml_threshold", 0.0))
    st.session_state["ui_ml_threshold"] = min(ml_max, max(ml_min, stored))
    ml["threshold"] = st.sidebar.slider(
        "ML P(particle) threshold",
        min_value=ml_min,
        max_value=ml_max,
        step=0.05,
        key="ui_ml_threshold",
        help="Keep blobs with at least this probability. 0 keeps every proposal.",
    )
    if not ml.get("model_path"):
        ml["model_path"] = (base.get("ml") or {}).get(
            "model_path", "models/particle_clf_v3.joblib"
        )

    st.sidebar.header("Measurement / report")
    meas = config.setdefault("measurement", {})
    meas["merge_radius_px"] = st.sidebar.number_input(
        "Merge radius (px)", min_value=0.0, key="ui_merge_radius_px"
    )
    meas["size_aggregation"] = st.sidebar.selectbox(
        "Cluster size aggregation",
        options=["max", "mean"],
        key="ui_size_aggregation",
    )
    report = config.setdefault("report", {})
    report["target_mb"] = st.sidebar.number_input(
        "Stitched mosaic size (MB)",
        min_value=1.0,
        max_value=200.0,
        step=1.0,
        key="ui_target_mb",
        help="Each tile is downsampled before stitching so the full mosaic is about this large.",
    )
    report["downsample"] = st.sidebar.number_input(
        "Overlay downsample (0 = auto)",
        min_value=0,
        key="ui_downsample",
        help="0 chooses a factor from the target size. A value > 0 forces that integer factor.",
    )
    return config


def _run_detection(config: dict, progress, status) -> None:
    if not config["input_dir"]:
        st.error("Set a tile folder in the sidebar.")
        return

    def _on_progress(current: int, total: int, name: str) -> None:
        fraction = current / max(total, 1)
        progress.progress(min(max(fraction, 0.0), 1.0), text=f"{current}/{total} {name}")

    try:
        with st.spinner("Running detection (first tile can take a minute)…"):
            table, mosaic = run_pipeline(config, progress_cb=_on_progress)
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.exception(exc)
        return

    st.session_state["table"] = table
    st.session_state["mosaic"] = mosaic
    st.session_state["config"] = config
    st.session_state.pop("overlay", None)
    st.session_state.pop("overlay_key", None)
    out_dir = resolve_output_dir(config["output_dir"])
    csv_path = out_dir / "particles.csv"
    xlsx_path = out_dir / "particles.xlsx"
    write_csv(table, csv_path)
    write_xlsx(table, xlsx_path, config)
    write_last_pipeline(config, csv_path)
    st.session_state["last_pipeline_csv"] = str(csv_path)
    progress.progress(1.0, text=f"Detection done · {len(table)} particles")
    status.success(f"Wrote {csv_path} and {xlsx_path} ({len(table)} particles).")


def _render_detection(config: dict) -> None:
    st.markdown(
        "Point at a folder of TIFF or BMP tiles. Names like **`R{run}_{row}_{col}_{mag}X.tif`** "
        "(for example `R3_5_32_5X.tif` or `R3_5_32_5X.bmp`) are placed on the grid and stitched. Files that "
        "do not match still run **per-tile** detection; they are omitted from the mosaic. "
        "Images named **N, S, E, W** are the same view under different lighting: a hit at "
        "the same place in more than one of them is **one particle** (size may differ). "
        "Matching tiles are downsampled and stitched into one overview of about the "
        "configured size (default **20 MB**). Scale is **1 pixel = 0.96 µm**; size "
        "limits are in **µm** (default 10–100 µm)."
    )

    run = st.button("Run pipeline", type="primary")
    progress = st.progress(0.0, text="Idle")
    status = st.empty()
    if run:
        _run_detection(config, progress, status)

    table: pd.DataFrame | None = st.session_state.get("table")
    mosaic = st.session_state.get("mosaic")
    active_config = deepcopy(st.session_state.get("config", config))
    active_config["report"] = config.get("report", active_config.get("report", {}))

    if table is None:
        st.info("Configure a folder and run the pipeline to see results.")
        return

    stats = summary_stats(table)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Particles", stats["particle_count"])
    c2.metric("Mean size (µm)", f"{_nm_to_um(stats['size_mean']):.2f}")
    c3.metric("Min size (µm)", f"{_nm_to_um(stats['size_min']):.2f}")
    c4.metric("Max size (µm)", f"{_nm_to_um(stats['size_max']):.2f}")

    st.subheader("Results")
    display = table.copy()
    if not display.empty:
        display.insert(display.columns.get_loc("size") + 1, "size_um", display["size"] / NM_PER_UM)
    preview_rows = 5_000
    if len(display) > preview_rows:
        st.caption(
            f"Showing the first {preview_rows:,} of {len(display):,} rows. "
            "Download the CSV or Excel file for the full table."
        )
        st.dataframe(display.head(preview_rows), width="stretch", hide_index=True)
    else:
        st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        "CSV `size` is equivalent circular diameter in nm "
        "(from the connected footprint on the photo, not the FFT/top-hat residual "
        "and not the DoG scale bin). `size_um` is the same value in µm."
    )

    st.subheader("Size distribution (µm)")
    if not table.empty:
        sizes_um = table["size"].to_numpy(dtype=float) / NM_PER_UM
        n_bins = min(20, max(int(table["size"].nunique()), 1))
        counts, bin_edges = np.histogram(sizes_um, bins=n_bins)
        hist_df = pd.DataFrame(
            {"count": counts},
            index=[f"{bin_edges[i]:.2f}" for i in range(len(counts))],
        )
        st.bar_chart(hist_df)

    dl_csv, dl_xlsx = st.columns(2)
    with dl_csv:
        st.download_button(
            "Download CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="particles.csv",
            mime="text/csv",
        )
    with dl_xlsx:
        st.download_button(
            "Download Excel",
            data=excel_bytes(table, active_config),
            file_name="particles.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if mosaic is None or not mosaic.placements:
        if table is not None:
            st.info(
                "No tiles had row/column (or x/y) in the filename, so nothing was "
                "stitched. Detection results are in the table above. Open the Tiles "
                "tab to inspect each image."
            )
        return

    placed_names = {placement.name for placement in mosaic.placements}
    if "source_tile" in table.columns:
        table_tiles = {
            str(name).replace("\\", "/").rsplit("/", 1)[-1]
            for name in table["source_tile"].astype(str)
        }
        leftover = table_tiles - placed_names
        if leftover:
            st.caption(
                f"{len(leftover)} file(s) did not match the filename pattern and were "
                "detected but not stitched into this mosaic."
            )

    st.subheader("Stitched mosaic")
    max_h = max(int(mosaic.full_height), 1)
    max_w = max(int(mosaic.full_width), 1)
    factor = overlay_downsample(mosaic, active_config, crop=None)
    out_h = max(int(np.ceil(mosaic.full_height / factor)), 0)
    out_w = max(int(np.ceil(mosaic.full_width / factor)), 0)
    st.caption(
        f"Full mosaic is {mosaic.full_height}×{mosaic.full_width} px. "
        f"Tiles are downsampled ×{factor} before stitching → {out_h}×{out_w} "
        f"(~{out_h * out_w * 3 / (1024 * 1024):.1f} MB uncompressed RGB)."
    )

    overlay_key = (
        "full",
        float(active_config.get("report", {}).get("target_mb", 20.0)),
        int(active_config.get("report", {}).get("downsample", 0) or 0),
        len(table),
    )
    if st.session_state.get("overlay_key") != overlay_key:

        def _stitch_progress(current: int, total: int, name: str) -> None:
            progress.progress(
                current / max(total, 1),
                text=f"Downsizing tiles {current}/{total} {name}",
            )

        try:
            st.session_state["overlay"] = overlay_markers(
                mosaic,
                table,
                active_config,
                crop=None,
                progress_cb=_stitch_progress,
            )
        except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
            st.error(f"Mosaic stitching failed: {exc}")
        else:
            st.session_state["overlay_key"] = overlay_key
            mosaic_path = resolve_output_dir(active_config.get("output_dir")) / "mosaic_overlay.jpg"
            mosaic_path.parent.mkdir(parents=True, exist_ok=True)
            mosaic_path.write_bytes(encode_overlay_jpeg(st.session_state["overlay"]))
            progress.progress(1.0, text="Done")
            status.success(f"Finished. Wrote {mosaic_path}")
    overlay = st.session_state.get("overlay")
    if overlay is not None:
        st.image(overlay, caption="All tiles, stitched at reduced resolution", width="stretch")
        st.download_button(
            "Download stitched mosaic",
            data=encode_overlay_jpeg(overlay),
            file_name="mosaic_overlay.jpg",
            mime="image/jpeg",
        )

    with st.expander("Inspect a crop (full-resolution mosaic pixels)"):
        col_a, col_b, col_c, col_d = st.columns(4)
        crop_y = col_a.number_input("Crop y", min_value=0, max_value=max_h, value=0)
        crop_x = col_b.number_input("Crop x", min_value=0, max_value=max_w, value=0)
        crop_h = col_c.number_input(
            "Crop height", min_value=1, max_value=max_h, value=min(max_h, 2048)
        )
        crop_w = col_d.number_input(
            "Crop width", min_value=1, max_value=max_w, value=min(max_w, 2048)
        )
        if st.button("Render crop"):
            crop = (int(crop_y), int(crop_x), int(crop_h), int(crop_w))
            cropped = overlay_markers(mosaic, table, active_config, crop=crop)
            st.image(cropped, caption="Cropped region with detections", width="stretch")


config = _sidebar(_base_config())

detect_tab, nsew_tab, tiles_tab, last_run_tab, label_tab, labeled_tab = st.tabs(
    ["Detection", "NSEW", "Tiles", "Last Run", "Tinder", "Database"],
    on_change="rerun",
    key="main_tabs",
)

with detect_tab:
    _render_detection(config)

if nsew_tab.open:
    with nsew_tab:
        render_nsew_tab()

if tiles_tab.open:
    with tiles_tab:
        render_tile_inspect_tab(
            config,
            ROOT,
            st.session_state.get("table"),
            st.session_state.get("mosaic"),
        )

if last_run_tab.open:
    with last_run_tab:
        render_labeled_tiles_tab(
            config,
            ROOT,
            st.session_state.get("mosaic"),
            st.session_state.get("table"),
        )

if label_tab.open:
    with label_tab:
        render_label_tab(
            config,
            ROOT,
            st.session_state.get("table"),
            st.session_state.get("mosaic"),
        )

if labeled_tab.open:
    with labeled_tab:
        particle_col, not_particle_col = st.columns(2)
        with particle_col:
            render_collection_tab(ROOT, "particle", "Particles", n_columns=2)
        with not_particle_col:
            render_collection_tab(ROOT, "not_particle", "Not particles", n_columns=2)

