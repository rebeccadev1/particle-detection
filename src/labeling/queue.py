"""Build the unlabeled review queue from detection CSVs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.config import DEFAULT_OUTPUT_DIR, cfg_get, resolve_output_dir

MIN_SIZE_NM = 10_000.0
RECALL_CSV_NAME = "recall/particles.csv"
FALLBACK_CSV_NAME = "allemaal 08/09/particles.csv"
LAST_RUN_CSV_NAME = RECALL_CSV_NAME
LAST_PIPELINE_POINTER = "last_pipeline.json"

REQUIRED = ("id", "x_global", "y_global", "size", "confidence", "source_tile")


def detection_key(row: Mapping[str, Any] | pd.Series) -> str:
    """Stable id across runs: tile plus rounded global coordinates."""
    tile = Path(str(row["source_tile"])).name
    x_nm = int(round(float(row["x_global"])))
    y_nm = int(round(float(row["y_global"])))
    return f"{Path(tile).stem}_{x_nm}_{y_nm}"


def last_run_csv(project_root: str | Path, config: Mapping[str, Any] | None = None) -> Path:
    """CSV used as the default labeling source (under ``Outputs``).

    Prefers a high-recall run (``recall/particles.csv``) so previously labeled
    keys are skipped and only new loosened-gate hits need review. Falls back
    to the last classical run when the recall table is missing.
    """
    names: list[str] = []
    if config is not None:
        custom = cfg_get(dict(config), "labeling.source_csv", None)
        if custom not in (None, ""):
            names.append(str(custom))
    for name in (RECALL_CSV_NAME, FALLBACK_CSV_NAME):
        if name not in names:
            names.append(name)
    for name in names:
        for folder in (DEFAULT_OUTPUT_DIR, Path(project_root)):
            path = folder / name
            if path.is_file():
                return path
    return DEFAULT_OUTPUT_DIR / names[0]


def pipeline_particles_csv(config: Mapping[str, Any] | None = None) -> Path:
    """``particles.csv`` written by Detection → Run pipeline."""
    output_dir = None if config is None else cfg_get(dict(config), "output_dir", None)
    return resolve_output_dir(output_dir) / "particles.csv"


def write_last_pipeline(
    config: Mapping[str, Any],
    csv_path: str | Path,
    pointer_dir: str | Path | None = None,
) -> Path:
    """Remember which CSV + folder the latest pipeline run used."""
    destination = Path(csv_path).resolve()
    payload = {
        "csv": str(destination),
        "input_dir": str(cfg_get(dict(config), "input_dir", "") or ""),
        "output_dir": str(cfg_get(dict(config), "output_dir", "") or ""),
        "pixel_size_nm": cfg_get(dict(config), "pixel_size_nm", 960.0),
        "overlap_fraction": cfg_get(dict(config), "overlap_fraction", 0.0),
        "filename_pattern": cfg_get(dict(config), "filename_pattern", None),
    }
    folder = Path(pointer_dir) if pointer_dir is not None else DEFAULT_OUTPUT_DIR
    pointer = folder / LAST_PIPELINE_POINTER
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return pointer


def load_last_pipeline(
    config: Mapping[str, Any] | None = None,
    pointer_dir: str | Path | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """CSV and run settings for the latest pipeline write.

    Prefers ``last_pipeline.json``, then the current ``output_dir``
    ``particles.csv``. Does not use the labeling recall table.
    """
    run_config: dict[str, Any] = dict(config or {})
    folder = Path(pointer_dir) if pointer_dir is not None else DEFAULT_OUTPUT_DIR
    pointer = folder / LAST_PIPELINE_POINTER
    if pointer.is_file():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            payload = {}
        if isinstance(payload, dict):
            csv_text = payload.get("csv")
            csv_path = Path(str(csv_text)) if csv_text else None
            for key in (
                "input_dir",
                "output_dir",
                "pixel_size_nm",
                "overlap_fraction",
                "filename_pattern",
            ):
                value = payload.get(key)
                if value not in (None, ""):
                    run_config[key] = value
            if csv_path is not None and csv_path.is_file():
                return csv_path, run_config
    csv_path = pipeline_particles_csv(run_config)
    if csv_path.is_file():
        return csv_path, run_config
    return None, run_config


def load_particles_csv(path: str | Path) -> pd.DataFrame:
    """Read a detector table and tag it with ``source_csv`` and ``key``."""
    destination = Path(path)
    df = pd.read_csv(destination)
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"Results table missing columns: {missing}")
    tagged = df.copy()
    tagged["source_csv"] = str(destination)
    tagged["key"] = [_row_key(row) for _, row in tagged.iterrows()]
    return tagged


def filter_min_size(df: pd.DataFrame, min_size_nm: float = MIN_SIZE_NM) -> pd.DataFrame:
    """Keep detections at least ``min_size_nm`` (default 10 µm)."""
    if df.empty:
        return df.copy()
    return df.loc[df["size"].astype(float) >= float(min_size_nm)].reset_index(drop=True)


def tag_table(df: pd.DataFrame, source_csv: str | Path | None = None) -> pd.DataFrame:
    """Add ``key`` / ``source_csv`` to a pipeline table without mutating it."""
    tagged = df.copy()
    if "source_csv" not in tagged.columns:
        tagged["source_csv"] = "" if source_csv is None else str(source_csv)
    elif source_csv is not None:
        tagged["source_csv"] = tagged["source_csv"].where(
            tagged["source_csv"].astype(str).str.len() > 0, str(source_csv)
        )
    tagged["key"] = [_row_key(row) for _, row in tagged.iterrows()]
    return tagged


def merge_detection_tables(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate tables. Later rows update the same key; first-seen order is kept."""
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for frame in tables:
        if frame is None or frame.empty:
            continue
        tagged = tagged_if_needed(frame)
        for record in tagged.to_dict(orient="records"):
            key = _row_key(record)
            record["key"] = key
            if key not in merged:
                order.append(key)
            merged[key] = record
    if not order:
        return pd.DataFrame()
    return pd.DataFrame([merged[key] for key in order])


def unlabeled_queue(
    tables: Iterable[pd.DataFrame],
    labeled_keys: Iterable[str],
    min_size_nm: float = MIN_SIZE_NM,
) -> pd.DataFrame:
    """Detections at or above the size floor that are not yet in the label store."""
    merged = merge_detection_tables(tables)
    if merged.empty:
        return merged
    above = filter_min_size(tagged_if_needed(merged), min_size_nm)
    skip = {str(key) for key in labeled_keys}
    if not skip:
        return above.reset_index(drop=True)
    return above.loc[~above["key"].astype(str).isin(skip)].reset_index(drop=True)


def tagged_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if "key" in df.columns:
        return df
    return tag_table(df)


def _row_key(row: Mapping[str, Any] | pd.Series) -> str:
    existing = None
    if "key" in row:
        existing = row["key"]
    if existing is None or (isinstance(existing, float) and pd.isna(existing)):
        return detection_key(row)
    text = str(existing)
    if text in ("", "nan"):
        return detection_key(row)
    return text
