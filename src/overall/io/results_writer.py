"""Write detected-particle tables to CSV, JSON, or Excel."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

EXPORT_COLUMNS = (
    "id",
    "x_global_um",
    "y_global_um",
    "size_um",
    "source_tile",
    "confidence",
)
PARTICLES_SHEET = "particles"
PARAMETERS_SHEET = "parameters"
NM_PER_UM = 1000.0


def result_table(df: pd.DataFrame) -> pd.DataFrame:
    """id, global x/y in µm, size in µm, source tile, and confidence."""
    if df.empty and "id" not in df.columns:
        return pd.DataFrame(columns=list(EXPORT_COLUMNS))
    frame = df.copy()
    for source, dest in (
        ("x_global", "x_global_um"),
        ("y_global", "y_global_um"),
        ("size", "size_um"),
    ):
        if source in frame.columns:
            frame[dest] = frame[source].astype(float) / NM_PER_UM
        elif dest not in frame.columns:
            raise ValueError(f"Results table missing columns: [{source!r}]")
    missing = [col for col in EXPORT_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Results table missing columns: {missing}")
    exported = frame.loc[:, list(EXPORT_COLUMNS)].copy()
    for column in ("x_global_um", "y_global_um", "size_um"):
        exported[column] = exported[column].astype(float).round(1)
    return exported


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the result table as CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result_table(df).to_csv(destination, index=False)
    return destination


def parameters_payload(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Nested settings used for this run."""
    return _json_ready(dict(config) if config else {})


def write_run_settings(
    config: Mapping[str, Any] | None,
    path: str | Path,
    csv_path: str | Path | None = None,
) -> Path:
    """Write this run's settings into the output folder as ``last_pipeline.json``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = parameters_payload(config)
    if csv_path is not None:
        payload["csv"] = str(Path(csv_path).resolve())
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stale = destination.parent / "parameters.json"
    if stale.is_file() and stale.resolve() != destination.resolve():
        stale.unlink()
    return destination


def parameters_table(config: Mapping[str, Any] | None) -> pd.DataFrame:
    """Flatten nested run settings into ``parameter`` / ``value`` rows."""
    rows: list[dict[str, Any]] = []
    if config:
        _flatten_config(dict(config), "", rows)
    return pd.DataFrame(rows, columns=["parameter", "value"])


def write_xlsx(
    df: pd.DataFrame,
    path: str | Path,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Write particles on sheet 1 and this run's settings on sheet 2.

    Sheet 2 includes preprocessing and detection (and the rest of the config
    that was passed in, such as pixel size and ML).
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_workbook(destination, result_table(df), parameters_table(config))
    return destination


def excel_bytes(
    df: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> bytes:
    """In-memory workbook with the same two sheets as ``write_xlsx``."""
    buffer = BytesIO()
    _write_workbook(buffer, result_table(df), parameters_table(config))
    return buffer.getvalue()


def _write_workbook(
    destination: Path | BytesIO,
    particles: pd.DataFrame,
    parameters: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(destination, engine="xlsxwriter") as writer:
        particles.to_excel(writer, sheet_name=PARTICLES_SHEET, index=False)
        parameters.to_excel(writer, sheet_name=PARAMETERS_SHEET, index=False)
        workbook = writer.book
        header = workbook.add_format({"bold": True})
        for sheet_name, frame in (
            (PARTICLES_SHEET, particles),
            (PARAMETERS_SHEET, parameters),
        ):
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, None, header)
            one_digit = workbook.add_format({"num_format": "0.0"})
            for index, column in enumerate(frame.columns):
                width = min(_column_width(frame, column), 60)
                if sheet_name == PARTICLES_SHEET and column in {
                    "x_global_um",
                    "y_global_um",
                    "size_um",
                }:
                    worksheet.set_column(index, index, width, one_digit)
                else:
                    worksheet.set_column(index, index, width)


def _flatten_config(
    node: Mapping[str, Any],
    prefix: str,
    rows: list[dict[str, Any]],
) -> None:
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and not isinstance(value, (str, bytes)):
            _flatten_config(value, dotted, rows)
        else:
            rows.append({"parameter": dotted, "value": _excel_value(value)})


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping) and not isinstance(value, (str, bytes)):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _excel_value(value)


def _excel_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=True)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return json.dumps(value, default=str, ensure_ascii=True)


def _column_width(frame: pd.DataFrame, column: str) -> float:
    header = len(str(column))
    if frame.empty:
        return max(header, 12) + 2
    sample = frame[column].astype(str).head(200)
    if sample.empty:
        return max(header, 12) + 2
    longest = sample.str.len().max()
    body = int(longest) if pd.notna(longest) else 0
    return max(header, body, 12) + 2
