"""Write detected-particle tables to CSV, JSON, or Excel."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.measurement.measurer import RESULT_COLUMNS


REQUIRED_COLUMNS = RESULT_COLUMNS[:6]
PARTICLES_SHEET = "particles"
PARAMETERS_SHEET = "parameters"


def _ordered_columns(df: pd.DataFrame) -> list[str]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Results table missing columns: {missing}")
    extra = [col for col in df.columns if col not in REQUIRED_COLUMNS]
    preferred = [col for col in RESULT_COLUMNS if col in df.columns]
    rest = [col for col in extra if col not in RESULT_COLUMNS]
    return preferred + rest


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, _ordered_columns(df)]


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the standard particle table as CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _validate(df).to_csv(destination, index=False)
    return destination


def write_json(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the standard particle table as JSON records."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _validate(df).to_json(destination, orient="records", indent=2)
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
    """Write particles on sheet 1 and selected run parameters on sheet 2."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_workbook(destination, _validate(df), parameters_table(config))
    return destination


def excel_bytes(
    df: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> bytes:
    """In-memory workbook with the same two sheets as ``write_xlsx``."""
    buffer = BytesIO()
    _write_workbook(buffer, _validate(df), parameters_table(config))
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
            for index, column in enumerate(frame.columns):
                width = min(_column_width(frame, column), 60)
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
    body = int(sample.str.len().max()) if len(sample) else 0
    return max(header, body, 12) + 2
