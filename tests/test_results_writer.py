from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

from src.io.results_writer import (
    PARAMETERS_SHEET,
    PARTICLES_SHEET,
    excel_bytes,
    parameters_table,
    write_xlsx,
)


def _particles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1],
            "x_global": [100.0],
            "y_global": [200.0],
            "size": [15000.0],
            "confidence": [0.9],
            "source_tile": ["R3_1_1_5X.tif"],
        }
    )


def test_parameters_table_flattens_nested_run_settings() -> None:
    table = parameters_table(
        {
            "input_dir": "R3 04-08",
            "pixel_size_nm": 960.0,
            "detection": {
                "min_size_nm": 20000.0,
                "recall_mode": False,
                "recall": {"min_confidence": 0.4},
            },
            "preprocessing": {"contrast_percentiles": [1.0, 99.0]},
        }
    )
    values = dict(zip(table["parameter"], table["value"]))
    assert values["input_dir"] == "R3 04-08"
    assert values["pixel_size_nm"] == 960.0
    assert values["detection.min_size_nm"] == 20000.0
    assert values["detection.recall_mode"] is False
    assert values["detection.recall.min_confidence"] == 0.4
    assert values["preprocessing.contrast_percentiles"] == "[1.0, 99.0]"


def test_write_xlsx_adds_parameters_sheet(tmp_path: Path) -> None:
    config = {
        "output_dir": "Single v7",
        "detection": {"min_size_nm": 20000.0, "method": "fft"},
    }
    path = write_xlsx(_particles(), tmp_path / "particles.xlsx", config)
    sheets = pd.read_excel(path, sheet_name=None)
    assert list(sheets) == [PARTICLES_SHEET, PARAMETERS_SHEET]
    particles = sheets[PARTICLES_SHEET]
    assert list(particles["source_tile"]) == ["R3_1_1_5X.tif"]
    params = dict(
        zip(sheets[PARAMETERS_SHEET]["parameter"], sheets[PARAMETERS_SHEET]["value"])
    )
    assert params["output_dir"] == "Single v7"
    assert params["detection.min_size_nm"] == 20000.0
    assert params["detection.method"] == "fft"


def test_excel_bytes_includes_parameters_sheet() -> None:
    book = pd.ExcelFile(BytesIO(excel_bytes(_particles(), {"input_dir": "tiles"})))
    assert book.sheet_names == [PARTICLES_SHEET, PARAMETERS_SHEET]
    params = pd.read_excel(book, sheet_name=PARAMETERS_SHEET)
    assert list(params["parameter"]) == ["input_dir"]
    assert list(params["value"]) == ["tiles"]
