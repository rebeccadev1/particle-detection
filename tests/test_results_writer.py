from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

from src.overall.io.results_writer import (
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


def test_result_table_keeps_only_reported_columns() -> None:
    from src.overall.io.results_writer import result_table

    exported = result_table(_particles())
    assert list(exported.columns) == [
        "id",
        "x_global_um",
        "y_global_um",
        "size_um",
        "source_tile",
        "confidence",
    ]
    assert float(exported.loc[0, "x_global_um"]) == 0.1
    assert float(exported.loc[0, "y_global_um"]) == 0.2
    assert float(exported.loc[0, "size_um"]) == 15.0
    one_digit = result_table(
        pd.DataFrame(
            {
                "id": [1],
                "x_global": [335260.0],
                "y_global": [10060.0],
                "size": [15620.0],
                "confidence": [0.9],
                "source_tile": ["a.tif"],
            }
        )
    )
    assert float(one_digit.loc[0, "x_global_um"]) == 335.3
    assert float(one_digit.loc[0, "y_global_um"]) == 10.1
    assert float(one_digit.loc[0, "size_um"]) == 15.6


def test_count_table_is_only_as_wide_as_its_text() -> None:
    from src.analyse.report.report_generator import (
        _BODY,
        _LABEL,
        _MARGIN,
        _PAGE_W,
        _count_table_edges,
        _font,
        particles_at_least,
    )

    counts = particles_at_least(pd.DataFrame({"size": [10_000.0, 75_000.0]}))
    _left, right = _count_table_edges(counts, _font(_BODY), _font(_LABEL, bold=True))
    assert right < _PAGE_W - _MARGIN


def test_unstitched_paths_are_names_without_a_grid(tmp_path: Path) -> None:
    from src.analyse.report.report_generator import unstitched_tile_paths

    (tmp_path / "loose.png").write_bytes(b"")
    (tmp_path / "R1_2_3.png").write_bytes(b"")
    (tmp_path / "(4,5).png").write_bytes(b"")
    grid = r"R(?P<run>\d+)_(?P<row>\d+)_(?P<col>\d+)"
    names = [
        path.name
        for path in unstitched_tile_paths(
            {"input_dir": str(tmp_path), "filename_pattern": grid}
        )
    ]
    assert names == ["(4,5).png", "loose.png"]
    paren = [
        path.name
        for path in unstitched_tile_paths(
            {
                "input_dir": str(tmp_path),
                "filename_pattern": r"\((?P<row>\d+),(?P<col>\d+)\)",
            }
        )
    ]
    assert "(4,5).png" not in paren
    assert "loose.png" in paren
    blank = {
        path.name
        for path in unstitched_tile_paths(
            {"input_dir": str(tmp_path), "filename_pattern": ""}
        )
    }
    assert blank == {"loose.png", "R1_2_3.png", "(4,5).png"}


def test_pdf_puts_unstitched_images_on_their_own_page(tmp_path: Path) -> None:
    import numpy as np

    from src.analyse.report.report_generator import write_last_run_pdf

    path = write_last_run_pdf(
        pd.DataFrame({"size": [10_000.0]}),
        {"input_dir": str(tmp_path), "detection": {"min_size_nm": 10000.0}},
        tmp_path / "last_run.pdf",
        unstitched=[
            (
                "loose.png",
                np.full((24, 32, 3), (255, 140, 40), dtype=np.uint8),
                np.full((24, 32, 3), (20, 40, 80), dtype=np.uint8),
            )
        ],
    )
    assert path.read_bytes().count(b"/Type /Page") >= 2
    from src.analyse.report.report_generator import _unstitched_pages

    page = np.asarray(
        _unstitched_pages(
            [
                (
                    "loose.png",
                    np.full((30, 60, 3), (255, 140, 40), dtype=np.uint8),
                    np.full((30, 60, 3), (20, 40, 80), dtype=np.uint8),
                )
            ]
        )[0]
    )
    orange = np.where(np.all(page == (255, 140, 40), axis=2))
    blue = np.where(np.all(page == (20, 40, 80), axis=2))
    assert orange[0].size > 0
    assert blue[0].size > 0
    assert float(orange[0].mean()) < float(blue[0].mean())


def test_last_run_pdf_includes_summary_and_mosaic(tmp_path: Path) -> None:
    import numpy as np

    from src.analyse.report.report_generator import write_last_run_pdf

    table = pd.DataFrame(
        {
            "size": [20_000.0, 46_620.0, 99_940.0],
            "x_global": [0.0, 1.0, 2.0],
            "y_global": [0.0, 1.0, 2.0],
        }
    )
    mosaic = np.zeros((40, 80, 3), dtype=np.uint8)
    mosaic[10:30, 20:40] = (255, 140, 40)
    path = write_last_run_pdf(
        table,
        {"input_dir": "Recall audit", "detection": {"min_size_nm": 20000.0}},
        tmp_path / "last_run.pdf",
        mosaic_rgb=mosaic,
    )
    raw = path.read_bytes()
    assert raw.startswith(b"%PDF")
    assert b"Last run" in raw or len(raw) > 1000
    pages = raw.count(b"/Type /Page")
    assert pages >= 2


def test_mosaic_page_puts_labels_above_the_plain_mosaic() -> None:
    import numpy as np

    from src.analyse.report.report_generator import _mosaic_page

    labeled = np.full((30, 60, 3), (255, 140, 40), dtype=np.uint8)
    plain = np.full((30, 60, 3), (20, 40, 80), dtype=np.uint8)
    page = np.asarray(_mosaic_page(labeled, plain))
    orange = np.where(np.all(page == (255, 140, 40), axis=2))
    blue = np.where(np.all(page == (20, 40, 80), axis=2))
    assert orange[0].size > 0
    assert blue[0].size > 0
    assert float(orange[0].mean()) < float(blue[0].mean())


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


def test_particles_at_least_counts_cumulative_sizes() -> None:
    from src.analyse.report.report_generator import particles_at_least

    table = pd.DataFrame({"size": [9_000.0, 10_000.0, 20_000.0, 25_000.0, 49_999.0, 75_000.0]})
    counts = dict(zip(particles_at_least(table)["Size"], particles_at_least(table)["Particles"]))
    assert int(counts["≥10 µm"]) == 5
    assert int(counts["≥20 µm"]) == 4
    assert int(counts["≥25 µm"]) == 3
    assert int(counts["≥50 µm"]) == 1
    assert int(counts["≥75 µm"]) == 1
    empty = particles_at_least(pd.DataFrame())
    assert list(empty["Size"]) == ["≥10 µm", "≥20 µm", "≥25 µm", "≥50 µm", "≥75 µm"]
    assert list(empty["Particles"]) == [0, 0, 0, 0, 0]


def test_size_distribution_steps_from_min_size() -> None:
    from src.analyse.report.report_generator import size_distribution

    table = pd.DataFrame({"size": [20_000.0, 24_999.0, 25_000.0, 34_000.0, 40_000.0]})
    counts = size_distribution(table, min_um=20.0)["count"]
    assert list(counts.index) == ["20–25", "25–30", "30–35", "35–40"]
    assert list(counts) == [2, 1, 1, 1]


def test_size_distribution_starts_at_detection_floor() -> None:
    from src.analyse.report.report_generator import size_distribution

    table = pd.DataFrame({"size": [23_200.0, 26_000.0, 31_000.0]})
    counts = size_distribution(table, min_um=20.0)["count"]
    assert list(counts.index) == ["20–25", "25–30", "30–35"]
    assert list(counts) == [1, 1, 1]


def test_excel_bytes_includes_parameters_sheet() -> None:
    book = pd.ExcelFile(BytesIO(excel_bytes(_particles(), {"input_dir": "tiles"})))
    assert book.sheet_names == [PARTICLES_SHEET, PARAMETERS_SHEET]
    params = pd.read_excel(book, sheet_name=PARAMETERS_SHEET)
    assert list(params["parameter"]) == ["input_dir"]
    assert list(params["value"]) == ["tiles"]
