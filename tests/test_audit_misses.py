"""Tests for miss autopsy: unmatched labels, trace captions, tile_inspect writes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.labeling.audit_misses import (
    audit_marks,
    format_trace_caption,
    promote_unmatched_to_inspect,
    trace_click,
    unmatched_particle_marks,
)
from src.labeling.inspect import MISS_SOURCE
from src.labeling.store import LabelStore
from tests.fixtures.synthetic import (
    detector_test_config,
    make_structured_tile,
    write_tile_tiff,
)


def _tile_setup(tmp_path: Path) -> tuple[Path, dict, pd.DataFrame]:
    folder = tmp_path / "tiles"
    path = write_tile_tiff(
        folder / "R3_1_1_5X.tif",
        make_structured_tile(particles=[(40.0, 42.0, 3.0)]),
    )
    config = detector_test_config(
        input_dir=str(folder),
        overlap_fraction=0.0,
        pixel_size_nm=1.0,
    )
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [42.0],
            "y_global": [40.0],
            "size": 12.0,
            "confidence": 0.9,
            "source_tile": [path.name],
        }
    )
    return path, config, detections


def test_format_trace_caption_includes_gate_and_residual() -> None:
    text = format_trace_caption(
        {"reason": "confidence", "residual": 0.44, "edge_distance_px": 12.5}
    )
    assert text.startswith("Gate: confidence")
    assert "0.440" in text
    assert "12.5" in text


def test_unmatched_particle_marks_skips_nearby_detections(tmp_path: Path) -> None:
    path, config, detections = _tile_setup(tmp_path)
    labels = pd.DataFrame(
        {
            "key": ["kept", "miss"],
            "label": ["particle", "particle"],
            "source_tile": [path.name, path.name],
            "x_global": [42.0, 10.0],
            "y_global": [40.0, 8.0],
            "size": [12.0, 15.0],
            "source_csv": ["run.csv", "run.csv"],
        }
    )
    marks = unmatched_particle_marks(labels, detections, config, near_px=5.0)
    assert list(marks["key"]) == ["miss"]
    assert marks.iloc[0]["x_local"] == 10.0
    assert marks.iloc[0]["y_local"] == 8.0
    assert marks.iloc[0]["nearest_kept_px"] > 5.0


def test_unmatched_particle_marks_inspect_only(tmp_path: Path) -> None:
    path, config, detections = _tile_setup(tmp_path)
    labels = pd.DataFrame(
        {
            "key": ["a", "b"],
            "label": ["particle", "particle"],
            "source_tile": [path.name, path.name],
            "x_global": [10.0, 11.0],
            "y_global": [8.0, 9.0],
            "size": [15.0, 15.0],
            "source_csv": ["other.csv", MISS_SOURCE],
        }
    )
    marks = unmatched_particle_marks(
        labels, detections, config, near_px=5.0, inspect_only=True
    )
    assert list(marks["key"]) == ["b"]


def test_trace_click_kept_on_synthetic_blob(tmp_path: Path) -> None:
    _path, config, _detections = _tile_setup(tmp_path)
    image = make_structured_tile(particles=[(40.0, 42.0, 3.0)])
    kept = trace_click(image, config, 40.0, 42.0)
    assert kept["reason"] == "already_kept"
    empty = trace_click(image, config, 5.0, 5.0)
    assert empty["reason"] != "already_kept"


def test_audit_marks_and_promote_inspect(tmp_path: Path) -> None:
    path, config, detections = _tile_setup(tmp_path)
    labels = pd.DataFrame(
        {
            "key": ["miss"],
            "label": ["particle"],
            "source_tile": [path.name],
            "x_global": [10.0],
            "y_global": [8.0],
            "size": [15.0],
            "source_csv": ["run.csv"],
        }
    )
    marks = unmatched_particle_marks(labels, detections, config, near_px=5.0)
    audited = audit_marks(marks, config)
    assert len(audited) == 1
    assert audited.iloc[0]["reason"] != "already_kept"

    store = LabelStore(tmp_path / "labels")
    written = promote_unmatched_to_inspect(marks, config, store)
    assert len(written) == 1
    assert written[0]["source_csv"] == MISS_SOURCE
    saved = store.load()
    assert saved.iloc[0]["source_csv"] == MISS_SOURCE
    assert saved.iloc[0]["label"] == "particle"
    again = promote_unmatched_to_inspect(marks, config, store)
    assert again == []


def test_promote_retags_existing_particle_as_inspect(tmp_path: Path) -> None:
    path, config, detections = _tile_setup(tmp_path)
    store = LabelStore(tmp_path / "labels")
    crop = np.zeros((32, 32, 3), dtype="uint8")
    store.apply_label(
        {
            "key": "R3_1_1_5X_10_8",
            "id": "",
            "source_tile": path.name,
            "x_global": 10.0,
            "y_global": 8.0,
            "size": 15.0,
            "confidence": 0.0,
            "source_csv": "older.csv",
        },
        "particle",
        crop,
    )
    labels = store.load()
    marks = unmatched_particle_marks(labels, detections, config, near_px=5.0)
    written = promote_unmatched_to_inspect(marks, config, store)
    assert len(written) == 1
    saved = store.load()
    assert list(saved["source_csv"]) == [MISS_SOURCE]
    assert len(saved) == 1
