"""Tests for the per-tile inspect overlay (no Streamlit)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.labeling.crops import local_px_to_global_nm
from src.labeling.inspect import (
    OverlayView,
    cell_bounds,
    circles_from_rows,
    detections_on_tile,
    display_xy_to_local,
    expand_direction_tile_names,
    extra_grid_placements,
    unhappiness_score,
    labeled_detections,
    labeled_particle_recovery,
    last_run_class_counts,
    last_run_class_counts_by_tile,
    local_xy_on_tile,
    missed_particle_record,
    overlay_circles,
    preview_click_crop,
    tile_names_for_hits,
    undetected_size_floor_nm,
)
from src.stitching.stitcher import TilePlacement


def test_cell_bounds_splits_tile_into_equal_grid() -> None:
    y0, x0, y1, x1 = cell_bounds(90, 120, 1, 2, grid=3)
    assert (y0, x0) == (30, 80)
    assert (y1, x1) == (60, 120)


def test_detections_on_tile_matches_basename() -> None:
    table = pd.DataFrame(
        {
            "source_tile": ["R3_5_17_5X.tif", "other/R3_2_31_5X.tif", "R3_5_17_5X.tif"],
            "size": [1.0, 2.0, 3.0],
        }
    )
    hits = detections_on_tile(table, "R3_5_17_5X.tif")
    assert list(hits["size"]) == [1.0, 3.0]


def test_detections_on_tile_share_nsew_hits() -> None:
    table = pd.DataFrame(
        {
            "source_tile": ["N.bmp", "loose.tif"],
            "size": [10.0, 11.0],
            "x_global": [100.0, 1.0],
            "y_global": [200.0, 2.0],
            "id": [1, 2],
            "confidence": [0.9, 0.8],
            "nsew_dirs": ["N,S,E", ""],
        }
    )
    hits = detections_on_tile(table, "S.bmp")
    assert list(hits["size"]) == [10.0]
    assert list(hits["source_tile"]) == ["S.bmp"]
    assert expand_direction_tile_names(["N.bmp"], ["N.bmp", "S.bmp", "E.bmp", "W.bmp"]) == [
        "N.bmp",
        "S.bmp",
        "E.bmp",
        "W.bmp",
    ]


def test_detections_on_tile_does_not_share_particles_only_versions() -> None:
    table = pd.DataFrame(
        {
            "source_tile": [
                "particles_only.png",
                "particles_only_4of4.png",
            ],
            "size": [10.0, 11.0],
            "x_global": [100.0, 100.0],
            "y_global": [200.0, 200.0],
            "id": [1, 2],
            "confidence": [0.9, 0.8],
            "key": ["NSEW_100_200", "NSEW_100_201"],
        }
    )
    two = detections_on_tile(table, "particles_only.png")
    four = detections_on_tile(table, "particles_only_4of4.png")
    assert list(two["id"]) == [1]
    assert list(four["id"]) == [2]


def test_labeled_detections_keeps_particle_keys_and_crops() -> None:
    detections = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x_global": [10.0, 20.0, 30.0],
            "y_global": [11.0, 21.0, 31.0],
            "size": [16000.0, 17000.0, 18000.0],
            "confidence": [0.9, 0.8, 0.7],
            "source_tile": ["a.tif", "b.tif", "a.tif"],
            "key": ["a_10_11", "b_20_21", "a_30_31"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["a_10_11", "b_20_21", "a_30_31"],
            "label": ["particle", "not_particle", "particle"],
            "crop_path": ["crops/a.jpg", "crops/b.jpg", "crops/c.jpg"],
            "particle_id": [1, 2, 3],
        }
    )
    hits = labeled_detections(detections, labels, label="particle")
    assert list(hits["key"]) == ["a_10_11", "a_30_31"]
    assert list(hits["crop_path"]) == ["crops/a.jpg", "crops/c.jpg"]
    assert tile_names_for_hits(hits) == ["a.tif"]


def test_labeled_detections_empty_when_no_overlap() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10.0],
            "y_global": [11.0],
            "size": [16000.0],
            "confidence": [0.9],
            "source_tile": ["a.tif"],
            "key": ["a_10_11"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["other"],
            "label": ["particle"],
            "crop_path": ["crops/x.jpg"],
            "particle_id": [9],
        }
    )
    hits = labeled_detections(detections, labels)
    assert hits.empty
    assert tile_names_for_hits(hits) == []


def test_labeled_particle_recovery_counts_last_run_keys_only() -> None:
    detections = pd.DataFrame(
        {
            "id": [1, 2],
            "x_global": [10_000.0, 50_000.0],
            "y_global": [11_000.0, 51_000.0],
            "size": [16000.0, 17000.0],
            "confidence": [0.9, 0.8],
            "source_tile": ["a.tif", "a.tif"],
            "key": ["a_10000_11000", "a_50000_51000"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["a_10000_11000", "a_90000_90000", "b_1_1"],
            "label": ["particle", "particle", "particle"],
            "source_tile": ["a.tif", "a.tif", "other.tif"],
            "x_global": [10_000.0, 90_000.0, 1.0],
            "y_global": [11_000.0, 90_000.0, 1.0],
        }
    )
    stats = labeled_particle_recovery(detections, labels, tile_names=["a.tif"])
    assert stats == {"n_labeled": 1, "n_found": 1, "pct": 100.0}


def test_labeled_particle_recovery_ignores_older_keys_not_in_run() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10_000.0],
            "y_global": [11_000.0],
            "size": [16000.0],
            "confidence": [0.9],
            "source_tile": ["a.tif"],
            "key": ["a_10000_11000"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["a_10000_11000", "old_tophat", "old_ml"],
            "label": ["particle", "particle", "particle"],
            "source_tile": ["a.tif", "a.tif", "a.tif"],
            "x_global": [10_000.0, 10_005.0, 90_000.0],
            "y_global": [11_000.0, 11_002.0, 90_000.0],
        }
    )
    stats = labeled_particle_recovery(detections, labels, tile_names=["a.tif"])
    assert stats == {"n_labeled": 1, "n_found": 1, "pct": 100.0}


def test_last_run_class_counts_splits_real_fake_and_inspect_misses() -> None:
    detections = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x_global": [10.0, 20.0, 30.0],
            "y_global": [11.0, 21.0, 31.0],
            "size": [16000.0, 17000.0, 18000.0],
            "confidence": [0.9, 0.8, 0.7],
            "source_tile": ["a.tif", "a.tif", "a.tif"],
            "key": ["real", "fake", "open"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["real", "fake", "miss", "old_miss"],
            "label": ["particle", "not_particle", "particle", "particle"],
            "source_tile": ["a.tif", "a.tif", "a.tif", "a.tif"],
            "source_csv": ["run.csv", "run.csv", "tile_inspect", "older.csv"],
        }
    )
    stats = last_run_class_counts(detections, labels, tile_names=["a.tif"])
    assert stats["n_detected"] == 3
    assert stats["n_real"] == 1
    assert stats["n_fake"] == 2
    assert stats["n_unlabeled"] == 1
    assert stats["n_undetected"] == 2
    assert stats["n_real_total"] == stats["n_real"] + stats["n_undetected"]
    assert stats["n_real_total"] == 3
    assert stats["precision"] == 1 / 3
    assert stats["recall"] == 1 / 3
    assert stats["unhappiness"] == unhappiness_score(1 / 3, 1 / 3)
    assert stats["unhappiness"] == 100.0 * (
        (1 / 3 - 0.95) ** 2 + (1 / 3 - 0.95) ** 2
    )


def test_last_run_class_counts_drops_undetected_below_run_size_floor() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10.0],
            "y_global": [11.0],
            "size": [20000.0],
            "confidence": [0.9],
            "source_tile": ["a.tif"],
            "key": ["hit"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["big_miss", "small_miss"],
            "label": ["particle", "particle"],
            "source_tile": ["a.tif", "a.tif"],
            "size": [25000.0, 15000.0],
        }
    )
    stats = last_run_class_counts(
        detections, labels, tile_names=["a.tif"], min_size_nm=20000.0
    )
    assert stats["n_undetected"] == 1
    assert stats["n_undetected_below"] == 1
    assert stats["n_real_total"] == 1
    assert stats["precision"] == 0.0
    assert stats["recall"] == 0.0


def test_last_run_class_counts_by_tile_splits_folder() -> None:
    detections = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x_global": [10.0, 20.0, 30.0],
            "y_global": [11.0, 21.0, 31.0],
            "size": [20000.0, 20000.0, 20000.0],
            "confidence": [0.9, 0.8, 0.7],
            "source_tile": ["a.tif", "a.tif", "b.tif"],
            "key": ["a_real", "a_fake", "b_real"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["a_real", "a_fake", "a_miss", "b_real"],
            "label": ["particle", "not_particle", "particle", "particle"],
            "source_tile": ["a.tif", "a.tif", "a.tif", "b.tif"],
            "size": [25000.0, 20000.0, 25000.0, 20000.0],
        }
    )
    table = last_run_class_counts_by_tile(
        detections, labels, ["a.tif", "b.tif"], min_size_nm=20000.0
    )
    by_tile = table.set_index("tile")
    assert list(table["tile"]) == ["a.tif", "b.tif"]
    assert int(by_tile.loc["a.tif", "n_detected"]) == 2
    assert int(by_tile.loc["a.tif", "n_real"]) == 1
    assert int(by_tile.loc["a.tif", "n_undetected"]) == 1
    assert float(by_tile.loc["a.tif", "precision"]) == 0.5
    assert float(by_tile.loc["a.tif", "recall"]) == 0.5
    assert float(by_tile.loc["a.tif", "unhappiness"]) == unhappiness_score(0.5, 0.5)
    assert int(by_tile.loc["b.tif", "n_detected"]) == 1
    assert float(by_tile.loc["b.tif", "precision"]) == 1.0
    assert float(by_tile.loc["b.tif", "recall"]) == 1.0
    assert float(by_tile.loc["b.tif", "unhappiness"]) == unhappiness_score(1.0, 1.0)


def test_last_run_class_counts_by_tile_splits_particles_only() -> None:
    detections = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x_global": [10.0, 20.0, 30.0],
            "y_global": [11.0, 21.0, 31.0],
            "size": [20000.0, 20000.0, 20000.0],
            "confidence": [0.9, 0.8, 0.7],
            "source_tile": [
                "particles_only.png",
                "particles_only.png",
                "particles_only_4of4.png",
            ],
            "key": ["NSEW_10_11", "NSEW_20_21", "NSEW_30_31"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["NSEW_10_11", "NSEW_30_31", "NSEW_99_99"],
            "label": ["particle", "particle", "particle"],
            "source_tile": ["N.png", "N.png", "E.png"],
            "size": [20000.0, 20000.0, 20000.0],
            "x_global": [10.0, 30.0, 99.0],
            "y_global": [11.0, 31.0, 99.0],
        }
    )
    table = last_run_class_counts_by_tile(
        detections,
        labels,
        ["particles_only.png", "particles_only_4of4.png"],
        min_size_nm=20000.0,
        folder_tiles=["particles_only.png", "particles_only_4of4.png", "N.png", "E.png"],
    )
    by_tile = table.set_index("tile")
    assert int(by_tile.loc["particles_only.png", "n_detected"]) == 2
    assert int(by_tile.loc["particles_only.png", "n_real"]) == 1
    assert int(by_tile.loc["particles_only_4of4.png", "n_detected"]) == 1
    assert int(by_tile.loc["particles_only_4of4.png", "n_real"]) == 1


def test_last_run_ignores_nsew_labels_from_another_folder() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10.0],
            "y_global": [11.0],
            "size": [20000.0],
            "confidence": [0.9],
            "source_tile": ["N.png"],
            "key": ["NSEW_10_11"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["NSEW_10_11", "NSEW_old_v2", "NSEW_miss_v3"],
            "label": ["particle", "particle", "particle"],
            "source_tile": ["N.png", "N.bmp", "E.png"],
            "size": [20000.0, 25000.0, 25000.0],
            "x_global": [10.0, 99.0, 50.0],
            "y_global": [11.0, 98.0, 51.0],
        }
    )
    stats = last_run_class_counts(
        detections,
        labels,
        tile_names=["N.png", "E.png", "S.png", "W.png"],
        min_size_nm=20000.0,
    )
    assert stats["n_real"] == 1
    assert stats["n_undetected"] == 1
    assert stats["n_real_total"] == 2


def test_last_run_particles_only_uses_same_scene_nsew_labels() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10.0],
            "y_global": [11.0],
            "size": [20000.0],
            "confidence": [0.9],
            "source_tile": ["particles_only_2of4.png"],
            "key": ["NSEW_10_11"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["NSEW_10_11", "NSEW_50_51", "NSEW_99_98"],
            "label": ["particle", "particle", "particle"],
            "source_tile": ["N.png", "E.png", "N.bmp"],
            "size": [20000.0, 25000.0, 25000.0],
            "x_global": [10.0, 50.0, 99.0],
            "y_global": [11.0, 51.0, 98.0],
        }
    )
    stats = last_run_class_counts(
        detections,
        labels,
        tile_names=["particles_only_2of4.png", "particles_only_4of4.png"],
        min_size_nm=20000.0,
        folder_tiles=[
            "particles_only_2of4.png",
            "particles_only_4of4.png",
            "N.png",
            "E.png",
            "S.png",
            "W.png",
        ],
    )
    assert stats["n_real"] == 1
    assert stats["n_undetected"] == 1
    assert stats["n_real_total"] == 2


def test_last_run_snaps_particles_only_to_nearby_nsew_label() -> None:
    from src.labeling.queue import snap_detection_keys_to_labels, tag_table

    detections = tag_table(
        pd.DataFrame(
            {
                "id": [1],
                "x_global": [18.0],
                "y_global": [11.0],
                "size": [20000.0],
                "confidence": [0.9],
                "source_tile": ["particles_only_2of4.png"],
            }
        )
    )
    labels = pd.DataFrame(
        {
            "key": ["NSEW_10_11"],
            "label": ["particle"],
            "source_tile": ["N.png"],
            "size": [20000.0],
            "x_global": [10.0],
            "y_global": [11.0],
        }
    )
    snapped = snap_detection_keys_to_labels(
        detections, labels, merge_radius_nm=24.0, size_match_fraction=0.0
    )
    stats = last_run_class_counts(
        snapped,
        labels,
        tile_names=["particles_only_2of4.png"],
        min_size_nm=20000.0,
        folder_tiles=["particles_only_2of4.png", "N.png"],
    )
    assert stats["n_real"] == 1
    assert stats["n_unlabeled"] == 0
    assert stats["n_undetected"] == 0


def test_related_scene_tile_names_keeps_groundup_v3_suffix(tmp_path: Path) -> None:
    from src.labeling.inspect import related_scene_tile_names

    v3 = tmp_path / "Groundup v3"
    v3.mkdir()
    (v3 / "N.png").write_bytes(b"n")
    (v3 / "S.png").write_bytes(b"s")
    po = v3 / "Output Groundup v3" / "Particles only"
    po.mkdir(parents=True)
    (po / "particles_only_2of4.png").write_bytes(b"p")
    (po / "particles_only_4of4.png").write_bytes(b"q")
    v2 = tmp_path / "Groundup v2"
    v2.mkdir()
    (v2 / "N.bmp").write_bytes(b"old")

    names = related_scene_tile_names(po, ["particles_only_2of4.png", "particles_only_4of4.png"])
    assert "N.png" in names
    assert "S.png" in names
    assert "particles_only_4of4.png" in names
    assert "N.bmp" not in names

    relative = related_scene_tile_names(
        "Groundup v3/Output Groundup v3/Particles only",
        ["particles_only_2of4.png"],
    )
    # Even without resolving the real Inputs tree, png particles-only
    # still keeps N.png labels (not v2 N.bmp).
    assert "N.png" in relative
    assert "N.bmp" not in relative


def test_unhappiness_score_is_zero_at_target() -> None:
    assert unhappiness_score(0.95, 0.95) == 0.0
    assert unhappiness_score(1.0, 1.0) == 100.0 * (
        (1.0 - 0.95) ** 2 + (1.0 - 0.95) ** 2
    )


def test_undetected_size_floor_uses_this_run_minimum() -> None:
    detections = pd.DataFrame({"size": [20000.0, 31000.0]})
    floor = undetected_size_floor_nm(detections, {"detection": {"min_size_nm": 10000.0}})
    assert floor == 20000.0


def test_labeled_particle_recovery_ignores_non_particle_labels() -> None:
    detections = pd.DataFrame(
        {
            "id": [1],
            "x_global": [10.0],
            "y_global": [11.0],
            "size": [16000.0],
            "confidence": [0.9],
            "source_tile": ["a.tif"],
            "key": ["a_10_11"],
        }
    )
    labels = pd.DataFrame(
        {
            "key": ["a_10_11"],
            "label": ["not_particle"],
            "source_tile": ["a.tif"],
            "x_global": [10.0],
            "y_global": [11.0],
        }
    )
    stats = labeled_particle_recovery(detections, labels)
    assert stats == {"n_labeled": 0, "n_found": 0, "pct": 0.0}


def test_overlay_draws_labeled_circle_in_crop() -> None:
    image = np.full((80, 90), 0.2, dtype=np.float32)
    image[20:28, 30:38] = 1.0
    rgb = overlay_circles(
        image,
        [
            {
                "x_local": 34.0,
                "y_local": 24.0,
                "size": 20.0,
                "color": (40, 180, 70),
            }
        ],
        pixel_size_nm=1.0,
        crop=(0, 0, 80, 90),
        max_side=None,
    )
    assert rgb.shape == (80, 90, 3)
    green = (rgb[:, :, 1] > 150) & (rgb[:, :, 0] < 80) & (rgb[:, :, 2] < 100)
    assert int(green.sum()) > 20


def test_circles_from_rows_uses_label_color() -> None:
    rows = pd.DataFrame(
        {
            "key": ["a", "b"],
            "size": [10.0, 12.0],
        }
    )
    circles = circles_from_rows(rows, [1.0, 2.0], [3.0, 4.0], {"a": "particle"})
    assert circles[0]["color"] == (40, 180, 70)
    assert circles[1]["color"] == (255, 140, 40)


def test_display_xy_to_local_inverts_overlay_scale() -> None:
    view = OverlayView(rgb=np.zeros((50, 80, 3), dtype=np.uint8), crop_x0=10, crop_y0=20, scale=0.5)
    x_local, y_local = display_xy_to_local(15.0, 8.0, view)
    assert x_local == 10.0 + 15.0 / 0.5
    assert y_local == 20.0 + 8.0 / 0.5


def test_missed_particle_record_uses_tile_inspect_source() -> None:
    record = missed_particle_record("R3_5_17_5X.tif", 100.0, 200.0, 15000.0)
    assert record["source_csv"] == "tile_inspect"
    assert record["key"] == "R3_5_17_5X_100_200"
    assert record["size"] == 15000.0


def test_n2_pattern_keeps_detector_and_inspect_circles_on_tile() -> None:
    """R4 n2_row_col names need that pattern or mosaic globals fall off the tile."""
    base = TilePlacement(
        path=Path("n2_1_17.tif"),
        name="n2_1_17.tif",
        y0=0,
        x0=0,
        height=2076,
        width=3088,
    )
    config = {
        "filename_pattern": r"n2_(?P<row>\d+)_(?P<col>\d+)\.tiff?",
        "overlap_fraction": 0.0,
    }
    extras = extra_grid_placements(base, config)
    placements = [base, *extras]
    x0 = 16 * 3088
    det_x, det_y = local_xy_on_tile(
        (100.0 + x0) * 960.0, 200.0 * 960.0, placements, 960.0, 3088, 2076
    )
    miss_x, miss_y = local_xy_on_tile(
        100.0 * 960.0, 200.0 * 960.0, placements, 960.0, 3088, 2076
    )
    assert 0.0 <= det_x < 3088.0
    assert 0.0 <= det_y < 2076.0
    assert 0.0 <= miss_x < 3088.0
    assert 0.0 <= miss_y < 2076.0


def test_extra_grid_placement_maps_last_run_csv_onto_tile() -> None:
    base = TilePlacement(
        path=Path("R3_1_26_5X.tif"),
        name="R3_1_26_5X.tif",
        y0=0,
        x0=0,
        height=2076,
        width=3088,
    )
    extras = extra_grid_placements(base, {"filename_pattern": r"R(?P<run>\d+)_(?P<row>\d+)_(?P<col>\d+)_(?P<mag>[\d.]+)X\.tiff?", "overlap_fraction": 0.0})
    x_local, y_local = local_xy_on_tile(
        68686080.0, 219840.0, [base, *extras], 960.0, 3088, 2076
    )
    assert 0.0 <= x_local < 3088.0
    assert 0.0 <= y_local < 2076.0


def test_preview_click_crop_keeps_full_circle_near_tile_edge() -> None:
    image = np.full((80, 90), 0.2, dtype=np.float32)
    image[2:10, 2:10] = 1.0
    rgb = preview_click_crop(image, 6.0, 6.0, 20.0, 1.0, color=(40, 180, 70))
    assert rgb.shape[0] >= 80
    assert rgb.shape[1] >= 90
    green = (rgb[:, :, 1] > 150) & (rgb[:, :, 0] < 80) & (rgb[:, :, 2] < 100)
    assert int(green.sum()) > 20


def test_local_xy_on_tile_prefers_in_frame_placement() -> None:
    folder = TilePlacement(
        path=Path("R3_2_31_5X.tif"),
        name="R3_2_31_5X.tif",
        y0=0,
        x0=38909,
        height=2076,
        width=3088,
    )
    wafer = TilePlacement(
        path=Path("R3_2_31_5X.tif"),
        name="R3_2_31_5X.tif",
        y0=1868,
        x0=83376,
        height=2076,
        width=3088,
    )
    x_global, y_global = local_px_to_global_nm(100.0, 80.0, wafer, 960.0)
    x_local, y_local = local_xy_on_tile(
        x_global, y_global, [folder, wafer], 960.0, 3088, 2076
    )
    assert x_local == 100.0
    assert y_local == 80.0

    x_click, y_click = local_px_to_global_nm(50.0, 60.0, folder, 960.0)
    x_local, y_local = local_xy_on_tile(
        x_click, y_click, [folder, wafer], 960.0, 3088, 2076
    )
    assert x_local == 50.0
    assert y_local == 60.0


def test_local_px_roundtrip_to_global_nm() -> None:
    placement = TilePlacement(
        path=Path("R3_1_2_5X.tif"),
        name="R3_1_2_5X.tif",
        y0=100,
        x0=80,
        height=128,
        width=128,
    )
    x_global, y_global = local_px_to_global_nm(5.0, 4.0, placement, pixel_size_nm=2.0)
    from src.labeling.crops import global_nm_to_local_px

    x_local, y_local = global_nm_to_local_px(x_global, y_global, placement, pixel_size_nm=2.0)
    assert x_local == 5.0
    assert y_local == 4.0
