"""Tests for labeling crops, queue filtering, and the on-disk store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.labeling.crops import (
    CIRCLE_COLOR,
    TileImageCache,
    config_for_hit_placement,
    crop_direction_views,
    crop_particle,
    crop_window,
    find_tile_path,
    global_nm_to_local_px,
)
from src.labeling.inspect import detections_on_tile
from src.labeling.queue import (
    MIN_SIZE_NM,
    detection_key,
    filter_min_size,
    tag_table,
    unlabeled_queue,
    nsew_key_aliases,
)
from src.labeling.store import LabelStore
from src.stitching.stitcher import TilePlacement
from tests.fixtures.synthetic import write_tile_tiff

PATTERN = r"R(?P<run>\d+)_(?P<row>\d+)_(?P<col>\d+)_(?P<mag>[\d.]+)X\.tiff?"


def _config(folder: Path, pixel_size_nm: float = 1.0) -> dict:
    return {
        "input_dir": str(folder),
        "filename_pattern": PATTERN,
        "overlap_fraction": 0.0,
        "pixel_size_nm": pixel_size_nm,
    }


def _red_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels close to the unlabeled orange marker (legacy name)."""
    return (
        (rgb[:, :, 0] > 200)
        & (rgb[:, :, 1] > 80)
        & (rgb[:, :, 1] < 180)
        & (rgb[:, :, 2] < 80)
    )


def test_global_nm_to_local_px_subtracts_origin() -> None:
    placement = TilePlacement(
        path=Path("R3_1_2_5X.tif"),
        name="R3_1_2_5X.tif",
        y0=100,
        x0=80,
        height=128,
        width=128,
    )
    x_local, y_local = global_nm_to_local_px(960.0, 480.0, placement, pixel_size_nm=2.0)
    assert x_local == 960.0 / 2.0 - 80
    assert y_local == 480.0 / 2.0 - 100


def test_find_tile_path_accepts_bmp_for_tif_name(tmp_path: Path) -> None:
    import cv2

    gray = np.zeros((8, 8), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "R3_1_1_5X.bmp"), gray)
    found = find_tile_path("R3_1_1_5X.tif", input_dir=tmp_path)
    assert found.name == "R3_1_1_5X.bmp"


def test_crop_window_clamps_to_tile_edges() -> None:
    y0, x0, y1, x1 = crop_window(5.0, 4.0, diameter_px=10.0, height=80, width=90)
    assert (y0, x0) == (0, 0)
    assert 0 < y1 <= 80
    assert 0 < x1 <= 90
    assert y1 - y0 <= 80
    assert x1 - x0 <= 90


def test_hit_placement_overlap_puts_the_circle_on_the_crop(tmp_path: Path) -> None:
    image = np.zeros((80, 100), dtype=np.float32)
    write_tile_tiff(tmp_path / "R3_1_1_5X.tif", image)
    write_tile_tiff(tmp_path / "R3_1_2_5X.tif", image)
    row = {
        "id": 1,
        "source_tile": "R3_1_2_5X.tif",
        "x_global": 95.0,
        "y_global": 40.0,
        "size": 10.0,
        "confidence": 1.0,
    }
    config = _config(tmp_path, pixel_size_nm=1.0)
    config["overlap_fraction"] = 0.0
    placed = config_for_hit_placement(config, pd.DataFrame([row]))
    assert float(placed["overlap_fraction"]) > 0.0
    crop = crop_particle(row, placed)
    assert int(_red_mask(crop.rgb).sum()) > 20


def test_crop_window_stays_local_when_point_is_outside_the_tile() -> None:
    y0, x0, y1, x1 = crop_window(-5000.0, 229.0, diameter_px=40.0, height=2076, width=3088)
    assert x1 - x0 == 192
    assert y1 - y0 == 192
    assert x0 == 0
    assert y0 <= 229 < y1


def test_crop_contains_blob_and_clamps_at_edge(tmp_path: Path) -> None:
    image = np.zeros((80, 90), dtype=np.float32)
    image[2:8, 3:8] = 1.0
    path = write_tile_tiff(tmp_path / "R3_1_1_5X.tif", image)
    row = {
        "id": 1,
        "source_tile": path.name,
        "x_global": 5.0,
        "y_global": 5.0,
        "size": 10.0,
        "confidence": 1.0,
    }
    crop = crop_particle(row, _config(tmp_path))
    assert crop.crop_x0 == 0
    assert crop.crop_y0 == 0
    assert abs(crop.x_local - 5.0) < 1.0
    assert abs(crop.y_local - 5.0) < 1.0
    assert crop.rgb.shape[0] == 80
    assert crop.rgb.shape[1] == 90
    local_x = int(round(crop.x_local)) - crop.crop_x0
    local_y = int(round(crop.y_local)) - crop.crop_y0
    assert 0 <= local_x < crop.rgb.shape[1]
    assert 0 <= local_y < crop.rgb.shape[0]


def test_overlay_draws_one_circle(tmp_path: Path) -> None:
    image = np.zeros((128, 128), dtype=np.float32)
    image[38:43, 38:43] = 1.0
    image[88:93, 88:93] = 1.0
    path = write_tile_tiff(tmp_path / "R3_1_1_5X.tif", image)
    row = {
        "id": 2,
        "source_tile": path.name,
        "x_global": 40.0,
        "y_global": 40.0,
        "size": 20.0,
        "confidence": 1.0,
    }
    crop = crop_particle(row, _config(tmp_path))
    marker = _red_mask(crop.rgb)
    assert int(marker.sum()) > 20
    yy, xx = np.nonzero(marker)
    assert np.mean(np.hypot(yy - 40.0, xx - 40.0)) < 30.0
    other = marker[85:96, 85:96]
    assert int(other.sum()) == 0


def test_tile_cache_reuses_decoded_array(tmp_path: Path) -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[10:14, 10:14] = 1.0
    path = write_tile_tiff(tmp_path / "R3_1_1_5X.tif", image)
    cache = TileImageCache()
    row = {
        "id": 1,
        "source_tile": path.name,
        "x_global": 12.0,
        "y_global": 12.0,
        "size": 8.0,
        "confidence": 1.0,
    }
    crop_particle(row, _config(tmp_path), cache=cache)
    first = cache.image
    crop_particle(row, _config(tmp_path), cache=cache)
    assert cache.image is first


def test_min_size_filter_drops_below_15_um() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x_global": [1.0, 2.0, 3.0],
            "y_global": [1.0, 2.0, 3.0],
            "size": [9_999.0, 10_000.0, 20_000.0],
            "confidence": [1.0, 1.0, 1.0],
            "source_tile": ["A.tif", "B.tif", "C.tif"],
        }
    )
    kept = filter_min_size(df, MIN_SIZE_NM)
    assert list(kept["id"]) == [2, 3]
    assert float(kept["size"].min()) >= MIN_SIZE_NM


def test_unlabeled_queue_skips_labeled_keys() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "x_global": [100.0, 200.0],
            "y_global": [10.0, 20.0],
            "size": [20_000.0, 21_000.0],
            "confidence": [1.0, 1.0],
            "source_tile": ["R3_1_1_5X.tif", "R3_1_1_5X.tif"],
        }
    )
    tagged = tag_table(df, "run.csv")
    skip = {detection_key(tagged.iloc[0])}
    queue = unlabeled_queue([tagged], skip)
    assert len(queue) == 1
    assert int(queue.iloc[0]["id"]) == 2


def test_nsew_label_covers_all_four_directions(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels")
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    south = {
        "id": 1,
        "source_tile": "S.bmp",
        "x_global": 100.4,
        "y_global": 200.4,
        "size": 20_000.0,
        "confidence": 0.8,
        "source_csv": "particles.csv",
    }
    store.apply_label(south, "particle", rgb)
    assert detection_key(south) == "NSEW_100_200"
    assert store.labeled_keys() == {"NSEW_100_200"}
    detections = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x_global": [100.4, 100.4, 50.0],
            "y_global": [200.4, 200.4, 50.0],
            "size": [20_000.0, 21_000.0, 22_000.0],
            "confidence": [0.9, 0.7, 0.6],
            "source_tile": ["N.bmp", "E.bmp", "loose.tif"],
        }
    )
    queue = unlabeled_queue([tag_table(detections)], store.labeled_keys())
    assert list(queue["source_tile"]) == ["loose.tif"]
    hits = detections_on_tile(tag_table(detections), "W.bmp")
    assert "NSEW_100_200" in set(hits["key"].astype(str))
    assert set(hits["source_tile"].astype(str)) == {"W.bmp"}


def test_particles_only_shares_nsew_key_and_skips_labeled_location(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels")
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    north = {
        "id": 1,
        "source_tile": "N.png",
        "x_global": 100.4,
        "y_global": 200.4,
        "size": 20_000.0,
        "confidence": 0.8,
        "source_csv": "particles.csv",
    }
    store.apply_label(north, "particle", rgb)
    only = {
        "id": 2,
        "source_tile": "particles_only_2of4.png",
        "x_global": 100.4,
        "y_global": 200.4,
        "size": 21_000.0,
        "confidence": 0.9,
    }
    four = {
        "id": 3,
        "source_tile": "particles_only_4of4.png",
        "x_global": 100.4,
        "y_global": 200.4,
        "size": 22_000.0,
        "confidence": 0.7,
    }
    other = {
        "id": 4,
        "source_tile": "particles_only_2of4.png",
        "x_global": 1_000_000.0,
        "y_global": 50.0,
        "size": 23_000.0,
        "confidence": 0.6,
    }
    assert detection_key(only) == "NSEW_100_200"
    assert detection_key(four) == detection_key(north)
    detections = pd.DataFrame([only, four, other])
    queue = unlabeled_queue(
        [tag_table(detections)],
        store.labeled_keys(),
        nsew_merge_radius_nm=24.0,
        nsew_size_match_fraction=0.4,
    )
    assert list(queue["source_tile"]) == ["particles_only_2of4.png"]
    assert float(queue.iloc[0]["x_global"]) == 1_000_000.0


def test_snap_keys_reuses_nearby_nsew_label() -> None:
    from src.labeling.queue import snap_detection_keys_to_labels

    labels = pd.DataFrame(
        {
            "key": ["NSEW_100_200"],
            "label": ["particle"],
            "source_tile": ["N.png"],
            "x_global": [100.0],
            "y_global": [200.0],
            "size": [20_000.0],
        }
    )
    detections = pd.DataFrame(
        {
            "id": [1, 2],
            "source_tile": ["particles_only_2of4.png", "particles_only_2of4.png"],
            "x_global": [118.0, 1_000_000.0],
            "y_global": [204.0, 50.0],
            "size": [21_000.0, 22_000.0],
            "confidence": [0.9, 0.8],
        }
    )
    snapped = snap_detection_keys_to_labels(
        tag_table(detections), labels, merge_radius_nm=24.0, size_match_fraction=0.0
    )
    assert str(snapped.iloc[0]["key"]) == "NSEW_100_200"
    assert str(snapped.iloc[1]["key"]) != "NSEW_100_200"
    queue = unlabeled_queue(
        [tag_table(detections)],
        labels["key"],
        nsew_merge_radius_nm=24.0,
        nsew_size_match_fraction=0.0,
        labels=labels,
    )
    assert list(queue["x_global"]) == [1_000_000.0]


def test_tinder_queue_merges_nwes_offset_hits() -> None:
    detections = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "x_global": [100.0, 118.0, 108.0, 95.0, 800.0],
            "y_global": [200.0, 204.0, 198.0, 205.0, 200.0],
            "size": [20_000.0, 21_000.0, 22_000.0, 19_000.0, 23_000.0],
            "confidence": [0.9, 0.5, 0.7, 0.8, 0.6],
            "source_tile": ["N.bmp", "W.bmp", "E.bmp", "S.bmp", "N.bmp"],
        }
    )
    queue = unlabeled_queue(
        [tag_table(detections)],
        labeled_keys=(),
        nsew_merge_radius_nm=24.0,
        nsew_size_match_fraction=0.4,
    )
    assert len(queue) == 2
    same = queue.loc[queue["x_global"].astype(float) < 400]
    other = queue.loc[queue["x_global"].astype(float) > 400]
    assert len(same) == 1
    assert len(other) == 1
    assert str(same.iloc[0]["key"]).startswith("NSEW_")
    assert int(same.iloc[0]["nsew_count"]) == 4
    assert set(str(same.iloc[0]["nsew_dirs"]).split(",")) == {"N", "W", "E", "S"}


def test_crop_direction_views_shows_nwes(tmp_path: Path) -> None:
    import cv2

    for name in ("N", "W", "E", "S"):
        gray = np.zeros((48, 48), dtype=np.uint8)
        gray[20:28, 20:28] = 200
        assert cv2.imwrite(str(tmp_path / f"{name}.bmp"), gray)
    row = {
        "id": 1,
        "source_tile": "N.bmp",
        "x_global": 24.0,
        "y_global": 24.0,
        "size": 8.0,
        "confidence": 0.9,
    }
    views = crop_direction_views(row, _config(tmp_path))
    assert list(views) == ["N", "W", "E", "S"]
    assert all(view.rgb.size > 0 for view in views.values())


def test_brightest_direction_crop_picks_the_bright_angle(tmp_path: Path) -> None:
    import cv2

    from src.labeling.crops import brightest_direction_crop

    angles = tmp_path / "Inputs" / "Groundup v4" / "v3"
    stage = tmp_path / "Inputs" / "Groundup v4 ML off"
    angles.mkdir(parents=True)
    stage.mkdir()
    for name, value in (("N", 20), ("S", 40), ("E", 220), ("W", 30)):
        gray = np.full((64, 64), 5, dtype=np.uint8)
        if name == "E":
            gray[28:36, 28:36] = value
        assert cv2.imwrite(str(angles / f"{name}.bmp"), gray)
    mask = np.zeros((64, 64), dtype=np.uint8)
    assert cv2.imwrite(str(stage / "v3_2of4.bmp"), mask)
    row = {
        "id": 1,
        "source_tile": "v3_2of4.bmp",
        "x_global": 32.0,
        "y_global": 32.0,
        "size": 8.0,
        "confidence": 0.9,
    }
    crop = brightest_direction_crop(row, _config(stage))
    assert crop is not None
    assert crop.direction == "E"
    assert crop.tile_path.name == "E.bmp"
    assert _red_mask(crop.rgb).any()


def test_brightest_direction_crop_uses_groundup_v5_exposure_folder(tmp_path: Path) -> None:
    import cv2

    from src.labeling.crops import brightest_direction_crop

    angles = tmp_path / "Inputs" / "Groundup v5" / "v3 (10^5)"
    stage = tmp_path / "Inputs" / "Groundup v5 combi"
    angles.mkdir(parents=True)
    stage.mkdir()
    for name, value in (("N", 20), ("S", 40), ("E", 220), ("W", 30)):
        gray = np.full((64, 64), 5, dtype=np.uint8)
        if name == "E":
            gray[28:36, 28:36] = value
        assert cv2.imwrite(str(angles / f"{name}.png"), gray)
    mask = np.zeros((64, 64), dtype=np.uint8)
    assert cv2.imwrite(str(stage / "v3_2of4.png"), mask)
    row = {
        "id": 1,
        "source_tile": "v3_2of4.png",
        "x_global": 32.0,
        "y_global": 32.0,
        "size": 8.0,
        "confidence": 0.9,
    }
    crop = brightest_direction_crop(row, _config(stage))
    assert crop is not None
    assert crop.direction == "E"
    assert crop.tile_path.parent == angles


def test_store_round_trip_and_undo(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels")
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[:, :] = CIRCLE_COLOR
    record = {
        "id": 7,
        "source_tile": "R3_1_1_5X.tif",
        "x_global": 111.4,
        "y_global": 222.6,
        "size": 20_000.0,
        "confidence": 0.99,
        "source_csv": "particles.csv",
    }
    key = detection_key(record)
    dest = store.apply_label(record, "particle", rgb)
    assert dest.is_file()
    assert key in store.labeled_keys()
    loaded = store.by_label("particle")
    assert len(loaded) == 1
    assert int(loaded.iloc[0]["particle_id"]) == 7

    restored = store.undo_last()
    assert restored is not None
    assert str(restored["key"]) == key
    assert key not in store.labeled_keys()
    assert not dest.is_file()
    assert store.by_label("particle").empty


def test_store_apply_labels_writes_particle_and_not_particle(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels")
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    particle = {
        "id": 1,
        "source_tile": "R3_1_1_5X.tif",
        "x_global": 10.0,
        "y_global": 20.0,
        "size": 20_000.0,
        "confidence": 0.9,
    }
    other = {
        "id": 2,
        "source_tile": "R3_1_1_5X.tif",
        "x_global": 80.0,
        "y_global": 90.0,
        "size": 25_000.0,
        "confidence": 0.4,
    }
    dests = store.apply_labels(
        [(particle, "particle", rgb), (other, "not_particle", rgb)]
    )
    assert len(dests) == 2
    assert all(path.is_file() for path in dests)
    counts = store.counts()
    assert counts["particle"] == 1
    assert counts["not_particle"] == 1
    assert detection_key(particle) in store.labeled_keys()
    assert detection_key(other) in store.labeled_keys()


def test_last_run_csv_queue_floor_is_10_um() -> None:
    from src.labeling.queue import last_run_csv, load_particles_csv

    path = last_run_csv(Path(__file__).resolve().parents[1])
    assert path.is_file()
    table = load_particles_csv(path)
    kept = filter_min_size(table)
    assert len(kept) > 0
    assert float(kept["size"].min()) >= MIN_SIZE_NM


def test_last_run_csv_prefers_custom_recall_table(tmp_path: Path) -> None:
    from src.labeling.queue import last_run_csv, load_particles_csv

    folder = tmp_path / "my_recall"
    folder.mkdir()
    table = pd.DataFrame(
        {
            "id": [1],
            "x_global": [100.0],
            "y_global": [10.0],
            "size": [20_000.0],
            "confidence": [1.0],
            "source_tile": ["R3_1_1_5X.tif"],
        }
    )
    csv_path = folder / "particles.csv"
    table.to_csv(csv_path, index=False)
    found = last_run_csv(
        tmp_path, {"labeling": {"source_csv": "my_recall/particles.csv"}}
    )
    assert found == csv_path
    loaded = load_particles_csv(found)
    assert len(loaded) == 1
    assert "key" in loaded.columns


def test_load_last_pipeline_uses_pointer_not_recall_csv(tmp_path: Path) -> None:
    from src.labeling.queue import load_last_pipeline, write_last_pipeline

    csv_path = tmp_path / "out" / "particles.csv"
    csv_path.parent.mkdir()
    pd.DataFrame(
        {
            "id": [1],
            "x_global": [100.0],
            "y_global": [10.0],
            "size": [20_000.0],
            "confidence": [1.0],
            "source_tile": ["R3_1_1_5X.tif"],
        }
    ).to_csv(csv_path, index=False)
    write_last_pipeline(
        {"input_dir": "My tiles", "output_dir": str(tmp_path / "out"), "pixel_size_nm": 960.0},
        csv_path,
        pointer_dir=tmp_path,
    )
    found, run_config = load_last_pipeline({"input_dir": "other"}, pointer_dir=tmp_path)
    assert found == csv_path.resolve()
    assert run_config["input_dir"] == "My tiles"


def test_set_scoped_nsew_label_stays_on_one_field() -> None:
    north = {
        "source_tile": "v1/N.bmp",
        "x_global": 100.0,
        "y_global": 200.0,
        "size": 20_000.0,
        "confidence": 0.0,
    }
    assert detection_key(north) == "v1_NSEW_100_200"
    assert detection_key({**north, "source_tile": "v1/S.bmp"}) == "v1_NSEW_100_200"
    assert detection_key({**north, "source_tile": "v2/N.bmp"}) == "v2_NSEW_100_200"
    assert detection_key({**north, "source_tile": "N.bmp"}) == "NSEW_100_200"
    aliases = nsew_key_aliases("v1_NSEW_100_200")
    assert "v1_2of4_100_200" in aliases
    assert "NSEW_100_200" not in aliases
    assert "v2_2of4_100_200" not in aliases
    detections = pd.DataFrame(
        [
            {
                "id": 1,
                "source_tile": "v1_2of4.bmp",
                "x_global": 100.0,
                "y_global": 200.0,
                "size": 20_000.0,
                "confidence": 0.8,
            },
            {
                "id": 2,
                "source_tile": "v2_2of4.bmp",
                "x_global": 100.0,
                "y_global": 200.0,
                "size": 20_000.0,
                "confidence": 0.8,
            },
        ]
    )
    queue = unlabeled_queue([tag_table(detections)], ["v1_NSEW_100_200"])
    assert list(queue["key"].astype(str)) == ["v2_2of4_100_200"]
