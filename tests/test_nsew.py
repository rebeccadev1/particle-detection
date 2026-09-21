"""Tests for N/S/E/W folder discovery and combined output size."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.nsew import (
    MIN_COUNTS,
    combine_nsew_folder,
    find_nsew_images,
    list_direction_images,
    nsew_output_dir,
    symmetry_signal_4way,
)


def _write_gray(path: Path, value: int, size: tuple[int, int] = (24, 32), dpi=(254, 254)) -> None:
    array = np.full(size, value, dtype=np.uint8)
    Image.fromarray(array).save(path, dpi=dpi)


def test_find_nsew_images_matches_stems(tmp_path: Path) -> None:
    _write_gray(tmp_path / "N.bmp", 10)
    _write_gray(tmp_path / "S.bmp", 20)
    _write_gray(tmp_path / "E.png", 30)
    _write_gray(tmp_path / "W.png", 40)
    found = find_nsew_images(tmp_path)
    assert found["N"].name == "N.bmp"
    assert found["S"].name == "S.bmp"
    assert found["E"].name == "E.png"
    assert found["W"].name == "W.png"


def test_list_direction_images_allows_partial_nwes(tmp_path: Path) -> None:
    _write_gray(tmp_path / "N.bmp", 10)
    _write_gray(tmp_path / "W.bmp", 20)
    found = list_direction_images(tmp_path)
    assert set(found) == {"N", "W"}


def test_combine_writes_same_size_in_folder(tmp_path: Path) -> None:
    height, width = 24, 32
    for name, value in (("N", 40), ("S", 50), ("E", 60), ("W", 70)):
        _write_gray(tmp_path / f"{name}.bmp", value, size=(height, width))
    result = combine_nsew_folder(tmp_path)
    dest = nsew_output_dir(tmp_path)
    assert result["output_dir"] == dest
    assert dest.name == f"Output {tmp_path.name}"
    assert result["width"] == width
    assert result["height"] == height
    for min_count in MIN_COUNTS:
        for kind, folder in (
            ("combi", "Combi"),
            ("particles_only", "Particles only"),
            ("symmetry_map", "Symmetry"),
        ):
            path = dest / folder / f"{kind}_{min_count}of4.bmp"
            assert path.is_file()
            with Image.open(path) as image:
                assert image.size == (width, height)
                # BMP inputs often have dpi (0,0); we must not copy that.
                dpi = image.info.get("dpi")
                if dpi is not None:
                    assert dpi[0] > 0 and dpi[1] > 0
    for folder, name in (
        ("Combi", "combi.bmp"),
        ("Particles only", "particles_only.bmp"),
        ("Symmetry", "symmetry_map.bmp"),
    ):
        assert (dest / folder / name).is_file()
        assert not (tmp_path / name).is_file()
        assert not (dest / name).is_file()


def test_combine_matches_png_inputs(tmp_path: Path) -> None:
    height, width = 16, 20
    for name, value in (("N", 40), ("S", 50), ("E", 60), ("W", 70)):
        array = np.full((height, width), value, dtype=np.uint8)
        Image.fromarray(array).save(tmp_path / f"{name}.png", dpi=(254, 254))
    result = combine_nsew_folder(tmp_path)
    dest = nsew_output_dir(tmp_path)
    assert result["output_dir"] == dest
    path = dest / "Combi" / "combi_2of4.png"
    assert path.is_file()
    with Image.open(path) as image:
        assert image.size == (width, height)
        assert image.info.get("dpi") == (254.0, 254.0)


def test_nsew_output_dir_uses_workspace_outputs(tmp_path: Path, monkeypatch) -> None:
    from src import nsew as nsew_mod

    inputs = tmp_path / "Inputs"
    outputs = tmp_path / "Outputs"
    folder = inputs / "Groundup v3"
    folder.mkdir(parents=True)
    monkeypatch.setattr(nsew_mod, "DEFAULT_INPUT_DIR", inputs)
    monkeypatch.setattr(
        nsew_mod, "resolve_output_dir", lambda name: (outputs / name).resolve()
    )
    dest = nsew_mod.nsew_output_dir(folder)
    assert dest == (outputs / "Output Groundup v3").resolve()


def test_min_count_uses_matching_rank() -> None:
    dark = np.zeros((3, 3), dtype=np.float32)
    bright = np.ones((3, 3), dtype=np.float32)
    stack = np.stack([bright, bright, dark, dark], axis=0)
    sorted_stack = np.sort(stack, axis=0)
    sum_val = np.sum(stack, axis=0)
    iso2, _ = symmetry_signal_4way(sorted_stack, sum_val, 2)
    iso3, _ = symmetry_signal_4way(sorted_stack, sum_val, 3)
    iso4, _ = symmetry_signal_4way(sorted_stack, sum_val, 4)
    assert float(iso2.mean()) > 0.5
    assert float(iso3.mean()) == 0.0
    assert float(iso4.mean()) == 0.0
