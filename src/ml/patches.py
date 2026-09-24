"""Unmarked two-channel crops for the FP filter.

Reads the source TIFF and the DoG residual. Never uses circled JPEGs under
``labels/crops/`` — the red marker would leak into any patch model.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd

from src.config import cfg_get, load_config, PACKAGE_ROOT
from src.detection.detector import compute_tile_residual
from src.io.tile_loader import DEFAULT_FILENAME_PATTERN, parse_tile_filename
from src.labeling.crops import (
    TileImageCache,
    find_tile_path,
    global_nm_to_local_px,
    placement_for_tile,
)
from src.preprocessing.corrections import apply_corrections

PATCH_SIZE = 96
POSITIVE = "particle"
NEGATIVE = "not_particle"


@dataclass(frozen=True)
class UnmarkedPatch:
    """Two-channel crop (raw, residual) with label metadata."""

    key: str
    label: str
    source_tile: str
    x_local: float
    y_local: float
    channels: np.ndarray  # (2, PATCH_SIZE, PATCH_SIZE) float32


class TileResidualCache:
    """Decode a tile once and reuse the corrected image plus residual."""

    def __init__(self) -> None:
        self._images = TileImageCache()
        self.path: Path | None = None
        self.corrected: np.ndarray | None = None
        self.residual: np.ndarray | None = None

    def maps(
        self,
        path: Path,
        config: dict[str, Any],
        fft_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        resolved = Path(path).resolve()
        if (
            self.path == resolved
            and self.corrected is not None
            and self.residual is not None
        ):
            return self.corrected, self.residual
        image = self._images.load(resolved)
        corrected = apply_corrections(image, config)
        _array, residual = compute_tile_residual(corrected, config, fft_mask=fft_mask)
        self.path = resolved
        self.corrected = np.asarray(corrected, dtype=np.float32)
        self.residual = np.asarray(residual, dtype=np.float32)
        return self.corrected, self.residual


def default_train_config(path: str | Path | None = None) -> dict[str, Any]:
    destination = Path(path) if path is not None else PACKAGE_ROOT / "config.yaml"
    if destination.is_file():
        return load_config(destination)
    return {}


def extract_centered(
    image: np.ndarray,
    x_local: float,
    y_local: float,
    size: int = PATCH_SIZE,
) -> np.ndarray:
    """Square crop centred on ``(x_local, y_local)``, reflect-padded at edges."""
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D tile, got shape {array.shape}")
    half = int(size) // 2
    cx = int(round(float(x_local)))
    cy = int(round(float(y_local)))
    y0, y1 = cy - half, cy - half + int(size)
    x0, x1 = cx - half, cx - half + int(size)
    pad_y0 = max(0, -y0)
    pad_x0 = max(0, -x0)
    pad_y1 = max(0, y1 - array.shape[0])
    pad_x1 = max(0, x1 - array.shape[1])
    if pad_y0 or pad_x0 or pad_y1 or pad_x1:
        array = np.pad(
            array,
            ((pad_y0, pad_y1), (pad_x0, pad_x1)),
            mode="reflect",
        )
        y0 += pad_y0
        y1 += pad_y0
        x0 += pad_x0
        x1 += pad_x0
    patch = np.asarray(array[y0:y1, x0:x1], dtype=np.float32)
    if patch.shape != (int(size), int(size)):
        patch = np.resize(patch, (int(size), int(size)))
    return patch


def stack_channels(raw: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """``(2, H, W)`` float32 pair. Independent 1–99 percentile stretch."""
    return np.stack([_stretch(raw), _stretch(residual)], axis=0).astype(np.float32)


def extract_unmarked_channels(
    corrected: np.ndarray,
    residual: np.ndarray,
    x_local: float,
    y_local: float,
    size: int = PATCH_SIZE,
) -> np.ndarray:
    """Two-channel unmarked crop from in-memory tile maps."""
    raw = extract_centered(corrected, x_local, y_local, size=size)
    res = extract_centered(residual, x_local, y_local, size=size)
    return stack_channels(raw, res)


def local_xy_for_record(
    record: Mapping[str, Any],
    config: dict[str, Any],
    origin_cache: dict[str, tuple[int, int]] | None = None,
) -> tuple[Path, float, float]:
    """Map a label row to ``(tile_path, x_local, y_local)`` in pixels."""
    tile_name = str(record["source_tile"])
    tile_path = find_tile_path(tile_name, input_dir=cfg_get(config, "input_dir", None))
    placement = placement_for_tile(tile_path, config, origin_cache=origin_cache)
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    x_local, y_local = global_nm_to_local_px(
        float(record["x_global"]),
        float(record["y_global"]),
        placement,
        pixel_size,
    )
    return tile_path, float(x_local), float(y_local)


def _limit_blas_threads() -> None:
    """Keep each worker on one core so a process pool does not oversubscribe."""
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1).__enter__()
    except Exception:
        pass


def _patches_for_tile(
    tile_name: str,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    size: int,
) -> list[UnmarkedPatch]:
    """Extract every unmarked crop on one tile. Missing tiles return []."""
    if not records:
        return []
    loader = TileResidualCache()
    origin_cache: dict[str, tuple[int, int]] = {}
    try:
        tile_path, _, _ = local_xy_for_record(records[0], config, origin_cache)
        corrected, residual = loader.maps(tile_path, config)
    except FileNotFoundError:
        return []
    patches: list[UnmarkedPatch] = []
    for record in records:
        try:
            _path, x_local, y_local = local_xy_for_record(record, config, origin_cache)
        except FileNotFoundError:
            continue
        patches.append(
            UnmarkedPatch(
                key=str(record.get("key", "")),
                label=str(record["label"]),
                source_tile=str(tile_name),
                x_local=x_local,
                y_local=y_local,
                channels=extract_unmarked_channels(
                    corrected, residual, x_local, y_local, size=size
                ),
            )
        )
    return patches


def iter_unmarked_patches(
    labels: pd.DataFrame,
    config: dict[str, Any],
    size: int = PATCH_SIZE,
    cache: TileResidualCache | None = None,
) -> Iterator[UnmarkedPatch]:
    """Yield unmarked patches for particle / not_particle rows.

    Circled JPEG paths in the CSV are ignored. Missing tiles are skipped.
    """
    keep = labels.loc[labels["label"].astype(str).isin((POSITIVE, NEGATIVE))].copy()
    if keep.empty:
        return
    loader = cache if cache is not None else TileResidualCache()
    origin_cache: dict[str, tuple[int, int]] = {}
    grouped = keep.groupby(keep["source_tile"].astype(str), sort=False)
    for tile_name, group in grouped:
        first = group.iloc[0].to_dict()
        try:
            tile_path, _, _ = local_xy_for_record(first, config, origin_cache)
            corrected, residual = loader.maps(tile_path, config)
        except FileNotFoundError:
            continue
        print(f"  {tile_name}  n={len(group)}", flush=True)
        for record in group.to_dict(orient="records"):
            try:
                _path, x_local, y_local = local_xy_for_record(
                    record, config, origin_cache
                )
            except FileNotFoundError:
                continue
            channels = extract_unmarked_channels(
                corrected, residual, x_local, y_local, size=size
            )
            yield UnmarkedPatch(
                key=str(record.get("key", "")),
                label=str(record["label"]),
                source_tile=str(tile_name),
                x_local=x_local,
                y_local=y_local,
                channels=channels,
            )


def collect_unmarked_dataset(
    labels: pd.DataFrame,
    config: dict[str, Any],
    size: int = PATCH_SIZE,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Stack patches, labels, tile groups, and a metadata table.

    ``workers`` is the process-pool size. ``None`` uses every CPU core.
    One tile, or ``workers=1``, stays in this process.
    """
    keep = labels.loc[labels["label"].astype(str).isin((POSITIVE, NEGATIVE))].copy()
    grouped = [
        (str(tile_name), group.to_dict(orient="records"))
        for tile_name, group in keep.groupby(keep["source_tile"].astype(str), sort=False)
    ]
    n_workers = os.cpu_count() or 1 if workers is None else max(1, int(workers))
    n_workers = min(n_workers, max(1, len(grouped)))
    found_by_tile: list[list[UnmarkedPatch]] = []
    if n_workers <= 1:
        for tile_name, records in grouped:
            found = _patches_for_tile(tile_name, records, config, size)
            if found:
                print(f"  {tile_name}  n={len(found)}", flush=True)
                found_by_tile.append(found)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_limit_blas_threads,
        ) as pool:
            futures = {
                pool.submit(_patches_for_tile, tile_name, records, config, size): index
                for index, (tile_name, records) in enumerate(grouped)
            }
            slots: list[list[UnmarkedPatch] | None] = [None] * len(grouped)
            for future in as_completed(futures):
                index = futures[future]
                found = future.result()
                slots[index] = found
                if found:
                    print(f"  {grouped[index][0]}  n={len(found)}", flush=True)
            found_by_tile = [slot for slot in slots if slot]
    rows: list[dict[str, Any]] = []
    patches: list[np.ndarray] = []
    for tile_patches in found_by_tile:
        for patch in tile_patches:
            patches.append(patch.channels)
            rows.append(
                {
                    "key": patch.key,
                    "label": patch.label,
                    "source_tile": patch.source_tile,
                    "x_local": patch.x_local,
                    "y_local": patch.y_local,
                }
            )
    if not patches:
        raise ValueError(
            "No unmarked patches could be extracted. Check that the labeled "
            "tiles exist under Inputs/ and that labels.csv has coordinates."
        )
    meta = pd.DataFrame(rows)
    y = (meta["label"].astype(str) == POSITIVE).astype(int).to_numpy()
    groups = meta["source_tile"].astype(str).to_numpy()
    stacked = np.stack(patches, axis=0)
    return stacked, y, groups, meta


def tile_row_col(
    tile_name: str,
    pattern: str = DEFAULT_FILENAME_PATTERN,
) -> tuple[float, float]:
    """Grid indices from the filename, or (0, 0) when parsing fails."""
    try:
        meta = parse_tile_filename(Path(str(tile_name)).name, pattern)
    except (ValueError, TypeError):
        return 0.0, 0.0
    return float(meta.get("row", 0) or 0), float(meta.get("col", 0) or 0)


def _stretch(channel: np.ndarray) -> np.ndarray:
    array = np.asarray(channel, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
