"""Explain why labeled flakes are missing from the current detector run.

``python -m src.labeling.audit_misses`` maps particle / tile_inspect labels
onto the input tiles, keeps those with no nearby detection, and runs
``trace_locations`` so each miss shows the first dropping gate.

Use ``--write-inspect`` to copy unmatched labels into ``tile_inspect`` rows
with the current folder's global coordinates (Tiles-tab marks).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.config import PACKAGE_ROOT, cfg_get, load_config
from src.detection.detector import trace_locations
from src.io.tile_loader import (
    DEFAULT_FILENAME_PATTERN,
    matching_tile_paths,
    peek_tile_hw,
)
from src.labeling.crops import (
    TileImageCache,
    find_tile_path,
    local_px_to_global_nm,
    placement_for_tile,
)
from src.labeling.inspect import (
    MISS_SOURCE,
    candidate_placements,
    local_xy_on_tile,
    missed_particle_record,
    preview_click_crop,
)
from src.labeling.queue import load_particles_csv
from src.labeling.store import LabelStore
from src.preprocessing.corrections import apply_corrections

NEAR_PX = 20.0


POSITIVE_LABEL = "particle"


def format_trace_caption(row: Mapping[str, Any]) -> str:
    """One-line Tiles-tab caption for a ``trace_locations`` row."""
    reason = str(row.get("reason", "unknown"))
    residual = float(row.get("residual", float("nan")))
    edge = float(row.get("edge_distance_px", float("nan")))
    return f"Gate: {reason} · residual {residual:.3f} · edge {edge:.1f} px"


def trace_click(
    image: Any,
    config: dict[str, Any],
    y_local: float,
    x_local: float,
) -> dict[str, Any]:
    """Preprocess ``image`` and return the gate that drops ``(y, x)``."""
    corrected = apply_corrections(image, config)
    rows = trace_locations(corrected, config, [(float(y_local), float(x_local))])
    if not rows:
        return {
            "reason": "unknown",
            "residual": float("nan"),
            "edge_distance_px": float("nan"),
            "nearest_blob_px": float("nan"),
            "nearest_kept_px": float("nan"),
        }
    return dict(rows[0])


def input_tile_names(config: Mapping[str, Any]) -> set[str]:
    """Basenames of TIFFs in ``config.input_dir``."""
    folder = cfg_get(dict(config), "input_dir", "")
    if not folder:
        return set()
    pattern = str(cfg_get(dict(config), "filename_pattern", DEFAULT_FILENAME_PATTERN))
    try:
        paths = matching_tile_paths(folder, pattern)
    except FileNotFoundError:
        return set()
    return {path.name for path in paths}


def particle_label_rows(
    labels: pd.DataFrame,
    tile_names: Iterable[str] | None = None,
    inspect_only: bool = False,
) -> pd.DataFrame:
    """Labeled real particles, optionally restricted to tiles / tile_inspect."""
    if labels.empty:
        return labels.copy()
    keep = labels.loc[labels["label"].astype(str) == POSITIVE_LABEL].copy()
    if inspect_only:
        keep = keep.loc[keep["source_csv"].astype(str) == MISS_SOURCE]
    if tile_names is not None:
        names = {Path(str(name)).name for name in tile_names}
        tile = keep["source_tile"].astype(str).map(lambda path: Path(path).name)
        keep = keep.loc[tile.isin(names)]
    return keep.reset_index(drop=True)


def _tile_name(row: Mapping[str, Any]) -> str:
    return Path(str(row["source_tile"])).name


def localize_label_row(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    placements: Sequence[Any],
    width: int,
    height: int,
) -> tuple[float, float]:
    """Map a label's global nm onto the tile using folder / wafer placements."""
    pixel_size = float(cfg_get(dict(config), "pixel_size_nm", 960.0))
    return local_xy_on_tile(
        float(row["x_global"]),
        float(row["y_global"]),
        placements,
        pixel_size,
        width,
        height,
    )


def unmatched_particle_marks(
    labels: pd.DataFrame,
    detections: pd.DataFrame | None,
    config: dict[str, Any],
    near_px: float = NEAR_PX,
    inspect_only: bool = False,
    keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Particle labels on the input tiles with no detection within ``near_px``."""
    tiles = input_tile_names(config)
    marked = particle_label_rows(labels, tiles, inspect_only=inspect_only)
    if keys is not None:
        wanted = {str(key) for key in keys}
        marked = marked.loc[marked["key"].astype(str).isin(wanted)].reset_index(drop=True)
    if marked.empty:
        return pd.DataFrame()

    det_by_tile: dict[str, list[tuple[float, float]]] = {}
    if detections is not None and not detections.empty:
        for _, row in detections.iterrows():
            det_by_tile.setdefault(Path(str(row["source_tile"])).name, []).append(
                (float(row["x_global"]), float(row["y_global"]))
            )

    pixel_size = float(cfg_get(config, "pixel_size_nm", 960.0))
    folder = cfg_get(config, "input_dir", "")
    origin_cache: dict[str, tuple[int, int]] = {}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for record in marked.to_dict(orient="records"):
        tile_name = _tile_name(record)
        path = find_tile_path(tile_name, input_dir=folder)
        height, width = peek_tile_hw(path)
        placements = candidate_placements(path, config, origin_cache=origin_cache)
        x_local, y_local = localize_label_row(record, config, placements, width, height)
        if not (0.0 <= x_local < float(width) and 0.0 <= y_local < float(height)):
            continue
        dedupe = (tile_name, int(round(x_local)), int(round(y_local)))
        if dedupe in seen:
            continue
        nearest = float("inf")
        folder_placement = placement_for_tile(path, config, origin_cache=origin_cache)
        det_pts = det_by_tile.get(tile_name, [])
        for x_g, y_g in det_pts:
            dx, dy = local_xy_on_tile(x_g, y_g, placements, pixel_size, width, height)
            dist = float(((dx - x_local) ** 2 + (dy - y_local) ** 2) ** 0.5)
            if dist < nearest:
                nearest = dist
        if nearest <= float(near_px):
            continue
        seen.add(dedupe)
        x_global, y_global = local_px_to_global_nm(
            x_local, y_local, folder_placement, pixel_size
        )
        out = dict(record)
        out["source_tile"] = tile_name
        out["x_local"] = float(x_local)
        out["y_local"] = float(y_local)
        out["folder_x_global"] = float(x_global)
        out["folder_y_global"] = float(y_global)
        out["nearest_kept_px"] = float(nearest)
        rows.append(out)
    return pd.DataFrame(rows)


def audit_marks(
    marks: pd.DataFrame,
    config: dict[str, Any],
    cache: TileImageCache | None = None,
) -> pd.DataFrame:
    """Run ``trace_locations`` on each unmatched mark (one TIFF decode per tile)."""
    if marks.empty:
        return marks.copy()
    cache = cache or TileImageCache()
    folder = cfg_get(config, "input_dir", "")
    traces: list[dict[str, Any]] = []
    for tile_name, group in marks.groupby(
        marks["source_tile"].map(lambda path: Path(str(path)).name),
        sort=True,
    ):
        path = find_tile_path(str(tile_name), input_dir=folder)
        image = cache.load(path)
        corrected = apply_corrections(image, config)
        locations = [
            (float(row["y_local"]), float(row["x_local"])) for _, row in group.iterrows()
        ]
        explained = trace_locations(corrected, config, locations)
        for record, trace in zip(group.to_dict(orient="records"), explained):
            merged = dict(record)
            merged.update(trace)
            traces.append(merged)
    return pd.DataFrame(traces)


def promote_unmatched_to_inspect(
    marks: pd.DataFrame,
    config: dict[str, Any],
    store: LabelStore,
    cache: TileImageCache | None = None,
) -> list[dict[str, Any]]:
    """Write Tiles-tab ``tile_inspect`` rows for unmatched flakes."""
    if marks.empty:
        return []
    cache = cache or TileImageCache()
    folder = cfg_get(config, "input_dir", "")
    pixel_size = float(cfg_get(config, "pixel_size_nm", 960.0))
    existing = store.load()
    known = set(existing["key"].astype(str)) if not existing.empty else set()
    written: list[dict[str, Any]] = []
    for record in marks.to_dict(orient="records"):
        tile_name = _tile_name(record)
        path = find_tile_path(tile_name, input_dir=folder)
        image = cache.load(path)
        x_global = float(record.get("folder_x_global", record["x_global"]))
        y_global = float(record.get("folder_y_global", record["y_global"]))
        size_nm = float(record.get("size") or 15_000.0)
        inspect = missed_particle_record(tile_name, x_global, y_global, size_nm)
        key = str(inspect["key"])
        if key in known:
            df = store.load()
            mask = df["key"].astype(str) == key
            if mask.any() and str(df.loc[mask, "source_csv"].iloc[-1]) != MISS_SOURCE:
                df.loc[mask, "source_csv"] = MISS_SOURCE
                store._write(df)
                inspect["key"] = key
                written.append(inspect)
            continue
        crop = preview_click_crop(
            image,
            float(record["x_local"]),
            float(record["y_local"]),
            size_nm,
            pixel_size,
        )
        store.apply_label(inspect, POSITIVE_LABEL, crop)
        known.add(key)
        written.append(inspect)
    return written


def default_label_path() -> Path:
    return PACKAGE_ROOT / "labels" / "labels.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--labels", type=Path, default=default_label_path())
    parser.add_argument(
        "--detections",
        type=Path,
        default=None,
        help="particles.csv to treat as the current proposal set.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only autopsy existing tile_inspect rows.",
    )
    parser.add_argument(
        "--write-inspect",
        action="store_true",
        help="Copy unmatched labels into tile_inspect marks (current-folder globals).",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="Optional label keys to include.",
    )
    parser.add_argument("--near-px", type=float, default=NEAR_PX)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    labels = pd.read_csv(args.labels)
    detections = load_particles_csv(args.detections) if args.detections else None
    marks = unmatched_particle_marks(
        labels,
        detections,
        config,
        near_px=float(args.near_px),
        inspect_only=bool(args.inspect_only),
        keys=args.keys,
    )
    if marks.empty:
        print("No unmatched particle labels on the input tiles.")
        return 0
    audited = audit_marks(marks, config)
    cols = [
        col
        for col in (
            "source_tile",
            "key",
            "reason",
            "residual",
            "edge_distance_px",
            "nearest_blob_px",
            "nearest_kept_px",
            "size",
        )
        if col in audited.columns
    ]
    print(audited[cols].to_string(index=False))
    if "reason" in audited.columns:
        print("\nreason counts:")
        print(audited["reason"].value_counts().to_string())
    if args.write_inspect:
        store = LabelStore(Path(args.labels).parent)
        written = promote_unmatched_to_inspect(marks, config, store)
        print(f"\nWrote {len(written)} tile_inspect marks to {store.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
