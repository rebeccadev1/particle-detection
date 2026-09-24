"""1% ML-threshold sweep on R7 holdout rows 12–14 for every saved model."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import cfg_get, load_config  # noqa: E402
from src.detection.detector import (  # noqa: E402
    CANDIDATE_FEATURE_FIELDS,
    ParticleCandidate,
    compute_tile_residual,
)
from src.io.tile_loader import DEFAULT_FILENAME_PATTERN, load_tile_image  # noqa: E402
from src.labeling.crops import find_tile_path, global_nm_to_local_px, placement_for_tile  # noqa: E402
from src.labeling.inspect import last_run_class_counts  # noqa: E402
from src.labeling.queue import detection_key  # noqa: E402
from src.ml.features import dataframe_feature_matrix  # noqa: E402
from src.ml.infer import (  # noqa: E402
    _positive_proba,
    calibrate_scores,
    combine_cascade_scores,
    load_artifact,
)
from src.ml.patch_features import layout_from_candidate, patches_to_matrix  # noqa: E402
from src.ml.patches import PATCH_SIZE, extract_unmarked_channels  # noqa: E402
from src.preprocessing.corrections import apply_corrections  # noqa: E402

R7_CSV = ROOT.parent / "Outputs" / "R7 ML off 20 micron" / "particles.csv"
LABELS = ROOT / "labels" / "labels.csv"
OUT = ROOT.parent / "Outputs" / "R7 holdout rows 12-14 ML sweep.xlsx"
R7_DIR = str(ROOT.parent / "Inputs" / "R7 17 and 18-09")
N2_V2_PATTERN = r"n2_v2_(?P<row>\d+)_(?P<col>\d+)\.tiff?"
HOLDOUT_ROWS = {12, 13, 14}
MIN_SIZE_NM = 20_000.0
MODELS = (
    ("v3", ROOT / "models" / "particle_clf_v3.joblib"),
    ("v4", ROOT / "models" / "particle_clf_v4_residual.joblib"),
    ("v5", ROOT / "models" / "particle_clf_v5.joblib"),
    ("v6", ROOT / "models" / "particle_clf_v6.joblib"),
)


def _holdout_row(name: str) -> int | None:
    stem = Path(str(name)).name
    if not stem.startswith("n2_v2_"):
        return None
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def _config() -> dict:
    cfg = load_config(ROOT / "config.yaml")
    cfg["input_dir"] = R7_DIR
    cfg["filename_pattern"] = N2_V2_PATTERN
    cfg["overlap_fraction"] = 0.0
    cfg["pixel_size_nm"] = 960.0
    return cfg


def _candidate(row: pd.Series, placement, pixel: float) -> ParticleCandidate:
    x_local, y_local = global_nm_to_local_px(
        float(row["x_global"]), float(row["y_global"]), placement, pixel
    )
    kwargs = {
        name: float(row[name])
        for name in CANDIDATE_FEATURE_FIELDS
        if name in row.index and pd.notna(row[name])
    }
    return ParticleCandidate(
        y_local=y_local,
        x_local=x_local,
        size=float(row["size"]) / pixel,
        confidence=float(row["confidence"]),
        **kwargs,
    )


_WORKER: dict = {}


def _init_workers(model_paths: list[tuple[str, str]], config: dict) -> None:
    global _WORKER
    _WORKER = {
        "config": config,
        "artifacts": [(name, load_artifact(path)) for name, path in model_paths],
    }


def _model_scores(
    candidates: list[ParticleCandidate],
    channels: np.ndarray,
    layouts: list[np.ndarray],
    artifact: dict,
    config: dict,
) -> np.ndarray:
    matrix = patches_to_matrix(channels, layouts)
    hog_scores = calibrate_scores(artifact, _positive_proba(artifact["model"], matrix))
    if str(artifact.get("kind") or "patch") != "cascade":
        return hog_scores
    band_low = float(artifact.get("band_low") if artifact.get("band_low") is not None else cfg_get(config, "ml.band_low", 0.20))
    band_high = float(artifact.get("band_high") if artifact.get("band_high") is not None else cfg_get(config, "ml.band_high", 0.80))
    uncertain = (hog_scores >= band_low) & (hog_scores < band_high)
    cnn_subset = (
        artifact["cnn"].predict_proba(channels[uncertain])
        if uncertain.any()
        else np.empty((0,), dtype=np.float64)
    )
    return combine_cascade_scores(hog_scores, cnn_subset, band_low, band_high, uncertain=uncertain)


def _score_tile(payload: tuple[str, list[dict]]) -> tuple[list[int], dict[str, list[float]]]:
    tile_name, records = payload
    config = _WORKER["config"]
    pixel = float(config["pixel_size_nm"])
    path = find_tile_path(tile_name, input_dir=config["input_dir"])
    placement = placement_for_tile(path, config)
    image = apply_corrections(load_tile_image(path), config)
    frame = pd.DataFrame(records)
    candidates = [_candidate(row, placement, pixel) for _, row in frame.iterrows()]
    pattern = str(cfg_get(config, "filename_pattern", DEFAULT_FILENAME_PATTERN))
    _corrected, residual = compute_tile_residual(image, config)
    corrected = np.asarray(image, dtype=np.float32)
    by_size: dict[int, np.ndarray] = {}
    layouts = [
        layout_from_candidate(cand, pixel, tile_name, pattern) for cand in candidates
    ]
    scored: dict[str, list[float]] = {}
    for name, artifact in _WORKER["artifacts"]:
        patch_size = int(artifact.get("patch_size") or PATCH_SIZE)
        channels = by_size.get(patch_size)
        if channels is None:
            channels = np.stack(
                [
                    extract_unmarked_channels(
                        corrected,
                        residual,
                        float(cand.x_local),
                        float(cand.y_local),
                        size=patch_size,
                    )
                    for cand in candidates
                ],
                axis=0,
            )
            by_size[patch_size] = channels
        scored[name] = _model_scores(candidates, channels, layouts, artifact, config).tolist()
    return frame["index"].astype(int).tolist(), scored


def _patch_scores(table: pd.DataFrame, artifacts: list[tuple[str, dict]], config: dict) -> dict[str, np.ndarray]:
    scores = {name: np.zeros(len(table), dtype=np.float64) for name, _ in artifacts}
    grouped = table.groupby(table["source_tile"].map(lambda s: Path(str(s)).name), sort=False)
    payloads = []
    for tile_name, group in grouped:
        records = group.reset_index().to_dict(orient="records")
        payloads.append((tile_name, records))
    workers = 5
    print(f"Scoring {len(payloads)} tiles on {workers} cores…", flush=True)
    paths = [(name, str(path)) for name, path in MODELS if name in scores]
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_workers,
        initargs=(paths, config),
    ) as pool:
        futures = [pool.submit(_score_tile, payload) for payload in payloads]
        for future in as_completed(futures):
            indexes, tile_scores = future.result()
            done += 1
            print(f"  {done}/{len(payloads)} tiles", flush=True)
            for name, values in tile_scores.items():
                scores[name][np.asarray(indexes, dtype=int)] = np.asarray(values, dtype=np.float64)
    return scores


def _sweep_rows(
    version: str,
    table: pd.DataFrame,
    scores: np.ndarray,
    labels: pd.DataFrame,
    tiles: list[str],
) -> list[dict]:
    rows = []
    for step in range(0, 101):
        threshold = step / 100.0
        keep = table.loc[scores >= threshold]
        counts = last_run_class_counts(
            keep,
            labels,
            tile_names=tiles,
            min_size_nm=MIN_SIZE_NM,
            folder_tiles=tiles,
        )
        rows.append(
            {
                "version": version,
                "threshold": threshold,
                "hits": int(counts["n_detected"]),
                "real": int(counts["n_real"]),
                "fake": int(counts["n_fake"]),
                "missed": int(counts["n_undetected"]),
                "P": float(counts["precision"]),
                "R": float(counts["recall"]),
                "H": float(counts["unhappiness"]),
            }
        )
    return rows


def main() -> int:
    config = _config()
    proposals = pd.read_csv(R7_CSV)
    proposals["source_tile"] = proposals["source_tile"].map(lambda s: Path(str(s)).name)
    row = proposals["source_tile"].map(_holdout_row)
    proposals = proposals.loc[row.isin(HOLDOUT_ROWS)].reset_index(drop=True)
    proposals["key"] = [detection_key(record) for _, record in proposals.iterrows()]
    labels = pd.read_csv(LABELS)
    labels["source_tile"] = labels["source_tile"].astype(str).map(lambda s: Path(s).name)
    label_row = labels["source_tile"].map(_holdout_row)
    labels = labels.loc[label_row.isin(HOLDOUT_ROWS)].copy()
    tiles = sorted(set(proposals["source_tile"]) | set(labels["source_tile"]))
    print(
        f"Holdout proposals {len(proposals)} on {proposals['source_tile'].nunique()} tiles. "
        f"Labels {len(labels)} on {len(tiles)} tiles.",
        flush=True,
    )
    artifacts = [(name, load_artifact(path)) for name, path in MODELS]
    score_bank: dict[str, np.ndarray] = {}
    residual = [(name, art) for name, art in artifacts if art.get("kind") == "residual"]
    patches = [(name, art) for name, art in artifacts if art.get("kind") != "residual"]
    for name, art in residual:
        print(f"Scoring {name} from residual features…", flush=True)
        score_bank[name] = _positive_proba(art["model"], dataframe_feature_matrix(proposals))
    if patches:
        print("Scoring patch/cascade models from tiles…", flush=True)
        score_bank.update(_patch_scores(proposals, patches, config))
    rows: list[dict] = []
    for name, _art in artifacts:
        rows.extend(_sweep_rows(name, proposals, score_bank[name], labels, tiles))
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="sweep", index=False)
    best = table.loc[table.groupby("version")["H"].idxmin()]
    print(f"Wrote {OUT}", flush=True)
    print(best.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
