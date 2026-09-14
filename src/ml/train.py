"""Train a tile-grouped particle vs not-particle classifier from labels.csv.

Does not read circled crop JPEGs. Default ``--mode cascade`` extracts unmarked
TIFF/residual patches, trains HOG ExtraTrees, then a tiny CNN on the uncertain
band. ``--mode residual`` is the old detector-float ExtraTrees.

Usage (from particle_detection/):

    python -m src.ml.train
    python -m src.ml.train --mode patch
    python -m src.ml.train --mode residual --detections ../Outputs/recall/particles.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

from src.config import DEFAULT_OUTPUT_DIR, PACKAGE_ROOT, cfg_get
from src.io.tile_loader import DEFAULT_FILENAME_PATTERN
from src.labeling.queue import detection_key
from src.ml.cnn import train_tiny_cnn
from src.ml.features import FEATURE_COLUMNS, dataframe_feature_matrix
from src.ml.infer import KIND_CASCADE, KIND_PATCH, KIND_RESIDUAL, resolve_model_path
from src.ml.patch_features import (
    PATCH_FEATURE_NAMES,
    layout_from_record,
    patches_to_matrix,
)
from src.ml.patches import (
    PATCH_SIZE,
    POSITIVE,
    NEGATIVE,
    collect_unmarked_dataset,
    default_train_config,
)

MIN_RECALL = 0.95
DEFAULT_BAND_LOW = 0.20
DEFAULT_BAND_HIGH = 0.80


def default_label_path() -> Path:
    return PACKAGE_ROOT / "labels" / "labels.csv"


def default_detection_paths() -> list[Path]:
    preferred = DEFAULT_OUTPUT_DIR / "recall" / "particles.csv"
    if preferred.is_file():
        return [preferred]
    fallback = DEFAULT_OUTPUT_DIR / "allemaal 08" / "09" / "particles.csv"
    if fallback.is_file():
        return [fallback]
    return []


def load_labeled_features(
    labels_path: str | Path,
    detection_paths: list[str | Path],
) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    if labels.empty:
        raise ValueError(f"No labels in {labels_path}")
    keep = labels.loc[labels["label"].astype(str).isin((POSITIVE, NEGATIVE))].copy()
    keep["key"] = keep["key"].astype(str)
    detections = _load_feature_tables(detection_paths)
    merged = keep.merge(detections, on="key", how="inner", suffixes=("", "_det"))
    if merged.empty:
        raise ValueError(
            "No labeled rows matched a detections table with features. "
            "Re-run the pipeline so particles.csv includes circularity and "
            "related columns, then train again."
        )
    return merged.reset_index(drop=True)


def _load_feature_tables(paths: list[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        missing = [name for name in FEATURE_COLUMNS[2:] if name not in df.columns]
        if missing:
            raise ValueError(
                f"{path} is missing feature columns {missing}. "
                "Circled JPEG crops are not used; re-run detection to persist features."
            )
        tagged = df.copy()
        tagged["key"] = [detection_key(row) for _, row in tagged.iterrows()]
        frames.append(tagged)
    if not frames:
        raise ValueError("No detections CSVs were given.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["key"], keep="last")
    return combined


def choose_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    min_recall: float = MIN_RECALL,
) -> tuple[float, dict[str, float]]:
    """Lowest threshold that still meets ``min_recall``, maximizing precision."""
    best_t = 0.0
    best_prec = -1.0
    best_rec = 0.0
    fallback_t = 0.35
    fallback_score = -1.0
    for threshold in np.linspace(0.05, 0.9, 18):
        pred = proba >= threshold
        rec = float(recall_score(y_true, pred, zero_division=0))
        prec = float(precision_score(y_true, pred, zero_division=0))
        f2 = 0.0
        denom = (4.0 * rec) + prec
        if denom > 0:
            f2 = (5.0 * prec * rec) / denom
        if rec >= min_recall and prec > best_prec:
            best_prec = prec
            best_rec = rec
            best_t = float(threshold)
        if f2 > fallback_score:
            fallback_score = f2
            fallback_t = float(threshold)
    if best_prec < 0:
        pred = proba >= fallback_t
        return fallback_t, {
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
        }
    return best_t, {"precision": best_prec, "recall": best_rec}


def _trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=80,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=0,
        n_jobs=1,
    )


def _oof_proba(
    model: ExtraTreesClassifier,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    n_pos = int(y.sum())
    n_neg = int(len(y) - y.sum())
    n_groups = int(pd.unique(groups).size)
    n_splits = int(min(5, n_groups, n_pos, n_neg))
    if n_splits >= 2 and n_groups >= 2:
        cv = GroupKFold(n_splits=n_splits)
        oof = cross_val_predict(model, x, y, cv=cv, groups=groups, method="predict_proba")
    else:
        cv = StratifiedKFold(n_splits=min(5, n_pos, n_neg), shuffle=True, random_state=0)
        oof = cross_val_predict(model, x, y, cv=cv, method="predict_proba")
    if oof.ndim == 1:
        return np.asarray(oof, dtype=np.float64)
    if oof.shape[1] == 1:
        return np.asarray(oof[:, 0], dtype=np.float64)
    return np.asarray(oof[:, 1], dtype=np.float64)


def train_classifier(table: pd.DataFrame) -> dict[str, Any]:
    """Fit ExtraTrees with tile-grouped CV; return a serializable artifact."""
    y = (table["label"].astype(str) == POSITIVE).astype(int).to_numpy()
    x = dataframe_feature_matrix(table)
    groups = table["source_tile"].astype(str).to_numpy()
    n_pos = int(y.sum())
    n_neg = int(len(y) - y.sum())
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            f"Need at least 2 particle and 2 not_particle rows with features "
            f"(got {n_pos} / {n_neg})."
        )

    model = _trees()
    proba = _oof_proba(model, x, y, groups)
    threshold, metrics = choose_threshold(y, proba)
    model.fit(x, y)
    importances = {
        name: float(value)
        for name, value in zip(FEATURE_COLUMNS, model.feature_importances_)
    }
    return {
        "kind": KIND_RESIDUAL,
        "model": model,
        "feature_names": FEATURE_COLUMNS,
        "threshold": threshold,
        "metrics": metrics,
        "n_samples": int(len(y)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "feature_importances": importances,
    }


def _require_classes(y: np.ndarray) -> tuple[int, int]:
    n_pos = int(y.sum())
    n_neg = int(len(y) - y.sum())
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            f"Need at least 2 particle and 2 not_particle unmarked patches "
            f"(got {n_pos} / {n_neg})."
        )
    return n_pos, n_neg


def _layouts_for_meta(
    meta: pd.DataFrame,
    labels: pd.DataFrame,
    config: dict[str, Any],
) -> np.ndarray:
    pattern = str(cfg_get(config, "filename_pattern", DEFAULT_FILENAME_PATTERN))
    unique = labels.drop_duplicates(subset=["key"], keep="last").copy()
    unique["key"] = unique["key"].astype(str)
    lookup = unique.set_index("key", drop=False)
    rows = []
    for record in meta.to_dict(orient="records"):
        key = str(record.get("key", ""))
        if key in lookup.index:
            src = lookup.loc[key]
            if isinstance(src, pd.DataFrame):
                src = src.iloc[-1]
            rows.append(layout_from_record(src.to_dict(), pattern))
        else:
            rows.append(layout_from_record(record, pattern))
    return np.vstack(rows)


def _hog_importances(model: ExtraTreesClassifier) -> dict[str, float]:
    values = np.asarray(model.feature_importances_, dtype=np.float64)
    names = PATCH_FEATURE_NAMES
    if values.size != len(names):
        return {f"f{i}": float(v) for i, v in enumerate(values)}
    ranked = sorted(zip(names, values), key=lambda item: item[1], reverse=True)
    return {name: float(value) for name, value in ranked[:24]}


def train_patch_classifier(
    channels: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    layouts: np.ndarray | None = None,
) -> dict[str, Any]:
    """HOG/LBP ExtraTrees on unmarked crops. GroupKFold by tile."""
    n_pos, n_neg = _require_classes(y)
    x = patches_to_matrix(channels, layouts)
    model = _trees()
    proba = _oof_proba(model, x, y, groups)
    threshold, metrics = choose_threshold(y, proba)
    model.fit(x, y)
    return {
        "kind": KIND_PATCH,
        "model": model,
        "cnn": None,
        "feature_names": PATCH_FEATURE_NAMES,
        "patch_size": PATCH_SIZE,
        "threshold": threshold,
        "metrics": metrics,
        "hog_metrics": metrics,
        "n_samples": int(len(y)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "feature_importances": _hog_importances(model),
        "oof_proba": proba,
    }


def _cnn_oof(
    channels: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    epochs: int,
) -> np.ndarray:
    n_pos, n_neg = _require_classes(y)
    n_groups = int(pd.unique(groups).size)
    n_splits = int(min(5, n_groups, n_pos, n_neg))
    oof = np.zeros(len(y), dtype=np.float64)
    if n_splits < 2:
        model = train_tiny_cnn(channels, y, epochs=epochs)
        return model.predict_proba(channels)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(y, y, groups)):
        print(
            f"  CNN fold {fold + 1}/{n_splits}  train={len(train_idx)} val={len(test_idx)}",
            flush=True,
        )
        model = train_tiny_cnn(
            channels[train_idx],
            y[train_idx],
            epochs=epochs,
            rng=np.random.default_rng(fold),
        )
        oof[test_idx] = model.predict_proba(channels[test_idx])
    return oof


def _select_cascade_band(
    y: np.ndarray,
    hog_oof: np.ndarray,
    cnn_oof: np.ndarray,
    hog_threshold: float,
    requested: tuple[float, float] | None = None,
) -> tuple[float, float, np.ndarray, float, dict[str, float], int]:
    """Pick the HOG/CNN band with the best OOF precision at 95% recall."""
    from src.ml.infer import combine_cascade_scores

    hog_t, hog_metrics = choose_threshold(y, hog_oof)
    options: list[tuple[float, float]] = [
        (1.0, 1.0),
        (max(0.05, hog_threshold - 0.10), min(0.90, hog_threshold + 0.20)),
        (0.15, 0.40),
        (0.20, 0.45),
        (0.25, 0.50),
        (0.30, 0.55),
    ]
    if requested is not None:
        options.append(requested)
    best: tuple[float, float, np.ndarray, float, dict[str, float], int] | None = None
    best_key = (-1.0, -1.0, 1)
    for lo, hi in options:
        uncertain = (hog_oof >= float(lo)) & (hog_oof < float(hi))
        combined = combine_cascade_scores(
            hog_oof, cnn_oof[uncertain], lo, hi, uncertain=uncertain
        )
        threshold, metrics = choose_threshold(y, combined)
        key = (float(metrics["precision"]), float(metrics["recall"]), -int(uncertain.sum()))
        if key > best_key:
            best_key = key
            best = (float(lo), float(hi), combined, float(threshold), metrics, int(uncertain.sum()))
    if best is None:
        return 1.0, 1.0, hog_oof, float(hog_t), hog_metrics, 0
    return best


def train_cascade_classifier(
    channels: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    layouts: np.ndarray | None = None,
    band_low: float = DEFAULT_BAND_LOW,
    band_high: float = DEFAULT_BAND_HIGH,
    cnn_epochs: int = 8,
) -> dict[str, Any]:
    """HOG ExtraTrees plus CNN on the uncertain probability band."""
    patch = train_patch_classifier(channels, y, groups, layouts)
    hog_oof = np.asarray(patch["oof_proba"], dtype=np.float64)
    hog_metrics = patch["metrics"]
    print(
        f"HOG ExtraTrees OOF precision={hog_metrics['precision']:.3f} "
        f"recall={hog_metrics['recall']:.3f} threshold={patch['threshold']:.2f}",
        flush=True,
    )
    print("Training CNN with GroupKFold…", flush=True)
    cnn_oof = _cnn_oof(channels, y, groups, epochs=cnn_epochs)
    hog_threshold, hog_metrics = choose_threshold(y, hog_oof)
    print("Selecting cascade band from OOF scores…", flush=True)
    band_low, band_high, combined, threshold, metrics, n_uncertain = _select_cascade_band(
        y,
        hog_oof,
        cnn_oof,
        hog_threshold,
        requested=(float(band_low), float(band_high)),
    )
    print(
        f"  cascade band=[{band_low:.2f}, {band_high:.2f}) uncertain={n_uncertain} "
        f"OOF precision={metrics['precision']:.3f} recall={metrics['recall']:.3f}",
        flush=True,
    )
    cnn = train_tiny_cnn(channels, y, epochs=cnn_epochs)
    return {
        "kind": KIND_CASCADE,
        "model": patch["model"],
        "cnn": cnn,
        "feature_names": PATCH_FEATURE_NAMES,
        "patch_size": PATCH_SIZE,
        "threshold": threshold,
        "metrics": metrics,
        "hog_metrics": hog_metrics,
        "hog_threshold": hog_threshold,
        "cnn_metrics": dict(zip(("precision", "recall"), _metrics_at(y, cnn_oof, threshold))),
        "band_low": float(band_low),
        "band_high": float(band_high),
        "n_samples": patch["n_samples"],
        "n_positive": patch["n_positive"],
        "n_negative": patch["n_negative"],
        "feature_importances": patch["feature_importances"],
        "n_uncertain": int(n_uncertain),
        "oof_proba": combined,
        "oof_hog": hog_oof,
        "oof_cnn": cnn_oof,
    }


def _metrics_at(
    y: np.ndarray, proba: np.ndarray, threshold: float
) -> tuple[float, float]:
    pred = np.asarray(proba) >= float(threshold)
    return (
        float(precision_score(y, pred, zero_division=0)),
        float(recall_score(y, pred, zero_division=0)),
    )


def inspect_mark_recall(
    meta: pd.DataFrame,
    labels: pd.DataFrame,
    oof: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """How many tile_inspect flakes the filter would keep (OOF scores)."""
    inspect_keys = set(
        labels.loc[labels["source_csv"].astype(str) == "tile_inspect", "key"].astype(str)
    )
    if meta.empty or not inspect_keys:
        return {"n": 0.0, "kept": 0.0, "recall": 1.0}
    mask = meta["key"].astype(str).isin(inspect_keys)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0.0, "kept": 0.0, "recall": 1.0}
    kept = int((oof[mask.to_numpy()] >= float(threshold)).sum())
    return {"n": float(n), "kept": float(kept), "recall": float(kept / n)}


def save_artifact(artifact: dict[str, Any], path: str | Path) -> Path:
    import joblib

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"oof_proba", "oof_hog", "oof_cnn"}
    }
    joblib.dump(payload, destination)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=default_label_path())
    parser.add_argument(
        "--mode",
        choices=(KIND_CASCADE, KIND_PATCH, KIND_RESIDUAL),
        default=KIND_CASCADE,
        help="cascade = HOG ExtraTrees + CNN on the uncertain band (default).",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        nargs="*",
        default=None,
        help="particles.csv files with residual features (residual mode only).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "config.yaml",
        help="Pipeline YAML so unmarked crops use the same residual as DoG.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=resolve_model_path("models/particle_clf.joblib"),
    )
    parser.add_argument("--cnn-epochs", type=int, default=8)
    parser.add_argument("--band-low", type=float, default=DEFAULT_BAND_LOW)
    parser.add_argument("--band-high", type=float, default=DEFAULT_BAND_HIGH)
    return parser.parse_args(argv)


def _print_artifact(artifact: dict[str, Any], saved: Path) -> None:
    metrics = artifact["metrics"]
    kind = artifact.get("kind", KIND_RESIDUAL)
    print(
        f"Wrote {saved}  kind={kind}  n={artifact['n_samples']} "
        f"(particle {artifact['n_positive']} / not {artifact['n_negative']})  "
        f"threshold={artifact['threshold']:.2f}  "
        f"OOF precision={metrics['precision']:.3f} recall={metrics['recall']:.3f}",
        flush=True,
    )
    if kind == KIND_CASCADE:
        hog = artifact.get("hog_metrics") or {}
        print(
            f"  HOG-only OOF precision={hog.get('precision', 0):.3f} "
            f"recall={hog.get('recall', 0):.3f}  "
            f"uncertain={artifact.get('n_uncertain', 0)} "
            f"band=[{artifact.get('band_low'):.2f}, {artifact.get('band_high'):.2f})",
            flush=True,
        )
    ranked = artifact.get("feature_importances") or {}
    if ranked:
        print("feature importances:", flush=True)
        items = list(ranked.items())
        if kind != KIND_RESIDUAL:
            items = items[:12]
        else:
            items = sorted(items, key=lambda item: item[1], reverse=True)
        for name, value in items:
            print(f"  {name:22s} {value:.4f}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == KIND_RESIDUAL:
        detections = list(args.detections) if args.detections else default_detection_paths()
        if not detections:
            raise SystemExit(
                "No detections CSV with features found. Pass --detections "
                "path/to/particles.csv after a pipeline run."
            )
        print(f"Loading labels from {args.labels}", flush=True)
        print("Detections:", ", ".join(str(path) for path in detections), flush=True)
        table = load_labeled_features(args.labels, detections)
        print(
            f"Joined {len(table)} labeled rows with features. Training ExtraTrees…",
            flush=True,
        )
        artifact = train_classifier(table)
        saved = save_artifact(artifact, args.output)
        _print_artifact(artifact, saved)
        return 0

    labels = pd.read_csv(args.labels)
    if labels.empty:
        raise SystemExit(f"No labels in {args.labels}")
    config = default_train_config(args.config)
    print(f"Loading unmarked patches from {args.labels} (circled JPEGs ignored)", flush=True)
    channels, y, groups, meta = collect_unmarked_dataset(labels, config)
    layouts = _layouts_for_meta(meta, labels, config)
    print(
        f"Extracted {len(y)} unmarked crops "
        f"(particle {int(y.sum())} / not {int(len(y) - y.sum())}) "
        f"from {pd.unique(groups).size} tiles.",
        flush=True,
    )
    if args.mode == KIND_PATCH:
        print("Training HOG/LBP ExtraTrees…", flush=True)
        artifact = train_patch_classifier(channels, y, groups, layouts)
    else:
        print("Training cascade: HOG ExtraTrees, then CNN on the uncertain band…", flush=True)
        artifact = train_cascade_classifier(
            channels,
            y,
            groups,
            layouts,
            band_low=float(args.band_low),
            band_high=float(args.band_high),
            cnn_epochs=int(args.cnn_epochs),
        )
    inspect = inspect_mark_recall(
        meta, labels, np.asarray(artifact["oof_proba"]), float(artifact["threshold"])
    )
    artifact["inspect_metrics"] = inspect
    saved = save_artifact(artifact, args.output)
    _print_artifact(artifact, saved)
    if inspect["n"]:
        print(
            f"  tile_inspect OOF kept {int(inspect['kept'])}/{int(inspect['n'])} "
            f"(recall {inspect['recall']:.3f})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
