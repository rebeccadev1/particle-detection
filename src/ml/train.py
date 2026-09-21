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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

from src.config import DEFAULT_OUTPUT_DIR, PACKAGE_ROOT, cfg_get
from src.io.tile_loader import DEFAULT_FILENAME_PATTERN
from src.labeling.queue import detection_key
from src.ml.cnn import train_tiny_cnn
from src.ml.features import FEATURE_COLUMNS, dataframe_feature_matrix
from src.ml.infer import (
    KIND_CASCADE,
    KIND_PATCH,
    KIND_RESIDUAL,
    load_artifact,
    next_versioned_model_path,
)
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


def apply_label_filters(
    labels: pd.DataFrame,
    *,
    manifest: Path | None,
    min_size_nm: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    keep = labels.copy()
    keep["key"] = keep["key"].astype(str)
    table = None
    if manifest is not None:
        table = pd.read_csv(manifest)
        table["key"] = table["key"].astype(str)
        keys = set(table["key"])
        keep = keep.loc[keep["key"].isin(keys)].copy()
    jpeg = keep["source_tile"].astype(str).str.lower().str.endswith((".jpg", ".jpeg"))
    keep = keep.loc[~jpeg].copy()
    if min_size_nm > 0 and "size" in keep.columns:
        keep = keep.loc[keep["size"].astype(float) >= float(min_size_nm)].copy()
    keep = keep.loc[keep["label"].astype(str).isin((POSITIVE, NEGATIVE))].copy()
    return keep, table


def overlay_detection_features(labels: pd.DataFrame, detections: Path) -> pd.DataFrame:
    """Copy residual feature columns from a particles.csv onto labeled keys."""
    table = pd.read_csv(detections)
    table["key"] = [detection_key(row) for _, row in table.iterrows()]
    skip = {"label", "crop_path", "labeled_at", "source_csv", "particle_id"}
    cols = [name for name in table.columns if name not in skip]
    extra = table[cols].drop_duplicates(subset=["key"], keep="last")
    merged = labels.merge(extra, on="key", how="left", suffixes=("", "_det"))
    for name in extra.columns:
        if name == "key":
            continue
        det_name = f"{name}_det"
        if det_name in merged.columns:
            merged[name] = merged[name].where(merged[name].notna(), merged[det_name])
            merged = merged.drop(columns=[det_name])
    return merged


def v4_scores_for_layouts(layouts: np.ndarray, v4_path: Path) -> np.ndarray:
    from src.ml.infer import _positive_proba

    artifact = load_artifact(v4_path)
    width = len(FEATURE_COLUMNS)
    matrix = np.asarray(layouts, dtype=np.float64)[:, :width]
    return _positive_proba(artifact["model"], matrix)


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


def keep_all_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Highest threshold that still keeps every positive (min positive score)."""
    pos = np.asarray(proba, dtype=np.float64)[np.asarray(y_true) == 1]
    if pos.size == 0:
        return 0.0
    return float(np.min(pos))


def fit_isotonic_calibrator(raw: np.ndarray, y: np.ndarray) -> IsotonicRegression | None:
    scores = np.asarray(raw, dtype=np.float64)
    labels = np.asarray(y)
    if scores.size < 4 or np.unique(scores).size < 2:
        return None
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    try:
        calibrator.fit(scores, labels)
    except ValueError:
        return None
    return calibrator


def hard_negative_weights(
    y: np.ndarray,
    v4_scores: np.ndarray | None,
    cutoff: float = 0.20,
    heavy: float = 3.0,
) -> np.ndarray:
    """Up-weight not-particle rows that v4 still keeps at ``cutoff``."""
    weights = np.ones(len(y), dtype=np.float64)
    if v4_scores is None or heavy <= 1.0:
        return weights
    scores = np.asarray(v4_scores, dtype=np.float64)
    hard = (np.asarray(y) == 0) & (scores >= float(cutoff))
    weights[hard] = float(heavy)
    return weights


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
    *,
    sample_weight: np.ndarray | None = None,
    holdout_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """HOG/LBP ExtraTrees on unmarked crops. GroupKFold by tile."""
    n_pos, n_neg = _require_classes(y)
    x = patches_to_matrix(channels, layouts)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if holdout_mask is None:
        train_mask = np.ones(len(y), dtype=bool)
        hold_mask = np.zeros(len(y), dtype=bool)
    else:
        hold_mask = np.asarray(holdout_mask, dtype=bool)
        train_mask = ~hold_mask
        if int(y[train_mask].sum()) < 2 or int((1 - y[train_mask]).sum()) < 2:
            train_mask = np.ones(len(y), dtype=bool)
            hold_mask = np.zeros(len(y), dtype=bool)
    model = _trees()
    x_train, y_train, g_train = x[train_mask], y[train_mask], groups[train_mask]
    oof_raw = np.zeros(len(y), dtype=np.float64)
    oof_raw[train_mask] = _oof_proba(model, x_train, y_train, g_train)
    weights = None if sample_weight is None else np.asarray(sample_weight)[train_mask]
    model.fit(x_train, y_train, sample_weight=weights)
    if hold_mask.any():
        oof_raw[hold_mask] = _positive_proba_matrix(model, x[hold_mask])
    calibrator = fit_isotonic_calibrator(oof_raw[train_mask], y_train)
    calibrated = np.asarray(oof_raw, dtype=np.float64)
    if calibrator is not None:
        calibrated = np.asarray(calibrator.predict(oof_raw), dtype=np.float64)
    thresh_src_y = y[hold_mask] if hold_mask.any() else y
    thresh_src_p = calibrated[hold_mask] if hold_mask.any() else calibrated
    if int(thresh_src_y.sum()) < 1:
        thresh_src_y, thresh_src_p = y, calibrated
    keep_all = keep_all_threshold(thresh_src_y, thresh_src_p)
    precision_t, metrics = choose_threshold(thresh_src_y, thresh_src_p)
    threshold = keep_all if hold_mask.any() else precision_t
    return {
        "kind": KIND_PATCH,
        "model": model,
        "cnn": None,
        "calibrator": calibrator,
        "feature_names": PATCH_FEATURE_NAMES,
        "patch_size": PATCH_SIZE,
        "threshold": float(threshold),
        "threshold_keep_all": float(keep_all),
        "threshold_precision": float(precision_t),
        "metrics": metrics,
        "hog_metrics": metrics,
        "n_samples": int(len(y)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_train": int(train_mask.sum()),
        "n_holdout": int(hold_mask.sum()),
        "feature_importances": _hog_importances(model),
        "oof_proba": calibrated,
    }


def _positive_proba_matrix(model: ExtraTreesClassifier, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    proba = model.predict_proba(matrix)
    classes = list(model.classes_)
    if 1 in classes:
        return np.asarray(proba[:, classes.index(1)], dtype=np.float64)
    return np.zeros(len(matrix), dtype=np.float64)


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
        default=None,
        help="Defaults to the next unused models/particle_clf_vN.joblib.",
    )
    parser.add_argument("--cnn-epochs", type=int, default=8)
    parser.add_argument("--band-low", type=float, default=DEFAULT_BAND_LOW)
    parser.add_argument("--band-high", type=float, default=DEFAULT_BAND_HIGH)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--min-size-nm", type=float, default=0.0)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--filename-pattern", type=str, default=None)
    parser.add_argument("--v4-model", type=Path, default=None)
    parser.add_argument("--hard-neg-weight", type=float, default=3.0)
    parser.add_argument("--hard-neg-cutoff", type=float, default=0.20)
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
    keep_all = artifact.get("threshold_keep_all")
    precision_t = artifact.get("threshold_precision")
    if keep_all is not None or precision_t is not None:
        print(
            f"  keep_all={float(keep_all or 0):.3f}  "
            f"precision={float(precision_t or 0):.3f}  "
            f"train={artifact.get('n_train', artifact['n_samples'])}  "
            f"holdout={artifact.get('n_holdout', 0)}",
            flush=True,
        )
    print(
        f"Point ml.model_path at {saved.name} in config.yaml to run this version.",
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
    if args.output is None:
        args.output = next_versioned_model_path()
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
    labels, manifest = apply_label_filters(
        labels, manifest=args.manifest, min_size_nm=float(args.min_size_nm)
    )
    if labels.empty:
        raise SystemExit("No labeled rows left after --manifest / --min-size-nm filters.")
    detections = list(args.detections) if args.detections else []
    if detections:
        labels = overlay_detection_features(labels, detections[0])
    config = default_train_config(args.config)
    if args.input_dir:
        config["input_dir"] = args.input_dir
    if args.filename_pattern:
        config["filename_pattern"] = args.filename_pattern
    print(f"Loading unmarked patches from {args.labels} (circled JPEGs ignored)", flush=True)
    channels, y, groups, meta = collect_unmarked_dataset(labels, config)
    layouts = _layouts_for_meta(meta, labels, config)
    print(
        f"Extracted {len(y)} unmarked crops "
        f"(particle {int(y.sum())} / not {int(len(y) - y.sum())}) "
        f"from {pd.unique(groups).size} tiles.",
        flush=True,
    )
    sample_weight = None
    if args.v4_model is not None:
        v4_scores = v4_scores_for_layouts(layouts, args.v4_model)
        sample_weight = hard_negative_weights(
            y, v4_scores, cutoff=float(args.hard_neg_cutoff), heavy=float(args.hard_neg_weight)
        )
        print(
            f"Hard-negative weight {args.hard_neg_weight:g} on "
            f"{int(((y == 0) & (v4_scores >= args.hard_neg_cutoff)).sum())} "
            f"v4≥{args.hard_neg_cutoff} FPs.",
            flush=True,
        )
    holdout_mask = None
    if manifest is not None and "split" in manifest.columns:
        hold_keys = set(
            manifest.loc[manifest["split"].astype(str) == "holdout_tile", "key"].astype(str)
        )
        holdout_mask = meta["key"].astype(str).isin(hold_keys).to_numpy()
        print(
            f"Holdout tiles: {int(holdout_mask.sum())} rows / "
            f"{pd.unique(groups[holdout_mask]).size if holdout_mask.any() else 0} tiles.",
            flush=True,
        )
    if args.mode == KIND_PATCH:
        print("Training HOG/LBP ExtraTrees…", flush=True)
        artifact = train_patch_classifier(
            channels,
            y,
            groups,
            layouts,
            sample_weight=sample_weight,
            holdout_mask=holdout_mask,
        )
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
