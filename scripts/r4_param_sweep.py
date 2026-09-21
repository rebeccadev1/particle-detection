"""R4 one-factor (then 2-factor) preprocess/detect sweep with ML v4 at 0.20."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config, resolve_output_dir
from src.io.results_writer import write_csv
from src.labeling.crops import find_tile_path, global_nm_to_local_px, placement_for_tile
from src.pipeline.runner import run_pipeline
from src.report.report_generator import write_overlay_image

PATTERN = r"n2_(?P<row>\d+)_(?P<col>\d+)\.tiff?"
INPUT_DIR = "R4 14-09"
OUTPUT_NAME = "R4 16-09 param sweep"
TINDER_CSV = "R4 14-09 holdout ML off 20um"
INSPECT_KEYS = (
    "n2_1_7_17928928_923312",
    "n2_2_16_46001936_2091776",
)
V4_PATH = "models/particle_clf_v4_residual.joblib"
ML_THRESHOLD = 0.20
MIN_SIZE_NM = 20_000.0
NEAR_PX = 20.0
N_GOLD = 41
MIN_TP = 39  # recall > 95% on 41 gold particles

RECALL_OVERLAY_KEYS = {
    "edge_exclude_px",
    "min_confidence",
    "min_circularity",
    "structure_min_neighbors",
    "structure_line_bin_px",
}

ONE_FACTOR: list[tuple[str, list[Any]]] = [
    ("denoise_sigma", [0.5, 1.0, 1.5, 2.0, "off"]),
    ("flatten_sigma", [100.0, 160.0, 220.0]),
    ("contrast", ["1_99", "2_98", "off"]),
    ("blob_min_sigma", [3.7, 7.4]),
    ("blob_threshold", [0.06, 0.08, 0.10, 0.12]),
    ("fft_peak_threshold", [0.25, 0.35, 0.45]),
    ("recall.min_confidence", [0.40, 0.50, 0.60]),
    ("recall.edge_exclude_px", [12.0, 24.0]),
    ("structure_min_neighbors", [2, 3]),
    ("structure_line_bin_px", [8.0, 10.0, 12.0]),
    ("min_circularity", [0.0, 0.15, 0.30]),
]
BASELINE_KNOBS: dict[str, Any] = {
    "denoise_sigma": 1.0,
    "flatten_sigma": 160.0,
    "contrast": "1_99",
    "blob_min_sigma": 3.7,
    "blob_threshold": 0.08,
    "fft_peak_threshold": 0.35,
    "recall.min_confidence": 0.40,
    "recall.edge_exclude_px": 12.0,
    "structure_min_neighbors": 2,
    "structure_line_bin_px": 10.0,
    "min_circularity": 0.0,
}


def _labels_path() -> Path:
    return Path(__file__).resolve().parents[1] / "labels" / "labels.csv"


def load_gold(labels_path: Path | None = None) -> pd.DataFrame:
    """39 Tinder particles on the R4 holdout CSV plus 2 inspect FNs ≥ 20 µm."""
    path = labels_path or _labels_path()
    labels = pd.read_csv(path)
    tinder = labels.loc[
        labels["source_csv"].astype(str).str.contains(TINDER_CSV, regex=False)
        & (labels["label"].astype(str) == "particle")
        & (labels["size"].astype(float) >= MIN_SIZE_NM)
    ].copy()
    inspect = labels.loc[labels["key"].astype(str).isin(INSPECT_KEYS)].copy()
    gold = pd.concat([tinder, inspect], ignore_index=True)
    gold = gold.drop_duplicates(subset=["key"]).reset_index(drop=True)
    if len(gold) != N_GOLD:
        raise RuntimeError(f"Expected {N_GOLD} gold rows, got {len(gold)}")
    return gold


def _set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur: dict[str, Any] = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def apply_knob(cfg: dict[str, Any], knob: str, value: Any) -> dict[str, Any]:
    """Mutate a copy of ``cfg`` for one named trial value."""
    out = deepcopy(cfg)
    pre = out.setdefault("preprocessing", {})
    det = out.setdefault("detection", {})
    recall = det.setdefault("recall", {})
    if knob == "denoise_sigma":
        if value == "off":
            pre["denoise"] = False
        else:
            pre["denoise"] = True
            pre["denoise_sigma"] = float(value)
        return out
    if knob == "flatten_sigma":
        pre["flatten_illumination"] = True
        pre["flatten_sigma"] = float(value)
        return out
    if knob == "contrast":
        if value == "off":
            pre["contrast_stretch"] = False
        elif value == "1_99":
            pre["contrast_stretch"] = True
            pre["contrast_percentiles"] = [1.0, 99.0]
        elif value == "2_98":
            pre["contrast_stretch"] = True
            pre["contrast_percentiles"] = [2.0, 98.0]
        else:
            raise ValueError(f"Unknown contrast value {value!r}")
        return out
    if knob.startswith("recall."):
        name = knob.split(".", 1)[1]
        recall[name] = value
        det[name] = value
        return out
    if knob in RECALL_OVERLAY_KEYS:
        recall[knob] = value
        det[knob] = value
        return out
    det[knob] = value
    return out


def r4_base_config(*, mosaic: bool = False) -> dict[str, Any]:
    cfg = deepcopy(load_config("config.yaml"))
    cfg["input_dir"] = INPUT_DIR
    cfg["output_dir"] = OUTPUT_NAME
    cfg["filename_pattern"] = PATTERN
    cfg["run"] = ""
    cfg["magnification"] = ""
    report = cfg.setdefault("report", {})
    report["downsample"] = 0 if mosaic else 1
    det = cfg.setdefault("detection", {})
    det["min_size_nm"] = MIN_SIZE_NM
    for knob, value in BASELINE_KNOBS.items():
        cfg = apply_knob(cfg, knob, value)
    det = cfg.setdefault("detection", {})
    ml = cfg.setdefault("ml", {})
    ml["enabled"] = True
    ml["model_path"] = V4_PATH
    ml["threshold"] = ML_THRESHOLD
    ml["score_before_structure"] = False
    return cfg


def _basename(name: object) -> str:
    return Path(str(name)).name


def _localize(
    row: pd.Series,
    config: dict[str, Any],
    origin_cache: dict[str, tuple[int, int]],
) -> tuple[str, float, float]:
    tile = _basename(row["source_tile"])
    path = find_tile_path(tile, input_dir=config["input_dir"])
    placement = placement_for_tile(path, config, origin_cache=origin_cache)
    pixel_size = float(config.get("pixel_size_nm", 960.0))
    x_local, y_local = global_nm_to_local_px(
        float(row["x_global"]),
        float(row["y_global"]),
        placement,
        pixel_size,
    )
    return tile, x_local, y_local


def score_table(
    table: pd.DataFrame,
    gold: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    origin_cache: dict[str, tuple[int, int]] = {}
    det_local: dict[str, list[tuple[float, float]]] = {}
    kept = table.loc[table["size"].astype(float) >= MIN_SIZE_NM]
    for _, row in kept.iterrows():
        tile, x_local, y_local = _localize(row, config, origin_cache)
        det_local.setdefault(tile, []).append((x_local, y_local))
    counted = int(sum(len(pts) for pts in det_local.values()))
    tp = 0
    missed: list[str] = []
    for _, row in gold.iterrows():
        tile, x_local, y_local = _localize(row, config, origin_cache)
        pts = det_local.get(tile, [])
        if not pts:
            missed.append(str(row["key"]))
            continue
        dist = min(((px - x_local) ** 2 + (py - y_local) ** 2) ** 0.5 for px, py in pts)
        if dist <= NEAR_PX:
            tp += 1
        else:
            missed.append(str(row["key"]))
    fp = counted - tp
    fn = int(len(gold) - tp)
    precision = (tp / counted) if counted else 0.0
    recall = (tp / len(gold)) if len(gold) else 0.0
    return {
        "counted": counted,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "feasible": int(tp >= MIN_TP),
        "missed_keys": ",".join(missed),
    }


def _progress(done: int, total: int, msg: str = "") -> None:
    if done == 0 or done == total or done % 10 == 0:
        print(f"    [{done}/{total}] {msg}", flush=True)


def run_trial(
    trial_id: str,
    cfg: dict[str, Any],
    gold: pd.DataFrame,
    out: Path,
    *,
    save_particles: bool = False,
    mosaic: bool = False,
) -> dict[str, Any]:
    print(f"\n=== {trial_id} ===", flush=True)
    t0 = time.time()
    table, mosaic_obj = run_pipeline(cfg, progress_cb=_progress)
    elapsed = time.time() - t0
    metrics = score_table(table, gold, cfg)
    metrics["trial_id"] = trial_id
    metrics["seconds"] = round(elapsed, 1)
    metrics["n_raw"] = int(len(table))
    print(
        f"  counted={metrics['counted']} TP={metrics['tp']} FP={metrics['fp']} "
        f"FN={metrics['fn']} P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"in {elapsed:.0f}s",
        flush=True,
    )
    if save_particles:
        write_csv(table, out / "particles.csv")
        table.to_pickle(out / "particles.pkl")
    if mosaic:
        try:
            write_overlay_image(mosaic_obj, table, cfg, out / "mosaic_overlay.jpg")
        except Exception as exc:  # noqa: BLE001
            print(f"  mosaic failed: {exc}", flush=True)
    return metrics


def _results_path(out: Path) -> Path:
    return out / "sweep_results.csv"


def _load_done(out: Path) -> dict[str, dict[str, Any]]:
    path = _results_path(out)
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    return {str(row["trial_id"]): row.to_dict() for _, row in df.iterrows()}


def _write_results(out: Path, rows: list[dict[str, Any]]) -> None:
    path = _results_path(out)
    pd.DataFrame(rows).to_csv(path, index=False)


def _record(
    out: Path,
    rows: list[dict[str, Any]],
    done: dict[str, dict[str, Any]],
    trial_id: str,
    knobs: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    payload = _trial_payload(trial_id, knobs, metrics)
    rows.append(payload)
    done[trial_id] = payload
    _write_results(out, rows)
    return payload


def _trial_payload(trial_id: str, knobs: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    payload = dict(metrics)
    payload["knobs_json"] = json.dumps(knobs, sort_keys=True)
    for key, value in knobs.items():
        payload[f"knob_{key}"] = value
    payload["trial_id"] = trial_id
    return payload


def _best_feasible(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [r for r in rows if int(r.get("feasible") or 0) == 1]
    if not feasible:
        raise RuntimeError("No trial met TP >= 39 (recall > 95%).")
    return max(
        feasible,
        key=lambda r: (float(r["precision"]), int(r["tp"]), -int(r["counted"])),
    )


def _knob_gains(rows: list[dict[str, Any]], baseline_p: float) -> list[tuple[str, float, Any]]:
    """Best feasible precision gain per one-factor knob."""
    by_knob: dict[str, tuple[float, Any]] = {}
    for row in rows:
        if str(row["trial_id"]) in {"baseline", "winner"}:
            continue
        if not str(row["trial_id"]).startswith("1f:"):
            continue
        if int(row.get("feasible") or 0) != 1:
            continue
        knobs = json.loads(row["knobs_json"]) if isinstance(row.get("knobs_json"), str) else {}
        if len(knobs) != 1:
            continue
        name, value = next(iter(knobs.items()))
        prec = float(row["precision"])
        prev = by_knob.get(name)
        if prev is None or prec > prev[0]:
            by_knob[name] = (prec, value)
    ranked = sorted(
        ((name, prec - baseline_p, value) for name, (prec, value) in by_knob.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked


def _combo_id(knobs: dict[str, Any]) -> str:
    parts = [f"{k}={v}" for k, v in sorted(knobs.items())]
    return "2f:" + ",".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-one-factor", action="store_true")
    parser.add_argument("--skip-two-factor", action="store_true")
    parser.add_argument("--skip-winner-mosaic", action="store_true")
    parser.add_argument(
        "--one-factor-knob",
        default="",
        help="If set, run only this one-factor knob (e.g. min_circularity).",
    )
    args = parser.parse_args(argv)

    gold = load_gold()
    out = resolve_output_dir(OUTPUT_NAME)
    out.mkdir(parents=True, exist_ok=True)
    gold.to_csv(out / "gold_41.csv", index=False)
    print(f"Gold N={len(gold)} -> {out / 'gold_41.csv'}", flush=True)

    done = _load_done(out)
    rows = list(done.values())
    base = r4_base_config(mosaic=False)

    METRIC_KEYS = (
        "counted", "tp", "fp", "fn", "precision", "recall",
        "feasible", "missed_keys", "seconds", "n_raw",
    )

    def _copy_metrics(source: dict[str, Any], trial_id: str) -> dict[str, Any]:
        metrics = {k: source[k] for k in METRIC_KEYS if k in source}
        metrics["trial_id"] = trial_id
        return metrics

    if "baseline" not in done:
        metrics = run_trial("baseline", deepcopy(base), gold, out, save_particles=True)
        _record(out, rows, done, "baseline", {}, metrics)
    else:
        print("skip baseline (already in sweep_results.csv)", flush=True)

    if not args.skip_one_factor:
        one_factor = ONE_FACTOR
        if args.one_factor_knob:
            one_factor = [item for item in ONE_FACTOR if item[0] == args.one_factor_knob]
            if not one_factor:
                raise ValueError(f"Unknown one-factor knob {args.one_factor_knob!r}")
        for knob, values in one_factor:
            for value in values:
                trial_id = f"1f:{knob}={value}"
                if trial_id in done:
                    print(f"skip {trial_id}", flush=True)
                    continue
                if BASELINE_KNOBS.get(knob) == value:
                    print(f"alias {trial_id} -> baseline", flush=True)
                    _record(
                        out, rows, done, trial_id, {knob: value},
                        _copy_metrics(done["baseline"], trial_id),
                    )
                    continue
                cfg = apply_knob(deepcopy(base), knob, value)
                metrics = run_trial(trial_id, cfg, gold, out)
                _record(out, rows, done, trial_id, {knob: value}, metrics)

    baseline_p = float(done["baseline"]["precision"])
    if not args.skip_two_factor:
        ranked = _knob_gains(rows, baseline_p)
        print("\nOne-factor precision gains vs baseline:", flush=True)
        for name, gain, value in ranked:
            print(f"  {name}={value}  ΔP={gain:+.4f}", flush=True)
        if len(ranked) >= 2:
            k1, _g1, _v1 = ranked[0]
            k2, _g2, _v2 = ranked[1]
            vals1 = [v for k, v in ONE_FACTOR if k == k1][0]
            vals2 = [v for k, v in ONE_FACTOR if k == k2][0]
            print(f"\n2-factor grid: {k1} × {k2}", flush=True)
            for a in vals1:
                for b in vals2:
                    knobs = {k1: a, k2: b}
                    trial_id = _combo_id(knobs)
                    if trial_id in done:
                        print(f"skip {trial_id}", flush=True)
                        continue
                    if knobs.get(k1) == BASELINE_KNOBS.get(k1) and knobs.get(k2) == BASELINE_KNOBS.get(k2):
                        print(f"alias {trial_id} -> baseline", flush=True)
                        _record(out, rows, done, trial_id, knobs, _copy_metrics(done["baseline"], trial_id))
                        continue
                    if knobs.get(k1) == BASELINE_KNOBS.get(k1):
                        alias = f"1f:{k2}={b}"
                        if alias in done:
                            print(f"alias {trial_id} -> {alias}", flush=True)
                            _record(out, rows, done, trial_id, knobs, _copy_metrics(done[alias], trial_id))
                            continue
                    if knobs.get(k2) == BASELINE_KNOBS.get(k2):
                        alias = f"1f:{k1}={a}"
                        if alias in done:
                            print(f"alias {trial_id} -> {alias}", flush=True)
                            _record(out, rows, done, trial_id, knobs, _copy_metrics(done[alias], trial_id))
                            continue
                    cfg = deepcopy(base)
                    cfg = apply_knob(cfg, k1, a)
                    cfg = apply_knob(cfg, k2, b)
                    metrics = run_trial(trial_id, cfg, gold, out)
                    _record(out, rows, done, trial_id, knobs, metrics)
        else:
            print("Not enough one-factor gains for a 2-factor grid.", flush=True)

    winner = _best_feasible(rows)
    print(
        f"\nWINNER {winner['trial_id']}  "
        f"P={float(winner['precision']):.3f} R={float(winner['recall']):.3f} "
        f"TP={winner['tp']} counted={winner['counted']}",
        flush=True,
    )
    knobs = json.loads(winner["knobs_json"]) if winner.get("knobs_json") else {}
    (out / "winner.json").write_text(json.dumps(winner, indent=2, default=str), encoding="utf-8")

    if not args.skip_winner_mosaic:
        cfg = deepcopy(base)
        for knob, value in knobs.items():
            cfg = apply_knob(cfg, knob, value)
        cfg = r4_base_config(mosaic=True)
        for knob, value in knobs.items():
            cfg = apply_knob(cfg, knob, value)
        metrics = run_trial("winner", cfg, gold, out, save_particles=True, mosaic=True)
        if "winner" not in done:
            _record(out, rows, done, "winner", knobs, metrics)
        print(f"Wrote winner particles + mosaic under {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
