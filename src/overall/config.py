from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = PACKAGE_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "Outputs"
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "Inputs"


def resolve_output_dir(folder: str | Path | None = None) -> Path:
    
    text = str(folder or "").strip().strip('"').strip("'")
    if not text:
        return DEFAULT_OUTPUT_DIR
    raw = Path(text)
    if raw.is_absolute():
        return raw
    first = raw.parts[0]
    if first in (".", ".."):
        return (PACKAGE_ROOT / raw).resolve()
    return (DEFAULT_OUTPUT_DIR / raw).resolve()


def next_versioned_output_name(
    input_folder: str | Path | None,
    outputs: str | Path | None = None,
) -> str:
    
    text = str(input_folder or "").strip().strip('"').strip("'").rstrip("/\\")
    if not text:
        return ""
    name = Path(text).name
    root = DEFAULT_OUTPUT_DIR if outputs is None else Path(outputs)
    highest = 0
    if root.is_dir():
        pattern = re.compile(rf"^{re.escape(name)} v(\d+)$")
        for child in root.iterdir():
            match = pattern.fullmatch(child.name)
            if match and child.is_dir():
                highest = max(highest, int(match.group(1)))
    return f"{name} v{highest + 1}"


def advance_versioned_output_name(
    input_folder: str | Path | None,
    current: str | Path | None,
    outputs: str | Path | None = None,
) -> str:
    
    fresh = next_versioned_output_name(input_folder, outputs)
    label = output_choice_label(current).strip()
    if not fresh:
        return label
    name = Path(
        str(input_folder or "").strip().strip('"').strip("'").rstrip("/\\")
    ).name
    pattern = re.compile(rf"^{re.escape(name)} v(\d+)$")
    used = pattern.fullmatch(label)
    if label and used is None:
        return label
    used_n = int(used.group(1)) if used else 0
    fresh_match = pattern.fullmatch(fresh)
    fresh_n = int(fresh_match.group(1)) if fresh_match else 1
    return f"{name} v{max(fresh_n, used_n + 1)}"


def reserve_fresh_output_dir(config: dict[str, Any]) -> Path:
    chosen = str(cfg_get(config, "output_dir", "") or "").strip()
    if not chosen:
        return resolve_output_dir(None)
    dest = resolve_output_dir(chosen)
    if not dest.exists():
        return dest
    fresh = _next_free_output_path(config, dest)
    config["output_dir"] = _output_dir_config_value(chosen, fresh)
    return fresh


def _next_free_output_path(config: dict[str, Any], occupied: Path) -> Path:
    root = occupied.parent
    fresh = next_versioned_output_name(cfg_get(config, "input_dir", ""), root)
    if fresh:
        candidate = root / fresh
        if candidate != occupied and not candidate.exists():
            return candidate
    match = re.fullmatch(r"(?P<base>.*) v(?P<n>\d+)", occupied.name)
    if match:
        base = match.group("base")
        number = int(match.group("n")) + 1
    else:
        base = occupied.name
        number = 2
    while True:
        candidate = root / f"{base} v{number}"
        if not candidate.exists():
            return candidate
        number += 1


def _output_dir_config_value(chosen: str, path: Path) -> str:
    raw = Path(chosen.strip().strip('"').strip("'"))
    if raw.is_absolute() or (raw.parts and raw.parts[0] in (".", "..")):
        return str(path)
    return path.name


def output_choice_label(folder: str | Path | None) -> str:
    
    text = str(folder or "").strip().strip('"').strip("'")
    if not text:
        return ""
    try:
        resolved = resolve_output_dir(text)
        relative = resolved.relative_to(DEFAULT_OUTPUT_DIR.resolve())
    except (OSError, ValueError):
        return text
    if relative == Path("."):
        return text
    return relative.as_posix()


def output_folder_labels(root: str | Path | None = None) -> list[str]:
    
    base = DEFAULT_OUTPUT_DIR if root is None else Path(root)
    if not base.is_dir():
        return []
    try:
        children = list(base.iterdir())
    except OSError:
        return []
    return sorted(
        child.name for child in children if child.is_dir() and not child.name.startswith(".")
    )


def output_folder_choices(
    current: str | None,
    standard: str | None,
    root: str | Path | None = None,
) -> list[str]:
    
    choices = output_folder_labels(root)
    for name in (output_choice_label(current), str(standard or "").strip()):
        if name and name not in choices:
            choices.insert(0, name)
    return choices


def load_config(path: str | Path) -> dict[str, Any]:
    
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


def cfg_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def deep_update(base: dict[str, Any], overlay: dict[str, Any] | None) -> dict[str, Any]:
    
    out = deepcopy(base)
    if not overlay:
        return out
    for key, value in overlay.items():
        current = out.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            out[key] = deep_update(current, value)
        else:
            out[key] = deepcopy(value)
    return out


def nsew_config_path(path: str | Path | None = None) -> Path:
    
    if path is not None:
        return Path(path)
    return CONFIG_DIR / "nsew_config.yaml"


def load_nsew_overlay(path: str | Path | None = None) -> dict[str, Any]:
    
    destination = nsew_config_path(path)
    if not destination.is_file():
        return {}
    data = load_config(destination)
    data.pop("nsew_profile", None)
    return data


def apply_nsew_settings(
    config: dict[str, Any],
    overlay: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    
    profile = overlay if overlay is not None else load_nsew_overlay(path)
    if not profile:
        profile = (config.get("nsew_profile") or {}) if isinstance(config, dict) else {}
    return deep_update(config, profile)


def beta_preset_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    
    raw = cfg_get(config, "report.beta_presets", None)
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "report.beta_presets is empty. Add a beta, model_path, and threshold."
        )
    rows: list[dict[str, Any]] = []
    seen: set[float] = set()
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("Each report.beta_presets entry must be a mapping.")
        try:
            beta = round(float(row["beta"]), 2)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("A beta preset is missing a numeric beta.") from exc
        path = row.get("model_path")
        threshold = row.get("threshold")
        if path is None or str(path).strip() == "" or threshold is None:
            raise ValueError(f"Beta {beta:.2f} is missing model_path or threshold.")
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Beta {beta:.2f} has a non-numeric threshold.") from exc
        if beta in seen:
            raise ValueError(f"Beta {beta:.2f} is listed more than once.")
        seen.add(beta)
        rows.append(
            {
                "beta": beta,
                "model_path": str(path),
                "threshold": threshold_value,
            }
        )
    return rows


def preset_for_beta(presets: list[dict[str, Any]], beta: float) -> dict[str, Any]:
    
    try:
        target = round(float(beta), 2)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Beta must be a number, got {beta!r}.") from exc
    for row in presets:
        if round(float(row["beta"]), 2) == target:
            return {
                "beta": target,
                "model_path": str(row["model_path"]),
                "threshold": float(row["threshold"]),
            }
    known = ", ".join(f"{float(row['beta']):.2f}" for row in presets) or "(none)"
    raise ValueError(f"No ML preset for beta {target:.2f}. Known betas: {known}.")


def apply_beta_ml(config: dict[str, Any]) -> dict[str, Any]:
    """Point the run's classifier at the model and threshold for report.f_beta."""
    raw = cfg_get(config, "report.beta_presets", None)
    if not isinstance(raw, list) or not raw:
        return config
    beta = cfg_get(config, "report.f_beta", 1.0)
    preset = preset_for_beta(beta_preset_rows(config), 1.0 if beta is None else beta)
    ml = config.setdefault("ml", {})
    ml["model_path"] = preset["model_path"]
    ml["threshold"] = preset["threshold"]
    return config
