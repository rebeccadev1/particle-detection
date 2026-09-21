"""Load and access nested pipeline configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "Outputs"
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "Inputs"


def resolve_output_dir(folder: str | Path | None = None) -> Path:
    """Map ``output_dir`` onto ``ASML SE/Outputs`` unless an absolute path is given.

    Bare names such as ``Single v6`` become ``Outputs/Single v6``. Relative
    paths that start with ``.`` or ``..`` are resolved from ``particle_detection/``.
    """
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


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a nested dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


def cfg_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch a nested value using dotted keys, e.g. ``detection.method``."""
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def with_recall_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Return ``config`` unchanged.

    High-recall gates (edge exclude 12 px, confidence 0.50, circularity 0)
    are the standard detection values, not a separate overlay.
    """
    return config


def deep_update(base: dict[str, Any], overlay: dict[str, Any] | None) -> dict[str, Any]:
    """Copy ``base`` and recursively overlay mapping values from ``overlay``."""
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
    """``nsew_config.yaml`` next to ``config.yaml`` unless ``path`` is given."""
    if path is not None:
        return Path(path)
    return PACKAGE_ROOT / "nsew_config.yaml"


def load_nsew_overlay(path: str | Path | None = None) -> dict[str, Any]:
    """Read Groundup / N/S/E/W standard values, or ``{}`` when the file is missing."""
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
    """Return ``config`` with N/S/E/W standard values merged on top."""
    profile = overlay if overlay is not None else load_nsew_overlay(path)
    if not profile:
        profile = (config.get("nsew_profile") or {}) if isinstance(config, dict) else {}
    return deep_update(config, profile)


def save_nsew_overlay(overlay: dict[str, Any], path: str | Path | None = None) -> Path:
    """Write N/S/E/W standard values for the sidebar checkbox."""
    destination = nsew_config_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = deepcopy(overlay)
    payload.pop("nsew_profile", None)
    text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    destination.write_text(
        "# Groundup / N/S/E/W standard values.\n"
        "# Sidebar checkbox: Apply NSEW settings.\n"
        f"{text}",
        encoding="utf-8",
    )
    return destination
