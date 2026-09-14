"""Load and access nested pipeline configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

RECALL_DEFAULTS = {
    "edge_exclude_px": 12.0,
    "min_confidence": 0.40,
    "min_circularity": 0.0,
    "structure_min_neighbors": 2,
    "structure_line_bin_px": 10.0,
}


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
    """Return a copy with loosened proposal gates when ``recall_mode`` is on.

    FFT/DoG, the size window, and cluster/grid rejection stay as in the
    recall overlay. Edge, confidence, and circularity are relaxed so a
    later classifier can recover rim debris the strict classical run drops.
    """
    if not cfg_get(config, "detection.recall_mode", False):
        return config
    out = deepcopy(config)
    det = out.setdefault("detection", {})
    overlay = dict(RECALL_DEFAULTS)
    custom = det.get("recall")
    if isinstance(custom, dict):
        overlay.update(custom)
    det.update(overlay)
    return out
