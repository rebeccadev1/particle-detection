"""On-disk label table and crop JPEGs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.labeling.queue import detection_key, nsew_key_aliases
from src.measurement.measurer import is_nsew_family_tile
from src.report.report_generator import encode_overlay_jpeg

LABEL_VALUES = ("particle", "not_sure", "not_particle")

COLUMNS = (
    "key",
    "particle_id",
    "source_tile",
    "x_global",
    "y_global",
    "size",
    "confidence",
    "label",
    "crop_path",
    "source_csv",
    "labeled_at",
)


class LabelStore:
    """CSV + JPEG collections under ``root`` (default ``labels/``)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.csv_path = self.root / "labels.csv"
        self.crops_dir = self.root / "crops"
        for label in LABEL_VALUES:
            (self.crops_dir / label).mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return _empty_table()
        df = pd.read_csv(self.csv_path)
        for column in COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA
        return df.loc[:, list(COLUMNS)]

    def labeled_keys(self) -> set[str]:
        df = self.load()
        if df.empty:
            return set()
        return {str(key) for key in df["key"].tolist()}

    def counts(self) -> dict[str, int]:
        df = self.load()
        counts = {label: 0 for label in LABEL_VALUES}
        if df.empty:
            return counts
        for label, group in df.groupby("label"):
            name = str(label)
            if name in counts:
                counts[name] = int(len(group))
        return counts

    def by_label(self, label: str) -> pd.DataFrame:
        if label not in LABEL_VALUES:
            raise ValueError(f"Unknown label {label!r}")
        df = self.load()
        if df.empty:
            return df
        return df.loc[df["label"] == label].reset_index(drop=True)

    def apply_label(
        self,
        record: Mapping[str, Any],
        label: str,
        crop: np.ndarray,
    ) -> Path:
        """Write the crop JPEG and append/replace the CSV row immediately."""
        return self.apply_labels([(record, label, crop)])[0]

    def apply_labels(
        self,
        items: list[tuple[Mapping[str, Any], str, np.ndarray]],
    ) -> list[Path]:
        """Write many crops and replace their CSV rows in one save."""
        if not items:
            return []
        prepared: dict[str, tuple[Mapping[str, Any], str, np.ndarray]] = {}
        order: list[str] = []
        for record, label, crop in items:
            if label not in LABEL_VALUES:
                raise ValueError(f"Unknown label {label!r}")
            key = _label_key(record)
            if key not in prepared:
                order.append(key)
            prepared[key] = (record, label, crop)

        drop_keys: set[str] = set()
        for key in order:
            drop_keys.update(nsew_key_aliases(key))

        df = self.load()
        if not df.empty:
            mask = df["key"].astype(str).isin(drop_keys)
            for crop_path in df.loc[mask, "crop_path"].tolist():
                if crop_path in (None, ""):
                    continue
                path = Path(str(crop_path))
                if path.is_file():
                    path.unlink()
            df = df.loc[~mask].reset_index(drop=True)

        stamped = _now()
        rows: list[dict[str, Any]] = []
        dests: list[Path] = []
        for key in order:
            record, label, crop = prepared[key]
            dest = self.crops_dir / label / f"{key}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(encode_overlay_jpeg(np.asarray(crop)))
            rows.append(_label_row(record, key, label, dest, stamped))
            dests.append(dest)
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        self._write(df)
        return dests

    def unlabel(self, key: str) -> dict[str, Any] | None:
        """Remove a labeled record and its crop. Returns the dropped row if any."""
        df = self.load()
        if df.empty:
            return None
        mask = df["key"].astype(str).isin(nsew_key_aliases(key))
        if not mask.any():
            return None
        dropped = df.loc[mask].iloc[-1].to_dict()
        for crop_path in df.loc[mask, "crop_path"].tolist():
            if crop_path in (None, ""):
                continue
            path = Path(str(crop_path))
            if path.is_file():
                path.unlink()
        self._write(df.loc[~mask].reset_index(drop=True))
        return {str(k): v for k, v in dropped.items()}

    def undo_last(self) -> dict[str, Any] | None:
        """Remove the most recently labeled row and return it to the queue."""
        df = self.load()
        if df.empty:
            return None
        times = df["labeled_at"].astype(str)
        idx = times.idxmax()
        key = str(df.loc[idx, "key"])
        return self.unlabel(key)

    def _write(self, df: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if df.empty:
            _empty_table().to_csv(self.csv_path, index=False)
            return
        df.loc[:, list(COLUMNS)].to_csv(self.csv_path, index=False)


def _label_key(record: Mapping[str, Any]) -> str:
    key = str(record.get("key") or detection_key(record))
    if is_nsew_family_tile(str(record.get("source_tile", "") or "")):
        return detection_key(record)
    return key


def _label_row(
    record: Mapping[str, Any],
    key: str,
    label: str,
    dest: Path,
    stamped: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "particle_id": record.get("id", record.get("particle_id", "")),
        "source_tile": record.get("source_tile", ""),
        "x_global": record.get("x_global", ""),
        "y_global": record.get("y_global", ""),
        "size": record.get("size", ""),
        "confidence": record.get("confidence", ""),
        "label": label,
        "crop_path": str(dest),
        "source_csv": record.get("source_csv", ""),
        "labeled_at": stamped,
    }


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in COLUMNS})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
