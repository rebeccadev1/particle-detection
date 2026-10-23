"""Summary statistics and downsampled overlay images for reporting."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.overall.config import cfg_get
from src.overall.io.tile_loader import (
    load_tile_image,
    matching_tile_paths,
    resolve_input_dir,
    try_parse_tile_filename,
)
from src.analyse.nsew import prepare_detection_paths
from src.analyse.stitching.stitcher import (
    LazyMosaic,
    downsample_for_target,
    downsample_image,
)
from src.overall.io.markers import CIRCLE_COLOR, draw_marker, marker_radius

SIZE_COUNT_THRESHOLDS_UM = (10, 20, 25, 50, 75)
SIZE_BIN_STEP_UM = 5.0
NM_PER_UM = 1000.0

def summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Return count and size-distribution summary for a particle table."""
    if df.empty:
        return {
            "particle_count": 0,
            "size_mean": 0.0,
            "size_std": 0.0,
            "size_min": 0.0,
            "size_max": 0.0,
        }
    sizes = df["size"].astype(float)
    return {
        "particle_count": int(len(df)),
        "size_mean": float(sizes.mean()),
        "size_std": float(sizes.std(ddof=0)),
        "size_min": float(sizes.min()),
        "size_max": float(sizes.max()),
    }

def particles_at_least(
    df: pd.DataFrame | None,
    thresholds_um: tuple[float, ...] = SIZE_COUNT_THRESHOLDS_UM,
) -> pd.DataFrame:
    """One row per size: how many particles are at least that large.

    ``size`` is equivalent circular diameter in nm.
    """
    labels = [
        f"≥{int(threshold) if float(threshold).is_integer() else threshold} µm"
        for threshold in thresholds_um
    ]
    if df is None or df.empty or "size" not in df.columns:
        counts = [0] * len(thresholds_um)
    else:
        sizes_um = df["size"].astype(float).to_numpy() / NM_PER_UM
        counts = [int(np.sum(sizes_um >= float(threshold))) for threshold in thresholds_um]
    return pd.DataFrame({"Size": labels, "Particles": counts})

def size_bin_edges(
    min_um: float,
    max_um: float,
    step_um: float = SIZE_BIN_STEP_UM,
) -> np.ndarray:
    """``[min, min+step, ...]`` covering ``max_um``.

    The last edge is the first step boundary at or above the largest size.
    A size that lands on ``min`` still gets one bin.
    """
    start = float(min_um)
    stop = max(float(max_um), start)
    step = float(step_um)
    steps = int(np.ceil((stop - start) / step - 1e-9))
    if steps < 1:
        steps = 1
    return start + np.arange(steps + 1, dtype=float) * step

def _size_bin_label(left: float, right: float) -> str:
    def fmt(value: float) -> str:
        rounded = round(value)
        if abs(value - rounded) < 1e-6:
            return str(int(rounded))
        return f"{value:.2f}"

    return f"{fmt(left)}–{fmt(right)}"

def size_distribution(
    df: pd.DataFrame | None,
    min_um: float,
    step_um: float = SIZE_BIN_STEP_UM,
) -> pd.DataFrame:
    """Particle counts in bins of ``step_um`` (5 µm) starting at ``min_um``.

    ``size`` is equivalent circular diameter in nm. ``min_um`` is the minimum
    size set for detection, not the smallest particle in the table. The first
    bin is ``[min, min+5)``, then ``[min+5, min+10)``, through the largest size.
    """
    if df is None or df.empty or "size" not in df.columns:
        edges = size_bin_edges(min_um, min_um, step_um)
        counts = np.zeros(len(edges) - 1, dtype=int)
    else:
        sizes_um = df["size"].astype(float).to_numpy() / NM_PER_UM
        edges = size_bin_edges(min_um, float(np.max(sizes_um)), step_um)
        counts, edges = np.histogram(sizes_um, bins=edges)
    labels = [_size_bin_label(edges[i], edges[i + 1]) for i in range(len(counts))]
    return pd.DataFrame({"count": counts}, index=labels)


def overlay_downsample(
    mosaic: LazyMosaic,
    config: dict[str, Any],
    crop: tuple[int, int, int, int] | None = None,
) -> int:
    """Pick the integer factor used to downsample tiles before stitching.

    ``report.downsample`` > 0 is an explicit override. Otherwise the factor is
    chosen so the stitched RGB overlay is about ``report.target_mb`` megabytes.
    """
    configured = int(cfg_get(config, "report.downsample", 0) or 0)
    if configured > 0:
        return configured
    if crop is None:
        height, width = mosaic.full_height, mosaic.full_width
    else:
        height, width = int(crop[2]), int(crop[3])
    target_mb = float(cfg_get(config, "report.target_mb", 20.0))
    return downsample_for_target(height, width, target_mb=target_mb, channels=3)

def overlay_rgb(
    mosaic: LazyMosaic,
    config: dict[str, Any],
    crop: tuple[int, int, int, int] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> np.ndarray:
    """Downsampled mosaic with no particle markers.

    ``crop`` is ``(y, x, h, w)`` in full-resolution mosaic pixels. ``crop=None``
    stitches every tile into one overview.
    """
    downsample = overlay_downsample(mosaic, config, crop=crop)
    preview = mosaic.preview(downsample=downsample, crop=crop, progress_cb=progress_cb)
    return photo_display_rgb(preview)

def draw_overlay_markers(
    rgb: np.ndarray,
    mosaic: LazyMosaic,
    particles: pd.DataFrame,
    config: dict[str, Any],
    crop: tuple[int, int, int, int] | None = None,
    colors: Mapping[str, tuple[int, int, int]] | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Copy ``rgb`` and draw circles on detected particles.

    Marker positions are converted back to mosaic pixels using ``pixel_size_nm``.
    ``crop`` is ``(y, x, h, w)`` in full-resolution mosaic pixels, matching the
    crop used to build ``rgb``. Unlabeled hits stay orange. ``colors`` maps a
    detection key to the review colour (green particle, red not a particle,
    blue not sure). ``scale`` is display pixels per full-resolution mosaic
    pixel; it overrides the downsample used when ``rgb`` is an already saved image.
    """
    if particles.empty or np.asarray(rgb).size == 0:
        return rgb

    placed_names = {placement.name for placement in mosaic.placements}
    if placed_names and "source_tile" in particles.columns:
        names = particles["source_tile"].astype(str).map(
            lambda path: str(path).replace("\\", "/").rsplit("/", 1)[-1]
        )
        particles = particles.loc[names.isin(placed_names)]
        if particles.empty:
            return rgb

    if scale is None:
        scale = 1.0 / float(overlay_downsample(mosaic, config, crop=crop))
    pixel_size = float(cfg_get(config, "pixel_size_nm", 1.0))
    if pixel_size <= 0:
        pixel_size = 1.0
    y_off = 0 if crop is None else crop[0]
    x_off = 0 if crop is None else crop[1]
    marked = np.array(rgb, copy=True)
    unlabeled = CIRCLE_COLOR

    for _, row in particles.iterrows():
        x_px = float(row["x_global"]) / pixel_size
        y_px = float(row["y_global"]) / pixel_size
        x_d = int(round((x_px - x_off) * scale))
        y_d = int(round((y_px - y_off) * scale))
        color = unlabeled
        if colors:
            key = str(row["key"]) if "key" in row.index and pd.notna(row["key"]) else ""
            if key:
                color = colors.get(key, unlabeled)
        draw_marker(
            marked,
            x_d,
            y_d,
            marker_radius(float(row["size"]), pixel_size, scale),
            color=color,
        )
    return marked

def photo_display_rgb(image: np.ndarray) -> np.ndarray:
    """RGB of a photograph at its saved gray levels.

    An 8-bit tile stays as saved. A 0–1 float is scaled by 255. Wider counts
    are scaled by their own maximum. There is no percentile stretch, which
    was lifting dark microscope photos.
    """
    array = np.asarray(image, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    finite = array[np.isfinite(array)]
    peak = float(finite.max()) if finite.size else 0.0
    if peak <= 1.0:
        scaled = np.clip(array, 0.0, 1.0) * 255.0
    elif peak <= 255.0:
        scaled = np.clip(array, 0.0, 255.0)
    else:
        scaled = np.clip(array, 0.0, None) * (255.0 / peak)
    gray = np.nan_to_num(scaled, nan=0.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def to_display_rgb(image: np.ndarray) -> np.ndarray:
    """Display a photo at its saved gray levels. No contrast stretch."""
    return photo_display_rgb(image)

_DPI = 220.0
_BASE_DPI = 150.0

def _s(px: float) -> int:
    """Scale a length chosen for the 150 dpi page."""
    return max(1, round(px * _DPI / _BASE_DPI))

_PAGE_W = _s(1240)
_PAGE_H = _s(1754)
_MARGIN = _s(64)
_INK = (28, 32, 38)
_MUTED = (90, 98, 110)
_ACCENT = (0x0F, 0x23, 0x8C)
_BAR = _ACCENT
_RULE = (220, 224, 230)

def write_last_run_pdf(
    table: pd.DataFrame,
    config: Mapping[str, Any],
    path: str | Path,
    mosaic_rgb: np.ndarray | None = None,
    mosaic_plain: np.ndarray | None = None,
    unstitched: Sequence[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> Path:
    """Write the run summary, size histogram, and mosaic or separate images.

    When ``mosaic_plain`` is set, the mosaic page shows the marked image and
    the same mosaic without markers beneath it. ``unstitched`` is
    ``(filename, labeled rgb, plain rgb)`` for images whose names have no
    grid pattern; each is placed on its own, not in the mosaic.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pages = _last_run_pages(table, config, mosaic_rgb, mosaic_plain, unstitched)
    pages[0].save(
        destination,
        "PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=_DPI,
        quality=95,
    )
    return destination

def _last_run_pages(
    table: pd.DataFrame,
    config: Mapping[str, Any],
    mosaic_rgb: np.ndarray | None,
    mosaic_plain: np.ndarray | None = None,
    unstitched: Sequence[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> list[Image.Image]:
    pages = [_summary_page(table, config)]
    if mosaic_rgb is not None and np.asarray(mosaic_rgb).size > 0:
        pages.append(_mosaic_page(np.asarray(mosaic_rgb), mosaic_plain))
    if unstitched:
        _path, folder = _input_path_and_folder(config)
        pages.extend(_unstitched_pages(unstitched, folder))
    return pages

_PT_PX = _DPI / 72.0
_H1 = round(18 * _PT_PX)
_H2 = round(14 * _PT_PX)
_FIGURE = round(13 * _PT_PX)
_BODY = round(11 * _PT_PX)
_LABEL = round(9 * _PT_PX)

_FONT_REGULAR = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)
_FONT_BOLD = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)
_FONT_ITALIC = (
    "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
)
_FONT_FILE: dict[bool, str | None] = {False: None, True: None}

def _font_file(*, bold: bool = False) -> str | None:
    cached = _FONT_FILE[bold]
    if cached is not None:
        return cached or None
    for candidate in (_FONT_BOLD if bold else _FONT_REGULAR):
        if Path(candidate).is_file():
            _FONT_FILE[bold] = candidate
            return candidate
    _FONT_FILE[bold] = ""
    return None

def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = _font_file(bold=bold)
    if path:
        return ImageFont.truetype(path, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _italic_font(size: int) -> ImageFont.ImageFont:
    for candidate in _FONT_ITALIC:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return _font(size)

def summary_pdf_name(config: Mapping[str, Any]) -> str:
    """``Summary of {input folder}.pdf``."""
    _path, folder = _input_path_and_folder(config)
    label = folder.strip() or "run"
    for char in '/\\:*?"<>|':
        label = label.replace(char, " ")
    label = " ".join(label.split()) or "run"
    return f"Summary of {label}.pdf"

def _input_path_and_folder(config: Mapping[str, Any]) -> tuple[str, str]:
    raw = str(config.get("input_dir") or "").strip().strip('"').strip("'").rstrip("/\\")
    if not raw:
        return "", ""
    try:
        resolved = resolve_input_dir(raw)
    except (FileNotFoundError, OSError):
        return raw, Path(raw).name
    return str(resolved), resolved.name

def _text_width(font: ImageFont.ImageFont, text: str) -> int:
    box = font.getbbox(text)
    return int(box[2] - box[0])

def _wrap_pil_text(text: str, font: ImageFont.ImageFont, max_w: int) -> list[str]:
    """Break ``text`` so each line fits ``max_w``, preferring ``/`` and spaces."""
    if _text_width(font, text) <= max_w:
        return [text]
    lines: list[str] = []
    rest = text
    while rest:
        if _text_width(font, rest) <= max_w:
            lines.append(rest)
            break
        cut = 1
        for index in range(1, len(rest) + 1):
            if _text_width(font, rest[:index]) > max_w:
                break
            cut = index
        split = cut
        for index in range(cut, 0, -1):
            if rest[index - 1] in "/ ":
                split = index
                break
        lines.append(rest[:split])
        rest = rest[split:]
    return lines

def _summary_page(table: pd.DataFrame, config: Mapping[str, Any]) -> Image.Image:
    image = Image.new("RGB", (_PAGE_W, _PAGE_H), "white")
    draw = ImageDraw.Draw(image)
    metric_label = _font(_LABEL)
    metric_value = _font(_FIGURE)
    header = _font(_LABEL, bold=True)

    y = _MARGIN
    title = _font(_H1, bold=True)
    body = _font(_BODY)
    path_font = _italic_font(_BODY)
    path, _folder = _input_path_and_folder(config)
    max_text = _PAGE_W - 2 * _MARGIN
    draw.text((_MARGIN, y), "Particle Detection Output", fill=_INK, font=title)
    y += _H1 + _s(14)
    if path:
        draw.text((_MARGIN, y), "Path to analysed image(s):", fill=_INK, font=body)
        y += _BODY + _s(6)
        path_box = path_font.getbbox("Hg/()")
        path_step = int(path_box[3] - path_box[1]) + _s(8)
        for wrapped in _wrap_pil_text(path, path_font, max_text):
            draw.text((_MARGIN, y), wrapped, fill=_ACCENT, font=path_font)
            y += path_step
        y += _s(8)
        draw.line([(_MARGIN, y), (_PAGE_W - _MARGIN, y)], fill=_RULE, width=_s(2))
        y += _s(14)
        y += _s(12)

    stats = summary_stats(table)
    metrics = (
        ("Particles", f"{stats['particle_count']}"),
        ("Mean size (µm)", f"{stats['size_mean'] / NM_PER_UM:.2f}"),
        ("Min size (µm)", f"{stats['size_min'] / NM_PER_UM:.2f}"),
        ("Max size (µm)", f"{stats['size_max'] / NM_PER_UM:.2f}"),
    )
    slot = (_PAGE_W - 2 * _MARGIN) // len(metrics)
    for index, (label, value) in enumerate(metrics):
        x = _MARGIN + index * slot
        draw.text((x, y), label, fill=_MUTED, font=metric_label)
        draw.text((x, y + _LABEL + _s(6)), value, fill=_INK, font=metric_value)
    y += _LABEL + _s(6) + _FIGURE + _s(22)
    draw.line([(_MARGIN, y), (_PAGE_W - _MARGIN, y)], fill=_RULE, width=_s(2))
    y += _s(22)

    caption = "Detected particles at least this large (equivalent circular diameter)."
    draw.text((_MARGIN, y), caption, fill=_INK, font=body)
    y += _BODY + _s(14)
    counts = particles_at_least(table)
    y = _draw_count_table(draw, counts, y, body, header)
    y += _s(16)
    draw.line([(_MARGIN, y), (_PAGE_W - _MARGIN, y)], fill=_RULE, width=_s(2))
    y += _s(22)

    draw.text((_MARGIN, y), "Size distribution (µm)", fill=_INK, font=_font(_H2, bold=True))
    y += _H2 + _s(16)
    min_size_nm = float((config.get("detection") or {}).get("min_size_nm") or 0.0)
    distribution = size_distribution(table, min_size_nm / NM_PER_UM)
    _draw_histogram(image, distribution, y, y + _s(640))
    return image


def _count_table_edges(
    counts: pd.DataFrame,
    body: ImageFont.ImageFont,
    header: ImageFont.ImageFont,
) -> tuple[int, int]:
    """Left and right of the size table, wide enough for the text only."""
    size_w = _text_width(header, "Size")
    count_w = _text_width(header, "Particles")
    for _, row in counts.iterrows():
        size_w = max(size_w, _text_width(body, str(row["Size"])))
        count_w = max(count_w, _text_width(body, str(int(row["Particles"]))))
    left = _MARGIN
    right = min(left + size_w + _s(36) + count_w, _PAGE_W - _MARGIN)
    return left, right

def _draw_count_table(
    draw: ImageDraw.ImageDraw,
    counts: pd.DataFrame,
    y: int,
    body: ImageFont.ImageFont,
    header: ImageFont.ImageFont,
) -> int:
    left, right = _count_table_edges(counts, body, header)
    row_h = _BODY + _s(14)
    draw.text((left, y), "Size", fill=_MUTED, font=header)
    particles = "Particles"
    draw.text((right - _text_width(header, particles), y), particles, fill=_MUTED, font=header)
    y += _LABEL + _s(10)
    draw.line([(left, y), (right, y)], fill=_RULE, width=_s(1))
    y += _s(8)
    for _, row in counts.iterrows():
        draw.text((left, y), str(row["Size"]), fill=_INK, font=body)
        value = str(int(row["Particles"]))
        draw.text((right - _text_width(body, value), y), value, fill=_INK, font=body)
        y += row_h
    return y

def _axis_ticks(peak: float, count: int = 4) -> list[float]:
    if peak <= 0:
        return [0.0]
    raw = peak / max(count, 1)
    power = 10 ** math.floor(math.log10(raw))
    step = 10 * power
    for factor in (1, 2, 2.5, 5, 10):
        if factor * power >= raw - 1e-9:
            step = factor * power
            break
    limit = math.ceil((peak - 1e-9) / step) * step
    ticks: list[float] = []
    value = 0.0
    while value <= limit + step * 0.01:
        ticks.append(value)
        value += step
        if len(ticks) > 8:
            break
    return ticks

def _tick_label(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.1f}"

def _paste_vertical(
    image: Image.Image,
    text: str,
    font: ImageFont.ImageFont,
    x: int,
    y: int,
) -> None:
    """Draw ``text`` rotated so it reads upward beside the y-axis."""
    bbox = font.getbbox(text)
    width = int(bbox[2] - bbox[0]) + 4
    height = int(bbox[3] - bbox[1]) + 4
    plate = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(plate).text((2 - bbox[0], 2 - bbox[1]), text, font=font, fill=_INK + (255,))
    rotated = plate.rotate(90, expand=True)
    image.paste(rotated, (x, y), rotated)

def _draw_histogram(
    image: Image.Image,
    distribution: pd.DataFrame,
    top: int,
    bottom: int,
) -> None:
    draw = ImageDraw.Draw(image)
    ticks_font = _font(_LABEL)
    axis_font = _font(_BODY)
    counts = [int(value) for value in distribution["count"].tolist()]
    labels = [str(label) for label in distribution.index]
    peak = max(counts) if counts else 0
    ticks = _axis_ticks(float(peak))
    scale = ticks[-1] if ticks[-1] > 0 else 1.0
    axis_y = "Amount of particles"
    y_box = axis_font.getbbox(axis_y)
    y_title_span = int(y_box[2] - y_box[0]) + _s(4)
    y_title_thick = int(y_box[3] - y_box[1]) + _s(4)
    tick_w = max((_text_width(ticks_font, _tick_label(tick)) for tick in ticks), default=0)
    left = _MARGIN + y_title_thick + _s(10) + tick_w + _s(12)
    right = _PAGE_W - _MARGIN
    plot_top = top + _s(8)
    plot_bottom = bottom - _BODY - _LABEL - _s(28)
    for tick in ticks:
        if tick <= 0:
            continue
        y = plot_bottom - (tick / scale) * (plot_bottom - plot_top)
        draw.line([(left, y), (right, y)], fill=_RULE, width=_s(1))
    draw.line([(left, plot_top), (left, plot_bottom)], fill=_INK, width=_s(2))
    draw.line([(left, plot_bottom), (right, plot_bottom)], fill=_INK, width=_s(2))
    for tick in ticks:
        y = plot_bottom - (tick / scale) * (plot_bottom - plot_top)
        draw.line([(left - _s(6), y), (left, y)], fill=_INK, width=_s(2))
        label = _tick_label(tick)
        draw.text(
            (left - _s(10) - _text_width(ticks_font, label), y - _LABEL // 2),
            label,
            fill=_INK,
            font=ticks_font,
        )
    title_top = plot_top + max((plot_bottom - plot_top - y_title_span) // 2, 0)
    _paste_vertical(image, axis_y, axis_font, _MARGIN, title_top)
    if not counts:
        return
    slot = (right - left) / len(counts)
    gap = min(_s(10), max(_s(2), slot * 0.25))
    for index, (count, label) in enumerate(zip(counts, labels)):
        bar_left = left + index * slot + gap / 2
        bar_right = left + (index + 1) * slot - gap / 2
        height = (count / scale) * (plot_bottom - plot_top)
        bar_top = plot_bottom - height
        draw.rectangle([bar_left, bar_top, bar_right, plot_bottom], fill=_BAR)
        center = (bar_left + bar_right) / 2
        draw.line([(center, plot_bottom), (center, plot_bottom + _s(6))], fill=_INK, width=_s(2))
        text_w = _text_width(ticks_font, label)
        label_x = min(max(center - text_w / 2, _MARGIN), right - text_w)
        draw.text((label_x, plot_bottom + _s(8)), label, fill=_INK, font=ticks_font)
    axis = "Size (µm)"
    draw.text(
        ((left + right) / 2 - _text_width(axis_font, axis) / 2, bottom - _BODY - _s(4)),
        axis,
        fill=_INK,
        font=axis_font,
    )

def _rgb_tile(mosaic_rgb: np.ndarray) -> Image.Image:
    rgb = np.asarray(mosaic_rgb)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(np.ascontiguousarray(rgb[:, :, :3].astype(np.uint8)), mode="RGB")

def _paste_fitted(page: Image.Image, tile: Image.Image, top: int, max_h: int) -> int:
    max_w = _PAGE_W - 2 * _MARGIN
    scale = min(max_w / tile.width, max_h / max(tile.height, 1), 1.0)
    resized = tile.resize(
        (max(int(tile.width * scale), 1), max(int(tile.height * scale), 1)),
        Image.Resampling.LANCZOS,
    )
    page.paste(resized, ((_PAGE_W - resized.width) // 2, top))
    return top + resized.height

def _mosaic_page(
    mosaic_rgb: np.ndarray,
    mosaic_plain: np.ndarray | None = None,
) -> Image.Image:
    image = Image.new("RGB", (_PAGE_W, _PAGE_H), "white")
    draw = ImageDraw.Draw(image)
    draw.text((_MARGIN, _MARGIN), "Stitched mosaic", fill=_INK, font=_font(_H1, bold=True))
    y = _MARGIN + _H1 + _s(12)
    plain = None if mosaic_plain is None or np.asarray(mosaic_plain).size == 0 else mosaic_plain
    if plain is None:
        draw.text(
            (_MARGIN, y),
            "Downsampled mosaic with detected particles marked.",
            fill=_MUTED,
            font=_font(_BODY),
        )
        y += _BODY + _s(20)
        _paste_fitted(image, _rgb_tile(mosaic_rgb), y, _PAGE_H - y - _MARGIN)
        return image

    caption_h = _BODY + _s(10)
    gap = _s(28)
    available = _PAGE_H - y - _MARGIN - 2 * caption_h - gap
    slot = max(available // 2, 1)
    draw.text((_MARGIN, y), "With labels", fill=_MUTED, font=_font(_BODY))
    y += caption_h
    y = _paste_fitted(image, _rgb_tile(mosaic_rgb), y, slot)
    y += gap
    draw.text((_MARGIN, y), "Without labels", fill=_MUTED, font=_font(_BODY))
    y += caption_h
    _paste_fitted(image, _rgb_tile(plain), y, slot)
    return image

def _filename_is_placeable(filename: str, pattern: str) -> bool:
    """True when the name has row/column or x/y, so it can join the mosaic."""
    meta = try_parse_tile_filename(filename, pattern)
    if not meta:
        return False
    if "x" in meta and "y" in meta:
        return True
    return "row" in meta and "col" in meta

def unstitched_tile_paths(config: Mapping[str, Any]) -> list[Path]:
    """Images detection ran that have no grid pattern.

    Directional N/S/E/W shots are combined first and left out, the same way
    the pipeline does. A combi with no row and column is shown on its own
    page. Grid combis are stitched instead.
    """
    folder = str(config.get("input_dir") or "").strip()
    if not folder:
        return []
    pattern = str(config.get("filename_pattern") or "")
    try:
        paths = matching_tile_paths(
            folder,
            pattern,
            config.get("run"),
            config.get("magnification"),
            include_unmatched=True,
        )
        paths = prepare_detection_paths(paths)
    except (FileNotFoundError, OSError, ValueError):
        return []
    return [path for path in paths if not _filename_is_placeable(path.name, pattern)]

def _particles_for_path(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    if table is None or table.empty or "source_tile" not in table.columns:
        return table.iloc[0:0] if isinstance(table, pd.DataFrame) else pd.DataFrame()
    sources = table["source_tile"].astype(str).str.replace("\\", "/", regex=False)
    base = sources.map(lambda value: value.rsplit("/", 1)[-1])
    parent = sources.map(
        lambda value: value.rsplit("/", 1)[0].rsplit("/", 1)[-1] if "/" in value else ""
    )
    mask = (base == path.name) & (
        (parent == "") | (parent == ".") | (parent == path.parent.name)
    )
    return table.loc[mask]

def _tile_preview_factor(height: int, width: int, config: Mapping[str, Any]) -> int:
    """Downsample one loose tile. An explicit factor wins; otherwise cap the preview."""
    settings = dict(config)
    configured = int(cfg_get(settings, "report.downsample", 0) or 0)
    if configured > 0:
        return configured
    target_mb = float(cfg_get(settings, "report.target_mb", 8.0))
    return downsample_for_target(
        height, width, target_mb=min(target_mb, 8.0), channels=3
    )

def _draw_local_markers(
    rgb: np.ndarray,
    particles: pd.DataFrame,
    config: Mapping[str, Any],
    downsample: int,
) -> np.ndarray:
    """Orange circles for particles whose origin on this tile is the top-left."""
    if (
        particles.empty
        or np.asarray(rgb).size == 0
        or "x_global" not in particles.columns
        or "y_global" not in particles.columns
        or "size" not in particles.columns
    ):
        return rgb
    pixel_size = float(config.get("pixel_size_nm") or 1.0)
    if pixel_size <= 0:
        pixel_size = 1.0
    factor = max(int(downsample), 1)
    marked = np.array(rgb, copy=True)
    for _, row in particles.iterrows():
        x_d = int(round((float(row["x_global"]) / pixel_size) / factor))
        y_d = int(round((float(row["y_global"]) / pixel_size) / factor))
        if not (0 <= x_d < marked.shape[1] and 0 <= y_d < marked.shape[0]):
            continue
        draw_marker(
            marked,
            x_d,
            y_d,
            marker_radius(float(row["size"]), pixel_size, 1.0 / factor),
        )
    return marked

def unstitched_tile_pair(
    path: str | Path,
    table: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Downsampled RGB of one loose image, with labels and without them."""
    tile_path = Path(path)
    image = load_tile_image(tile_path)
    factor = _tile_preview_factor(int(image.shape[0]), int(image.shape[1]), config)
    small = downsample_image(np.asarray(image, dtype=np.float32), factor)
    plain = photo_display_rgb(small)
    labeled = _draw_local_markers(
        plain, _particles_for_path(table, tile_path), config, factor
    )
    return labeled, plain

def unstitched_tile_rgb(
    path: str | Path,
    table: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    labels: bool = True,
) -> np.ndarray:
    """Downsampled RGB of one image that is not part of the mosaic."""
    labeled, plain = unstitched_tile_pair(path, table, config)
    return labeled if labels else plain

def _unstitched_pages(
    tiles: Sequence[tuple[str, np.ndarray, np.ndarray]],
    folder: str = "",
) -> list[Image.Image]:
    """Pages of images that were not stitched, with and without labels."""
    pages: list[Image.Image] = []
    image = Image.new("RGB", (_PAGE_W, _PAGE_H), "white")
    draw = ImageDraw.Draw(image)
    y = _MARGIN
    title = f"Stitches of {folder.strip()}" if str(folder).strip() else "Stitches"
    draw.text((_MARGIN, y), title, fill=_INK, font=_font(_H1, bold=True))
    y += _H1 + _s(16)
    caption_font = _font(_BODY)
    first_on_page = True
    for _name, labeled_rgb, plain_rgb in tiles:
        if np.asarray(labeled_rgb).size == 0 and np.asarray(plain_rgb).size == 0:
            continue
        label_h = _BODY + _s(8)
        gap = _s(16)
        room = _PAGE_H - _MARGIN - y
        if room < _s(320) and not first_on_page:
            pages.append(image)
            image = Image.new("RGB", (_PAGE_W, _PAGE_H), "white")
            draw = ImageDraw.Draw(image)
            y = _MARGIN
            first_on_page = True
            room = _PAGE_H - _MARGIN - y
        slot = max((min(room, _s(900)) - 2 * label_h - gap) // 2, 1)
        plain = None if np.asarray(plain_rgb).size == 0 else plain_rgb
        if plain is None:
            draw.text((_MARGIN, y), "With labels", fill=_MUTED, font=caption_font)
            y += label_h
            y = _paste_fitted(image, _rgb_tile(labeled_rgb), y, slot * 2)
        else:
            draw.text((_MARGIN, y), "With labels", fill=_MUTED, font=caption_font)
            y += label_h
            y = _paste_fitted(image, _rgb_tile(labeled_rgb), y, slot)
            y += gap
            draw.text((_MARGIN, y), "Without labels", fill=_MUTED, font=caption_font)
            y += label_h
            y = _paste_fitted(image, _rgb_tile(plain), y, slot)
        y += _s(28)
        first_on_page = False
    pages.append(image)
    return pages

def encode_overlay_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    """JPEG bytes for download; ``rgb`` is uint8 RGB."""
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("Failed to encode mosaic JPEG.")
    return buffer.tobytes()
