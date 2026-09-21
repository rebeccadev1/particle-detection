"""Combine N/S/E/W directional images and write results next to them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from skimage import measure

from src.config import DEFAULT_INPUT_DIR, resolve_output_dir
from src.io.tile_loader import resolve_input_dir

DIRECTIONS = ("N", "S", "E", "W")
IMAGE_EXTS = {".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}
MIN_COUNTS = (2, 3, 4)
KIND_NAMES = ("combi", "particles_only", "symmetry_map")
KIND_FOLDERS = {
    "combi": "Combi",
    "particles_only": "Particles only",
    "symmetry_map": "Symmetry",
}
# Prefer matching the input suffix (usually .bmp); fall back to .png.
OUTPUT_NAMES = tuple(
    f"{kind}_{min_count}of4" for min_count in MIN_COUNTS for kind in KIND_NAMES
)


def list_direction_images(folder: str | Path) -> dict[str, Path]:
    """Images in ``folder`` whose stem is N, S, E, or W (any common suffix)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")

    found: dict[str, Path] = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        name = path.stem.upper()
        if name not in DIRECTIONS:
            continue
        if name in found:
            raise FileExistsError(
                f"Multiple images named {name} in {folder}: "
                f"{found[name].name} and {path.name}"
            )
        found[name] = path
    return {d: found[d] for d in DIRECTIONS if d in found}


def find_nsew_images(folder: str | Path) -> dict[str, Path]:
    """Locate four images named N, S, E, W in ``folder`` (any common image suffix)."""
    found = list_direction_images(folder)
    missing = [d for d in DIRECTIONS if d not in found]
    if missing:
        raise FileNotFoundError(
            f"Missing directional images in {folder}: {', '.join(missing)}. "
            "Expected files named N, S, E, W (e.g. N.bmp, S.png)."
        )
    return found


def nsew_output_dir(input_folder: str | Path) -> Path:
    """``Outputs/Output {input folder name}`` (or a sibling folder in tests)."""
    folder = Path(input_folder).resolve()
    name = f"Output {folder.name}"
    try:
        folder.relative_to(DEFAULT_INPUT_DIR.resolve())
    except ValueError:
        return folder.parent / name
    return resolve_output_dir(name)


def kind_output_dir(output_dir: str | Path, kind: str) -> Path:
    """Subfolder under the NSEW output folder: Combi, Particles only, or Symmetry."""
    name = KIND_FOLDERS.get(kind, kind)
    folder = Path(output_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _usable_dpi(dpi: Any) -> tuple[float, float] | None:
    """Return DPI only when both axes are positive (BMP often stores 0,0)."""
    if not dpi or len(dpi) < 2:
        return None
    x_dpi, y_dpi = float(dpi[0]), float(dpi[1])
    if x_dpi <= 0 or y_dpi <= 0:
        return None
    return (x_dpi, y_dpi)


def output_suffix(reference_path: str | Path) -> str:
    """Match the input image format so viewers treat size the same way."""
    suffix = Path(reference_path).suffix.lower()
    if suffix in IMAGE_EXTS:
        return suffix
    return ".png"


def save_same_resolution(path: str | Path, array: np.ndarray, reference_path: str | Path) -> None:
    """Write ``array`` with the same pixel size (and DPI when known) as ``reference_path``."""
    path = Path(path)
    h, w = array.shape[:2]
    with Image.open(reference_path) as ref:
        ref_w, ref_h = ref.size
        if (w, h) != (ref_w, ref_h):
            raise ValueError(
                f"{path}: output is {w}x{h} px but input is {ref_w}x{ref_h} px"
            )
        dpi = _usable_dpi(ref.info.get("dpi"))
    image = Image.fromarray(array)
    save_kw: dict[str, Any] = {}
    if dpi is not None:
        save_kw["dpi"] = dpi
    # Uncompressed BMP matches input size; PNG uses no palette downsample.
    if path.suffix.lower() == ".png":
        save_kw["compress_level"] = 1
    image.save(path, **save_kw)


def flatten_illumination(img: np.ndarray, sigma: float = 120) -> np.ndarray:
    """Remove the large-scale brightness gradient caused by directional lighting."""
    img_f = img.astype(np.float32)
    background = cv2.GaussianBlur(img_f, (0, 0), sigma)
    background = np.clip(background, 1.0, None)
    return img_f / background


def symmetry_signal_4way(
    sorted_stack: np.ndarray,
    sum_val: np.ndarray,
    min_count: int,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Score pixels that are bright in at least ``min_count`` of the four angles.

    ``sorted_stack`` is brightness sorted low→high. Index ``4 - min_count`` is the
    darkest of those ``min_count`` brightest angles (2nd-brightest for 2-of-4,
    2nd-darkest for 3-of-4, darkest for 4-of-4).
    """
    n_dirs = sorted_stack.shape[0]
    if min_count < 1 or min_count > n_dirs:
        raise ValueError(f"min_count must be between 1 and {n_dirs}, got {min_count}")
    min_val = sorted_stack[n_dirs - min_count]
    sym = (n_dirs * min_val) / (sum_val + eps)
    combined_brightness = sum_val / float(n_dirs)
    isotropic = sym * combined_brightness
    return isotropic, sym


def particles_from_isotropic(
    isotropic: np.ndarray,
    min_area: int = 3,
    max_area: int = 2000,
    max_eccentricity: float = 0.85,
    max_aspect: float = 3.0,
    min_circularity: float = 0.35,
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    iso_u8 = cv2.normalize(isotropic, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(iso_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
    )

    labels = measure.label(mask > 0, connectivity=2)
    props = measure.regionprops(labels, intensity_image=iso_u8)

    particles = []
    for p in props:
        area = p.area
        ecc = p.eccentricity
        major, minor = p.axis_major_length, p.axis_minor_length
        aspect = major / (minor + 1e-6)
        perim = p.perimeter if p.perimeter > 0 else 1
        circularity = 4 * np.pi * area / (perim ** 2)
        if (
            min_area <= area <= max_area
            and ecc < max_eccentricity
            and aspect < max_aspect
            and circularity > min_circularity
        ):
            particles.append(p)

    particle_mask = np.zeros(iso_u8.shape, dtype=np.uint8)
    for p in particles:
        for y, x in p.coords:
            particle_mask[y, x] = 255

    return particles, particle_mask, iso_u8


def load_flattened_nsew(
    path1: str | Path,
    path2: str | Path,
    path3: str | Path,
    path4: str | Path,
    flatten_sigma: float = 120,
) -> tuple[np.ndarray, np.ndarray]:
    paths = [path1, path2, path3, path4]
    imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    for i, img in enumerate(imgs):
        if img is None:
            raise FileNotFoundError(f"Could not read image: {paths[i]}")
        if img.shape != imgs[0].shape:
            raise ValueError("All images must be the same size and pixel-aligned")
    f_imgs = [flatten_illumination(img, flatten_sigma) for img in imgs]
    stack = np.array(f_imgs)
    return np.sort(stack, axis=0), np.sum(stack, axis=0)


def combine_nsew_folder(
    folder: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write 2-of-4, 3-of-4, and 4-of-4 combined images into an output folder."""
    folder = Path(folder)
    dest = Path(output_dir) if output_dir is not None else nsew_output_dir(folder)
    dest.mkdir(parents=True, exist_ok=True)
    nsew = find_nsew_images(folder)
    paths = [nsew[d] for d in DIRECTIONS]
    sorted_stack, sum_val = load_flattened_nsew(*paths)

    suffix = output_suffix(paths[0])
    variants: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for min_count in MIN_COUNTS:
        isotropic, sym = symmetry_signal_4way(sorted_stack, sum_val, min_count)
        particles, particle_mask, iso_u8 = particles_from_isotropic(isotropic)
        named: list[tuple[str, str, np.ndarray]] = [
            ("combi", f"combi_{min_count}of4{suffix}", iso_u8),
            ("particles_only", f"particles_only_{min_count}of4{suffix}", particle_mask),
            ("symmetry_map", f"symmetry_map_{min_count}of4{suffix}", (sym * 255).astype(np.uint8)),
        ]
        if min_count == 2:
            named.extend(
                (
                    ("combi", f"combi{suffix}", iso_u8),
                    ("particles_only", f"particles_only{suffix}", particle_mask),
                    ("symmetry_map", f"symmetry_map{suffix}", (sym * 255).astype(np.uint8)),
                )
            )
        for kind, filename, array in named:
            out_path = kind_output_dir(dest, kind) / filename
            save_same_resolution(out_path, array, paths[0])
            outputs.append(out_path)
        variants.append(
            {
                "min_count": min_count,
                "n_particles": len(particles),
                "outputs": [
                    str(kind_output_dir(dest, "combi") / f"combi_{min_count}of4{suffix}"),
                    str(
                        kind_output_dir(dest, "particles_only")
                        / f"particles_only_{min_count}of4{suffix}"
                    ),
                    str(
                        kind_output_dir(dest, "symmetry_map")
                        / f"symmetry_map_{min_count}of4{suffix}"
                    ),
                ],
            }
        )

    height, width = sorted_stack.shape[1], sorted_stack.shape[2]
    return {
        "folder": folder,
        "output_dir": dest,
        "nsew": nsew,
        "n_particles": variants[0]["n_particles"],
        "width": width,
        "height": height,
        "outputs": outputs,
        "variants": variants,
    }


def render_nsew_tab() -> None:
    """Main-screen tab: folder name under Inputs, then combine N/S/E/W."""
    st.caption(
        "Enter the name of a folder inside Inputs that contains images named N, S, E, and W. "
        "Each run writes 2-of-4, 3-of-4, and 4-of-4 combinations into "
        "Outputs/Output [folder name], in subfolders Combi, Particles only, and Symmetry, "
        "same pixel size and file type as N/S/E/W."
    )
    folder_name = st.text_input(
        "Folder name within Inputs",
        key="nsew_folder_name",
        placeholder="e.g. ringlight v2",
        help=f"Looks up this name under {DEFAULT_INPUT_DIR}.",
    )

    preview_folder = None
    preview_error = None
    if folder_name.strip():
        try:
            preview_folder = resolve_input_dir(folder_name)
            nsew = find_nsew_images(preview_folder)
        except (FileNotFoundError, FileExistsError, OSError) as exc:
            preview_error = str(exc)
            nsew = None
        else:
            st.write(f"Using `{preview_folder}`")
            for direction, path in nsew.items():
                st.write(f"- **{direction}**: {path.name}")

    if preview_error:
        st.warning(preview_error)

    if st.button("Combine N, S, E, W", type="primary", disabled=not folder_name.strip()):
        try:
            folder = resolve_input_dir(folder_name)
            with st.spinner("Combining directional images…"):
                result = combine_nsew_folder(folder)
        except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
            st.error(str(exc))
        else:
            counts = ", ".join(
                f"{v['min_count']}-of-4: {v['n_particles']}" for v in result["variants"]
            )
            st.session_state["nsew_last"] = {
                "folder": str(result["folder"]),
                "output_dir": str(result["output_dir"]),
                "n_particles": result["n_particles"],
                "width": result["width"],
                "height": result["height"],
                "variants": result["variants"],
            }
            st.success(
                f"Wrote {result['width']}×{result['height']} px images to {result['output_dir']}. "
                f"Particles {counts}."
            )

    last = st.session_state.get("nsew_last")
    if not last:
        return
    st.subheader("Last combination")
    st.write(
        f"{last['width']}×{last['height']} px · `{last.get('output_dir') or last['folder']}`"
    )
    captions = ("Combined (isotropic)", "Particles only", "Symmetry map")
    for variant in last.get("variants") or []:
        min_count = variant["min_count"]
        st.markdown(f"**{min_count} of 4 directions** · {variant['n_particles']} particles")
        cols = st.columns(3)
        for col, path_str, caption in zip(cols, variant["outputs"], captions):
            path = Path(path_str)
            if path.is_file():
                col.image(str(path), caption=f"{caption} ({min_count} of 4)", width="stretch")
            else:
                col.warning(f"Missing {path.name}")
