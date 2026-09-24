# Particle detection on structured wafer surfaces

This project finds **contaminant particles** on optical microscope images of a **patterned wafer**. The images are a grid of overlapping tiles (TIFF, BMP, PNG, or JPEG). The surface itself is periodic (circuit / lithography structure), so the hard problem is not “find bright spots” — it is **ignore the repeating pattern and keep only blob-like debris**.

This is a **classical image-processing pipeline** with an optional sklearn / CNN post-filter on detector hits. Parameters live in `config.yaml` (and the Streamlit sidebar in `app.py`). Changing wafer pitch, magnification, or lighting usually means retuning those numbers. The ML model only scores blobs DoG already found.

A second path handles **N/S/E/W directional lighting** of the same field: combine the four angles into a particles-only image, then run the same detector with `nsew_config.yaml` knobs.

---

## Contents

1. [The problem](#the-problem)
2. [What the pipeline produces](#what-the-pipeline-produces)
3. [Why detection runs per tile, not on the mosaic](#why-detection-runs-per-tile-not-on-the-mosaic)
4. [End-to-end data flow](#end-to-end-data-flow)
5. [Tile discovery, load, and placement](#1-tile-discovery-load-and-placement)
6. [Preprocessing](#2-preprocessing)
7. [Structured-background suppression](#3-structured-background-suppression)
8. [Local SNR and prominence](#4-local-snr-and-prominence)
9. [Compact residual islands](#5-compact-residual-islands)
10. [Region-border attenuation](#6-region-border-attenuation)
11. [Blob detection (difference of Gaussians)](#7-blob-detection-difference-of-gaussians)
12. [Per-blob filters](#8-per-blob-filters)
13. [Spatial structure filters](#9-spatial-structure-filters)
14. [Global coordinates and de-duplication](#10-global-coordinates-and-de-duplication)
15. [Outputs and the mosaic overlay](#11-outputs-and-the-mosaic-overlay)
16. [Scale: pixels, micrometres, nanometres](#scale-pixels-micrometres-nanometres)
17. [N/S/E/W directional lighting](#nsew-directional-lighting)
18. [Parameter reference](#parameter-reference)
19. [Tuning: if the result looks wrong](#tuning-if-the-result-looks-wrong)
20. [What this does not do](#what-this-does-not-do)
21. [ML cascade (optional)](#ml-cascade-optional)
22. [How to run](#how-to-run)

---

## The problem

A 5× optical tile of a patterned wafer is not a blank field with specks on it. Typical content:

- **Periodic lattice** — die / metal / resist pattern that repeats at a few-to-tens-of-pixels pitch. In the Fourier domain this is a handful of bright peaks. In the spatial domain it is a carpet of similar-looking nodes.
- **Layout pads and boxes** — large rectangular regions whose *corners* and *rims* look round and bright after a morphological top-hat. Difference-of-Gaussians (DoG) reports them as blobs of the same size as real debris.
- **Area boundaries** — step edges between regions. They are not a lattice (FFT notch does not remove them) and they are not compact (but DoG beads *just inside* the box along the ringing band).
- **Illumination** — slow vignetting and lamp drift across a multi-megapixel tile.
- **Overlapping tiles** — the same particle can appear on two neighbouring TIFFs, so a wafer-wide table must merge duplicates.
- **Mixed layout** — neighbouring tiles (and even regions inside one tile) do not share one pitch. A notch mask built on the first tile is wrong for the rest.

A particle of interest is a **compact, irregular bright flake** (clump, diamond, or potato-shaped — not a perfect circle). The **reported size window** is **20–100 µm** (`min_size_nm` / `max_size_nm`). DoG still searches from `blob_min_sigma: 3.7` (~10 µm at 0.96 µm/px): raising the search floor to match 20 µm flooded layout false positives on R4. Hits smaller than 20 µm are dropped after measurement.

Layout — letters, pad corners, array row starts, fiducials, lattice nodes, box frames — is a false positive the rest of the pipeline exists to reject.

---

## What the pipeline produces

| File | Content |
|------|---------|
| `particles.csv` | One row per merged particle: identity, global nm, size, confidence, source tile, N/S/E/W merge fields, plus residual features for the optional ML gate |
| `mosaic_overlay.jpg` | Downsampled stitch of every *placeable* tile with a red circle at each detection |

The Streamlit app (`streamlit run app.py`) has tabs: **Detection**, **NSEW**, **Tiles**, **Last Run**, **Tinder** (label queue), **Database**. Detection shows the table, a size histogram in **µm**, and can render a **full-resolution crop** of the mosaic.

---

## Why detection runs per tile, not on the mosaic

Tiles are large (multi-megapixel) and numerous. Assembling a full-resolution mosaic just to detect on it would:

- need tens of gigabytes of RAM,
- mix neighbouring tiles’ illumination and overlap twice,
- and not help the lattice problem (each tile already has its own pattern).

Instead, detection runs **one tile at a time**, in parallel across CPU **threads** (`ThreadPoolExecutor`). A process pool spawned from Streamlit hangs on macOS (workers re-import the server and the UI stays on “Idle”). Each worker:

1. Loads one image, converts it to grayscale, and **discards the RGB**.
2. Preprocesses, detects, optionally ML-filters, then measures.
3. Returns only particle rows, a placement record, and (optionally) a small thumbnail.
4. Drops the full-resolution array.

The mosaic is assembled later from those thumbnails. It is a **report**, not an input to detection.

`pipeline.workers: 0` means “use every CPU core.” When more than one worker is used, OpenCV is pinned to **one thread** so *N* workers do not oversubscribe the machine. Set `workers: 1` to debug a single tile serially.

---

## End-to-end data flow

```
Image folder
    │
    ├─ match R{run}_{row}_{col}_{mag}X.{tif,bmp,png,jpg}   (unmatched files still load)
    ├─ grid origin = min(row), min(col) among placeable tiles
    └─ place each tile: origin = (col−col0)×W×(1−overlap), same for row
            │
            ▼  (ThreadPoolExecutor, one image per worker)
    load RGB → gray → float32
            │
            ▼  preprocessing (corrections.py)
    denoise → flatten illumination → percentile stretch
            │
            ▼  background suppression (detector.py)
    FFT notch  ──no lattice peaks──►  white top-hat fallback
            │
            ▼
    invert if particles are dark
            │
            ▼
    local SNR (FFT residuals only) → compact-island mask
            │
            ▼
    soften region borders (islands keep ≥0.85 of the ramp)
            │
            ▼
    scale to [0, 1] → zero below min_prominence
            │
            ▼
    DoG blobs + island-centroid seeds (OpenCV Gaussians; LoG if blob_method: log)
            │
            ▼  per-blob gates
    size window · measured ECD · prominence · min_confidence
    edge distance (skipped on islands) · circularity
    local peak on preprocessed tile (skipped on islands) · compact support
            │
            ▼  among remaining blobs  (order can swap with ML)
    drop clustered neighbours · drop long axis-aligned rows
            │
            ▼  optional ml.enabled
    ExtraTrees HOG (± CNN on uncertain band)
            │
            ▼
    map (x, y, diameter) to global nm · tag source_tile
            │
            ▼  after all tiles
    KD-tree merge (overlap duplicates)
    N/S/E/W + particles_only → directional merge (at most one hit per direction)
            │
            ├─ particles.csv
            └─ downsample thumbnails → stitch → red circles → mosaic_overlay.jpg
```

Implemented in `src/pipeline/runner.py`. Modules:

| Step | Module | Role |
|------|--------|------|
| Find tiles | `src/io/tile_loader.py` | Regex match, optional run/mag filters, unmatched files, RGB→gray |
| Place tile | `src/stitching/stitcher.py` | Top-left mosaic origin from `row`/`col` and `overlap_fraction` |
| Preprocess | `src/preprocessing/corrections.py` | Denoise, flatten, stretch; stay **float32** |
| Detect | `src/detection/detector.py` | Lattice out, islands, compact high-prominence residual, DoG, filters |
| ML | `src/ml/infer.py` | Optional ExtraTrees / cascade CNN on detector hits |
| Measure | `src/measurement/measurer.py` | Local px → global nm; cluster nearby hits; N/S/E/W merge |
| NSEW combine | `src/nsew.py` | Four lighting angles → combi / particles-only / symmetry maps |
| Report | `src/report/report_generator.py` | Stats, markers, JPEG from cached thumbnails. UI: `app.py` |

Default order is **gates → structure filters → ML**. Set `ml.score_before_structure: true` to score gated DoG hits first, then cluster/line-filter the ML survivors (can add lattice neighbours that cluster-kill isolated flakes). `config.yaml` keeps this **false**.

---

## 1. Tile discovery, load, and placement

### Filenames

Default pattern (`filename_pattern`):

```
R(?P<run>\d+)_(?P<row>\d+)_(?P<col>\d+)_(?P<mag>[\d.]+)X\.(?:tiff?|bmp|png|jpe?g)
```

Example: `R3_5_32_5X.tif` → run 3, row 5, column 32, 5×. Matching is case-insensitive. `.tif`, `.tiff`, `.bmp`, `.png`, `.jpg`, and `.jpeg` all work.

Optional sidebar / config filters:

- `run` — keep only that run (empty = all).
- `magnification` — keep only that mag (empty = all).

Column 0 is **not** dropped. If `R3_2_0_5X.tif` is in the folder it is loaded. The mosaic origin is the **smallest row and column present**, so unused column 0 does not reserve empty space.

The pipeline also loads files that **do not** match the pattern (`include_unmatched=true`). Those tiles are detected but not placed on the mosaic (typical for `N.bmp` / `S.bmp` / `particles_only_2of4.bmp`).

If the pattern also has named groups `x` and `y`, those are used as the mosaic origin in pixels instead of the row/col grid.

Bare folder names such as `R3 04-08` are resolved under `ASML SE/Inputs` (`resolve_input_dir`). Relative paths that start with `.` or `..` are resolved from `particle_detection/`. Absolute paths are used as-is.

### Load

TIFF uses `tifffile.imread`. BMP / PNG / JPEG use OpenCV. Extra channels are dropped immediately:

- RGB / RGBA uint8 → OpenCV `COLOR_RGB2GRAY`
- other 3-channel → Rec. 601 weights `(0.299, 0.587, 0.114)`

Detection is intensity-based; colour is never used after this point.

Tile **height and width** for mosaic planning come from the file **header** (`peek_tile_hw`) without decoding pixels, so the grid can be laid out before any worker starts.

### Placement

A regular grid with fractional overlap (current `overlap_fraction: 0.0` = no assumed overlap):

```
step_x = tile_width  × (1 − overlap)
step_y = tile_height × (1 − overlap)

x0 = (col − col_origin) × step_x
y0 = (row − row_origin) × step_y
```

`row_origin` / `col_origin` are the **smallest row and column present in the folder**. Origins are rounded to integer pixels (`TilePlacement.x0`, `y0`).

This is **filename-grid placement**, not feature-based registration. If the stage overlap is not `overlap_fraction`, global positions and de-duplication will drift.

---

## 2. Preprocessing

Applied independently on each tile (`apply_corrections`). Arrays stay **float32**.

### Float conversion

Integer images (typical microscope uint8 / uint16) are divided by their max so values sit roughly in `[0, 1]`. Images already in `[0, 1]` are left alone (`vmax ≤ 1.5`).

### Gaussian denoise (`denoise_sigma`, default 1 px)

A small blur so blob detection does not fire on sensor speckle. Too large a sigma also rounds off real particles and merges close neighbours; 1 px is a noise filter, not a smoother. NSEW overlay uses **0.5 px**.

### Illumination flatten (`flatten_sigma`, default 160 px)

Divide the tile by a **wide** Gaussian of itself. That removes vignetting and slow intensity drift so they do not look like particles.

The kernel is large **on purpose**. A small sigma straddles die / area borders and paints a **halo** that later looks like debris. Illumination varies slowly, so the blur is computed at about **1/8 resolution** (`INTER_AREA` downsample → Gaussian with sigma scaled by the shrink factor → bilinear upsample) and the result is divided into the full-res tile. After the divide, the median is restored so the typical gray level matches the input.

Disable with `flatten_illumination: false` if a tile is already flat and flattening is ringing at pad edges.

### Percentile contrast stretch (default 1st–99th)

Maps those percentiles to `[0, 1]` and clips. Every tile then has comparable contrast before the FFT / top-hat. On multi-megapixel tiles the percentiles are estimated from a **512-bin histogram**, not a full sort.

---

## 3. Structured-background suppression

Particles sit on a **repeating lattice**. Two methods (`detection.method`), plus an automatic fallback.

### FFT notch (default `method: fft`)

Idea: a periodic pattern is a set of peaks in the 2-D Fourier transform. Zero those peaks, invert, and the lattice disappears while compact (non-periodic) blobs remain.

Concrete steps (`suppress_periodic_fft` / `_notch_mask_from_spectrum`):

1. Real 2-D FFT (`scipy.fft.rfft2`, one worker thread).
2. Magnitude, normalised so the strongest bin is 1.
3. Protect a disk around **DC** (the mean intensity and large-scale shading). Radius is `max(3 × fft_notch_radius, 6)` bins. Without this, flattening leftovers and the whole-tile mean would be notched out and the residual would be garbage.
4. Mark bins whose normalised magnitude ≥ `fft_peak_threshold` (default **0.35**) and that are outside the DC disk.
5. Dilate those peaks by `fft_notch_radius` (default **3**) with an elliptical kernel so a peak that leaked into neighbouring bins is fully removed. The height axis is circularly padded before dilation because `rfft2` wraps vertically.
6. Set those spectrum bins to 0 and inverse-transform (`irfft2`).

**Mask source** (`fft_mask`):

- `per_tile` (default) — build the notch from **this** tile’s spectrum. Mixed-layout wafers do not share one pitch; a mask from the first tile is wrong for the others.
- `shared` — compute the mask once on the first tile (after preprocessing) and reuse it. Only correct when every tile has the same lattice.

**Lattice probe (speed).** A full-resolution FFT on a multi-megapixel mixed-layout tile that has **no** peaks is wasted work. If `min(H, W) > 320`, the detector first downsamples to a ~256 px thumbnail and asks whether *that* spectrum has any notch peaks. No thumbnail peaks → skip the full FFT and go straight to top-hat. Small tiles skip the probe (the real FFT is cheap).

**Fallback.** Mixed-layout tiles often have **no FFT peaks** above the threshold. An empty notch would leave the preprocessed image unchanged, and DoG would then fire on every pad corner and grain. If the mask is empty, the detector **falls back to white top-hat** so compact bright debris is still extracted.

Area **boundaries** are step edges, not a lattice. They survive both methods and are handled in [§6](#6-region-border-attenuation) and [§5](#5-compact-residual-islands).

### Morphological white top-hat (`method: tophat`, and the FFT fallback)

White top-hat = image − morphological opening. It keeps bright objects **smaller than the structuring element** and treats anything larger as background.

- Kernel: OpenCV ellipse of radius `tophat_radius` (default **50 px**).
- At 0.96 µm/px, radius 50 ≈ 100 µm, so a ~100 px flake still survives.
- If the radius is **smaller than the debris**, the flake is treated as background and **disappears**. That is why a 17 px top-hat erases the large contaminants this pipeline is meant to find.

OpenCV’s `MORPH_TOPHAT` is used instead of `skimage.morphology.white_tophat`: same elliptical shape, tens of milliseconds instead of seconds on a large tile.

### Dark particles

If `particles_bright` is false, the residual is negated after suppression so subsequent steps (which look for bright blobs) still apply.

---

## 4. Local SNR and prominence

The residual is **not** min-max stretched against the brightest edge in the tile. That would make dark-field particles look weak next to a pad rim. Two steps replace a global stretch.

### Local SNR (FFT residuals only)

```
energy   = residual²
envelope = √(Gaussian(energy, σ))
SNR      = residual / max(envelope, MAD floor)
```

then scale a robust high percentile (99.5th) into `[0, 1]`.

- `σ` defaults to `3.5 × blob_max_sigma` (`local_snr_sigma: 0` = auto). It must be several times the particle scale so a blob does not inflate its own envelope.
- Isolated blobs in a **smooth** field have a small envelope → they are boosted.
- Long edges and busy texture have a **large** envelope → they are suppressed.
- The MAD floor (`1.4826 × median absolute deviation`, subsampled on huge tiles) stops division by ~0 in empty regions.

**Top-hat residuals skip auto SNR.** Top-hat is already a local-contrast map. Running auto SNR on it boosts dark-field noise into thousands of DoG hits; the neighbour filter then deletes the real specks because they sit in a cluster of noise. Set `local_snr_sigma` to a positive value to force SNR on top-hat, or to a negative value to disable it even on FFT residuals.

### Prominence floor (`min_prominence`, default 0.30)

After SNR and edge softening, the residual is scaled to `[0, 1]` again. Values below `min_prominence` are **zeroed** before DoG.

Bright compact particles stay. Mesh nodes, grain, and weak ringing do not become candidates. Raise this to drop texture; lower it (toward 0) to pick up dimmer specks. `0` disables the floor.

`config.yaml` currently omits this key; the detector and sidebar still default to **0.30**. NSEW overlay sets it explicitly.

A separate **`min_confidence` (0.50)** gate later requires the residual at the DoG peak to be at least that high. Prominence zeros the map; confidence is the keep/drop on the peak value.

---

## 5. Compact residual islands

Pad-edge softening zeros a wide band around layout ridges, which also erases real flakes that sit *on* a single die step. After the residual is scaled, `_compact_island_protect` finds connected components at the prominence threshold and marks a subset as **protected**.

Keep a component when all of these hold:

1. Equivalent circular diameter ≥ `island_min_nm` (default **20 µm**).
2. Bounding-box aspect ≤ `island_max_aspect` (default **3**).
3. Fill (area / box) ≥ `island_min_solidity` (default **0.35**).
4. Optional morphological open (`island_open_px`, default **4**) so thin ridges do not count as islands.

Then:

- Diameter ≥ `island_large_nm` (default **40 µm**) → **always** protected, even at an L-junction / pad corner.
- Diameter ≥ 20 µm but &lt; 40 µm → protected only if the preprocessed image is **not** an orthogonal (L) junction at the centroid. Pad corners stay unmasked.

Protected pixels:

- Keep at least **0.85** of the edge-soften ramp (the flake is not multiplied down to zero).
- Skip the hard `edge_exclude_px` drop.
- Skip the local-peak SNR test on the preprocessed photo.
- Have their **centroids seeded into DoG** at σ = ECD / (2√2), then overlap-pruned with ordinary DoG peaks. If the residual max is below `blob_threshold`, only these seeds are used.

---

## 6. Region-border attenuation

Pad corners and box rims look like round debris after top-hat / DoG. A narrow dip *on* the gradient ridge just moves the DoG peak **inside** the box, so the pipeline attenuates a **wide band** around every long coarse border.

### Finding the borders (`coarse_edge_distance`)

Computed on the **preprocessed tile**, not the residual (the residual has already lost the lattice and is a poor place to find layout geometry).

1. Heavy Gaussian blur (`edge_soften_sigma`, default **12 px**) so lattice / texture is ignored and only region-scale steps remain.
2. Sobel gradient magnitude.
3. **Hysteresis:** strong pixels (≥ max(12% of the 99.5th percentile, a MAD-based floor)) seed connected components of weaker pixels (≥ 4% of the peak). Weaker pad borders that connect to stronger corners still count; isolated speckle does not.
4. Keep only components that are **long or thinly filled**: long side ≥ `edge_min_length_px` (default **40**) and (aspect ≥ 2 **or** fill &lt; 0.35). Compact gradient rings around real particles are ignored.
5. Morphological close (7×7 ellipse) to rejoin broken segments, then the length filter again.
6. **Skeletonize** the mask so distance is to the border *line*, not a fat gradient band (a fat band would swallow real specks sitting inside a pad).
7. The **tile frame** is forced on (`ridge[0,:]`, last row, first/last column). Truncated pads at the crop edge would otherwise look like particles.
8. Euclidean distance transform to that skeleton.

`edge_soften_sigma: 0` skips all of this (distance is “far”).

### Softening the residual (`soften_structure_edges`)

```
margin = max(edge_exclude_px, edge_soften_sigma, 1)
t      = clip(distance / margin, 0, 1)
ramp   = smoothstep(t) ^ strength     # smoothstep = t² (3 − 2t)
if protect: ramp = max(ramp, 0.85)
residual ← residual × ramp
```

Default `edge_soften_strength: 2` and `edge_exclude_px: 12`. The ramp goes to zero **on** the border and recovers over ~12 px (or 12 px of blur, whichever is larger). Islands keep most of their residual.

Set `edge_soften_strength` to 0 to leave the residual unchanged (the hard distance gate below can still drop blobs).

### Hard exclude

After DoG, any hit whose distance to the skeleton is **≤ `edge_exclude_px`** is dropped, **unless it sits on a protected island**. Pad *corners* look compact and bright after top-hat; they fail the island L-junction test and still drop. Set `edge_exclude_px` to 0 to keep blobs next to borders (you will get pad corners).

---

## 7. Blob detection (difference of Gaussians)

Default `blob_method: dog` uses `blob_dog_fast`, an OpenCV implementation of the same scale convention as `skimage.feature.blob_dog`. (`blob_method: log` calls `skimage.feature.blob_log` instead.)

### Why DoG

A blob of radius *R* is a local maximum of the difference of two Gaussians whose sigmas straddle *R*. Scanning a range of scales finds compact objects without a fixed template. Lattice nodes that survived the notch are usually the wrong scale or the wrong shape and fail later filters.

### Scale convention (scikit-image)

```
radius              ≈ σ × √2
equivalent diameter  = 2 × radius  = 2√2 σ  ≈ 2.828 σ
```

Defaults at **0.96 µm/px**:

| Config | Sigma | Diameter (px) | Physical |
|--------|-------|----------------|----------|
| `blob_min_sigma` | 3.7 | ≈ 10.4 | **10 µm** (search floor) |
| `blob_max_sigma` | 36 | ≈ 102 | **100 µm** |
| `min_size_nm` | — | ≈ 20.8 | **20 µm** (keep floor) |

`tophat_radius: 50` must stay larger than the biggest flake or the flake never reaches DoG.

If you change `min_size_nm` / `max_size_nm`, change the matching `blob_min_sigma` / `blob_max_sigma` (and top-hat radius) or the detector hunts for the wrong scales and the size gate throws the hits away. Do **not** raise `blob_min_sigma` to 7.4 just because the product floor is 20 µm: that flooded layout FPs on R4 and dropped recall below 95%.

### Scale sampling

Sigmas are a geometric series:

```
σᵢ = blob_min_sigma × (blob_sigma_ratio)ⁱ
```

until `σ ≥ blob_max_sigma`. Default `blob_sigma_ratio: 1.4`. If you set `blob_num_sigma` instead (used by LoG, and as a fallback to derive the ratio), the ratio becomes `(max/min)^(1/(n−1))`.

Each pair of adjacent Gaussians is subtracted and multiplied by `1 / (ratio − 1)` so responses are comparable across scales. A pixel is a peak if it is ≥ its 8-neighbours **and** ≥ the dilated previous/next scale **and** ≥ `blob_threshold` (default **0.12** after prominence scaling).

Overlapping peaks: if two circles overlap by more than 0.5 (fraction of the smaller diameter, or full containment), the **smaller-sigma** blob is dropped.

Gaussians for DoG are computed at **full resolution** (`_gaussian_blur_full`) so blob scales match scikit-image. The reduced-resolution blur used for illumination / SNR / edges would shift σ and therefore reported size.

---

## 8. Per-blob filters

Each DoG peak is converted to a candidate only if it passes **all** of the following. Order matches `_gate_blob`.

### Size window and measured diameter

DoG only *finds* the blob. Reported `size` is **not** `2√2 σ`. After the other gates, a window on the **preprocessed photo** (not the FFT/top-hat residual) is split into peak vs local background (median of the outer ring). Pixels that stand out by `size_mass_fraction` of that contrast (default **0.4**) form the 8-connected component that contains the peak. Area *A* becomes equivalent circular diameter:

```
ECD = 2 √(A / π)
```

The DoG diameter `2√2 σ` is still used as a coarse scale gate so the detector does not keep blobs far outside `min_size_nm` … `max_size_nm`. The same window is then applied to the measured ECD, so a 3 px speck that DoG labelled with `blob_min_sigma` is dropped instead of reported as exactly 10.046 µm.

If the connected support is empty, the pipeline falls back to the DoG diameter.

Defaults: **20–100 µm**. At 0.96 µm/px that is about 20.8–104 px. `min_area_px: 4` is an absolute area floor on both the DoG disk and the measured ECD.

### Prominence at the peak

The residual value at the rounded `(y, x)` must still be ≥ `min_prominence`. (The map was already zeroed below that floor; this catches numerical leftovers.)

### Confidence (`min_confidence`, default 0.50)

Same residual value must also be ≥ `min_confidence`. This is the main intensity keep/drop after 0–1 scaling. `0` disables.

### Edge distance

`coarse_edge_distance[y, x] ≤ edge_exclude_px` → drop, unless the pixel is on a **protected island**. Pad corners, box rims, and the tile frame sit here.

### Circularity (`min_circularity`, default 0.0 = off)

Inertia-ratio of the local residual mass in a window of half-width `3 × radius`:

1. Take residual above `0.4 ×` the centre value as mass.
2. Compute the 2×2 covariance of that mass.
3. Circularity = `λ_min / λ_max` of that tensor.

**1** = round, **0** = line-like. Labeled debris is irregular (median circularity ~0.51); letters and fiducials are often *rounder*, so raising this floor keeps layout and drops potato-shaped flakes. Leave it at **0**.

### Local peak on the *preprocessed* tile

Top-hat and flattening can leave residual peaks on empty dark field that are not particles. The centre of a `3 × radius` window on the **preprocessed intensity** (not the residual) must exceed the local median by **2.5 × MAD**. If `particles_bright` is false, the test is inverted (must be a local dark extremum). **Skipped on protected islands.**

### Compact support

Count residual pixels in the same window that are ≥ `0.4 ×` centre. Plateaus and line segments fill most of the window; compact particles do not. Drop if that count exceeds `max_support_area_px`, or if that key is 0 / unset, **`6 ×` the blob’s disk area**. Layout pads that DoG reports at a small sigma fail this.

### Confidence (stored)

The residual intensity at the blob centre (0–1 after scaling and the prominence floor). Used later to pick the winner when overlap tiles (or N/S/E/W views) report the same particle.

---

## 9. Spatial structure filters

Applied to the **set** of survivors on this tile, not to each blob in isolation (`apply_structure_filters`).

### Neighbour / cluster filter (`reject_clustered_candidates`)

Isolated particles have no nearby hits. Region-border beads and repeating layout cells have several.

- Build a KD-tree of candidate centres.
- Drop a blob if it has **`structure_min_neighbors`** or more *other* blobs within `structure_neighbor_px` (defaults: **2** neighbours, **48 px**), **unless** it is a size/SNR outlier versus those neighbours (equivalent diameter ≥ `structure_size_ratio` × the neighbour median, default **1.30×**, or local-peak SNR ≥ **2.5×**). That keeps a ~20 µm flake sitting in a scatter of ~13 µm layout nodes.
- `structure_neighbor_px: 0` disables.

Equal-sized lattice nodes still drop as a group.

### Axis-aligned chain filter (`reject_axis_aligned_chains`)

Leftover “frame just inside the box”: a row or column of blobs along a pad rim that the neighbour radius did not fully catch.

- Bin `y` (then `x`) to `structure_line_bin_px` (default **10 px**).
- If a bin contains ≥ `structure_line_min_run` blobs (default **3**) whose span in the other axis is ≥ `structure_line_min_span_px` (default **48 px**), drop members of that run that are **not** size outliers (≥ 1.30× the run median).
- A run with a hole larger than `structure_line_max_gap_px` (default **48 px**) is **not** treated as a frame. That stops two unrelated flakes that happen to share a 10 px band from killing each other.
- `structure_line_bin_px: 0` disables. `structure_line_max_gap_px: 0` skips the hole check.

---

## 10. Global coordinates and de-duplication

For each surviving candidate (`measure_candidates`):

```
mosaic_x_px = tile.x0 + x_local
mosaic_y_px = tile.y0 + y_local

x_global = mosaic_x_px × pixel_size_nm     # nanometres
y_global = mosaic_y_px × pixel_size_nm
size     = ECD_px × pixel_size_nm          # equivalent circular diameter, nm
```

Unplaced tiles (no row/col in the filename) use origin `(0, 0)`. `source_tile` is the filename. `id` is 0 until the merge step.

### Overlap merge (`deduplicate`)

Overlapping *grid* tiles can report the same particle twice. Merge radius:

```
merge_radius_nm = merge_radius_px × pixel_size_nm
```

Default `merge_radius_px: 8` ≈ **7.7 µm** at 0.96 µm/px.

1. Sort remaining particles by **confidence, descending**.
2. KD-tree: every pair closer than `merge_radius_nm` is a neighbour.
3. Walk from highest confidence. Each unused seed claims all unused neighbours (including itself).
4. Kept position and `source_tile` come from the **seed** (highest confidence).
5. Cluster **size** is `max` or `mean` of members (`size_aggregation`, default **max** — the most complete view of a flake that was clipped on one tile).
6. Assign `id` 1…N in the order kept (highest confidence first).

`merge_radius_px: 0` skips clustering and only renumbers (unless a size-match fraction is set).

### N/S/E/W merge (`deduplicate_nsew`)

Hits whose `source_tile` stem is `N`, `S`, `E`, `W`, or `particles_only…` are merged **separately** with `directional=True`:

- Search radius is `max(merge_radius_px, nsew_merge_radius_px)` (default extra **24 px**) times `pixel_size_nm`.
- Two hits also merge if they are closer than `nsew_size_match_fraction` × the **larger** diameter (default **0.4**). Directional views of the same flake can sit farther apart than 8 px.
- At most **one member per direction / particles-only version**. Two hits on the same `N.bmp` never merge with each other through this path.
- `nsew_count` / `nsew_dirs` record which lighting angles contributed (`N,S,E` etc.).

Placeable grid tiles and unmatched directional files are concatenated after their own merges, then `id` is reassigned 1…N.

---

## 11. Outputs and the mosaic overlay

### CSV

Written to `{output_dir}/particles.csv`. Bare output names such as `Recall audit` become `ASML SE/Outputs/Recall audit`. Columns:

| Column | Unit | Meaning |
|--------|------|---------|
| `id` | — | 1…N after merge |
| `x_global` | nm | Mosaic X from the top-left of the grid (0 for unplaced tiles) |
| `y_global` | nm | Mosaic Y, **down** (image convention) |
| `size` | nm | Equivalent circular diameter from the photo footprint × `pixel_size_nm` |
| `confidence` | 0–1 | Residual at the peak |
| `source_tile` | filename | Tile that supplied the winning (highest-confidence) hit |
| `nsew_count` | count | Distinct N/S/E/W / particles-only members in the merge (0 for grid-only) |
| `nsew_dirs` | text | Those stems, comma-separated |
| `circularity` | 0–1 | Inertia-ratio circularity of the residual mass |
| `support_over_area` | — | Residual support pixels / DoG disk area |
| `edge_distance_px` | px | Distance to the nearest coarse region border |
| `tile_border_dist_px` | px | Distance to the tile frame |
| `n_neighbors_48` | count | Other blobs within 48 px (before structure filters) |
| `local_peak_snr` | — | Peak vs local median, MAD-scaled |
| `radial_inner` / `radial_mid` / `radial_outer` | 0–1 | Mean residual in rings 0–0.5R / 0.5–1R / 1–1.5R |

The UI also shows `size_um` = `size / 1000`. The CSV itself stays in nm. Feature columns are computed from the residual at detection time, not from labeled crop JPEGs.

### Mosaic overlay

Detection never builds a full-resolution stitch. During the detection pass, if the overlay will be downsampled by a factor `f > 1`, each worker also emits an `INTER_AREA` thumbnail (`H/f × W/f`). `LazyMosaic` later pastes those thumbnails onto one canvas. Unplaced tiles do not appear on the stitch.

Downsample factor (`overlay_downsample`):

- `report.downsample > 0` — use that integer.
- else choose `f` so uncompressed RGB ≈ `report.target_mb` (default **20 MB**):

```
f = ceil( √( H × W × 3  /  (target_mb × 1024²) ) )
```

Display:

1. Percentile-stretch the gray mosaic (1st–99.5th) to 8-bit.
2. Convert to RGB.
3. For each particle, convert `x_global, y_global` back to mosaic pixels (`/ pixel_size_nm`), divide by `f`, and draw a red circle. Radius is `size / pixel_size_nm / f / 2`, then padded to at least 14 px so markers stay visible on a 20 MB overview.
4. JPEG quality 90 → `mosaic_overlay.jpg`.

**Inspect a crop** in the app reads only the overlapping full-resolution patches (`crop = (y, x, h, w)` in mosaic pixels), so you can check a detection at native scale without stitching the whole wafer.

Later tiles overwrite earlier ones in overlap; there is no blending. That is fine for an overview. Positions are still from the filename grid, not from the pixels in the overlap.

---

## Scale: pixels, micrometres, nanometres

Typical 5× tiles: **`pixel_size_nm: 960`** → **1 pixel = 0.96 µm**.

Groundup / N/S/E/W images use **`pixel_size_nm: 3500`** (3.5 µm/px) when the sidebar **NSEW** toggle is on.

The UI talks in **µm**. The CSV and internal config talk in **nm**. Sidebar conversions: `nm = µm × 1000`.

Useful conversions at 0.96 µm/px:

| Physical | Pixels | DoG sigma (`d / 2√2`) |
|----------|--------|------------------------|
| 10 µm | 10.4 | 3.7 |
| 15 µm | 15.6 | 5.5 |
| 20 µm | 20.8 | 7.4 |
| 32 µm | 33.3 | 11.8 |
| 50 µm | 52.1 | 18.4 |
| 100 µm | 104.2 | 36.8 |
| merge 8 px | 7.7 µm | — |
| NSEW merge 24 px | 23 µm | — |
| edge exclude 12 px | 11.5 µm | — |
| top-hat radius 50 | objects ≲ 100 px survive | — |

If you change magnification, set `pixel_size_nm` first, then recompute sigmas and size bounds from the physical window you care about.

---

## NSEW directional lighting

Four images of the **same field** named `N`, `S`, `E`, `W` (any common suffix) under a folder. The **NSEW** tab (`src/nsew.py`) does **not** run DoG. It:

1. Flatten each angle with a Gaussian (σ **120**, local to this combine step).
2. For min-count **2, 3, and 4 of 4**, score pixels that are bright in at least that many angles (`symmetry_signal_4way`).
3. Write, at the same pixel size as the inputs:
   - `Combi/combi_{k}of4.*` — isotropic brightness
   - `Particles only/particles_only_{k}of4.*` — Otsu + shape-filtered mask of compact blobs
   - `Symmetry/symmetry_map_{k}of4.*` — symmetry score

Outputs go to `Outputs/Output {folder name}/`. 2-of-4 is also written without the `_2of4` suffix.

Detection of those combined images (and of the four raw angles) uses the main pipeline. The sidebar **NSEW** toggle merges `nsew_config.yaml` on top of `config.yaml`: 3.5 µm/px, denoise σ 0.5, `blob_min_sigma` 9, `blob_threshold` 0.3, ML threshold **0.55**, same v5 model. Off restores `config.yaml`, whose v5 threshold is **0.39**.

---

## Parameter reference

Defaults below are from current `config.yaml` unless noted. The Streamlit sidebar exposes most of them; keys only in YAML are marked *YAML only*. `min_prominence` is a code/sidebar default (not currently in YAML).

### Data / grid

| Key | Default | Role |
|-----|---------|------|
| `input_dir` | (path) | Folder of tiles under `Inputs/` or an absolute path |
| `output_dir` | (path) | CSV + JPEG destination under `Outputs/` |
| `filename_pattern` | `R(?P<run>…)_(?P<row>…)_(?P<col>…)_(?P<mag>…)X\.(tiff?\|bmp\|png\|jpe?g)` | Named groups for placement |
| `run` / `magnification` | empty | Optional filters |
| `overlap_fraction` | 0.0 | Assumed stage overlap (0 = tiles abut) |
| `pixel_size_nm` | 960 | 1 px = 0.96 µm on these 5× tiles |
| `pipeline.workers` | 0 | 0 = all cores; thread pool, 1 OpenCV thread when parallel |

### Preprocessing

| Key | Default | If you raise it | If you lower / disable it |
|-----|---------|-----------------|---------------------------|
| `denoise` | true | — | More speckle blobs |
| `denoise_sigma` | 1 | Softens real particles | Speckle |
| `flatten_illumination` | true | — | Vignetting looks like particles |
| `flatten_sigma` | 160 | Safer around pads | Small values halo region edges |
| `contrast_stretch` | true | — | Tiles with weak contrast miss specks |
| `contrast_percentiles` | [1, 99] *YAML* | Wider stretch, more noise | Narrower, clipped highlights |

### Background suppression

| Key | Default | Notes |
|-----|---------|--------|
| `method` | `fft` | `fft` or `tophat`. FFT falls back to top-hat when there are no lattice peaks. |
| `particles_bright` | true | false = invert residual (dark debris) |
| `fft_peak_threshold` | 0.35 | Lower = notch more (may eat real structure). Higher = leave lattice in. |
| `fft_notch_radius` | 3 *YAML* | Dilate each spectral peak before zeroing. |
| `fft_mask` | `per_tile` | `shared` only if every tile has the same pitch. |
| `tophat_radius` | 50 | **Must exceed the largest particle** (px). 50 ≈ 100 µm flakes at 0.96 µm/px. |

### Blob / size

| Key | Default | Notes |
|-----|---------|--------|
| `blob_method` | `dog` *YAML* | `log` = slower scikit-image LoG |
| `min_size_nm` / `max_size_nm` | 20000 / 100000 | **20–100 µm** keep window. DoG search still starts at σ 3.7. |
| `blob_min_sigma` / `blob_max_sigma` | 3.7 / 36 | Scale window; see table above |
| `blob_num_sigma` | 5 *YAML* | LoG levels; also derives ratio if `blob_sigma_ratio` unset |
| `blob_sigma_ratio` | 1.4 *YAML* | Geometric step between DoG scales |
| `blob_threshold` | 0.12 | DoG peak height after 0–1 scaling. Lower = more candidates. |
| `min_area_px` | 4 *YAML* | Absolute area floor (DoG disk and measured ECD) |
| `size_mass_fraction` | 0.4 *YAML* | Photo contrast vs local background that counts as particle area |
| `min_prominence` | 0.30 (code) | Zero residual below this before DoG. 0 disables. |
| `min_confidence` | 0.50 | Drop if residual-at-peak is below this. |

### Islands / residual / edges

| Key | Default | Notes |
|-----|---------|--------|
| `island_min_nm` | 20000 *YAML* | Smallest island ECD (nm) that can be protected |
| `island_large_nm` | 40000 *YAML* | Always protect at this ECD, even on L-junctions |
| `island_max_aspect` | 3 *YAML* | Drop elongated residual components |
| `island_min_solidity` | 0.35 *YAML* | area / bounding box |
| `island_open_px` | 4 *YAML* | Morphological open before labelling |
| `local_snr_sigma` | 0 | 0 = auto on FFT (`3.5 × blob_max_sigma`), off for top-hat. Negative = off always. |
| `edge_soften_sigma` | 12 | Coarse blur before the region-border gradient. 0 disables distance. |
| `edge_soften_strength` | 2 | How hard to ramp residual to 0 near borders. 0 = no ramp. |
| `edge_exclude_px` | 12 | Hard drop if distance ≤ this (except protected islands). 0 keeps border blobs. |
| `edge_min_length_px` | 40 *YAML* | Ignore compact gradient rings shorter than this. |

### Shape / structure

| Key | Default | Notes |
|-----|---------|--------|
| `min_circularity` | 0.0 | 0 keeps every blob. Do not raise this to reject letters. |
| `max_support_area_px` | 0 *YAML* | 0 → cap at `6 × π r²`. Layout pads exceed this. |
| `structure_neighbor_px` | 48 | Drop blobs with ≥ `structure_min_neighbors` others this close. 0 disables. |
| `structure_min_neighbors` | 2 *YAML* | |
| `structure_size_ratio` | 1.30 *YAML* | Keep a clustered blob if size ≥ this × neighbour median |
| `structure_line_bin_px` | 10 *YAML* | Row/column binning for frame filter. 0 disables. |
| `structure_line_min_run` | 3 *YAML* | |
| `structure_line_min_span_px` | 48 *YAML* | |
| `structure_line_max_gap_px` | 48 *YAML* | Hole larger than this → not a frame |

`detection.recall_mode` and `detection.recall.*` are **unused**. `with_recall_profile()` is a no-op. Edge 12 / confidence 0.50 / circularity 0 are the standard gates.

### ML

| Key | Default | Notes |
|-----|---------|--------|
| `ml.enabled` | true | Score proposals. Raises if the joblib is missing. |
| `ml.model_path` | `models/particle_clf_v5.joblib` | Relative to `particle_detection/`. Versioned; do not overwrite a vN file. |
| `ml.threshold` | 0.39 | Keep blobs with `P(particle)` ≥ this (v5 on the R4 holdout: P = R = 0.878). NSEW overlay uses **0.55**. |
| `ml.score_before_structure` | false | true = ML then cluster/line. false = cluster/line then ML. |
| `ml.band_low` / `band_high` | 0.20 / 0.80 | Fallbacks if a cascade joblib has no stored band. CNN replaces HOG only inside the band. |

### Measurement / report

| Key | Default | Notes |
|-----|---------|--------|
| `merge_radius_px` | 8 | Overlap de-dupe. 0 = no merge (grid tiles). |
| `size_aggregation` | `max` | `max` or `mean` of a merged cluster |
| `nsew_size_match_fraction` | 0.4 *YAML* | Merge directional hits closer than this × larger diameter |
| `nsew_merge_radius_px` | 24 *YAML* | Extra search radius (px) for N/S/E/W pairing |
| `target_mb` | 20 | Uncompressed RGB budget for the stitch |
| `downsample` | 0 | 0 = derive from `target_mb`; `> 0` forces that integer factor |

### Labeling

| Key | Default | Notes |
|-----|---------|--------|
| `labeling.source_csv` | `recall/particles.csv` | Label tab prefers this `particles.csv` under `Outputs/`, then falls back |

---

## Tuning: if the result looks wrong

Work from the **symptom**, not from every slider at once. After a change, re-run; the overlay and CSV are the ground truth.

**Too many hits on the lattice / grain**

- Raise `min_prominence` (0.30 → 0.4–0.5) or `min_confidence` (0.50 → 0.6+).
- Raise `fft_peak_threshold` only if the lattice is *not* being notched (you should see it vanish in a residual debug). More often the FFT already worked and prominence is the issue.
- Raise `blob_threshold`.
- Confirm `method` is `fft` on periodic tiles.
- If ML is on, raise `ml.threshold`.

**Pad corners and box rims circled**

- These are the main false-positive class. Confirm `edge_exclude_px` is at least ~12 and `edge_soften_sigma` > 0. Do not lower `island_large_nm` so far that corners become “large islands”.
- Do **not** raise `min_circularity` — labeled letters/fiducials are as round as (or rounder than) real flakes. Use edge and structure filters, or the ML layout gate.
- Raise `structure_neighbor_px` if a chain of rim beads survives as “isolated” pairs.

**Large flake missing (~50–100 µm)**

- `tophat_radius` must be **larger than the flake** (default 50). A 17 px top-hat erases ~100 px debris.
- `blob_max_sigma` must cover it (36 ≈ 100 µm at 0.96 µm/px).
- `max_size_nm` must not clip it (100000 = 100 µm).
- If it sits on a region border, island protect should keep ≥40 µm flakes; inspect a full-res crop before lowering `edge_exclude_px` (lowering it brings pad corners back).

**Small specks missing (~20 µm)**

- `min_size_nm` is 20 µm. Lower it together with checking `blob_min_sigma` if you truly want ~10 µm.
- Lower `min_prominence` / `min_confidence` / `blob_threshold` a little.
- Check `denoise_sigma` is not much above 1.
- A 20 µm flake on an L-junction is **not** island-protected.

**Thousands of hits on dark field, then real specks disappear**

- Auto SNR ran on a top-hat residual (or prominence is 0). Leave `local_snr_sigma` at 0 so top-hat skips SNR, and keep `min_prominence` around 0.30. The neighbour filter was deleting real particles because they sat in a cloud of noise.

**Same particle listed twice**

- Raise `merge_radius_px`, or check `overlap_fraction` matches the stage. If the grid is wrong, duplicates can sit farther apart than 8 px.
- For N/S/E/W, raise `nsew_merge_radius_px` or `nsew_size_match_fraction`.

**Positions look shifted on the mosaic**

- Stitching is the filename grid + `overlap_fraction`, not registration. Measure actual overlap on two neighbouring tiles and set the fraction to match.

**Run is slow (~minutes)**

- Expected: DoG at full resolution plus an FFT or a 50 px top-hat, once per tile, on all cores. `workers: 0` should already use every core (threads). Crops in the UI re-read images; the overview should use cached thumbnails from the detection pass.

---

## What this does *not* do

- It does **not** classify particle type (metal, resist, scratch, fibre, etc.). The optional model only answers particle vs not-particle.
- It does **not** replace DoG with a neural detector. Changing wafer pitch, magnification, or illumination still means retuning FFT threshold, DoG sigmas, size bounds, and the edge band. Labels are on detector hits, not missed debris.
- Mosaic stitching is **placement by filename grid + overlap**, not feature-based registration. Stage error becomes global-position error.
- It does **not** detect on the stitched mosaic. Overlap is handled only by the KD-tree merge after per-tile detection.
- It does **not** keep pad-corner L-junctions. Compact islands on a *single* edge (≥20 µm) or large flakes (≥40 µm) can survive the border band; corners cannot.
- It does **not** try to split overlapping real particles. DoG overlap pruning keeps the larger scale; the merge step keeps the higher-confidence seed.
- It does **not** train on the circled label JPEGs. The red marker would leak into any crop CNN. Training reads unmarked TIFF/BMP patches and residual channels.
- `detection.recall_mode` does **not** switch a looser profile. That overlay was removed; the current YAML *is* the high-recall-ish standard.

---

## ML cascade (optional)

DoG remains the only proposal generator. After cheap geometric gates (and usually after structure filters), each blob is scored.

`python -m src.ml.train` (default `--mode cascade`):

1. Joins `labels/labels.csv` to detections (GroupKFold by tile).
2. Extracts **unmarked** corrected-image + residual patches (circled crops under `labels/crops/` are ignored).
3. Trains class-weighted ExtraTrees on HOG / layout features.
4. Trains a tiny CNN on the **uncertain HOG band** (default 0.20–0.80).
5. Writes the next `models/particle_clf_v{N+1}.joblib` (never overwrites an existing vN).

`--mode patch` skips the CNN. `--mode residual` is the old detector-float ExtraTrees on CSV columns only.

At inference (`apply_ml_filter`):

- Residual models score the ~12-float vector (`size_um`, `confidence`, circularity, support, edges, neighbours, local-peak SNR, radial rings).
- Patch / cascade models rebuild unmarked patches on the **corrected tile** in the worker.
- Cascade: keep HOG `P` outside the band; replace with CNN `P` inside it. Band stored in the joblib wins over `ml.band_low` / `band_high`.
- Drop blobs with `P(particle) < ml.threshold`.

Scoring a few dozen vectors (or patches) per tile is small next to DoG. `ml.enabled: true` with a missing joblib **raises**; it will not silently run classical-only.

v5 at threshold **0.39** is the R4 holdout operating point (36 real, 5 fake, 5 missed; P = R = 0.878). NSEW uses a higher threshold (**0.55**) from the 2-of-4 sweep.

Typical loop:

1. Run Detection (structure filters on, ML on or off).
2. **Tinder** tab: label new hits; existing keys (including N/S/E/W aliases) are skipped.
3. Train a new vN. Point `ml.model_path` at it and set `ml.threshold`.

---

## How to run

```bash
cd particle_detection
streamlit run app.py
```

Set the tile folder in the sidebar (or `input_dir` in `config.yaml`) and click **Run pipeline**. **Reset to standard values** restores every sidebar widget from `config.yaml` and turns the **NSEW** toggle off. Turning that toggle on overlays `nsew_config.yaml`.

| Tab | Use |
|-----|-----|
| Detection | Run the pipeline, table, histogram, mosaic crop |
| NSEW | Combine N/S/E/W lighting into combi / particles-only / symmetry |
| Tiles | Walk one image at a time; zoom; mark a missed speck |
| Last Run | Review the last `particles.csv` against labels |
| Tinder | Label queue from detections |
| Database | Saved particle / not-particle crops |

Tests live under `tests/` (`pytest`). Synthetic fixtures in `tests/fixtures/synthetic.py` cover lattice tiles, dark-field flakes, and pad corners so the filters have regression locks.
