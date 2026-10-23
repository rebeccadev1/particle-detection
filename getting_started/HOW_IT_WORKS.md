# Particle Detection

This app photographs a patterned wafer and finds contaminant particles on those images. The surface is periodic, so detection has to ignore the repeating layout and keep compact debris.

Open it with `streamlit run app.py`. The sidebar has two modules: **Capture** and **Analyse**. The app opens on Capture. Leaving Capture stops the live camera view. The app listens on port 8502.

| Module | What it is for |
|--------|----------------|
| Capture | Ring light, camera, stage, focus, and a grid scan |
| Analyse | Run detection, then read the table, histogram, and mosaic |

Images land in `ASML SE/Inputs`. Detection writes to `ASML SE/Outputs`.

---

## Capture

Capture drives three devices, shown on one status line: the **Arduino** (ring light), the **printer** (the stage), and the **camera**. The Arduino and the printer need different serial ports.

### Lighting

**Uniform** turns on all four sides of the ring. **NSEW** lights one side at a time: N, then E, then S, then W (Arduino pins 2–5, commands `0`–`3`). Automatic NSEW is one click that saves all four frames, with a short gap between sides. Manual NSEW lights and saves one side when you press that side.

**Total exposure** is the camera shutter, in milliseconds. The standard is 100 ms for Uniform and 800 ms for NSEW, eight times the uniform standard. Uniform has all four sides on for that shutter. NSEW lights one side at a time, and each of those frames uses its own total exposure.

### Live view and a single frame

The left column is the live camera. The right column is the last saved capture. A Uniform **Capture** writes one frame. NSEW writes one frame per side.

### Stage, focus, and scan

The stage is a 3D-printer bed. You can jog it, then set the scan: field of view in millimetres, overlap fraction, columns, rows, and travel speed. The step size is the field of view times `(1 − overlap)`. The path is a serpentine: odd rows go left to right, even rows start at the right edge and walk back. The G-code for the current settings can be downloaded as `scanwafer.gcode`.

**Focus** hill-climbs Z from the current height using the live image. **Check and correct focus during the scan** scores the frame after every seventh stage step and runs autofocus. If the best score stays at or below 80% of the sharpness at the start of that search, autofocus runs again. Leave the checkbox off to keep the current Z for the whole scan.

**Scan and capture** moves through the grid and saves a picture at each stop, using the lighting mode and total exposure already chosen. Stop ends the scan. Sharpness at each stop, and each autofocus move, is written to `sharpness.json`.

### Where the pictures go

Each capture and each scan gets a new folder:

`ASML SE/Inputs/Groundup/vN YYYY-MM-DD`

`N` counts runs already saved that day, starting at 1. The folder is created only after the camera has accepted setup. `capture.json` in that folder records every stop.

| Mode | Filename |
|------|----------|
| Uniform | `(row,col).png` |
| NSEW | `(row,col)_N.png`, `_E.png`, `_S.png`, `_W.png` |

Analyse places those names on a grid. The overlap you set here should match `overlap_fraction` there, or global positions drift.

---

## Analyse

### What you do

**Adjust parameters** holds the detector settings. The profile is **Uniform** or **NSEW**. Uniform loads `config/config.yaml`. NSEW overlays `config/nsew_config.yaml` on top of it. **Reset to standard values** restores the selected profile and keeps the folder you just chose.

**Run/stop pipeline** detects on the input folder and writes the output folder. When it finishes, the page shows the particle table, a size histogram in micrometres, the mosaic, and any image that was detected but could not be placed on the grid.

Set the tile folder to a Capture run (`Inputs/Groundup/vN …`) or to any other folder of tiles. A bare folder name is resolved under `ASML SE/Inputs`.

### What a run writes

Bare output names such as `Recall audit` become `ASML SE/Outputs/Recall audit`.

| File | Content |
|------|---------|
| `particles.csv` | One row per merged particle: `id`, `x_global_um`, `y_global_um`, `size_um`, `source_tile`, `confidence` |
| `particles.xlsx` | The same six columns, plus a sheet of every setting used |
| `last_pipeline.json` | Those settings, and the path of the CSV |
| `mosaic_overlay.png` | Downsampled stitch of every placeable tile, orange circle on each detection |
| `mosaic_plain.png` | The same stitch with no circles |
| `Summary of {input folder}.pdf` | Counts, size histogram, marked mosaic above the plain one, and any image that was detected but not placed |

Positions in the files are micrometres, rounded to 0.1. The in-memory table still has nanometre coordinates and the residual features the optional model uses. Those features are not written to the CSV. An old `mosaic_overlay.jpg` in the folder is deleted.

Circle radius is the measured radius in display pixels, plus 10, and at least 14. Gray levels are the saved photo, with no extra contrast stretch. Later tiles overwrite earlier ones in the overlap. There is no blending.

### How a hit is found

Detection runs **one tile at a time**, on threads (`pipeline.workers`, **10** in `config/config.yaml`; `0` means every core). The mosaic is built afterwards from thumbnails. It is a report, not an input.

Before detection, a complete N/S/E/W set is combined and the four source images are left out:

- Files named `N`, `S`, `E`, `W` (or Groundup `quad0`–`quad3`) become `combi_2of4` beside them.
- Files named `(row,col)_N` and the other three sides become `(row,col)`, which is placed on the grid.

The combi keeps pixels that are bright in at least two of the four angles, after a wide illumination flatten (σ 120, only in this combine step). `python src/analyse/nsew_to_combi.py` is separate: it writes 2-of-4, 3-of-4, and 4-of-4 combi and symmetry maps under `Outputs/Output {folder}`.

On each image the worker:

1. Loads it as gray. The raw gray is kept for the size measurement.
2. Denoises (σ 1 px), flattens slow illumination (σ 160 px), and stretches contrast.
3. Removes a periodic lattice with an FFT notch built on **this** tile. If the spectrum has no lattice peaks, it uses a white top-hat instead. Area borders survive both and are handled next.
4. Boosts compact blobs relative to long edges (local SNR, FFT residuals only), then softens a band around long region borders. A compact residual island on a single edge (≥ 20 µm) or a large flake (≥ 40 µm) is protected so a real particle on a die step is not erased. Pad-corner L-junctions are not protected.
5. Finds blobs with difference-of-Gaussians (LoG if `blob_method` is `log`).
6. Drops peaks that fail, in order: residual prominence, confidence, distance to a border, circularity, a local peak on the preprocessed tile, compact support, then measured size.
7. Drops clusters and long axis-aligned rows of equal-sized hits. A blob much larger than its neighbours is kept.
8. Optionally scores the survivors with a frozen ExtraTrees model. Uniform uses residual features (`src/analyse/models/particle_clf_v4_residual.joblib`, threshold **0.23**). The NSEW profile uses `src/analyse/models/particle_clf_v8.joblib` at **0.37**. A missing joblib raises. The app does not train.

Reported **size** is the equivalent circular diameter of the footprint on the **raw photo**, not the DoG sigma. Pixels that stand out from the local background by `size_mass_fraction` (0.4) of the peak contrast form that footprint. The keep window is **20–100 µm**.

DoG itself searches pixel scales σ **3.7–36** (about 10–100 px across). Those sigmas were chosen when 1 px was 0.96 µm. At the current `pixel_size_nm: 3610` (3.61 µm/px) the same sigmas search about **38–368 µm**, so the 20 µm floor sits below the search and the 100 µm ceiling still cuts large DoG hits.

Placeable names match `filename_pattern`. Current `config/config.yaml` is `(row,col)` plus an image suffix. The mosaic origin is the smallest row and column in the folder, stepped by tile size times `(1 − overlap_fraction)`. Current overlap is **0.1**. Files that do not match are still detected. They are omitted from the stitch and shown on their own PDF pages. Overlapping grid tiles that report the same particle are merged (8 px, about 29 µm at 3.61 µm/px). The highest-confidence hit keeps the position. Cluster size is the max of the members.

The two profiles differ in a few gates. Everything else is shared.

| | Uniform (`config/config.yaml`) | NSEW (`config/nsew_config.yaml`) |
|--|-------------------------|---------------------------|
| Model | v4 residual at 0.23 | v8 at 0.37 |
| Border drop | 8 px | 12 px |
| Cluster rule | drop at 3 neighbours within 48 px | drop at 2 |
| Top-hat radius | 50 px | 25 px |
| Contrast stretch | 1st–99th | 0.1–99.9th |

Every other knob is in those YAML files and in the sidebar. `ml.score_before_structure` is **false**: cluster and line filters run, then the model. Turning it on scores first, then filters, and can delete an isolated flake because a lattice neighbour survived.

### If the overlay looks wrong

- Lattice or grain circled: raise `min_prominence` (0.30) or `min_confidence` (0.50), or the model threshold.
- Pad corners circled: keep the border drop on (8 px Uniform, 12 px NSEW). Do not raise circularity. Labeled letters are as round as real flakes.
- A large flake missing: the top-hat radius must be larger than the flake in pixels, and `max_size_nm` must be at least 100000.
- The same particle twice: raise `merge_radius_px`, or set `overlap_fraction` to the overlap Capture actually used.
- Positions shifted: placement is the filename grid plus overlap, not image registration.

---

## How to run

```bash
cd particle-detection
streamlit run app.py
```

Python 3.11 or newer. Setup is in [README.md](../README.md). The app uses port 8502. On a Mac you can also open **getting_started/Particle Detection.app**. On Windows, **getting_started/Start Particle Detection.bat**.

| Module | Use |
|--------|-----|
| Capture | Lights, live view, stage, focus, scan |
| Analyse | Settings, run, table, histogram, mosaic |

From this folder:

```bash
python src/analyse/nsew_to_combi.py [folder]
```

Tests are `pytest` after `pip install -r getting_started/requirements-dev.txt`.

---

## Where the files are

This is the GitHub repository. Photos and detection runs are created on your machine and are not in the clone.

```
particle-detection/
├── app.py                               Streamlit app: Capture and Analyse
├── README.md                            install and launch
├── getting_started/
│   ├── HOW_IT_WORKS.md                  this document
│   ├── requirements.txt                 packages to run the app
│   ├── requirements-dev.txt             packages for the tests
│   ├── Start Particle Detection.bat     Windows launcher
│   └── Particle Detection.app/          Mac launcher
├── config/
│   ├── config.yaml                      Analyse → Uniform settings
│   └── nsew_config.yaml                 Analyse → NSEW settings, overlaid on config.yaml
├── .streamlit/config.toml               app theme and port 8502
├── src/
│   ├── capture/                         ring light, camera, stage, focus, scan
│   ├── analyse/                         detect, measure, combine NSEW, write the report
│   │   └── models/                      frozen classifiers (v3 through v8)
│   ├── verification/                    reload the last Analyse run
│   └── overall/                         settings, filenames, placement, CSV and Excel
└── tests/                               pytest, including synthetic wafer tiles
```
