# Particle Detection (TL;DR) 

This app finds particles on darkfield images of a patterned wafer. It photographs the wafer and runs detection. How the detector actually works is all written down in [getting_started/HOW_IT_WORKS.md](getting_started/HOW_IT_WORKS.md), if you're interested :).

## Get the folder

The app is in this GitHub folder. Either clone it or download the ZIP. Both give you the same files, but cloning is easier if you want updates later with `git pull`.

**Clone**

```bash
git clone https://github.com/rebeccadev1/particle-detection.git
cd particle-detection
```

**Download ZIP**

On this GitHub page, click **Code**, then **Download ZIP**. Unzip it. The folder is named `particle-detection-main`. It should contain `app.py` and `getting_started/requirements.txt`.

**Mac:** press Command-Space, type `Terminal`, and press Return. Type `cd ` with a space after it, then drag the `particle-detection-main` folder from Finder onto the Terminal window. Press Return.

**Windows:** open `particle-detection-main` in File Explorer. Click the address bar, type `cmd`, and press Enter. That opens Command Prompt already inside that folder.

## Install

You need Python 3.11 or newer (3.13 is fine).

On Mac: From this folder in your terminal, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r getting_started/requirements.txt
```

On Windows: Activate with `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

Tests need the extra packages in `getting_started/requirements-dev.txt`:

```bash
pip install -r getting_started/requirements-dev.txt
pytest
```

## Run

With the virtual environment active, from this folder:

```bash
streamlit run app.py
```

The browser opens on **Particle Detection** at <http://127.0.0.1:8502>. Leave the terminal running. Stop the app with Ctrl+C.

## Starting the app

From now onwards, you can also double-click **getting_started/Particle Detection.app** (Mac) or **getting_started/Start Particle Detection.bat** (Windows). The first Mac launch may need Control-click → Open. A second launch only opens the browser if the app is already running. Quit the Mac app from the Dock, or close the Windows console, to stop it.

Congrats! You've read the whole README!
