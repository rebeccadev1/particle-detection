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

You need 64-bit Python 3.11 or newer (3.13 and 3.14 are fine). The camera package needs macOS 14 or newer.

**Mac.** From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r getting_started/requirements.txt
python -m pip install --no-deps "Printrun==2.2.0"
```

**Windows.** From this folder:

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r getting_started\requirements.txt
python -m pip install --no-deps "Printrun==2.2.0"
```

Printrun is how Capture talks to the printer stage, the same library Pronterface uses. `--no-deps` keeps Pillow 10. A normal Printrun install on Windows asks for Pillow 9, and the summary PDF needs Pillow 10. If Pronterface is already installed with pipx, Capture finds that copy and this line can be skipped.

On Windows, install the Basler pylon Camera Software Suite so the USB camera driver is present, and the WCH CH340 driver when Device Manager lists the Arduino or the printer as an unknown USB serial device. `pip` covers the Python packages. The drivers are separate installers.

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

Capture saves photos in an `Inputs` folder beside this project, and Analyse writes to `Outputs` beside it. A folder named `particle-detection` uses `../Inputs` and `../Outputs`.

## Starting the app

From now onwards, you can also double-click **getting_started/Particle Detection.app** (Mac) or **getting_started/Start Particle Detection.bat** (Windows). The first Mac launch may need Control-click → Open. A second launch only opens the browser if the app is already running. Quit the Mac app from the Dock, or close the Windows console, to stop it.

Congrats! You've read the whole README!
