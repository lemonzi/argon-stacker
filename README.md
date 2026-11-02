# Argon Stacker (`argon-stacker`)

A high-performance, open-source 16-bit focus stacking engine and **Adobe Lightroom Classic** plugin designed for extreme macro, landscape, and commercial product photography.

Co-developed in collaboration with **Argon**, **Argon Stacker** was built from the ground up to solve the critical limitations of legacy focus stacking software: 8-bit color truncation, posterization banding in deep RAW shadows, and boundary halo artifacts around fine macro structures. It pairs a high-precision **Edge-Guided Multi-Scale Decomposition (EG-MSD)** algorithm with an **interactive local web studio**, real-time multi-band retouching, and seamless **1-click Lightroom Classic workflow integration**.

---

## The Story Behind Argon Stacker

Macro focus stacking at 61 megapixels (such as on the Sony A7C R) requires uncompromising tonal precision. Standard tools often convert 16-bit camera RAW exports into 8-bit integers (`CV_8U`) on ingest, discard wide-gamut ICC profiles (`ProPhotoRGB`), and produce severe chromatic halos around delicate details like insect bristles, flower stamens, and fine mechanical edges.

Argon helped design, benchmark, and optimize a modern mathematical pipeline that solves these challenges natively on Apple Silicon and modern architectures:

1. **Uncompromised 16-Bit Dynamic Range**: Internal floating-point processing (`float32`) preserves all 65,536 tonal levels per channel and retains **Tag 34675 (ICC Profile)** so ProPhotoRGB colors match Lightroom's RAW develop module 1:1.
2. **EG-MSD Halo Suppression**: Instead of naive max-contrast selection, Argon Stacker models the physical Circle of Confusion (CoC) and depth discontinuity, completely suppressing out-of-focus background light bleeding around fine edges.
3. **Streaming Memory Architecture**: Memory-efficient 2-pass streaming keeps peak RAM usage below **2.5 GB**, allowing photographers to stack 30+ 61MP RAW frames on standard laptops without memory pressure or crashes.
4. **Interactive Web Retouching Studio**: A lightweight, zero-dependency browser-based interface offering 100% loupe inspection, focus peaking, and an interactive multi-band Laplacian brush to paint in details from individual source frames before rendering the master TIFF.

---

## Key Highlights

- **True 16-Bit Color Depth**: Full `uint16` / `float32` pipeline. Zero color posterization, banding, or color space shifts.
- **EG-MSD Engine**: Edge-Guided Multi-Scale Decomposition with adaptive polarity (`auto`, `bright`, `dark`, `energy`) for pristine edge isolation and natural bokeh transitions.
- **Low Memory Footprint**: 2-pass disk-backed streaming pipeline executes large focus sweeps ($9504 \times 6336$) in $< 2.5\text{ GB}$ RAM.
- **Interactive Web App**: Real-time 100% zoom crop loupe, focus peaking overlays, and instant preview parameter tweaking at `http://127.0.0.1:8765`.
- **Integrated Retouching Brush**: Multi-band Laplacian pyramid blending allows seamlessly painting sharp details from specific bracket frames with full undo/redo.
- **1-Click Lightroom Classic Plugin**: Select bracketed frames in Lightroom Classic, right-click $\rightarrow$ **Export** $\rightarrow$ **Argon Stacker**. The stacked master image is automatically imported and stacked directly above source frames in your catalog.
- **Standalone CLI & macOS Launcher**: Run headlessly from the terminal, integrate into automated scripts, or double-click `FocusStack.command`.

---

## Quick Start (Installation)

### 1. Requirements
* macOS 12+ (Apple Silicon M1/M2/M3/M4 or Intel)
* Python 3.10 or higher
* Adobe Lightroom Classic (Optional, for plugin workflow)

### 2. Environment Setup (1 Step)
In your terminal, run the setup script from the repository root:
```bash
./setup.sh
```
This automatically provisions an isolated virtual environment (`numpy`, `scipy`, `opencv-python-headless`, `pillow`, `tifffile`, `pywebview`) inside the plugin bundle.

### 3. Add Plugin to Lightroom Classic
1. Open **Adobe Lightroom Classic**.
2. Go to **File $\rightarrow$ Plug-in Manager** (`Opt + Cmd + ,`).
3. Click **Add** in the bottom-left corner and select the `FocusStack.lrplugin` folder.
4. The status indicator will turn green: **Installed and Running**.

---

## How to Use

### Method A: Lightroom 1-Click Context Menu (Recommended)
1. In the Lightroom **Library** grid, select your bracketed photos.
2. **Right-click** on any selected frame $\rightarrow$ **Export** $\rightarrow$ **Argon Stacker**.
3. Lightroom exports 16-bit ProPhotoRGB TIFFs in the background, launches the stacking engine, and imports the finished `_stacked.tif` master image right above your originals in the grid.

### Method B: Via Export Dialog (To Customize Settings)
1. Select your bracketed photos and press `Shift + Cmd + E` (or **File $\rightarrow$ Export...**).
2. In the **Export To:** dropdown at the top, select **Argon Stacker**.
3. Adjust settings:
   - **Interactive Preview Mode**: Launches the web GUI to review alignment and adjust parameters before rendering.
   - **Background Polarity Mode**: `Auto`, `Dark` (for bright subjects on dark backgrounds), `Bright` (for dark subjects on high-key backgrounds), or `Energy`.
   - **Subpixel Alignment**: Automatically aligns frames and compensates for lens focus breathing.
4. Click **Export**.

### Method C: Standalone CLI / macOS Double-Click
You can stack images directly from the command line:
```bash
# Basic stack
./FocusStack.command --output final_stack.tif image1.jpg image2.jpg image3.jpg

# Interactive Web Studio
./FocusStack.command --interactive --output final_stack.tif /path/to/photos/*.tif
```

---

## Verification & Testing

Argon Stacker includes an automated test suite verifying halo suppression, 2-pass memory streaming, and TIFF metadata integrity:

```bash
./run_tests.sh
```

All 8 core tests and 9 algorithmic QA acceptance benchmarks run and report status in ~7 seconds.

---

## Project Structure

```
argon-stacker/
├── FocusStack.lrplugin/         # Lightroom Classic Plugin Bundle
│   ├── Info.lua                 # Plugin metadata & version manifest
│   ├── FocusStackExportServiceProvider.lua # Export session & settings
│   ├── FocusStackMenuAction.lua # Context menu hook
│   ├── RegisterExistingPhoto.lua# 1-click catalog ingestion tool
│   ├── Export Presets/          # Lightroom export template
│   └── bin/
│       ├── focus_stack          # Shell launcher
│       ├── setup_venv.sh        # Dependency installer
│       └── py_focus_stack/      # Core Python stacking package
├── tests/                       # Automated test suite
│   ├── test_egmsd_qa.py         # 9-point QA acceptance tests
│   ├── test_streaming_pipeline.py # 2-pass streaming & retouch tests
│   └── test_tiff_io.py          # 16-bit TIFF & ICC roundtrip tests
├── requirements.txt             # Core Python dependencies
├── requirements-dev.txt         # Development & testing dependencies
├── run_tests.sh                 # Unit test runner
├── setup.sh                     # Quick environment setup script
├── DEVELOPMENT.md               # Developer instructions & architecture guide
└── README.md                    # Project overview & documentation
```

---

## Contributing & Developer Documentation

For technical details on the algorithmic pipeline, multi-band retouching protocol, and development guidelines, see [DEVELOPMENT.md](DEVELOPMENT.md).

---

## Acknowledgments & Credits

- Co-developed with **Argon**.
- Built upon principles from Burt & Adelson (Laplacian Pyramid Fusion), Mertens et al. (Exposure & Contrast Fusion), and Petteri Aimonen's open-source focus-stack research.

---

## License

Argon Stacker is open-source software licensed under the MIT License.
