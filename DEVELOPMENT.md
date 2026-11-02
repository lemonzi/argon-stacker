# Developer Instructions & Architecture Guide

Welcome to the **Argon Stacker** development guide. This document provides technical instructions for developers, contributors, and maintainers working on the focus stacking engine and Adobe Lightroom Classic plugin.

---

## Architecture Overview

Argon Stacker is designed with a decoupled 3-tier architecture:

```
+--------------------------------------------------------------+
|               Adobe Lightroom Classic (macOS)                |
|  • FocusStackExportServiceProvider.lua (Export dialog hook)  |
|  • FocusStackMenuAction.lua (1-click right-click context)    |
|  • RegisterExistingPhoto.lua (Standalone catalog ingestion)  |
+------------------------------+-------------------------------+
                               | JSON / CLI Subprocess
                               v
+--------------------------------------------------------------+
|               Orchestration & UI Tier (Python)               |
|  • cli.py (CLI argument parser, progress reporter, dispatch) |
|  • server_ui.py (Local HTTP/WebSocket web app, HTML5 Canvas) |
|  • FocusStack.command (macOS double-clickable launcher)      |
+------------------------------+-------------------------------+
                               | NumPy / SciPy / OpenCV
                               v
+--------------------------------------------------------------+
|            Algorithmic Core (16-Bit float32 / SIMD)          |
|  • egmsd_stacker.py (Edge-Guided Multi-Scale Decomposition)   |
|  • core_fusion.py (Streaming 2-pass Laplacian accumulation)  |
|  • core_align.py (Multi-scale ECC subpixel affine alignment) |
|  • tiff_io.py (16-bit TIFF read/write + Tag 34675 ICC profile)|
+--------------------------------------------------------------+
```

### Key Architectural Concepts

1. **True 16-Bit Color Pipeline (`tiff_io.py`)**:
   - Reads inputs into uncompressed 16-bit integer arrays (`uint16`, 0–65535).
   - Extracts and preserves **Tag 34675** (`ICC_Profile`) so wide-gamut spaces like `ProPhotoRGB` or `AdobeRGB` survive the roundtrip without color clipping or white balance distortion.
   - Enforces TIFF 6.0 compliance with 300 DPI tags and clean metadata headers (avoiding duplicate tag 270 entries).

2. **Edge-Guided Multi-Scale Decomposition (EG-MSD) (`egmsd_stacker.py`)**:
   - Solves the classic focus stacking "halo problem" (where blurred foreground edges bleed out-of-focus background light into the composite).
   - Computes robust focus energy volumes using high-frequency gradients.
   - Performs bilateral / depth-consistent filtering to segment foreground occlusion masks from background bokeh.
   - Supports polarities: `auto`, `bright` (dark subject on high-key background), `dark` (bright subject on dark background), and `energy` (unconstrained).

3. **Memory-Efficient 2-Pass Streaming Engine (`core_fusion.py`)**:
   - **Pass 1 (Alignment & Depth Estimation)**: Frames are read sequentially at downscaled resolution to compute affine transformation matrices and preliminary depth maps. Peak RAM $< 150\text{ MB}$.
   - **Pass 2 (Pyramid Accumulation)**: High-resolution frames are loaded one-by-one, warped, decomposed into multi-octave Laplacian pyramids, and accumulated directly into streaming numerator and denominator buffers.
   - Enables stacking 30+ 61MP ($9504 \times 6336$) RAW frames with $< 2.5\text{ GB}$ peak memory.

4. **Interactive Web UI & Retouching Engine (`server_ui.py`)**:
   - Built-in local HTTP server (`http://127.0.0.1:8765`) serving a zero-dependency dark-mode web application.
   - Features 100% zoom loupe, focus peaking overlay, real-time ROI preview, and an interactive retouching brush.
   - Retouching brush encodes overrides via an `A=255` Canvas protocol to prevent HTML5 premultiplied-alpha color degradation, followed by server-side multi-band Laplacian blending.

---

## Developer Setup

### Prerequisites
- macOS (Apple Silicon M1/M2/M3/M4 or Intel)
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### 1. Clone the Repository
```bash
git clone <repository_url>
cd quick-hypatia
```

### 2. Set Up the Python Environment
You can automatically create and populate the plugin's internal virtual environment:
```bash
./setup.sh
```
Or set up your own virtualenv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

---

## Running Unit Tests

The test suite is located in `tests/` and includes end-to-end algorithmic verification, edge-case QA suites, and image I/O regression tests.

### Run All Tests via Runner
```bash
./run_tests.sh
```

### Run via Standard Python unittest
```bash
python3 -m unittest discover -s tests -v
```

### Test Suite Breakdown

| Test File | Description | Key Assertions |
|-----------|-------------|----------------|
| `test_egmsd_qa.py` | 9-point algorithmic QA acceptance test suite | Boundary halo MAE $< 0.035$, bristle detail MAE $< 0.065$, noise change $\le +0.5\text{ dB}$, specular bokeh max error $< 0.080$, tile seam continuity $\Delta = 0$, moat MAE $< 0.015$ |
| `test_streaming_pipeline.py` | Memory streaming and interactive retouching verification | Validates 2-pass streaming, `far_to_near` vs `near_to_far` depth ordering, 4-level Laplacian retouch blend, Canvas $A=255$ encoding |
| `test_tiff_io.py` | Image I/O and color space validation | 16-bit uint16 roundtrip, Tag 34675 ICC profile persistence, 300 DPI metadata compliance, 8-bit to 16-bit normalization |

---

## Lightroom Plugin Development

The Lightroom plugin bundle lives in `FocusStack.lrplugin/`.

### Directory Layout
```
FocusStack.lrplugin/
├── Info.lua                           # Plugin manifest & version declaration
├── FocusStackExportServiceProvider.lua # Export session & settings dialog
├── FocusStackMenuAction.lua           # 1-click right-click context menu action
├── RegisterExistingPhoto.lua          # Menu item to ingest existing stacked TIFFs
├── Export Presets/
│   └── Argon Stacker.lrtemplate       # Default export preset (16-bit ProPhotoRGB)
├── FocusStack.command                 # Double-clickable macOS launcher
└── bin/
    ├── focus_stack                    # Standalone shell wrapper
    ├── fix_orientation.py             # EXIF orientation normalizer
    ├── setup_venv.sh                  # Bootstrap script for plugin's venv
    └── py_focus_stack/                # Core Python package
        ├── cli.py
        ├── core_align.py
        ├── core_fusion.py
        ├── egmsd_stacker.py
        ├── server_ui.py
        └── tiff_io.py
```

### Reloading the Plugin in Lightroom Classic
1. In Lightroom Classic, open **File $\rightarrow$ Plug-in Manager** (`Opt + Cmd + ,`).
2. Select **Argon Stacker**.
3. Click **Reload Plug-in**.

### Debugging Logs
Plugin diagnostic events and catalog transactions are logged to:
```
/tmp/focus_stack_import.log
```

---

## Standalone CLI Usage

You can run the engine directly from the command line without Lightroom:

```bash
# Basic stack
./FocusStack.command --output result.tif frame1.tif frame2.tif frame3.tif

# Interactive UI mode with automatic browser launch
./FocusStack.command --interactive --output result.tif /path/to/frames/*.tif

# Explicit background mode for challenging bokeh
./FocusStack.command --bg-mode=dark --output result.tif *.tif
```

### Command-Line Arguments
- `--output <path>`: Destination path for stacked 16-bit TIFF or JPEG.
- `--interactive`: Launches the local web GUI on `http://127.0.0.1:8765`.
- `--bg-mode <auto|bright|dark|energy>`: Background polarity suppression mode.
- `--crop`: Automatically crops image to the valid bounding box overlap across all frames.
- `--align / --no-align`: Toggle subpixel ECC affine registration (default: on).
- `--lightroom`: Enables Lightroom-friendly progress reporting and lifecycle handling.
