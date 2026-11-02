#!/usr/bin/env bash
# Argon Stacker — Standalone Environment Bootstrap
# Run this script if copying this plugin to a new Mac without Python dependencies installed.
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PLUGIN_DIR="$( dirname "$DIR" )"
VENV_DIR="$DIR/venv"

echo "=========================================================="
echo "Argon Stacker — Standalone Environment Installer"
echo "=========================================================="

PYTHON_BIN=""
for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v "$candidate")"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: Python 3 was not found on your system."
    echo "Please install Python 3 (e.g. brew install python) and run this script again."
    exit 1
fi

echo "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version))"
echo "Target environment: $VENV_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment inside plugin bundle..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Installing required packages (numpy, scipy, opencv, pillow, tifffile, pywebview)..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install numpy scipy opencv-python-headless pillow tifffile pywebview

echo "=========================================================="
echo "✅ Argon Stacker is packaged and ready to run!"
echo "=========================================================="
