#!/usr/bin/env bash
# Test Runner for Focus Stacker
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Locate Python
PYTHON_BIN=""
if [ -x "${DIR}/FocusStack.lrplugin/bin/venv/bin/python3" ]; then
    PYTHON_BIN="${DIR}/FocusStack.lrplugin/bin/venv/bin/python3"
elif [ -x "${DIR}/.venv/bin/python3" ]; then
    PYTHON_BIN="${DIR}/.venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: Python 3 not found." >&2
    exit 1
fi

echo "Running Focus Stacker Test Suite using $PYTHON_BIN..."
exec "$PYTHON_BIN" -m unittest discover -s tests -v "$@"
