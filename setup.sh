#!/usr/bin/env bash
# Argon Stacker — Developer & Environment Setup
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Setting up virtual environment inside FocusStack.lrplugin/bin/venv..."
exec "${DIR}/FocusStack.lrplugin/bin/setup_venv.sh" "$@"
