#!/usr/bin/env bash
# Argon Stacker — 1-Click macOS Double-Clickable Launcher
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"

if [ -f "${DIR}/bin/focus_stack" ]; then
    exec "${DIR}/bin/focus_stack" "$@"
elif [ -f "${DIR}/FocusStack.lrplugin/bin/focus_stack" ]; then
    exec "${DIR}/FocusStack.lrplugin/bin/focus_stack" "$@"
else
    echo "Error: Could not locate focus_stack binary in ${DIR}" >&2
    exit 1
fi
