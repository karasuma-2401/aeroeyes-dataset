#!/usr/bin/env bash
set -euo pipefail

trap 'status=$?; echo "SDQM setup failed (exit=$status): $BASH_COMMAND" >&2; exit "$status"' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDQM_DIR="${SDQM_REPO_DIR:-$ROOT_DIR/third_party/SDQM}"
ULTRALYTICS_DIR="$SDQM_DIR/dataset_interpretability/v_info/ultralytics"
PYTHON="${PYTHON:-python}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/datastore/khanhnd/models/aeroeyes_model/pip}"
TMPDIR="${TMPDIR:-/datastore/khanhnd/models/aeroeyes_model/tmp}"

export PIP_CACHE_DIR TMPDIR
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

if [[ ! -d "$SDQM_DIR" ]]; then
  echo "SDQM repo not found at $SDQM_DIR"
  exit 1
fi

if [[ ! -d "$ULTRALYTICS_DIR" ]]; then
  echo "Custom ultralytics not found at $ULTRALYTICS_DIR"
  exit 1
fi

echo "Installing core SDQM dependencies."
"$PYTHON" -m pip install \
  --progress-bar off \
  -r "$ROOT_DIR/requirements-sdqm-core.txt"

echo "Installing TensorFlow for the SDQM separability metric."
"$PYTHON" -m pip install --progress-bar off tensorflow==2.21.0

echo "Installing AutoKeras for the SDQM separability metric."
"$PYTHON" -m pip install --progress-bar off autokeras

echo "Installing the SDQM Ultralytics fork."
"$PYTHON" -m pip install --no-deps --force-reinstall -e "$ULTRALYTICS_DIR"

echo "SDQM V-Info setup complete."
