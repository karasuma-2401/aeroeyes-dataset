#!/bin/bash
#SBATCH --job-name=aeroeyes_bbox
#SBATCH --output=logs/bounding_boxes_%j.out
#SBATCH --error=logs/bounding_boxes_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="$PROJECT_DIR/venv/bin/python"
OUTPUT_ROOT=/datastore/khanhnd/models/aeroeyes_output/output
METADATA_DIR="$OUTPUT_ROOT/_metadata"
BBOX_OUTPUT_DIR="$OUTPUT_ROOT/bounding_boxes"
LIMIT_PAIRS="${1:-5}"

cd "$PROJECT_DIR"

if [[ "$PROJECT_DIR" != /datastore/khanhnd/* ]]; then
  echo "Project must be stored below /datastore/khanhnd: $PROJECT_DIR" >&2
  exit 1
fi

if [[ ! "$LIMIT_PAIRS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Pair limit must be a positive integer: $LIMIT_PAIRS" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Generation environment is missing. Run scripts/setup_generation.slurm first." >&2
  exit 1
fi

if [[ ! -d "$METADATA_DIR" ]]; then
  echo "Generation metadata is missing: $METADATA_DIR" >&2
  exit 1
fi

module clear -f
module load slurm/slurm/25.05
module load cuda12.9/toolkit/12.9.1
module load shared python312

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export AEROEYES_MODEL_DIR=/datastore/khanhnd/models/aeroeyes_model
export HF_HOME="$AEROEYES_MODEL_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
export TORCH_HOME="$AEROEYES_MODEL_DIR/torch"
export XDG_CACHE_HOME="$AEROEYES_MODEL_DIR/xdg"

mkdir -p "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$HF_XET_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME"

"$PYTHON" -m unittest discover -s tests -p 'test_bounding_box_pipeline.py'
"$PYTHON" -m unittest discover -s tests -p 'test_change_detection.py'

"$PYTHON" - "$METADATA_DIR" "$LIMIT_PAIRS" <<'PY'
import sys
from pathlib import Path

import torch

metadata_dir = Path(sys.argv[1])
limit = int(sys.argv[2])
complete = 0
for metadata_path in sorted(metadata_dir.glob("*.json")):
    if metadata_path.stat().st_size > 0:
        complete += 1
if complete < limit:
    raise SystemExit(f"Only {complete} metadata records exist; requested {limit}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the Slurm GPU job")
print(f"Metadata records ready: {complete}; pair target: {limit}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

"$PYTHON" main_bounding_boxes.py \
  --limit "$LIMIT_PAIRS" \
  --metadata-dir "$METADATA_DIR" \
  --output-dir "$BBOX_OUTPUT_DIR"
