#!/bin/bash
#SBATCH --job-name=aeroeyes_eval_setup
#SBATCH --output=logs/setup_evaluation_%j.out
#SBATCH --error=logs/setup_evaluation_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00

set -euo pipefail

trap 'status=$?; echo "Evaluation setup failed (exit=$status): $BASH_COMMAND" >&2; exit "$status"' ERR

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="$PROJECT_DIR/venv/bin/python"

cd "$PROJECT_DIR"

if [[ "$PROJECT_DIR" != /datastore/khanhnd/* ]]; then
  echo "Project must be stored below /datastore/khanhnd: $PROJECT_DIR" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Generation environment is missing. Run scripts/setup_generation.slurm first." >&2
  exit 1
fi

module clear -f
module load slurm/slurm/25.05
module load shared python312

export PIP_CACHE_DIR=/datastore/khanhnd/models/aeroeyes_model/pip
export TMPDIR=/datastore/khanhnd/models/aeroeyes_model/tmp
export AEROEYES_MODEL_DIR=/datastore/khanhnd/models/aeroeyes_model
export SDQM_REPO_DIR="$PROJECT_DIR/third_party/SDQM"
export PYTHON

mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

"$PYTHON" -m pip install --progress-bar off ultralytics==8.4.118
bash scripts/setup_sdqm_vinfo.sh
"$PYTHON" -m pip install --progress-bar off \
  pyiqa==0.1.16 \
  -r cmmd-pytorch/requirements.txt
"$PYTHON" -m pip check

"$PYTHON" - <<'PY'
import absl
import autokeras
from importlib.metadata import version
import pandas
import pyiqa
import scipy
import skimage
import tensorflow
import torch

from src.evaluation.sdqm_vinfo import check_custom_ultralytics

is_ready, message = check_custom_ultralytics()
if not is_ready:
    raise SystemExit(f"Custom Ultralytics check failed: {message}")
if torch.version.cuda != "12.8":
    raise SystemExit(
        f"PyTorch CUDA mismatch after evaluation setup: {torch.version.cuda}"
    )

print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
print(f"pandas: {pandas.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"scikit-image: {skimage.__version__}")
print(f"pyiqa: {version('pyiqa')}")
print(f"tensorflow: {tensorflow.__version__}")
print(f"autokeras: {version('autokeras')}")
print("Evaluation environment is ready.")
PY
