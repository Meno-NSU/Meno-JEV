#!/usr/bin/env bash
# Create a virtualenv and install Meno-JEV.
#
#   bash scripts/setup_env.sh            # core only (works with the mock, no GPU)
#   WITH_HF=1 bash scripts/setup_env.sh  # + torch/transformers for real models
#
# For a specific CUDA build, install torch yourself first, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
# then re-run this script.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> using interpreter: $($PYTHON --version) ($(command -v "$PYTHON"))"
"$PYTHON" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel >/dev/null

echo "==> installing core (menoj + tqdm)"
pip install -e .

if [[ "${WITH_HF:-0}" == "1" ]]; then
  echo "==> installing HuggingFace backend"
  pip install -r requirements-hf.txt
fi

echo "==> environment:"
python -m jev.cli doctor
echo
echo "Activate with: source $VENV_DIR/bin/activate"
