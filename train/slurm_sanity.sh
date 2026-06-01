#!/bin/bash
#SBATCH --job-name=nemotron_sanity
#SBATCH --chdir=/home/pds981/nemotron-reasoning-challenge
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=ml4good
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

# One-stop setup + smoke test for the Nemotron QLoRA pipeline.
#
# What this does:
#   1. Creates $VENV (default $HOME/nemotron-reasoning-challenge/venv) if missing.
#   2. Installs torch + transformers + peft + bitsandbytes + mamba_ssm
#      + causal_conv1d + datasets + accelerate + kagglehub + pandas + numpy.
#   3. Runs train/sanity_check.py against the cached base model.
#
# Idempotent: re-running on a node where the venv already has the right
# packages skips reinstall and goes straight to the smoke test.
#
# Submit:    sbatch train/slurm_sanity.sh
# Or run interactively (needs an L40s):
#            srun -p ml4good --gres=gpu:l40s:1 --time=01:00:00 \
#                 --cpus-per-task=8 --mem=64GB --pty bash train/slurm_sanity.sh

set -euo pipefail

hostname
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
date

REPO_DIR=${REPO_DIR:-$HOME/nemotron-reasoning-challenge}
VENV=${VENV:-$REPO_DIR/venv}
MODEL_PATH=${MODEL_PATH:-$HOME/models/nemotron-3-nano-30b-a3b-bf16}

echo "=== config ==="
echo "REPO_DIR:   $REPO_DIR"
echo "VENV:       $VENV"
echo "MODEL_PATH: $MODEL_PATH"
echo "=============="

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# 1. venv
# ---------------------------------------------------------------------------
if [ ! -f "$VENV/bin/activate" ]; then
    echo "=> creating venv at $VENV"
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel

# ---------------------------------------------------------------------------
# 2. dependencies
# ---------------------------------------------------------------------------
# Sentinel file lets us skip reinstalls when nothing changed.
REQ_HASH=$(printf %s "torch==2.4.0|transformers|peft|bitsandbytes|datasets|accelerate|tensorboard|mamba_ssm|causal_conv1d|kagglehub|pandas|numpy" | md5sum | awk '{print $1}')
SENTINEL="$VENV/.deps_$REQ_HASH"
if [ ! -f "$SENTINEL" ]; then
    echo "=> installing core deps"
    # torch with CUDA 12.1 wheels — matches what mamba_ssm prebuilt wheels target
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.4.0"
    pip install \
        "transformers>=4.45" \
        "peft>=0.13" \
        "bitsandbytes>=0.43" \
        "datasets>=2.20" \
        "accelerate>=0.33" \
        "tensorboard" \
        "kagglehub" \
        "pandas" \
        "numpy<2.0"
    # mamba_ssm + causal_conv1d need ninja and a matching CUDA toolkit. Pre-
    # built wheels exist for cu121 + torch 2.4; if pip falls back to source
    # build, the install will take ~10 min. --no-build-isolation lets it find
    # the installed torch headers.
    pip install ninja packaging
    pip install causal-conv1d --no-build-isolation
    pip install mamba-ssm --no-build-isolation
    touch "$SENTINEL"
fi
echo "=> deps ready"
python -c "import torch, transformers, peft, bitsandbytes, mamba_ssm, causal_conv1d; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('transformers', transformers.__version__); \
print('peft', peft.__version__); \
print('bitsandbytes', bitsandbytes.__version__); \
print('mamba_ssm OK'); print('causal_conv1d OK')"

# ---------------------------------------------------------------------------
# 3. sanity check
# ---------------------------------------------------------------------------
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "!! base model not found at $MODEL_PATH"
    echo "   first run:  python -m train.download_model"
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=== starting sanity check ==="
python -u -m train.sanity_check --model-path "$MODEL_PATH"
echo "=== done $(date) ==="
