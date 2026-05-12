#!/bin/bash
#SBATCH --job-name=nemotron_sft
#SBATCH --chdir=/home/pds981/nemotron-reasoning-challenge
#SBATCH --output=/home/pds981/nemotron-reasoning-challenge/logs/sft_%j.out
#SBATCH --error=/home/pds981/nemotron-reasoning-challenge/logs/sft_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=ml4good
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192GB

# QLoRA SFT of Nemotron-3-Nano-30B on a single L40s 48GB.
# Run train/slurm_sanity.sh once first to provision the venv.

set -euo pipefail
mkdir -p logs runs

hostname
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

REPO_DIR=${REPO_DIR:-$HOME/nemotron-reasoning-challenge}
VENV=${VENV:-$REPO_DIR/venv}
if [ ! -f "$VENV/bin/activate" ]; then
    echo "!! venv missing at $VENV — run 'sbatch train/slurm_sanity.sh' first" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$REPO_DIR"

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HOME"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Base model. If MODEL_PATH isn't set or doesn't point to weights, we fetch
# from Kaggle Hub via train/download_model.py. Requires ~/.kaggle/kaggle.json
# (chmod 600) — create one at https://www.kaggle.com/settings.
MODEL_PATH=${MODEL_PATH:-$HOME/models/nemotron-3-nano-30b-a3b-bf16}
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "=> base model not found at $MODEL_PATH; fetching from Kaggle Hub"
    python -u -m train.download_model --dest "$HOME/models"
    MODEL_PATH=$(cat "$HOME/models/MODEL_PATH")
    echo "=> using MODEL_PATH=$MODEL_PATH"
fi

DATA=${DATA:-$REPO_DIR/synth/data.jsonl}
OUT_DIR=${OUT_DIR:-$REPO_DIR/runs/sft1}

echo "=== run config ==="
echo "model: $MODEL_PATH"
echo "data:  $DATA"
echo "out:   $OUT_DIR"
echo "=================="

srun python -u -m train.sft \
    --model-path "$MODEL_PATH" \
    --data "$DATA" \
    --out-dir "$OUT_DIR" \
    --quant nf4 \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 32 \
    --lr 2e-4 \
    --max-seq-len 2048 \
    --save-steps 500

echo "=== done $(date) ==="
