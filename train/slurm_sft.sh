#!/bin/bash
#SBATCH --job-name=nemotron_sft
#SBATCH --chdir=/home/pds981/nemotron
#SBATCH --output=/home/pds981/nemotron/logs/sft_%j.out
#SBATCH --error=/home/pds981/nemotron/logs/sft_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=ml4good
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192GB

# QLoRA SFT of Nemotron-3-Nano-30B on a single L40s 48GB.
# Edit the paths below for your checkout.

set -euo pipefail
mkdir -p logs runs

hostname
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# venv: created with python -m venv venv && pip install torch transformers peft \
#       bitsandbytes datasets accelerate tensorboard mamba_ssm causal_conv1d
VENV=${VENV:-$HOME/nemotron/venv}
source "$VENV/bin/activate"

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HOME"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Path to the Nemotron base model. Download once with:
#   huggingface-cli download <repo_id> --local-dir $HOME/models/nemotron-3-nano-30b
# or kagglehub.model_download(...) on a Kaggle box, then rsync it over.
MODEL_PATH=${MODEL_PATH:-$HOME/models/nemotron-3-nano-30b-a3b-bf16}
DATA=${DATA:-$HOME/nemotron/synth/data.jsonl}
OUT_DIR=${OUT_DIR:-$HOME/nemotron/runs/sft1}

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
