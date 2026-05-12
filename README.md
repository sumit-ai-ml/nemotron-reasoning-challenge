# Nemotron Reasoning Challenge

Code for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
on Kaggle. The challenge ships a base `nemotron-3-nano-30b-a3b-bf16` model and
9,500 "Wonderland" puzzles across six categories; submissions are a LoRA adapter
of rank ≤ 32.

The approach: write per-category Python solvers, generate chain-of-thought
training data from them, and fine-tune the base model with QLoRA to imitate the
correct reasoning traces. See `STRATEGY.md` for the full plan.

## Solver accuracy on the 9,500 training rows

| Category     |  Score   | Method |
|--------------|---------:|--------|
| `unit_conv`  | 100.00 % | OLS y = a·x + b |
| `gravity`    | 100.00 % | OLS through origin on (½t², d) |
| `numeral`    | 100.00 % | Roman-numeral encoder |
| `cipher`     | 100.00 % | 77-word closed vocabulary + pattern-match backtracking |
| `bit_manip`  |  87.45 % | Operator-family enumeration (unary + pair + maj/choice) |
| `eq_symbols` |   0.90 % | Layered numeric-arith + per-position; symbolic puzzles still open |
| **Overall**  | **81.66 %** | |

## Layout

```
data/                      # train.csv / test.csv (gitignored; sourced from Kaggle)
solvers/
    categorize.py          # 6-class detector by phrase match
    unit_conv.py           # linear regression
    gravity.py             # closed-form d = ½gt²
    numeral.py             # Roman numerals
    cipher.py              # monoalphabetic over a 77-word vocab
    bit_manip.py           # operator-family enumeration
    eq_symbols.py          # layered hypothesis testing (WIP)
eval/score.py              # offline eval mirroring the competition metric
synth/generate.py          # converts train.csv → JSONL of {prompt, completion} pairs
train/sft.py               # QLoRA + LoRA rank-32 SFT
train/slurm_sft.sh         # Hendrix submission (single L40s 48 GB)
train/package_submission.py# bundles adapter into submission.zip
STRATEGY.md                # full plan
```

## Pipeline

```bash
# 1. Get the data (place train.csv / test.csv into ./data/)
unzip nvidia-nemotron-model-reasoning-challenge.zip -d data/

# 2. Verify solver coverage on train
python -m eval.score

# 3. Generate the SFT dataset
python -m synth.generate --out synth/data.jsonl

# 4. Fine-tune (single L40s 48 GB, QLoRA)
sbatch train/slurm_sft.sh
# or locally:
python -m train.sft \
    --model-path /path/to/nemotron-3-nano-30b-a3b-bf16 \
    --data synth/data.jsonl --out-dir runs/sft1 \
    --quant nf4 --epochs 1 --batch-size 1 --grad-accum 32 --lr 2e-4

# 5. Package
python -m train.package_submission --adapter-dir runs/sft1/final \
    --out submission.zip
```

## Requirements

`pip install torch transformers peft bitsandbytes datasets accelerate tensorboard mamba_ssm causal_conv1d pandas numpy`

## License

Code: MIT. Competition data (`data/`) is CC-BY-4.0 from NVIDIA and not included
in this repo.
