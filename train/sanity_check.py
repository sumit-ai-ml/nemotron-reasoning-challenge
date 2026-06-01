"""Pre-flight check: confirm the Nemotron base + LoRA scaffold actually works.

Run this once on Hendrix (interactive or short SLURM) before submitting the
long QLoRA SFT job. Verifies:

  1. mamba_ssm and causal_conv1d import (required by the Mamba blocks).
  2. The tokenizer loads.
  3. The model loads in nf4 + bf16 compute dtype.
  4. The LoRA target_modules regex matches *some* parameters in the model.
  5. A single forward pass produces finite logits.
  6. The PEFT-wrapped model saves an adapter_config.json + safetensors.

Output is verbose; failure exits non-zero. Total time: ~5-10 minutes on an
L40s (most of it is loading the 4-bit base).

Usage:
    python -m train.sanity_check --model-path $HOME/models/nemotron-3-nano-30b-a3b-bf16
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--target-modules", default=r".*\.(in_proj|out_proj|up_proj|down_proj)$")
    ap.add_argument("--lora-rank", type=int, default=32)
    args = ap.parse_args()

    print("[1/6] importing required hybrid-Mamba kernels...")
    try:
        import mamba_ssm  # noqa: F401
        import causal_conv1d  # noqa: F401
    except ImportError as e:
        print(f"  FAIL: {e}")
        print("  pip install mamba_ssm causal_conv1d")
        sys.exit(1)
    print("  OK")

    print("[2/6] loading tokenizer...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"  OK. vocab_size={tok.vocab_size}  pad={tok.pad_token!r}  eos={tok.eos_token!r}")

    print("[3/6] loading model in nf4 + bf16 compute dtype "
          "(this is slow — coffee break)...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
    )
    print(f"  OK. type={type(model).__name__}  num_params={sum(p.numel() for p in model.parameters()):,}")

    print(f"[4/6] verifying LoRA target_modules regex {args.target_modules!r}...")
    pat = re.compile(args.target_modules)
    hits = [n for n, _ in model.named_modules() if pat.match(n)]
    if not hits:
        print("  FAIL: regex matched zero submodules.")
        all_leaves = sorted({n for n, m in model.named_modules() if not list(m.children())})
        suffixes = sorted({n.split(".")[-1] for n in all_leaves})
        print("  observed leaf suffixes (first 40):")
        for s in suffixes[:40]:
            print(f"    {s}")
        sys.exit(1)
    print(f"  OK. {len(hits)} matching submodules. examples:")
    for n in hits[:5]:
        print(f"    {n}")

    print("[5/6] forward pass...")
    test = tok("In Alice's Wonderland, 2+2 =", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**test)
    logits = out.logits
    if not torch.isfinite(logits).all():
        print(f"  FAIL: non-finite logits (any nan/inf).")
        sys.exit(1)
    print(f"  OK. logits shape={tuple(logits.shape)}  range=[{logits.min().item():.2f}, {logits.max().item():.2f}]")

    print(f"[6/6] applying LoRA rank={args.lora_rank} and saving an adapter...")
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=64, lora_dropout=0.05, bias="none",
        target_modules=args.target_modules, task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  trainable={n_trainable:,}  total={n_all:,}  pct={100*n_trainable/n_all:.2f}%")
    with tempfile.TemporaryDirectory() as td:
        model.save_pretrained(td)
        files = sorted(Path(td).iterdir())
        names = [f.name for f in files]
        if "adapter_config.json" not in names:
            print("  FAIL: adapter_config.json not produced.")
            sys.exit(1)
        if not any(n.endswith(".safetensors") for n in names):
            print("  FAIL: no .safetensors written.")
            sys.exit(1)
        print(f"  OK. wrote: {names}")

    print()
    print("=== all checks passed — safe to submit slurm_sft.sh ===")


if __name__ == "__main__":
    main()
