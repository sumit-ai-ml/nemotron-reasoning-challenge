"""Supervised fine-tuning of Nemotron-3-Nano-30B with LoRA rank 32.

Two paths supported, controlled by --quant {none,nf4}:

  --quant nf4  : QLoRA (4-bit base via bitsandbytes) — fits on a single L40s
                 48GB. Required on Hendrix's `ml4good` partition.

  --quant none : bf16 base + LoRA. Needs ≥2 large GPUs (or A100/H100 80GB).
                 Use this on Kaggle's high-VRAM session if available.

Training data is a JSONL with {prompt, completion} (see synth/generate.py).
Prompt tokens are loss-masked so only completion tokens contribute to the
loss.

LoRA targets mirror the official demo:
  target_modules = r".*\.(in_proj|out_proj|up_proj|down_proj)$"

This covers Mamba SSM blocks (in_proj/out_proj) plus MLPs (up_proj/down_proj).

Usage:
    python -m train.sft \
        --model-path /path/to/nemotron-3-nano-30b-a3b-bf16 \
        --data synth/data.jsonl \
        --out-dir runs/sft1 \
        --quant nf4 \
        --epochs 1 \
        --batch-size 1 \
        --grad-accum 32 \
        --lr 2e-4
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True,
                    help="Path or HF id of the Nemotron base model")
    ap.add_argument("--data", required=True, help="JSONL with prompt/completion")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quant", choices=["none", "nf4"], default="nf4")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default=r".*\.(in_proj|out_proj|up_proj|down_proj)$")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validation-frac", type=float, default=0.02)
    return ap.parse_args()


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    args = parse_args()

    print(f"=> loading tokenizer from {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_config = None
    if args.quant == "nf4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            # The fused Mamba kernel reaches into out_proj.weight directly and
            # bypasses Linear4bit.__call__, so out_proj must stay bf16 or the
            # training forward dies with a shape mismatch on the packed buffer.
            llm_int8_skip_modules=["out_proj"],
        )

    print(f"=> loading model from {args.model_path}  (quant={args.quant})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
    )
    print(f"   model loaded: {type(model).__name__} "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    model.config.use_cache = False
    if args.quant == "nf4":
        print("=> prepare_model_for_kbit_training", flush=True)
        model = prepare_model_for_kbit_training(model)
        print("   kbit prep done", flush=True)

    print(f"=> applying LoRA rank={args.lora_rank} alpha={args.lora_alpha}", flush=True)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"=> loading data from {args.data}")
    rows = load_jsonl(args.data)
    print(f"   {len(rows)} examples")

    def encode(example):
        prompt = example["prompt"]
        completion = example["completion"]
        # Build input as prompt + " " + completion
        # Tokenize each separately to know how many tokens belong to the prompt
        # (so we can mask the prompt portion in the loss).
        full_text = prompt + "\n\n" + completion + tok.eos_token
        prompt_text = prompt + "\n\n"
        full_ids = tok(full_text, add_special_tokens=False, truncation=True,
                       max_length=args.max_seq_len)["input_ids"]
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for i in range(prompt_len):
            labels[i] = -100
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
                "labels": labels}

    ds = Dataset.from_list(rows)
    ds = ds.map(encode, remove_columns=ds.column_names, num_proc=8)

    n_val = max(1, int(len(ds) * args.validation_frac))
    ds = ds.shuffle(seed=args.seed)
    val_ds = ds.select(range(n_val))
    train_ds = ds.select(range(n_val, len(ds)))
    print(f"   train={len(train_ds)} val={len(val_ds)}")

    collator = DataCollatorForSeq2Seq(tokenizer=tok, padding=True,
                                      label_pad_token_id=-100)

    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        report_to=["tensorboard"],
        seed=args.seed,
        optim="paged_adamw_8bit" if args.quant == "nf4" else "adamw_torch",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tok,
    )

    print("=> starting training")
    trainer.train()

    print(f"=> saving final adapter to {args.out_dir}/final")
    final_dir = Path(args.out_dir) / "final"
    model.save_pretrained(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print("done")


if __name__ == "__main__":
    main()
