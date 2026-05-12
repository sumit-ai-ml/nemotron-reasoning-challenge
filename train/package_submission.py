"""Package a trained LoRA adapter into submission.zip for Kaggle.

The eval pipeline expects:
  submission.zip
    ├── adapter_config.json
    └── adapter_model.safetensors

(Plus tokenizer files are harmless to include.)

Usage:
    python -m train.package_submission --adapter-dir runs/sft1/final \
        --out submission.zip
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

REQUIRED = ("adapter_config.json", "adapter_model.safetensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--include-tokenizer", action="store_true")
    args = ap.parse_args()

    adir = Path(args.adapter_dir)
    if not adir.is_dir():
        raise SystemExit(f"adapter dir {adir} does not exist")

    for fname in REQUIRED:
        if not (adir / fname).is_file():
            raise SystemExit(f"missing required file: {adir / fname}")

    # Sanity: verify adapter_config.json's max_lora_rank is ≤ 32
    cfg = json.loads((adir / "adapter_config.json").read_text())
    if cfg.get("r", 0) > 32:
        raise SystemExit(f"LoRA rank {cfg.get('r')} > 32 (competition limit)")

    out = Path(args.out)
    if out.exists():
        out.unlink()

    files = list(REQUIRED)
    if args.include_tokenizer:
        for extra in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            if (adir / extra).is_file():
                files.append(extra)

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fname in files:
            z.write(adir / fname, arcname=fname)
            print(f"  + {fname}")
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
