"""One-time fix for the Nemotron Kaggle Hub shards.

The shards ship without a metadata header, so transformers 4.46.3's
load_state_dict trips on `metadata.get("format")` because `metadata`
is None. Re-saves each shard with `{"format": "pt"}` metadata.
Idempotent — already-fixed shards are skipped.

    python -m train.fix_safetensors_metadata --model-dir $HOME/models/nemotron-3-nano-30b-a3b-bf16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    shards = sorted(Path(args.model_dir).glob("*.safetensors"))
    if not shards:
        print(f"no .safetensors files in {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    fixed = 0
    for path in shards:
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata()
        if meta and meta.get("format") in ("pt", "tf", "flax", "mlx"):
            print(f"ok    {path.name}")
            continue

        print(f"fix   {path.name}  ({path.stat().st_size / 1e9:.1f} GB)")
        tensors = load_file(str(path))
        tmp = path.with_suffix(path.suffix + ".tmp")
        save_file(tensors, str(tmp), metadata={"format": "pt"})
        tmp.replace(path)
        del tensors
        fixed += 1

    print(f"\n{fixed} of {len(shards)} shards rewritten.")


if __name__ == "__main__":
    main()
