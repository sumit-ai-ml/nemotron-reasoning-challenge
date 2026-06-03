"""Patch a dtype-mismatch bug in the Nemotron MoE forward.

The vendored modeling code does:

    final_hidden_states.index_add_(0, token_indices, weighted_output)

`weighted_output` ends up fp32 (router gate softmax is fp32 for
stability), while `final_hidden_states` is bf16. PyTorch's
`index_add_` requires matching dtypes, so training crashes on the
very first step.

Patch the call site to cast the source to the destination dtype.
Idempotent: re-running on a patched file is a no-op.

    python -m train.fix_modeling_moe --model-dir $HOME/models/nemotron-3-nano-30b-a3b-bf16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

OLD = "final_hidden_states.index_add_(0, token_indices, weighted_output)"
NEW = "final_hidden_states.index_add_(0, token_indices, weighted_output.to(final_hidden_states.dtype))"

CACHE_DIR = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"


def patch_file(path: Path) -> str:
    text = path.read_text()
    if NEW in text:
        return "already"
    if OLD not in text:
        return "missing"
    path.write_text(text.replace(OLD, NEW))
    return "patched"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    source = Path(args.model_dir) / "modeling_nemotron_h.py"
    if not source.exists():
        print(f"no modeling_nemotron_h.py in {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    targets = [source]
    targets += list(CACHE_DIR.rglob("modeling_nemotron_h.py"))

    for p in targets:
        status = patch_file(p)
        print(f"  {status:9s} {p}")
        if status == "missing":
            print("  (file did not contain the expected line — model version may "
                  "differ; inspect manually)", file=sys.stderr)


if __name__ == "__main__":
    main()
