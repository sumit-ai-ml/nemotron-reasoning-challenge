"""Pre-fetch the Nemotron-3-Nano-30B base weights to a local cache.

The competition model lives on Kaggle Hub:
    metric/nemotron-3-nano-30b-a3b-bf16/transformers/default

This script downloads it once and prints the resolved local path. Idempotent:
re-running on an already-downloaded model is a no-op.

Requires the Kaggle credentials file at ~/.kaggle/kaggle.json (chmod 600).
Create one from https://www.kaggle.com/settings → "Create New Token".

Usage:
    python -m train.download_model
    python -m train.download_model --dest $HOME/models

The script writes a small marker file `MODEL_PATH` in the destination dir so
downstream scripts (slurm_sft.sh) can `cat` it without re-importing kagglehub.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


KAGGLE_HANDLE = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.environ.get("MODEL_HOME", str(Path.home() / "models")),
                    help="Where to symlink the resolved model directory (default: $HOME/models)")
    ap.add_argument("--handle", default=KAGGLE_HANDLE,
                    help="Kaggle Hub model handle")
    args = ap.parse_args()

    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed. Run: pip install kagglehub", file=sys.stderr)
        sys.exit(1)

    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.is_file():
        print(
            f"Kaggle credentials not found at {cred_path}.\n"
            "Create one at https://www.kaggle.com/settings → 'Create New Token'\n"
            "then: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"=> downloading {args.handle}")
    resolved = kagglehub.model_download(args.handle)
    print(f"   resolved to: {resolved}")

    dest_root = Path(args.dest).expanduser()
    dest_root.mkdir(parents=True, exist_ok=True)
    marker = dest_root / "MODEL_PATH"
    marker.write_text(str(resolved) + "\n")
    print(f"=> wrote marker {marker} ({resolved})")

    symlink = dest_root / "nemotron-3-nano-30b-a3b-bf16"
    if symlink.is_symlink() or symlink.exists():
        try:
            symlink.unlink()
        except IsADirectoryError:
            shutil.rmtree(symlink)
    symlink.symlink_to(resolved)
    print(f"=> linked  {symlink} -> {resolved}")
    print()
    print(f"Use this path in train/sft.py: {symlink}")


if __name__ == "__main__":
    main()
