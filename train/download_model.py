"""Pre-fetch the Nemotron-3-Nano-30B base weights to a local cache.

The competition model lives on Kaggle Hub:
    metric/nemotron-3-nano-30b-a3b-bf16/transformers/default

This script downloads it once and prints the resolved local path. Idempotent:
re-running on an already-downloaded model is a no-op.

Auth: kagglehub looks for credentials in any of (in order):
    KAGGLE_USERNAME + KAGGLE_KEY env vars
    KAGGLEHUB_AUTH_TOKEN env var (newer access-token style)
    ~/.kaggle/kaggle.json   (older username+key file)
    ~/.kaggle/access_token  (newer access-token file)

Create one at https://www.kaggle.com/settings → "Create New Token", then:
    mkdir -p ~/.kaggle
    mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    # or for the access-token variant:
    echo $YOUR_TOKEN > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

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


def _have_kaggle_auth() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if os.environ.get("KAGGLEHUB_AUTH_TOKEN"):
        return True
    kdir = Path.home() / ".kaggle"
    return (kdir / "kaggle.json").is_file() or (kdir / "access_token").is_file()


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

    if not _have_kaggle_auth():
        print(
            "No Kaggle credentials found. Set one of:\n"
            "  - KAGGLE_USERNAME / KAGGLE_KEY  env vars\n"
            "  - KAGGLEHUB_AUTH_TOKEN          env var\n"
            "  - ~/.kaggle/kaggle.json         (username+key json)\n"
            "  - ~/.kaggle/access_token        (single-line access token)\n"
            "Create one at https://www.kaggle.com/settings → 'Create New Token'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # If the user put the token in ~/.kaggle/access_token but kagglehub on the
    # installed version only reads the env var, plumb it through ourselves.
    if not os.environ.get("KAGGLEHUB_AUTH_TOKEN"):
        tok_path = Path.home() / ".kaggle" / "access_token"
        if tok_path.is_file():
            os.environ["KAGGLEHUB_AUTH_TOKEN"] = tok_path.read_text().strip()

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
