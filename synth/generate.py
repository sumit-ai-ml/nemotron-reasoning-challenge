"""Generate SFT training data from train.csv using our solvers.

For each row:
  1. Categorize the prompt.
  2. If a solver exists for that category, generate a CoT trace via
     `solver.explain(prompt)`. If the trace's `\\boxed{}` answer matches the
     gold answer (with relative-numeric tolerance to mirror the eval metric),
     emit a CoT example.
  3. Otherwise (or on solver failure), emit a *no-CoT* example whose
     completion is just `\\boxed{<gold>}` — the model still sees the
     (prompt, answer) mapping and can pattern-match.

Output: a JSONL with fields {id, category, has_cot, prompt, completion, gold}.

The completion always ends with `\\boxed{<answer>}`. We add a simple instruction
prefix to the prompt that primes the model to produce the boxed answer; this
matches what we'll do at inference.

Usage:
    python -m synth.generate --out synth/data.jsonl
    python -m synth.generate --out synth/data.jsonl --no-skip-unsolved
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solvers.categorize import categorize  # noqa: E402
from solvers import SOLVERS  # noqa: E402
from eval.score import is_correct, extract_answer  # noqa: E402


SYSTEM_PREFIX = (
    "You are a careful reasoner. Solve the following Wonderland puzzle by "
    "first identifying the underlying rule from the examples, then applying "
    "it to the query. Reason step by step. Place the final answer inside "
    "\\boxed{}.\n\n"
)


def build_prompt(raw: str) -> str:
    return SYSTEM_PREFIX + raw.strip()


def build_completion_with_cot(trace: str) -> str:
    return trace.strip()


def build_completion_no_cot(gold: str) -> str:
    return f"\\boxed{{{gold}}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "train.csv"))
    ap.add_argument("--out",  default=str(ROOT / "synth" / "data.jsonl"))
    ap.add_argument("--skip-unsolved", action="store_true",
                    help="Skip examples where the solver answer doesn't match gold")
    ap.add_argument("--no-skip-unsolved", dest="skip_unsolved",
                    action="store_false")
    ap.set_defaults(skip_unsolved=False)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_cot = n_raw = n_skip = 0
    cat_stats = {}

    with out_path.open("w") as fout:
        for _, row in df.iterrows():
            n_total += 1
            cat = categorize(row["prompt"])
            gold = str(row["answer"])
            mod = SOLVERS.get(cat)

            example = {
                "id": row["id"],
                "category": cat,
                "gold": gold,
                "prompt": build_prompt(row["prompt"]),
                "has_cot": False,
                "completion": build_completion_no_cot(gold),
            }

            if mod is not None:
                try:
                    trace = mod.explain(row["prompt"])
                except Exception:
                    trace = None
                if trace:
                    extracted = extract_answer(trace)
                    if is_correct(extracted, gold):
                        example["has_cot"] = True
                        example["completion"] = build_completion_with_cot(trace)
                        n_cot += 1
                    elif args.skip_unsolved:
                        n_skip += 1
                        cat_stats.setdefault(cat, {"cot": 0, "raw": 0, "skip": 0})
                        cat_stats[cat]["skip"] += 1
                        continue
                    else:
                        n_raw += 1
                else:
                    if args.skip_unsolved:
                        n_skip += 1
                        cat_stats.setdefault(cat, {"cot": 0, "raw": 0, "skip": 0})
                        cat_stats[cat]["skip"] += 1
                        continue
                    n_raw += 1
            else:
                if args.skip_unsolved:
                    n_skip += 1
                    cat_stats.setdefault(cat, {"cot": 0, "raw": 0, "skip": 0})
                    cat_stats[cat]["skip"] += 1
                    continue
                n_raw += 1

            cat_stats.setdefault(cat, {"cot": 0, "raw": 0, "skip": 0})
            cat_stats[cat]["cot" if example["has_cot"] else "raw"] += 1
            fout.write(json.dumps(example) + "\n")

    print(f"Wrote {n_cot + n_raw} examples to {out_path}")
    print(f"  CoT  : {n_cot}")
    print(f"  raw  : {n_raw}")
    print(f"  skip : {n_skip}")
    print(f"  total processed: {n_total}")
    print()
    print(f"{'category':>12}  {'cot':>6}  {'raw':>6}  {'skip':>6}")
    for cat, s in sorted(cat_stats.items()):
        print(f"{cat:>12}  {s['cot']:>6}  {s['raw']:>6}  {s['skip']:>6}")


if __name__ == "__main__":
    main()
