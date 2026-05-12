"""Offline eval harness.

Mirrors the competition's answer-extraction + correctness rules as closely as we
can given the public description:

  - prefer the content of \\boxed{...}
  - fall back to other heuristic patterns ("Final answer:" lines, last numeric)
  - graded correct if exact string match OR within relative numeric tolerance

We assume relative tolerance ~1e-2 unless overridden (the competition page just
says "within a relative numerical tolerance"; 1e-2 is consistent with the 2dp
training labels). We additionally score with strict-string-match for diagnostic
purposes.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solvers.categorize import categorize  # noqa: E402
from solvers import unit_conv, gravity, numeral, cipher, bit_manip, eq_symbols  # noqa: E402

SOLVER_BY_CAT = {
    "unit_conv": unit_conv,
    "gravity": gravity,
    "numeral": numeral,
    "cipher": cipher,
    "bit_manip": bit_manip,
    "eq_symbols": eq_symbols,
}

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(generation: str) -> str:
    m = list(BOXED_RE.finditer(generation))
    if m:
        return m[-1].group(1).strip()
    # Fallback: last "Final answer: ..." line
    for line in reversed(generation.splitlines()):
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("final answer") or low.startswith("answer"):
            after = line.split(":", 1)[-1].strip()
            if after:
                return after
    nums = NUM_RE.findall(generation)
    if nums:
        return nums[-1]
    return generation.strip()


def is_correct(pred: str, gold: str, rel_tol: float = 1e-2) -> bool:
    pred = (pred or "").strip()
    gold = (gold or "").strip()
    if pred == gold:
        return True
    # Try numeric.
    try:
        p, g = float(pred), float(gold)
    except ValueError:
        return False
    if g == 0:
        return abs(p) <= rel_tol
    return abs(p - g) <= rel_tol * abs(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "train.csv"))
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, only evaluate this many rows per category")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--show-failures", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df["cat"] = df["prompt"].apply(categorize)

    cats = args.categories or list(SOLVER_BY_CAT.keys())

    overall_n = 0
    overall_c = 0
    fail_examples = defaultdict(list)

    for cat in cats:
        sub = df[df["cat"] == cat]
        if args.limit:
            sub = sub.head(args.limit)
        if cat not in SOLVER_BY_CAT:
            print(f"[{cat}] no solver registered, skipping {len(sub)} rows")
            continue
        mod = SOLVER_BY_CAT[cat]
        n = c = 0
        for _, row in sub.iterrows():
            try:
                pred = mod.solve(row["prompt"])
            except Exception as e:
                pred = f"<error: {e}>"
            ok = is_correct(pred, str(row["answer"]))
            n += 1
            c += int(ok)
            if not ok and len(fail_examples[cat]) < args.show_failures:
                fail_examples[cat].append((row["prompt"], row["answer"], pred))
        acc = c / max(n, 1)
        print(f"[{cat:>10}] {c}/{n} = {acc*100:.2f}%")
        overall_n += n
        overall_c += c

    print(f"[{'OVERALL':>10}] {overall_c}/{overall_n} = "
          f"{overall_c/max(overall_n,1)*100:.2f}%")

    if args.show_failures:
        for cat, fails in fail_examples.items():
            if not fails:
                continue
            print(f"\n--- failures in {cat} ---")
            for p, g, pr in fails:
                print(f"GOLD={g!r}  PRED={pr!r}")
                print("  prompt:", p[:200].replace("\n", " | "))


if __name__ == "__main__":
    main()
