"""Linear-regression solver for the unit_conv puzzles.

Format:
    X.XX m becomes Y.YY
    ...
    Now, convert the following measurement: Q.QQ m

Hypothesis: y = a*x + b for some hidden a, b. Fit OLS, apply, round to 2dp.
"""
import re
from typing import List, Tuple

import numpy as np

EXAMPLE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*m\s*becomes\s*(-?\d+(?:\.\d+)?)")
QUERY_RE = re.compile(
    r"Now,\s*convert\s*the\s*following\s*measurement:\s*(-?\d+(?:\.\d+)?)\s*m"
)


def parse(prompt: str) -> Tuple[List[Tuple[float, float]], float]:
    pairs = [(float(a), float(b)) for a, b in EXAMPLE_RE.findall(prompt)]
    m = QUERY_RE.search(prompt)
    if not m:
        raise ValueError("could not find unit_conv query in prompt")
    # The query value also matches EXAMPLE_RE if we're not careful — make sure
    # we excluded it from `pairs`. Re-derive pairs using only the lines that
    # contain "becomes".
    pairs = []
    for line in prompt.splitlines():
        if "becomes" in line:
            mm = EXAMPLE_RE.search(line)
            if mm:
                pairs.append((float(mm.group(1)), float(mm.group(2))))
    return pairs, float(m.group(1))


def fit(pairs):
    xs = np.array([p[0] for p in pairs], dtype=np.float64)
    ys = np.array([p[1] for p in pairs], dtype=np.float64)
    a, b = np.polyfit(xs, ys, 1)
    return float(a), float(b)


def solve(prompt: str) -> str:
    pairs, q = parse(prompt)
    a, b = fit(pairs)
    y = a * q + b
    return f"{y:.2f}"


def explain(prompt: str) -> str:
    pairs, q = parse(prompt)
    a, b = fit(pairs)
    y = a * q + b

    lines = []
    lines.append(
        "This is a unit-conversion puzzle. The hidden rule looks like a linear "
        "relation y = a*x + b. I'll fit a and b from the example pairs and "
        "apply the result to the query."
    )
    lines.append("")
    lines.append("Examples (input -> output):")
    for x, y_ex in pairs:
        lines.append(f"  {x:g} -> {y_ex:g}")
    lines.append("")
    lines.append(
        f"Least-squares fit gives a ≈ {a:.4f}, b ≈ {b:.4f}."
    )
    lines.append(
        f"Applying to the query x = {q:g}: y = {a:.4f} * {q:g} + ({b:.4f}) "
        f"= {y:.4f}."
    )
    lines.append(f"Rounded to two decimals: {y:.2f}.")
    lines.append("")
    lines.append(f"\\boxed{{{y:.2f}}}")
    return "\n".join(lines)
