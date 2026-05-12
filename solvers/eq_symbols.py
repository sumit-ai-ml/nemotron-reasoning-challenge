"""eq_symbols solver — layered hypothesis testing.

The eq_symbols category is the trickiest: inputs are 5-char strings, outputs are
1-4 char strings, drawn from a mix of digits and special characters. Several
rule families coexist:

  (A) Numeric arithmetic puzzles. Input has digits 0-9 and exactly one
      operator-like character at a fixed position; the operator denotes a
      basic op (+, -, *, ...) and may be repeated in the output as a marker.
      Different examples in the *same* puzzle may use different operators —
      each behaves consistently across the puzzle.

  (B) Per-position character substitution. Input and output are equal-length;
      out_i = f(in_i) for a fixed character map.

  (C) Character-set ciphers in a custom alphabet. Symbol-level monoalphabetic
      substitution where the char set spans printable ASCII.

  (D) None of the above — there are puzzles whose rule we haven't yet
      reverse-engineered. We return a best-effort guess so training-trace
      generation can still proceed.

This implementation handles (A) + (B). The fallback for (C, D) returns the
output of the example that most resembles the query (longest common
prefix/suffix), which is a weak baseline but better than nothing.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

EXAMPLE_RE = re.compile(r"^(\S[^\n=]*?)\s*=\s*([^\n]+?)\s*$", flags=re.MULTILINE)
QUERY_RE = re.compile(r"determine the result for:\s*([^\n]+)")


def parse(prompt: str) -> Tuple[List[Tuple[str, str]], str]:
    examples = []
    for lhs, rhs in EXAMPLE_RE.findall(prompt):
        if any(s in lhs for s in ("Below", "examples", "Now", "transformation")):
            continue
        examples.append((lhs, rhs))
    qm = QUERY_RE.search(prompt)
    if not qm:
        raise ValueError("could not find eq_symbols query in prompt")
    return examples, qm.group(1).strip()


# --------------------------------------------------------------------------
# (A) numeric-arithmetic detection
# --------------------------------------------------------------------------

_OPS = {
    "+":  lambda a, b: a + b,
    "-":  lambda a, b: a - b,
    "*":  lambda a, b: a * b,
    "/":  lambda a, b: a // b if b else None,
    "%":  lambda a, b: a % b if b else None,
    "abs-": lambda a, b: abs(a - b),
    "abs+": lambda a, b: a + b,
}


def _is_numeric_form(s: str) -> bool:
    """5-char string: digits with exactly one non-digit operator."""
    if len(s) != 5:
        return False
    nd = sum(1 for c in s if not c.isdigit())
    return nd == 1


def _parse_numeric(lhs: str) -> Optional[Tuple[str, str, str]]:
    if not _is_numeric_form(lhs):
        return None
    idx = next((i for i, c in enumerate(lhs) if not c.isdigit()), None)
    if idx in (None, 0, len(lhs) - 1):
        return None
    return lhs[:idx], lhs[idx], lhs[idx + 1:]


def _detect_arithmetic(examples: List[Tuple[str, str]]) -> Optional[Dict]:
    """For numeric-form puzzles, learn op_char -> (op_name, marker_pattern).

    The marker_pattern is one of:
      ""           : output is just the integer
      "<op>$"      : output is integer with the operator-char appended
      "^<op>"      : output is the operator-char prepended to the integer
    """
    parsed = []
    for lhs, rhs in examples:
        p = _parse_numeric(lhs)
        if p is None:
            return None
        a_s, op, b_s = p
        try:
            a, b = int(a_s), int(b_s)
        except ValueError:
            return None
        parsed.append((a, op, b, rhs))
    rules: Dict[str, Tuple[str, str]] = {}
    by_op: Dict[str, List[Tuple[int, int, str]]] = {}
    for a, op, b, rhs in parsed:
        by_op.setdefault(op, []).append((a, b, rhs))

    for op, items in by_op.items():
        candidate_rules: List[Tuple[str, str]] = []
        for op_name, op_fn in _OPS.items():
            for marker in ("", "$", "^"):
                ok = True
                for a, b, rhs in items:
                    val = op_fn(a, b)
                    if val is None:
                        ok = False
                        break
                    s = str(val)
                    if marker == "$":
                        s = s + op
                    elif marker == "^":
                        s = op + s
                    if s != rhs:
                        ok = False
                        break
                if ok:
                    candidate_rules.append((op_name, marker))
                    break  # take first matching marker
            if candidate_rules:
                break
        if not candidate_rules:
            return None
        rules[op] = candidate_rules[0]
    return rules


def _solve_numeric(rules: Dict[str, Tuple[str, str]], query: str) -> Optional[str]:
    p = _parse_numeric(query)
    if p is None:
        return None
    a_s, op, b_s = p
    if op not in rules:
        return None
    op_name, marker = rules[op]
    fn = _OPS[op_name]
    try:
        a, b = int(a_s), int(b_s)
    except ValueError:
        return None
    val = fn(a, b)
    if val is None:
        return None
    s = str(val)
    if marker == "$":
        s = s + op
    elif marker == "^":
        s = op + s
    return s


# --------------------------------------------------------------------------
# (B) equal-length per-position substitution
# --------------------------------------------------------------------------

def _detect_position_subst(examples: List[Tuple[str, str]]) -> Optional[Dict[str, str]]:
    if not all(len(l) == len(r) for l, r in examples):
        return None
    mapping: Dict[str, str] = {}
    for lhs, rhs in examples:
        for a, b in zip(lhs, rhs):
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    return mapping


def _apply_position_subst(mapping: Dict[str, str], query: str) -> Optional[str]:
    out = []
    for c in query:
        if c not in mapping:
            return None
        out.append(mapping[c])
    return "".join(out)


# --------------------------------------------------------------------------
# Fallback: nearest-example output
# --------------------------------------------------------------------------

def _fallback(examples: List[Tuple[str, str]], query: str) -> str:
    def lcp(a: str, b: str) -> int:
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return i
    best = examples[0][1]
    best_score = -1
    for lhs, rhs in examples:
        score = lcp(lhs, query) + lcp(lhs[::-1], query[::-1])
        if score > best_score:
            best_score = score
            best = rhs
    return best


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def _solve_inner(prompt: str) -> Tuple[str, str]:
    examples, query = parse(prompt)

    # Layer A: numeric arithmetic
    rules = _detect_arithmetic(examples)
    if rules is not None:
        ans = _solve_numeric(rules, query)
        if ans is not None:
            return ans, "arithmetic"

    # Layer B: per-position substitution
    subst = _detect_position_subst(examples)
    if subst is not None:
        ans = _apply_position_subst(subst, query)
        if ans is not None:
            return ans, "position-subst"

    # Fallback
    return _fallback(examples, query), "fallback"


def solve(prompt: str) -> str:
    return _solve_inner(prompt)[0]


def explain(prompt: str) -> str:
    examples, query = parse(prompt)
    ans, method = _solve_inner(prompt)

    lines = []
    lines.append(
        "This is a symbol-equation puzzle: a fixed transformation rule maps "
        "5-character inputs to outputs."
    )
    lines.append("")
    lines.append("Examples:")
    for lhs, rhs in examples:
        lines.append(f"  {lhs} -> {rhs}")
    lines.append("")
    if method == "arithmetic":
        rules = _detect_arithmetic(examples) or {}
        lines.append(
            "Each input is `AA <op> BB` with a single operator character. "
            "I'll learn what operation each operator denotes from the examples."
        )
        for op, (op_name, marker) in rules.items():
            lines.append(f"  '{op}' acts as {op_name} (output marker: {marker or 'none'})")
        p = _parse_numeric(query)
        if p:
            a, op, b = p
            lines.append(
                f"For the query `{query}` => operands {a}, {b} with operator '{op}': "
                f"applying the rule yields {ans}."
            )
    elif method == "position-subst":
        subst = _detect_position_subst(examples) or {}
        lines.append(
            "Each input character maps to a fixed output character; I'll build "
            "the substitution table from the examples and apply it."
        )
        items = sorted(subst.items())
        chunks = []
        for k, v in items:
            chunks.append(f"{k!r}->{v!r}")
        for i in range(0, len(chunks), 8):
            lines.append("  " + ", ".join(chunks[i:i+8]))
        lines.append(f"Applying to `{query}`: {ans}.")
    else:
        lines.append(
            "Neither a clean arithmetic rule nor a per-position substitution "
            "fully explains the examples; I'll use the closest matching example "
            "as my best guess."
        )
        lines.append(f"Best-effort guess: {ans}.")
    lines.append("")
    lines.append(f"\\boxed{{{ans}}}")
    return "\n".join(lines)
