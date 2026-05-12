from .categorize import categorize, CATEGORIES
from . import unit_conv, gravity, numeral, cipher, bit_manip, eq_symbols

SOLVERS = {
    "unit_conv": unit_conv,
    "gravity": gravity,
    "numeral": numeral,
    "cipher": cipher,
    "bit_manip": bit_manip,
    "eq_symbols": eq_symbols,
}


def solve(prompt: str):
    cat = categorize(prompt)
    mod = SOLVERS.get(cat)
    if mod is None:
        return None, cat
    return mod.solve(prompt), cat


def explain(prompt: str):
    cat = categorize(prompt)
    mod = SOLVERS.get(cat)
    if mod is None:
        return None, cat
    return mod.explain(prompt), cat
