CATEGORIES = ("bit_manip", "cipher", "numeral", "unit_conv", "gravity", "eq_symbols")


def categorize(prompt: str) -> str:
    p = prompt
    if "bit manipulation" in p:
        return "bit_manip"
    if "encryption" in p:
        return "cipher"
    if "numeral system" in p:
        return "numeral"
    if "unit conversion" in p:
        return "unit_conv"
    if "gravitational constant" in p:
        return "gravity"
    if "transformation rules is applied to equations" in p:
        return "eq_symbols"
    return "unknown"
