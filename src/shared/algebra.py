"""Polynomial coefficient-extraction sweep for the model's zone of proximal development.

Builds products of K linear factors and asks for the coefficient of x^2, run each
K times at non-zero temperature via shared.sweep.run_krate. The answer is a single
integer, so the model always commits; combining cross-terms across many factors is
error-prone, so failures present as wrong coefficients rather than blanks.

Usage:
    from shared.algebra import build_algebra_sweep
    variants = build_algebra_sweep([3, 5, 7])
"""

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import random

from shared.arithmetic import make_validator
from shared.cases import Case, NO_THINK_DIRECTIVE, user_message

try:
    import sympy as sp
except ImportError:
    sp = None

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
SEED_BASE = 5100
DEFAULT_MAX_TOKENS = 4096
COEFF_DEGREE = 2
CONSTANT_CHOICES = [c for c in range(-9, 10) if c != 0]
INSTRUCTION = ("Expand the following product of linear factors and report the coefficient of x^2.\n"
               "{expression}\nReply with only the integer coefficient.")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def build_product(n_factors, seed):
    """Return a printed product of n linear factors and the true coefficient of x^2."""
    rng = random.Random(seed)
    constants = [rng.choice(CONSTANT_CHOICES) for _ in range(n_factors)]
    expression = "".join(f"(x {'+' if c >= 0 else '-'} {abs(c)})" for c in constants)
    symbol = sp.symbols("x")
    polynomial = 1
    for constant in constants:
        polynomial *= (symbol + constant)
    coefficient = int(sp.expand(polynomial).coeff(symbol, COEFF_DEGREE))
    return expression, coefficient

# ─────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────
def build_algebra_sweep(factor_counts, instances=1, no_think=False, max_tokens=DEFAULT_MAX_TOKENS):
    """Return a list of (n_factors, [Case, ...]) variants, `instances` distinct products per count."""
    if sp is None:
        raise ImportError("sympy is required for the algebra sweep: pip install sympy")
    variants = []
    for n_factors in factor_counts:
        cases = []
        for instance in range(instances):
            expression, coefficient = build_product(n_factors, SEED_BASE + n_factors * 100 + instance)
            content = INSTRUCTION.format(expression=expression)
            if no_think:
                content = content + NO_THINK_DIRECTIVE
            cases.append(Case("algebra", f"factors_{n_factors:02d}_i{instance}", user_message(content),
                              validator=make_validator(coefficient), params={"max_tokens": max_tokens}))
        variants.append((n_factors, cases))
    return variants
