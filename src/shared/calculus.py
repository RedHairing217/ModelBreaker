"""Derivative-at-a-point sweep for the model's zone of proximal development.

Builds a product of D linear factors and asks for f'(a) at an integer point, run
each D times at non-zero temperature via shared.sweep.run_krate. The answer is a
single integer, so the model always commits; product-rule differentiation across
many factors is error-prone, and the narration is shorter than a full expansion,
so the corruption band should open before truncation.

Usage:
    from shared.calculus import build_calculus_sweep
    variants = build_calculus_sweep([3, 5, 7])
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
SEED_BASE = 6100
DEFAULT_MAX_TOKENS = 4096
CONSTANT_CHOICES = [c for c in range(-6, 7) if c != 0]
POINT_CHOICES = [1, 2, 3, 4, 5]
INSTRUCTION = ("Let f(x) be the product of the linear factors below. Compute the first derivative "
               "f'(x) and evaluate it at x = {point}.\n{expression}\n"
               "Reply with only the integer value of f'({point}).")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def build_problem(n_factors, seed):
    """Return a printed product of n linear factors, an integer point, and the true f'(point)."""
    rng = random.Random(seed)
    constants = [rng.choice(CONSTANT_CHOICES) for _ in range(n_factors)]
    point = rng.choice(POINT_CHOICES)
    expression = "".join(f"(x {'+' if c >= 0 else '-'} {abs(c)})" for c in constants)
    symbol = sp.symbols("x")
    polynomial = 1
    for constant in constants:
        polynomial *= (symbol + constant)
    value = int(sp.diff(polynomial, symbol).subs(symbol, point))
    return expression, point, value

# ─────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────
def build_calculus_sweep(factor_counts, instances=1, no_think=False, max_tokens=DEFAULT_MAX_TOKENS):
    """Return a list of (n_factors, [Case, ...]) variants, `instances` distinct problems per count."""
    if sp is None:
        raise ImportError("sympy is required for the calculus sweep: pip install sympy")
    variants = []
    for n_factors in factor_counts:
        cases = []
        for instance in range(instances):
            expression, point, value = build_problem(n_factors, SEED_BASE + n_factors * 100 + instance)
            content = INSTRUCTION.format(point=point, expression=expression)
            if no_think:
                content = content + NO_THINK_DIRECTIVE
            cases.append(Case("calculus", f"factors_{n_factors:02d}_i{instance}", user_message(content),
                              validator=make_validator(value), params={"max_tokens": max_tokens}))
        variants.append((n_factors, cases))
    return variants
