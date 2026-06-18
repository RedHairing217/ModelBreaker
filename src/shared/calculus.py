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
import re

from shared.arithmetic import make_validator
from shared.cases import Case, NO_THINK_DIRECTIVE, user_message
from shared.harvest import DEGENERATE, CORRECT, WRONG_COMPLETE, HarvestProblem

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

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
INT_TOKEN = re.compile(r"-?\d+")
GROUPING_COMMA = re.compile(r"(?<=\d),(?=\d)")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def format_expression(constants):
    """Render constants as a product of (x + c) factors, folding the sign in."""
    return "".join(f"(x {'+' if c >= 0 else '-'} {abs(c)})" for c in constants)


def build_problem(n_factors, seed):
    """Return a printed product of n linear factors, an integer point, and the true f'(point)."""
    rng = random.Random(seed)
    constants = [rng.choice(CONSTANT_CHOICES) for _ in range(n_factors)]
    point = rng.choice(POINT_CHOICES)
    expression = format_expression(constants)
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

# ─────────────────────────────────────────────────────────────
# Harvest support (per-problem, no sympy on the hot path)
# ─────────────────────────────────────────────────────────────
def oracle(constants, point):
    """Pure-Python f'(point) for f(x) = product of (x + c). Root of (x + c) is -c."""
    roots = [-c for c in constants]
    total = 0
    for i in range(len(roots)):
        term = 1
        for j in range(len(roots)):
            if j != i:
                term *= point - roots[j]
        total += term
    return total


def worked_example():
    """Known-answer check: roots {4, 4, -2, 1} are constants {-4, -4, 2, -1}, f'(2) = 4."""
    value = oracle([-4, -4, 2, -1], 2)
    if value != 4:
        raise AssertionError(f"calculus oracle self-test failed: expected 4, got {value}")
    return value


def extract_integer(text):
    """Strip any think block, drop digit-grouping commas, return the last integer."""
    stripped = THINK_BLOCK.sub("", text or "")
    stripped = GROUPING_COMMA.sub("", stripped)
    matches = INT_TOKEN.findall(stripped)
    if not matches:
        return None
    return int(matches[-1])


def classify_trial(finish_reason, text, problem):
    """Sort one trial into (bucket, committed_value) against the oracle.

    committed_value is the integer the model committed, or None when it committed
    nothing parseable (a degenerate trial).
    """
    if finish_reason != "stop":
        return DEGENERATE, None
    value = extract_integer(text)
    if value is None:
        return DEGENERATE, None
    return (CORRECT if value == problem.answer else WRONG_COMPLETE), value


def make_harvest_problem(constants, point, no_think, max_tokens):
    """Build one HarvestProblem from a constant list and an evaluation point."""
    expression = format_expression(constants)
    content = INSTRUCTION.format(point=point, expression=expression)
    if no_think:
        content = content + NO_THINK_DIRECTIVE
    answer = oracle(constants, point)
    return HarvestProblem(
        label=f"n{len(constants):02d}_p{point}",
        messages=user_message(content),
        answer=answer,
        key=(tuple(sorted(constants)), point),
        max_tokens=max_tokens,
        detail={
            "n_factors": len(constants),
            "expression": expression,
            "constants": list(constants),
            "point": point,
            "answer": answer,
        },
    )


def sample_harvest_problems(rng, factor_counts, const_pool, point_pool, size,
                            no_think=False, max_tokens=DEFAULT_MAX_TOKENS):
    """Sample up to `size` distinct calculus problems for the harvest engine."""
    problems = []
    keys = set()
    attempts = 0
    while len(problems) < size and attempts < size * 20:
        attempts += 1
        n_factors = rng.choice(factor_counts)
        constants = [rng.choice(const_pool) for _ in range(n_factors)]
        point = rng.choice(point_pool)
        key = (tuple(sorted(constants)), point)
        if key in keys:
            continue
        keys.add(key)
        problems.append(make_harvest_problem(constants, point, no_think, max_tokens))
    return problems
