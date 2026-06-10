"""Chained-arithmetic sweep for locating the model's zone of proximal development.

Builds left-to-right expressions with a tunable number of operations, to be run
each K times at non-zero temperature via shared.sweep.run_krate. The per-step
success rate shows where the model moves from reliable to unreliable.

Usage:
    from shared.arithmetic import build_arithmetic_sweep
    variants = build_arithmetic_sweep([2, 6, 10])
"""

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import random
import re

from shared.cases import Case, NO_THINK_DIRECTIVE, user_message

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
OPERATOR_SETS = {
    "mixed": ("+", "-", "+", "-", "*"),
    "addsub": ("+", "-"),
}
SEED_BASE = 4100
DEFAULT_MAX_TOKENS = 512
INSTRUCTION = ("Evaluate this expression strictly left to right, applying each operation in the "
               "order written (sequential, not standard operator precedence):\n{expression}\n"
               "Reply with only the final integer.")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def build_expression(steps, seed, operators=OPERATOR_SETS["mixed"]):
    """Return a left-to-right expression of `steps` operations and its true value."""
    rng = random.Random(seed)
    value = rng.randint(2, 9)
    parts = [str(value)]
    for _ in range(steps):
        operator = rng.choice(operators)
        operand = rng.randint(2, 9)
        if operator == "+":
            value += operand
        elif operator == "-":
            value -= operand
        else:
            value *= operand
        parts.append(f"{operator} {operand}")
    return " ".join(parts), value


def make_validator(target):
    """Return a validator that passes when the final integer in the reply equals target."""
    def check(text):
        numbers = re.findall(r"-?\d+", text)
        if not numbers:
            return False, "no number"
        final = numbers[-1]
        return final == str(target), "correct" if final == str(target) else f"got {final}"
    return check

# ─────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────
def build_arithmetic_sweep(step_counts, instances=1, no_think=False, max_tokens=DEFAULT_MAX_TOKENS, ops="mixed"):
    """Return a list of (steps, [Case, ...]) variants, with `instances` distinct expressions per step count."""
    operators = OPERATOR_SETS[ops]
    variants = []
    for steps in step_counts:
        cases = []
        for instance in range(instances):
            expression, answer = build_expression(steps, SEED_BASE + steps * 100 + instance, operators)
            content = INSTRUCTION.format(expression=expression)
            if no_think:
                content = content + NO_THINK_DIRECTIVE
            cases.append(Case("arith", f"steps_{steps:02d}_i{instance}", user_message(content),
                              validator=make_validator(answer), params={"max_tokens": max_tokens}))
        variants.append((steps, cases))
    return variants
