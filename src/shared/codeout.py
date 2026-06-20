"""Code-output prediction family: trace a small program, predict its stdout.

The difficulty here is execution tracing, an axis distinct from the arithmetic of the
calculus task and the tokenization of letter counting, so it should generate bands the
other families cannot. The programs are deterministic, import-free, and bounded, and the
ground truth is obtained by executing each one rather than re-deriving it, so the label
cannot disagree with what Python actually prints. Every program prints a single integer,
which lets grading reuse a tolerant last-integer extractor and sidesteps the quoting and
whitespace ambiguity that string outputs would invite.
"""

import hashlib
import random
import re
import subprocess
import sys

from shared.cases import NO_THINK_DIRECTIVE
from shared.harvest import CORRECT, DEGENERATE, WRONG_COMPLETE, HarvestProblem

DEFAULTS = {"max_tokens": 1024, "exec_timeout": 5, "max_size": 4}

CODE_OUTPUT_PROMPT = (
    "Here is a Python program:\n\n```python\n{code}\n```\n\n"
    "What integer does it print to stdout?\nReply with only that integer."
)


def extract_output(text):
    """The committed integer: last integer on the final non-empty line, else anywhere."""
    if not text:
        return None
    stripped = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I).strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    for candidate in ([lines[-1]] if lines else []) + [stripped]:
        nums = re.findall(r"-?\d+", candidate.replace(",", ""))
        if nums:
            return int(nums[-1])
    return None


def run_snippet(code, timeout):
    """Execute a generated snippet in an isolated subprocess; return stripped stdout or None."""
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", code],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _accumulate(rng, size):
    n = rng.randint(4 + size, 7 + size)
    c = rng.randint(2, 5)
    start = rng.randint(1, 4)
    return ("total = 0\n"
            f"for i in range({start}, {start + n}):\n"
            f"    total += i * {c}\n"
            "print(total)")


def _modular_fold(rng, size):
    x = rng.randint(1, 9)
    m = rng.randint(2, 6)
    a = rng.randint(1, 9)
    p = rng.choice([7, 11, 13, 17])
    return (f"x = {x}\n"
            f"for _ in range({4 + size}):\n"
            f"    x = (x * {m} + {a}) % {p}\n"
            "print(x)")


def _conditional_count(rng, size):
    hi = rng.randint(20 + 5 * size, 30 + 5 * size)
    k = rng.randint(3, 6)
    r = rng.randint(0, k - 1)
    return ("count = 0\n"
            f"for i in range(1, {hi}):\n"
            f"    if i % {k} == {r}:\n"
            "        count += 1\n"
            "print(count)")


def _nested_sum(rng, size):
    a = rng.randint(3, 4 + size)
    b = rng.randint(3, 4 + size)
    return ("s = 0\n"
            f"for i in range(1, {a}):\n"
            f"    for j in range(1, {b}):\n"
            "        s += i * j\n"
            "print(s)")


def _string_metric(rng, size):
    unit = rng.choice(["ab", "xyz", "aab", "ba"])
    target = rng.choice(sorted(set(unit)))
    return ("s = ''\n"
            f"for i in range({3 + size}):\n"
            f"    s += {unit!r}\n"
            f"print(s.count({target!r}))")


def _list_reduce(rng, size):
    n = rng.randint(5 + size, 8 + size)
    m = rng.randint(2, 4)
    off = rng.randint(0, 5)
    mod = rng.randint(7, 13)
    op = rng.choice(["max", "sum", "min"])
    return (f"xs = [(i * {m} + {off}) % {mod} for i in range({n})]\n"
            f"print({op}(xs))")


SHAPES = {
    "accumulate": _accumulate,
    "modular_fold": _modular_fold,
    "conditional_count": _conditional_count,
    "nested_sum": _nested_sum,
    "string_metric": _string_metric,
    "list_reduce": _list_reduce,
}


def build_code_output_problem(code, answer, shape, size, no_think, max_tokens):
    content = CODE_OUTPUT_PROMPT.format(code=code)
    if no_think:
        content += NO_THINK_DIRECTIVE
    digest = hashlib.md5(code.encode()).hexdigest()[:6]
    return HarvestProblem(
        label=f"code_{shape}_{size}_{digest}",
        messages=[{"role": "user", "content": content}],
        answer=answer,
        key=f"code_output:{digest}",
        max_tokens=max_tokens,
        detail={"family": "code_output", "shape": shape, "size": size,
                "code": code, "answer": answer},
    )


def classify_code_output(finish_reason, text, problem):
    """Bucket a trial and return the committed integer; truth was set by execution."""
    committed = extract_output(text)
    if finish_reason in (None, "error", "length", "loop") or committed is None:
        return DEGENERATE, None
    if committed == problem.answer:
        return CORRECT, committed
    return WRONG_COMPLETE, committed


def sample_code_output(pool, no_think, max_tokens, seed, max_size, exec_timeout):
    """Build a deduped pool of executable single-integer-output programs across shapes and sizes."""
    rng = random.Random(seed)
    names = sorted(SHAPES)
    problems, seen, attempts = [], set(), 0
    while len(problems) < pool and attempts < pool * 200:
        attempts += 1
        shape = rng.choice(names)
        size = rng.randint(0, max_size)
        code = SHAPES[shape](rng, size)
        if code in seen:
            continue
        out = run_snippet(code, exec_timeout)
        if out is None or not re.fullmatch(r"-?\d+", out):
            continue
        seen.add(code)
        problems.append(build_code_output_problem(code, int(out), shape, size, no_think, max_tokens))
    return problems
