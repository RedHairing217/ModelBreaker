"""Distractor-needle sweep for locating the model's zone of proximal development.

Builds needle-retrieval variants with a tunable number of revoked decoy codes
under an "active code" rule, runs each variant K times at non-zero temperature,
and reports the success rate per variant so the mid-band (the ZPD) is visible.

Usage:
    from shared.sweep import build_sweep, run_krate
    variants = build_sweep([0, 4, 16], context_tokens=8192)
"""

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import statistics
from dataclasses import dataclass

from shared.cases import Case, NO_THINK_DIRECTIVE, make_filler, user_message
from shared.lmstudio_client import probe

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
TRUE_CODE = "MAGENTA-4417"
CODE_LINE = "The vault access code is {code}."
REVOKED_SUFFIX = " (REVOKED)"
QUESTION = ("\n\nQuestion: Exactly one vault access code is active; every other code shown "
            "is revoked. What is the active code? Reply with only the code.")
DECOY_COLORS = ("CRIMSON", "INDIGO", "AMBER", "VIOLET", "TEAL", "SCARLET",
                "COBALT", "OLIVE", "MAROON", "CYAN", "SLATE", "BRONZE")

# ─────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────
@dataclass
class SweepResult:
    name: str
    level: int
    trials: int
    passes: int
    rate: float
    latency_median: float
    tokens_median: float
    in_band: bool
    sample_response: str = ""

# ─────────────────────────────────────────────────────────────
# Validators
# ─────────────────────────────────────────────────────────────
def expect_active_code(text):
    """Pass when the active (non-revoked) code appears in the response."""
    found = TRUE_CODE in text
    return found, "correct" if found else "wrong or missing"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def make_decoy_codes(count):
    """Return count unique decoy codes shaped like the true code."""
    return [f"{DECOY_COLORS[i % len(DECOY_COLORS)]}-{2000 + i:04d}" for i in range(count)]


def build_code_block(n_distractors):
    """Build the line block: n revoked decoys plus one active true code in the middle."""
    lines = [CODE_LINE.format(code=code) + REVOKED_SUFFIX for code in make_decoy_codes(n_distractors)]
    lines.insert(len(lines) // 2, CODE_LINE.format(code=TRUE_CODE))
    return " ".join(lines)

# ─────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────
def build_sweep(distractor_counts, context_tokens, depth=0.5, no_think=False):
    """Return a list of (n_distractors, Case) variants across the sweep."""
    filler = make_filler(int(context_tokens * 0.5))
    cut = int(len(filler) * depth)
    variants = []
    for n_distractors in distractor_counts:
        block = build_code_block(n_distractors)
        content = filler[:cut] + " " + block + " " + filler[cut:] + QUESTION
        if no_think:
            content = content + NO_THINK_DIRECTIVE
        case = Case("sweep", f"distractors_{n_distractors:02d}", user_message(content),
                    validator=expect_active_code, params={"max_tokens": 384})
        variants.append((n_distractors, [case]))
    return variants

# ─────────────────────────────────────────────────────────────
# K-rate
# ─────────────────────────────────────────────────────────────
def run_krate(client, model, level, cases, trials, temperature, band):
    """Run every instance case K times and aggregate the success rate for the level."""
    passes = 0
    attempts = 0
    latencies = []
    tokens = []
    last_response = ""
    for case in cases:
        for _ in range(trials):
            result = probe(client, model, case.category, case.name, case.messages,
                           validator=case.validator, temperature=temperature, **case.params)
            attempts += 1
            if result.passed:
                passes += 1
            if result.latency_seconds is not None:
                latencies.append(result.latency_seconds)
            if result.completion_tokens is not None:
                tokens.append(result.completion_tokens)
            last_response = result.response_text or result.error or ""
    rate = passes / attempts if attempts else 0.0
    band_low, band_high = band
    return SweepResult(
        name=cases[0].name if cases else f"level_{level}",
        level=level,
        trials=attempts,
        passes=passes,
        rate=rate,
        latency_median=round(statistics.median(latencies), 2) if latencies else 0.0,
        tokens_median=round(statistics.median(tokens), 1) if tokens else 0.0,
        in_band=band_low <= rate <= band_high,
        sample_response=last_response[:240],
    )
