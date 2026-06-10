"""Run the distractor-needle sweep to locate the model's zone of proximal development.

Each distractor count is run K times at non-zero temperature; the per-count
success rate shows where the model moves from reliable to unreliable. Counts
whose rate lands inside the target band are the ZPD.

Usage:
    python src/run_sweep.py
    python src/run_sweep.py --trials 10 --distractors 0 4 8 16 32
    python src/run_sweep.py --no-think --context-tokens 16384

Run from the repository root.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone

from shared.lmstudio_client import build_client
from shared.reporting import OK, WARN, print_header, print_status
from shared.sweep import build_sweep, run_krate

# ─────────────────────────────────────────────────────────────
# Constants and defaults
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "base_url": "http://127.0.0.1:1234/v1",
    "model": "qwen/qwen3-8b",
    "context_tokens": 8192,
    "depth": 0.5,
    "distractors": [0, 2, 4, 8, 16, 32],
    "trials": 8,
    "temperature": 0.7,
    "band_low": 0.3,
    "band_high": 0.7,
    "timeout": 120,
    "output": "reports/sweep.json",
}

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ModelBreaker distractor sweep (ZPD finder).")
    parser.add_argument("--base-url", default=DEFAULTS["base_url"])
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--context-tokens", type=int, default=DEFAULTS["context_tokens"])
    parser.add_argument("--depth", type=float, default=DEFAULTS["depth"])
    parser.add_argument("--distractors", type=int, nargs="+", default=DEFAULTS["distractors"])
    parser.add_argument("--trials", type=int, default=DEFAULTS["trials"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--band-low", type=float, default=DEFAULTS["band_low"])
    parser.add_argument("--band-high", type=float, default=DEFAULTS["band_high"])
    parser.add_argument("--no-think", action="store_true",
                        help="Append the Qwen3 /no_think directive to each prompt")
    parser.add_argument("--timeout", type=int, default=DEFAULTS["timeout"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────
def print_row(result):
    """Print one sweep row with a band marker."""
    marker = WARN if result.in_band else "    "
    print(
        f"{marker} N={result.level:<4} "
        f"{result.passes}/{result.trials:<4} rate={result.rate:<6.2f} "
        f"med_lat={result.latency_median:<7} med_tok={result.tokens_median}"
    )


def write_report(results, path, metadata):
    """Write the sweep report: metadata then one record per variant."""
    report = {
        "metadata": {**metadata, "generated_at": datetime.now(timezone.utc).isoformat()},
        "results": [asdict(result) for result in results],
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as report_file:
        json.dump(report, report_file, indent=2)

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    band = (args.band_low, args.band_high)

    print_header(f"Distractor sweep: {args.model}  (K={args.trials}, temp={args.temperature})")
    print_status(OK, f"{len(args.distractors)} counts x {args.trials} trials, "
                     f"band {args.band_low} to {args.band_high}  [{WARN} marks in-band]")

    client = build_client(args.base_url, timeout=args.timeout)
    variants = build_sweep(args.distractors, args.context_tokens, depth=args.depth, no_think=args.no_think)

    results = []
    for level, cases in variants:
        result = run_krate(client, args.model, level, cases, args.trials, args.temperature, band)
        results.append(result)
        print_row(result)

    in_band = [r.level for r in results if r.in_band]
    if in_band:
        print_status(WARN, f"ZPD candidates (counts in band): {in_band}")
    else:
        print_status(OK, "no count landed in band; sweep saturated (too easy or too hard)")

    write_report(results, args.output, metadata={
        "model": args.model,
        "context_tokens": args.context_tokens,
        "depth": args.depth,
        "trials": args.trials,
        "temperature": args.temperature,
        "band": [args.band_low, args.band_high],
        "no_think": args.no_think,
    })
    print_status(OK, f"report at {args.output}")


if __name__ == "__main__":
    main()
