"""Run the polynomial coefficient-extraction sweep to locate the zone of proximal development.

Each factor count is run K times at non-zero temperature; the per-count success
rate shows where the model moves from reliable to unreliable. Factor counts whose
rate lands inside the target band are the ZPD.

Usage:
    python src/run_algebra.py
    python src/run_algebra.py --factors 3 4 5 6 7 8 --instances 5 --trials 3

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

from shared.algebra import build_algebra_sweep
from shared.lmstudio_client import build_client
from shared.reporting import OK, WARN, print_header, print_status
from shared.sweep import run_krate

# ─────────────────────────────────────────────────────────────
# Constants and defaults
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "base_url": "http://127.0.0.1:1234/v1",
    "model": "qwen/qwen3-8b",
    "factors": [3, 4, 5, 6, 7, 8],
    "instances": 5,
    "trials": 3,
    "temperature": 0.7,
    "band_low": 0.3,
    "band_high": 0.7,
    "max_tokens": 4096,
    "timeout": 120,
    "output": "reports/algebra.json",
}

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ModelBreaker algebra sweep (ZPD finder).")
    parser.add_argument("--base-url", default=DEFAULTS["base_url"])
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--factors", type=int, nargs="+", default=DEFAULTS["factors"])
    parser.add_argument("--instances", type=int, default=DEFAULTS["instances"])
    parser.add_argument("--trials", type=int, default=DEFAULTS["trials"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--band-low", type=float, default=DEFAULTS["band_low"])
    parser.add_argument("--band-high", type=float, default=DEFAULTS["band_high"])
    parser.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
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
        f"{marker} factors={result.level:<4} "
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

    print_header(f"Algebra sweep: {args.model}  (K={args.trials}, temp={args.temperature})")
    print_status(OK, f"{len(args.factors)} factor counts x {args.instances} instances x {args.trials} trials, "
                     f"band {args.band_low} to {args.band_high}  [{WARN} marks in-band]")

    client = build_client(args.base_url, timeout=args.timeout)
    variants = build_algebra_sweep(args.factors, instances=args.instances,
                                   no_think=args.no_think, max_tokens=args.max_tokens)

    results = []
    for n_factors, cases in variants:
        result = run_krate(client, args.model, n_factors, cases, args.trials, args.temperature, band)
        results.append(result)
        print_row(result)

    in_band = [r.level for r in results if r.in_band]
    if in_band:
        print_status(WARN, f"ZPD candidates (factor counts in band): {in_band}")
    else:
        print_status(OK, "no factor count landed in band; sweep saturated (too easy or too hard)")

    write_report(results, args.output, metadata={
        "model": args.model,
        "factors": args.factors,
        "instances": args.instances,
        "trials": args.trials,
        "temperature": args.temperature,
        "band": [args.band_low, args.band_high],
        "max_tokens": args.max_tokens,
        "no_think": args.no_think,
    })
    print_status(OK, f"report at {args.output}")


if __name__ == "__main__":
    main()
