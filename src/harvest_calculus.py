"""Harvest individual calculus problems whose pass@k rate lands inside a band.

A thin entry point over shared.harvest: it samples calculus problems from the
sweep's own construction (shared.calculus), then hands them to the generic
per-problem harvest engine, which classifies every trial as correct /
wrong_complete / degenerate and keeps only clean in-band problems.

Usage:
    python src/harvest_calculus.py --self-test
    python src/harvest_calculus.py
    python src/harvest_calculus.py --factors 4 5 --const-range 4 --trials 16

Run from the repository root so that reports/ resolves correctly.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import argparse
import random

from shared.bands import BAND
from shared.calculus import (CONSTANT_CHOICES, POINT_CHOICES, classify_trial,
                             sample_harvest_problems, worked_example)
from shared.harvest import harvest, write_report
from shared.lmstudio_client import build_client
from shared.reporting import ERR, OK, WARN, print_header, print_status

# ─────────────────────────────────────────────────────────────
# Constants and defaults
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "base_url": "http://127.0.0.1:1234/v1",
    "model": "qwen/qwen3-8b",
    "factors": [4, 5, 6, 7, 8],
    "const_range": None,
    "point_max": None,
    "pool": 240,
    "prescreen_trials": 3,
    "trials": 16,
    "temperature": 0.7,
    "band_low": BAND[0],
    "band_high": BAND[1],
    "max_tokens": 4096,
    "degenerate_threshold": 0.1,
    "timeout": 360,
    "seed": 0,
    "output": "reports/calculus_harvest.json",
}

# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────
def print_row(result):
    """Print one kept-problem row with a band marker."""
    detail = result.detail
    breakdown = f"{result.correct}/{result.wrong_complete}/{result.degenerate}"
    print(
        f"{WARN} n={detail['n_factors']:<2} A={detail['point']:<3} "
        f"pass@{result.trials}={result.pass_at_k:<6.2f} "
        f"c/wc/deg={breakdown:<9} deg_frac={result.degenerate_fraction:<5.2f} "
        f"med_lat={result.latency_median:<7} med_tok={result.tokens_median}"
    )

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ModelBreaker per-problem calculus harvester.")
    parser.add_argument("--base-url", default=DEFAULTS["base_url"])
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--factors", type=int, nargs="+", default=DEFAULTS["factors"])
    parser.add_argument("--const-range", type=int, default=DEFAULTS["const_range"],
                        help="Sample constants from [-R, R] minus zero; default uses the sweep pool (-6..6)")
    parser.add_argument("--point-max", type=int, default=DEFAULTS["point_max"],
                        help="Sample the evaluation point from [1, P]; default uses the sweep pool (1..5)")
    parser.add_argument("--pool", type=int, default=DEFAULTS["pool"],
                        help="Target size of the deduped candidate pool")
    parser.add_argument("--prescreen-trials", type=int, default=DEFAULTS["prescreen_trials"])
    parser.add_argument("--trials", type=int, default=DEFAULTS["trials"], help="Full k for survivors")
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--band-low", type=float, default=DEFAULTS["band_low"])
    parser.add_argument("--band-high", type=float, default=DEFAULTS["band_high"])
    parser.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
    parser.add_argument("--degenerate-threshold", type=float, default=DEFAULTS["degenerate_threshold"])
    parser.add_argument("--no-think", action="store_true",
                        help="Append the Qwen3 /no_think directive; tags kept problems as no_think")
    parser.add_argument("--frequency-penalty", type=float, default=0.0,
                        help="Penalize repeated tokens to break reasoning loops; tag as a separate config")
    parser.add_argument("--stop", nargs="+", default=None,
                        help="Stop strings passed to the model (use with care; can cut a trace before it commits)")
    parser.add_argument("--disable-loop-detect", action="store_true",
                        help="Turn off the streaming loop detector even on think passes")
    parser.add_argument("--loop-ratio", type=float, default=0.15,
                        help="Compression-ratio threshold below which a think trace is judged to be looping")
    parser.add_argument("--timeout", type=int, default=DEFAULTS["timeout"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--self-test", action="store_true",
                        help="Run the oracle known-answer test and exit without touching the model")
    return parser.parse_args()


def resolve_pools(args):
    """Pick constant and point pools, defaulting to the sweep's own pools."""
    if args.const_range is not None:
        const_pool = [c for c in range(-args.const_range, args.const_range + 1) if c != 0]
    else:
        const_pool = CONSTANT_CHOICES
    point_pool = list(range(1, args.point_max + 1)) if args.point_max is not None else POINT_CHOICES
    return const_pool, point_pool

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    worked_example()
    print_status(OK, "oracle self-test passed: f'(2) = 4 for roots {4, 4, -2, 1}")
    if args.self_test:
        return

    thinking_mode = "no_think" if args.no_think else "think"
    const_pool, point_pool = resolve_pools(args)
    rng = random.Random(args.seed)

    print_header(f"Calculus harvest: {args.model}  (k={args.trials}, temp={args.temperature}, {thinking_mode})")
    print_status(OK, f"band {args.band_low} to {args.band_high}, "
                     f"degenerate threshold {args.degenerate_threshold}  [{WARN} marks kept]")

    problems = sample_harvest_problems(rng, args.factors, const_pool, point_pool, args.pool,
                                       no_think=args.no_think, max_tokens=args.max_tokens)
    print_status(OK, f"candidate pool: {len(problems)} distinct problems after dedupe")

    metadata = {
        "model": args.model,
        "base_url": args.base_url,
        "factors": args.factors,
        "const_pool": const_pool,
        "point_pool": point_pool,
        "pool_target": args.pool,
        "prescreen_trials": args.prescreen_trials,
        "trials": args.trials,
        "temperature": args.temperature,
        "band": [args.band_low, args.band_high],
        "max_tokens": args.max_tokens,
        "degenerate_threshold": args.degenerate_threshold,
        "thinking_mode": thinking_mode,
        "frequency_penalty": args.frequency_penalty,
        "stop": args.stop,
        "seed": args.seed,
    }

    extra_params = {}
    if args.frequency_penalty:
        extra_params["frequency_penalty"] = args.frequency_penalty
    if args.stop:
        extra_params["stop"] = args.stop

    detect_loops = (thinking_mode == "think") and not args.disable_loop_detect
    loop_params = {"ratio_threshold": args.loop_ratio} if detect_loops else None
    metadata["loop_detect"] = detect_loops
    if detect_loops:
        metadata["loop_ratio_threshold"] = args.loop_ratio

    def on_progress(done, total, survivors, elapsed):
        rate = elapsed / done if done else 0.0
        remaining = (total - done) * rate
        print_status(OK, f"prescreen {done}/{total}, {survivors} survivors, "
                         f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s left at {rate:.1f}s/problem")

    def checkpoint(kept, rejected, summary):
        write_report(kept, summary, args.output, metadata, rejected=rejected)

    client = build_client(args.base_url, timeout=args.timeout)
    kept, rejected, summary = harvest(
        client, args.model, problems,
        classify=classify_trial,
        trials=args.trials,
        prescreen_trials=args.prescreen_trials,
        band=(args.band_low, args.band_high),
        degenerate_threshold=args.degenerate_threshold,
        temperature=args.temperature,
        thinking_mode=thinking_mode,
        on_keep=print_row,
        on_progress=on_progress,
        checkpoint=checkpoint,
        extra_params=extra_params,
        detect_loops=detect_loops,
        loop_params=loop_params,
    )
    print_status(OK, f"prescreen at k={args.prescreen_trials}: {summary['prescreen_survivors']} survivors "
                     f"({summary['prescreen_always_pass']} always-pass, "
                     f"{summary['prescreen_always_fail']} always-fail dropped)")

    if kept:
        print_status(WARN, f"kept {len(kept)} in-band problems with clean correct/wrong-complete splits")
    else:
        print_status(ERR, "no problem landed in band with a low enough degenerate fraction")
    print_status(OK, f"retained {len(rejected)} rejected problems "
                     f"({summary['rejected_out_of_band']} out-of-band, "
                     f"{summary['rejected_degenerate']} degenerate)")

    write_report(kept, summary, args.output, metadata, rejected=rejected)
    print_status(OK, f"report at {args.output}")


if __name__ == "__main__":
    main()
