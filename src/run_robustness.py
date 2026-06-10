"""Run the functional robustness battery against a local LM Studio model.

Usage:
    python src/run_robustness.py
    python src/run_robustness.py --category long_context --context-tokens 32768
    python src/run_robustness.py --model qwen/qwen3-8b --output reports/qwen3.json

Run from the repository root so that reports/ resolves correctly.
"""

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import argparse
import os

from shared.cases import CATEGORIES, build_cases
from shared.lmstudio_client import DEFAULT_BASE_URL, build_client, probe
from shared.reporting import OK, ERR, format_results_table, print_header, print_status, write_report

# ─────────────────────────────────────────────────────────────
# Constants and defaults
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "base_url": DEFAULT_BASE_URL,
    "model": "qwen/qwen3-8b",
    "category": "all",
    "context_tokens": 32768,
    "output": "reports/robustness.json",
}

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LM Studio functional robustness probe.")
    parser.add_argument("--base-url", default=DEFAULTS["base_url"])
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--category", choices=("all",) + CATEGORIES, default=DEFAULTS["category"])
    parser.add_argument("--context-tokens", type=int, default=DEFAULTS["context_tokens"])
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max_tokens for every case (diagnostic use)")
    parser.add_argument("--no-think", action="store_true",
                        help="Append the Qwen3 /no_think directive to each prompt")
    parser.add_argument("--output", default=DEFAULTS["output"])
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────
def execute_cases(client, model, cases, max_tokens_override=None):
    """Run every case through probe() and collect the results."""
    results = []
    for case in cases:
        if max_tokens_override is not None:
            case.params["max_tokens"] = max_tokens_override
        result = probe(client, model, case.category, case.name, case.messages,
                       validator=case.validator, **case.params)
        results.append(result)
    return results


def ensure_output_dir(output_path):
    """Create the report's parent directory if it does not exist."""
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print_header(f"Robustness probe: {args.model}  [{args.category}]")
    print_status(OK, f"connecting to {args.base_url}")

    client = build_client(args.base_url)
    cases = build_cases(args.category, args.context_tokens, no_think=args.no_think)
    results = execute_cases(client, args.model, cases, args.max_tokens)

    format_results_table(results)

    ensure_output_dir(args.output)
    summary = write_report(
        results,
        args.output,
        metadata={
            "model": args.model,
            "category": args.category,
            "context_tokens": args.context_tokens,
            "max_tokens_override": args.max_tokens,
            "no_think": args.no_think,
        },
    )

    status = OK if summary["failed"] == 0 else ERR
    print_status(status, f"{summary['passed']}/{summary['total']} passed, report at {args.output}")


if __name__ == "__main__":
    main()
