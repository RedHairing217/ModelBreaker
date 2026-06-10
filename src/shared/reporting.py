"""Console and JSON reporting helpers shared by all probe scripts.

Usage:
    from shared.reporting import print_header, print_status, OK, ERR, WARN
    from shared.reporting import format_results_table, write_report
"""

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import json
import statistics
from dataclasses import asdict
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
OK = "[OK]"
ERR = "[ERR]"
WARN = "[WARN]"

_SEPARATOR = "─" * 72

# ─────────────────────────────────────────────────────────────
# Console output
# ─────────────────────────────────────────────────────────────
def print_header(title):
    """Print a section header bracketed by separators."""
    print(_SEPARATOR)
    print(title)
    print(_SEPARATOR)


def print_status(prefix, message):
    """Print a single status line with a consistent prefix."""
    print(f"{prefix} {message}")


def format_results_table(results):
    """Print one row per result with aligned columns."""
    print(f"{'CATEGORY':<13} {'TEST':<26} {'PASS':<6} {'LAT':<8} {'FINISH':<10} {'TOK':<6} NOTE")
    for result in results:
        detail = result.note or result.error
        print(
            f"{result.category:<13} {result.name:<26} {str(result.passed):<6} "
            f"{str(result.latency_seconds):<8} {str(result.finish_reason):<10} "
            f"{str(result.completion_tokens):<6} {detail}"
        )

# ─────────────────────────────────────────────────────────────
# JSON report
# ─────────────────────────────────────────────────────────────
def build_summary(results):
    """Return aggregate counts and latency stats for a result set."""
    latencies = [r.latency_seconds for r in results if r.latency_seconds is not None]
    failures = [r for r in results if not r.passed]
    return {
        "total": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "latency_median_seconds": round(statistics.median(latencies), 2) if latencies else None,
        "latency_max_seconds": max(latencies) if latencies else None,
    }


def write_report(results, path, metadata):
    """Write a consistent JSON report: metadata, summary, then results."""
    report = {
        "metadata": {**metadata, "generated_at": datetime.now(timezone.utc).isoformat()},
        "summary": build_summary(results),
        "results": [asdict(result) for result in results],
    }
    with open(path, "w") as report_file:
        json.dump(report, report_file, indent=2)
    return report["summary"]
