"""Aggregate harvested problems from every report into one labeled corpus.

Reads the JSON reports produced by the calculus harvester and the trap harvester, pulls the
kept problems (or all evaluated problems with --include all), assigns each a failure-mode label
from its statistics, reconstructs the prompt from the stored detail, dedupes, and writes one
JSONL dataset. The point is a single place where the inconsistently-solved problems collected
across families live together, each tagged with how the model fails on it.
"""

import argparse
import glob
import json
import os
from collections import Counter

from shared.reporting import OK, WARN, print_header, print_status

ANSWER_KEYS = ("answer", "correct", "count")


def answer_of(detail):
    for key in ANSWER_KEYS:
        if key in detail:
            return detail[key]
    return None


def family_of(detail, metadata):
    return detail.get("family") or ("calculus" if "constants" in detail else
                                    metadata.get("family", "unknown"))


def reconstruct_prompt(family, detail):
    """Rebuild the user prompt from stored detail using the real builders; None on any mismatch."""
    try:
        if family == "letter_count":
            from shared.traps import build_letter_count_problem
            problem = build_letter_count_problem(detail["word"], detail["letter"], False, 0)
        elif family == "average_speed":
            from shared.traps import build_average_speed_problem
            problem = build_average_speed_problem(detail["a"], detail["b"], False, 0)
        elif family == "novel_operator":
            from shared.traps import build_novel_operator_problem
            problem = build_novel_operator_problem(detail["operands"], False, 0)
        elif family == "calculus":
            from shared.calculus import make_harvest_problem
            problem = make_harvest_problem(detail["constants"], detail["point"], False, 4096)
        else:
            return None
        return problem.messages[0]["content"]
    except Exception:
        return None


def classify_mode(correct_share, top_wrong_share, degenerate_fraction, breakdown):
    """Label the failure mode from the per-problem statistics."""
    if degenerate_fraction >= 0.5:
        dominant = max(breakdown, key=breakdown.get) if breakdown else ""
        if dominant == "truncated":
            return "truncation"
        if dominant in ("loop", "empty", "error_or_timeout"):
            return "non_termination"
        return "degenerate"
    if top_wrong_share >= 0.5:
        return "confident_wrong"
    if correct_share >= 1.0:
        return "solved"
    return "scatter"


def record_from_item(item, metadata, source):
    detail = item.get("detail", {})
    family = family_of(detail, metadata)
    answer = answer_of(detail)
    trials = item.get("trials", 0) or 0
    cvc = item.get("committed_value_counts", {})
    correct = item.get("correct", 0)
    correct_share = correct / trials if trials else 0.0
    wrong = {value: n for value, n in cvc.items() if value != str(answer)}
    if wrong and trials:
        top_wrong_value, top_wrong_n = max(wrong.items(), key=lambda kv: kv[1])
        top_wrong_share = top_wrong_n / trials
    else:
        top_wrong_value, top_wrong_share = None, 0.0
    degenerate_fraction = item.get("degenerate_fraction", 0.0)
    breakdown = item.get("degenerate_breakdown", {})
    mode = classify_mode(correct_share, top_wrong_share, degenerate_fraction, breakdown)
    return {
        "source": source,
        "model": metadata.get("model"),
        "family": family,
        "thinking_mode": item.get("thinking_mode") or metadata.get("thinking_mode"),
        "label": item.get("label"),
        "prompt": reconstruct_prompt(family, detail),
        "answer": answer,
        "failure_mode": mode,
        "trials": trials,
        "pass_at_k": item.get("pass_at_k"),
        "correct": correct,
        "wrong_complete": item.get("wrong_complete"),
        "degenerate": item.get("degenerate"),
        "degenerate_fraction": degenerate_fraction,
        "top_wrong_share": round(top_wrong_share, 3),
        "top_wrong_value": top_wrong_value,
        "dispersion": len(cvc),
        "committed_value_counts": cvc,
        "detail": detail,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate harvested problems into one labeled corpus")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--output", default="reports/corpus.jsonl")
    parser.add_argument("--include", choices=["kept", "all"], default="kept",
                        help="kept uses each report's selected problems; all adds the rejected ones too")
    parser.add_argument("--drop-solved", action="store_true",
                        help="Exclude problems the model always solved (only relevant with --include all)")
    args = parser.parse_args()

    print_header(f"Corpus aggregation from {args.reports_dir} (include={args.include})")
    paths = sorted(glob.glob(os.path.join(args.reports_dir, "*.json")))
    records, skipped = [], 0
    for path in paths:
        if os.path.abspath(path) == os.path.abspath(args.output):
            continue
        try:
            report = json.load(open(path))
        except (ValueError, OSError):
            skipped += 1
            continue
        if "results" not in report:
            skipped += 1
            continue
        metadata = report.get("metadata", {})
        items = list(report.get("results", []))
        if args.include == "all":
            items += list(report.get("rejected", []))
        for item in items:
            records.append(record_from_item(item, metadata, os.path.basename(path)))

    best = {}
    for record in records:
        if args.drop_solved and record["failure_mode"] == "solved":
            continue
        key = (record["thinking_mode"], record["prompt"] or f"{record['family']}|{record['label']}")
        if key not in best or record["trials"] > best[key]["trials"]:
            best[key] = record
    corpus = sorted(best.values(), key=lambda r: (r["failure_mode"], r["family"], str(r["label"])))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        for record in corpus:
            handle.write(json.dumps(record) + "\n")

    by_mode = Counter(r["failure_mode"] for r in corpus)
    by_family = Counter(r["family"] for r in corpus)
    no_prompt = sum(1 for r in corpus if not r["prompt"])
    print_status(OK, f"read {len(paths) - skipped} reports, {len(records)} problems, {len(corpus)} after dedupe")
    print_status(OK, "by failure mode: " + ", ".join(f"{m}={n}" for m, n in by_mode.most_common()))
    print_status(OK, "by family: " + ", ".join(f"{f}={n}" for f, n in by_family.most_common()))
    if no_prompt:
        print_status(WARN, f"{no_prompt} records lack a reconstructed prompt; detail is still present")
    print_status(OK, f"wrote {args.output}")


if __name__ == "__main__":
    main()
