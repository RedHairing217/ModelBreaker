"""Aggregate harvested problems from every report into one labeled corpus.

Reads the JSON reports produced by every harvester, takes all evaluated problems (results and
rejected alike, since the band is re-derived here from stored trials rather than trusted from each
report), assigns each a failure-mode label, reconstructs the prompt from the stored detail,
dedupes, and splits on the pass band defined in shared/bands.py: problems with pass inside the band
go to the primary corpus of inconsistently-solved problems; those below the band that commit a
dominant wrong value go to a separate confident-wrong file; the rest are left out. The band is the
single filter, so widening or narrowing it is one edit in shared/bands.py.
"""

import argparse
import glob
import json
import os
from collections import Counter

from shared.bands import BAND, in_band
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
        elif family == "code_output":
            from shared.codeout import build_code_output_problem
            problem = build_code_output_problem(detail["code"], detail.get("answer"),
                                                detail.get("shape"), detail.get("size"), False, 0)
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
    parser.add_argument("--output", default="reports/corpus.jsonl",
                        help="Primary corpus of inconsistently-solved problems, pass in band")
    parser.add_argument("--confident-output", default="reports/corpus_confident_wrong.jsonl",
                        help="Reliably-wrong concentrated problems, pass below band, kept separate")
    parser.add_argument("--band-lo", type=float, default=BAND[0])
    parser.add_argument("--band-hi", type=float, default=BAND[1])
    parser.add_argument("--capture-threshold", type=float, default=0.5,
                        help="Min top wrong-value share for a below-band problem to count as confident-wrong")
    args = parser.parse_args()
    band = (args.band_lo, args.band_hi)

    print_header(f"Corpus aggregation from {args.reports_dir} (band {band[0]} to {band[1]})")
    paths = sorted(glob.glob(os.path.join(args.reports_dir, "*.json")))
    records, skipped = [], 0
    for path in paths:
        try:
            report = json.load(open(path))
        except (ValueError, OSError):
            skipped += 1
            continue
        if "results" not in report:
            skipped += 1
            continue
        metadata = report.get("metadata", {})
        items = list(report.get("results", [])) + list(report.get("rejected", []))
        for item in items:
            records.append(record_from_item(item, metadata, os.path.basename(path)))

    best = {}
    for record in records:
        key = (record["thinking_mode"], record["prompt"] or f"{record['family']}|{record['label']}")
        if key not in best or record["trials"] > best[key]["trials"]:
            best[key] = record
    corpus = sorted(best.values(), key=lambda r: (r["failure_mode"], r["family"], str(r["label"])))

    # Membership is decided by the pass band, the single filter in shared/bands.py. A problem in
    # band is inconsistent and goes to the primary corpus; one below band that commits a dominant
    # wrong value is confident-wrong; everything else (reliably solved, dispersed always-fail,
    # think-mode non-termination) is left out.
    primary, confident, dropped = [], [], 0
    for record in corpus:
        passk = record["pass_at_k"]
        if passk is None or record["degenerate_fraction"] >= 0.5:
            dropped += 1
        elif in_band(passk, band):
            primary.append(record)
        elif passk < band[0] and record["top_wrong_share"] >= args.capture_threshold:
            confident.append(record)
        else:
            dropped += 1

    def write_jsonl(path, rows):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as handle:
            for record in rows:
                handle.write(json.dumps(record) + "\n")

    write_jsonl(args.output, primary)
    write_jsonl(args.confident_output, confident)

    def families(rows):
        return ", ".join(f"{f}={n}" for f, n in Counter(r["family"] for r in rows).most_common())

    no_prompt = sum(1 for r in (primary + confident) if not r["prompt"])
    print_status(OK, f"read {len(paths) - skipped} reports, {len(records)} problems, {len(corpus)} after dedupe")
    print_status(OK, f"primary (in band) -> {len(primary)} [{families(primary)}] : {args.output}")
    print_status(OK, f"confident-wrong (below band) -> {len(confident)} [{families(confident)}] : {args.confident_output}")
    print_status(OK, f"left out (solved, dispersed-fail, non-termination) -> {dropped}")
    if no_prompt:
        print_status(WARN, f"{no_prompt} records lack a reconstructed prompt; detail is still present")


if __name__ == "__main__":
    main()
