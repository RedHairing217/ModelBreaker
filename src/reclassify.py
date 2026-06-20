"""Re-select a harvest report under a new band or concentration threshold, offline.

Reports store every trial's committed value and the per-problem pass rate, so the keep decision
is a pure function of stored data: widening the band or moving the concentration threshold needs
no new generations. This rewrites a report's statuses, recomputes its summary, and updates the
recorded band so the file reflects the new policy. Keep style follows the family: code-output
keeps on band or concentration, the trap families keep on concentration, and the calculus band
reports keep on the band; --keep overrides the inference.
"""

import argparse
import json
from statistics import mean

from shared.bands import BAND, in_band as band_contains
from shared.reporting import OK, print_header, print_status

ANSWER_KEYS = ("answer", "correct", "count")
CONCENTRATION_FAMILIES = ("average_speed", "novel_operator", "letter_count")


def answer_of(detail):
    for key in ANSWER_KEYS:
        if key in detail:
            return detail[key]
    return None


def concentration(item):
    cvc = item.get("committed_value_counts", {})
    trials = item.get("trials", 0) or 0
    answer = str(answer_of(item.get("detail", {})))
    wrong = {value: n for value, n in cvc.items() if value != answer}
    if not wrong or not trials:
        return None, 0.0
    value, n = max(wrong.items(), key=lambda kv: kv[1])
    return value, n / trials


def keep_style(metadata, override):
    if override != "auto":
        return override
    family = metadata.get("family", "")
    if family == "code_output":
        return "band_or_concentration"
    if family in CONCENTRATION_FAMILIES:
        return "concentration"
    return "band"


def main():
    parser = argparse.ArgumentParser(description="Re-select a harvest report under a new band/threshold")
    parser.add_argument("report")
    parser.add_argument("--band-lo", type=float, default=BAND[0])
    parser.add_argument("--band-hi", type=float, default=BAND[1])
    parser.add_argument("--capture-threshold", type=float, default=0.5)
    parser.add_argument("--keep", choices=["auto", "band", "concentration", "band_or_concentration"],
                        default="auto")
    parser.add_argument("--output", default=None, help="Defaults to overwriting the input report")
    args = parser.parse_args()
    out = args.output or args.report

    report = json.load(open(args.report))
    metadata = report.get("metadata", {})
    style = keep_style(metadata, args.keep)
    print_header(f"Reclassify {args.report}  style={style}  "
                 f"band=[{args.band_lo}, {args.band_hi}]  cap={args.capture_threshold}")

    items = list(report.get("results", [])) + list(report.get("rejected", []))
    kept, rejected = [], []
    kept_band = kept_concentrated = 0
    shares = []
    for item in items:
        passk = item.get("pass_at_k", 0.0)
        value, share = concentration(item)
        shares.append(share)
        item.setdefault("detail", {})
        item["detail"]["top_wrong_share"] = round(share, 3)
        item["detail"]["top_wrong_value"] = value
        in_band = band_contains(passk, (args.band_lo, args.band_hi))
        concentrated = share >= args.capture_threshold
        if style == "band":
            keep = in_band
        elif style == "concentration":
            keep = concentrated
        else:
            keep = in_band or concentrated
        kept_band += in_band
        kept_concentrated += concentrated
        item["in_band"] = keep
        item["status"] = "kept" if keep else "rejected"
        (kept if keep else rejected).append(item)

    report["results"] = kept
    report["rejected"] = rejected
    summary = report.setdefault("summary", {})
    summary["kept"] = len(kept)
    summary["rejected"] = len(rejected)
    summary["kept_band"] = kept_band
    summary["kept_concentrated"] = kept_concentrated
    summary["band"] = [args.band_lo, args.band_hi]
    summary["capture_threshold"] = args.capture_threshold
    summary["mean_top_wrong_share"] = round(mean(shares), 3) if shares else 0.0
    if metadata.get("band") is not None:
        metadata["band"] = [args.band_lo, args.band_hi]
    if metadata.get("capture_threshold") is not None:
        metadata["capture_threshold"] = args.capture_threshold

    json.dump(report, open(out, "w"), indent=2)
    print_status(OK, f"kept {len(kept)} (band {kept_band}, concentrated {kept_concentrated}), "
                     f"rejected {len(rejected)}  ->  {out}")


if __name__ == "__main__":
    main()
