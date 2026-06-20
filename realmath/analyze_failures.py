import json
import argparse
import collections


def mode_of(r, thresh):
    used = r.get("samples_used") or r.get("k", 1)
    degen_frac = r.get("degenerate", 0) / used if used else 0
    top = r.get("top_wrong_share", 0)
    if degen_frac >= 0.5:
        return "collapse_degenerate", degen_frac, top
    if top >= thresh:
        return "misdirection", degen_frac, top
    return "scatter", degen_frac, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/realmath_band.jsonl")
    ap.add_argument("--label", default="too_hard")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    rows = [r for r in rows if r["label"] == args.label]

    counts = collections.Counter()
    by_mode = collections.defaultdict(list)
    for r in rows:
        m, degen_frac, top = mode_of(r, args.thresh)
        counts[m] += 1
        by_mode[m].append((r, degen_frac, top))

    total = len(rows)
    misdir = counts["misdirection"]
    collapse = counts["scatter"] + counts["collapse_degenerate"]
    print(f"{args.label}: {total}")
    print(f"  misdirection (stable wrong commit): {misdir}")
    print(f"  collapse: {collapse}  "
          f"(scatter {counts['scatter']}, degenerate {counts['collapse_degenerate']})")

    print("\n--- misdirection: truth vs Qwen's committed wrong answer ---")
    for r, _, top in sorted(by_mode["misdirection"], key=lambda x: -x[2])[:args.show]:
        print(f"share={top:.2f} truth={r['truth'][:34]:34s} -> wrong={r['modal_wrong'][:40]}")

    print("\n--- scatter: distinct wrong answers per problem ---")
    for r, _, top in by_mode["scatter"][:args.show]:
        d = r.get("distinct_wrong", "?")
        print(f"distinct={d} top={top:.2f} truth={r['truth'][:40]}")

    if by_mode["collapse_degenerate"]:
        print("\n--- degenerate (no usable answer) ---")
        for r, df, _ in by_mode["collapse_degenerate"][:args.show]:
            print(f"degen_frac={df:.2f} truth={r['truth'][:40]}")


if __name__ == "__main__":
    main()
