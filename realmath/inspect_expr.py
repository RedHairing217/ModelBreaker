import json
import random
import argparse

import sympy

import verifier as V


def to_expr(s):
    return V.parse_expr(V.rhs(V.strip_delims(s or "")))


def numeric_equiv(a, b, trials=6, tol=1e-6):
    if a is None or b is None:
        return None
    try:
        syms = sorted(a.free_symbols | b.free_symbols, key=str)
    except Exception:
        return None
    ok = 0
    seen = 0
    for _ in range(trials * 4):
        if ok >= trials:
            break
        subs = {s: random.randint(2, 9) for s in syms}
        try:
            va = complex(a.evalf(subs=subs))
            vb = complex(b.evalf(subs=subs))
        except Exception:
            continue
        if any(map(lambda z: z != z, (va.real, vb.real))):
            continue
        seen += 1
        if abs(va - vb) > tol * (1 + abs(va)):
            return False
        ok += 1
    return True if seen else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/realmath_band.jsonl")
    ap.add_argument("--label", default="too_hard")
    ap.add_argument("--tier", default="expr")
    ap.add_argument("--max", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    rows = [r for r in rows if r["tier"] == args.tier
            and (args.label == "all" or r["label"] == args.label)]

    recoverable = 0
    shown = 0
    for r in rows:
        if shown >= args.max:
            break
        shown += 1
        truth = to_expr(r["answer"])
        print("=" * 70)
        print("Q:", r["question"][:90])
        print("truth:", str(truth)[:80])
        flagged = False
        for cand in r["samples"]:
            ce = to_expr(cand)
            sym = None
            try:
                sym = bool(truth is not None and ce is not None
                           and sympy.simplify(ce - truth) == 0)
            except Exception:
                sym = False
            num = numeric_equiv(truth, ce)
            mark = ""
            if num is True and not sym:
                mark = "  <-- VERIFIER MISS"
                flagged = True
            print(f"  cand={str(cand)[:48]:48s} sym={sym} num={num}{mark}")
        if flagged:
            recoverable += 1

    print("\n" + "=" * 70)
    print(f"shown: {shown} | rows with a numeric-equiv sample the symbolic "
          f"check missed: {recoverable}")
    if recoverable:
        print("=> verifier false-negatives confirmed; harden with numeric sampling")
    else:
        print("=> no recoverable cases; these expr are genuinely failed by Qwen")


if __name__ == "__main__":
    main()
