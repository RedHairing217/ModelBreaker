import json
import argparse

import sympy

import verifier as V

JUNK = ["mathrm", "mathbb", "mathcal", "mathbf", "mathsf", "operatorname",
        "Bigl", "Bigr", "bigl", "bigr", "widetilde", "widehat", "cdots",
        "ldots", "efsub", "displaystyle", "text", "boldsymbol", "mathfrak",
        "mathscr", "langle", "rangle", "nabla", "pmod"]
REL = ["<", ">", "\\le", "\\ge", "\\leq", "\\geq", "\\in", "\\equiv",
       "\\subset", "\\supset", "\\neq", "\\to", "\\mapsto"]


def truth_bad(answer, truth):
    if truth is None:
        return "unparseable"
    s = str(truth)
    if any(j in s for j in JUNK):
        return "junk"
    try:
        if truth.is_Symbol:
            return "bare_symbol"
    except Exception:
        pass
    a = V.strip_delims(answer)
    if any(t in a for t in REL):
        return "relation"
    if isinstance(truth, (sympy.core.relational.Relational,
                          sympy.logic.boolalg.Boolean)):
        return "relation"
    return None


def mw_mismatch(mw):
    if not mw:
        return False
    if any(j in mw for j in JUNK) or any(t in mw for t in REL):
        return True
    return V.parse_expr(V.rhs(V.strip_delims(mw))) is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/k8_corpus.jsonl")
    ap.add_argument("--output", default="reports/realmath_corpus.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    kept = []
    dropped = {}
    for r in rows:
        tier, truth = V.classify(r["answer"])
        bad = truth_bad(r["answer"], truth)
        if bad:
            dropped[bad] = dropped.get(bad, 0) + 1
            continue
        if r["label"] == "misdirection" and mw_mismatch(r.get("modal_wrong", "")):
            dropped["kind_mismatch"] = dropped.get("kind_mismatch", 0) + 1
            continue
        kept.append(r)

    with open(args.output, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    import collections
    klabels = collections.Counter(r["label"] for r in kept)
    print(f"input: {len(rows)}  kept: {len(kept)}  -> {args.output}")
    print("kept by label:", dict(klabels))
    print("dropped:", dropped)


if __name__ == "__main__":
    main()
