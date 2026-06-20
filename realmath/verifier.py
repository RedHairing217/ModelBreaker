import re
import json
import argparse

import sympy
from sympy import sympify, simplify, Tuple, nsimplify

try:
    from sympy.parsing.latex import parse_latex
    HAVE_LATEX = True
except Exception:
    HAVE_LATEX = False


PROSE_RE = re.compile(r"[A-Za-z]{3,}")
TUPLE_RE = re.compile(r"^\(\s*[-+0-9.,\s/]+\)$")


def strip_delims(s):
    s = (s or "").strip()
    for a, b in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)")):
        if s.startswith(a) and s.endswith(b):
            s = s[len(a):-len(b)].strip()
    s = s.strip("$").strip()
    return s


def english_words(s):
    s2 = re.sub(r"\\text\{[^}]*\}", " ", s)
    s2 = re.sub(r"\\[a-zA-Z]+", " ", s2)
    return PROSE_RE.findall(s2)


def has_cases(s):
    return "\\begin{cases}" in s or "\\begin{dcases" in s


def multi_part(s):
    parts = re.findall(r"\(\s*[a-d]\s*\)", s)
    return len(parts) >= 2


def rhs(s):
    s2 = re.sub(r"\\(leq|geq|neq|equiv|approx|sim|le|ge)\b", " ", s)
    if "=" in s2:
        tail = s2.rsplit("=", 1)[1].strip()
        if tail:
            return tail
    return s


def parse_expr(s):
    s = s.strip()
    if HAVE_LATEX:
        try:
            return parse_latex(s)
        except Exception:
            pass
    try:
        return sympify(s)
    except Exception:
        return None


def as_number(s):
    e = parse_expr(s)
    if e is None:
        return None
    try:
        if e.is_number:
            return e
    except Exception:
        pass
    return None


def as_tuple(s):
    if not TUPLE_RE.match(s):
        return None
    inner = s[1:-1]
    items = [p.strip() for p in inner.split(",") if p.strip()]
    vals = []
    for it in items:
        try:
            vals.append(sympify(it))
        except Exception:
            return None
    return Tuple(*vals)


def classify(answer):
    raw = answer or ""
    s = strip_delims(raw)
    if not s:
        return "empty", None
    if has_cases(raw):
        return "piecewise", None
    if multi_part(raw) or raw.count("$$") >= 3:
        return "multi", None

    t = as_tuple(s)
    if t is not None:
        return "tuple", t

    r = rhs(s)
    n = as_number(r)
    if n is not None:
        return "number", n

    wc = len(english_words(s))
    if wc >= 2:
        return "prose", None

    e = parse_expr(r)
    if e is not None:
        return "expr", e
    return "reject", None


def verify(candidate, truth_obj, tier):
    c = strip_delims(candidate or "")
    if tier in ("number", "expr"):
        ce = parse_expr(rhs(c))
        if ce is None:
            return False
        try:
            return bool(simplify(ce - truth_obj) == 0)
        except Exception:
            try:
                return bool(nsimplify(ce) == nsimplify(truth_obj))
            except Exception:
                return False
    if tier == "tuple":
        ct = as_tuple(c)
        if ct is None or len(ct) != len(truth_obj):
            return False
        return all(simplify(a - b) == 0 for a, b in zip(ct, truth_obj))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="Math_arXiv")
    ap.add_argument("--output", default="realmath_verifiable.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("ethz-spylab/realmath")[args.split]

    buckets = {}
    examples = {}
    kept = []
    for row in ds:
        tier, obj = classify(row["answer"])
        buckets[tier] = buckets.get(tier, 0) + 1
        examples.setdefault(tier, [])
        if len(examples[tier]) < 6:
            examples[tier].append(strip_delims(row["answer"])[:110])
        if tier in ("number", "tuple", "expr"):
            kept.append({
                "link": row["link"],
                "question": row["question"],
                "answer": row["answer"],
                "tier": tier,
                "truth": str(obj),
            })

    total = len(ds)
    print(f"total: {total}")
    for k in sorted(buckets, key=lambda x: -buckets[x]):
        print(f"{k}: {buckets[k]} ({100*buckets[k]/total:.1f}%)")
    verifiable = sum(buckets.get(k, 0) for k in ("number", "tuple", "expr"))
    print(f"\nVERIFIABLE (number+tuple+expr): {verifiable} ({100*verifiable/total:.1f}%)")

    for k in ("number", "tuple", "expr", "prose", "piecewise", "multi", "reject"):
        if examples.get(k):
            print(f"\n--- {k} ---")
            for e in examples[k]:
                print(repr(e))

    with open(args.output, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(kept)} verifiable rows to {args.output}")


if __name__ == "__main__":
    main()
