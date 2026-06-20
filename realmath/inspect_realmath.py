import json
import re

from datasets import load_dataset

import sympy
from sympy import sympify

try:
    from sympy.parsing.latex import parse_latex
    HAVE_LATEX = True
except Exception:
    HAVE_LATEX = False


def clean(ans):
    s = (ans or "").strip()
    s = s.strip("$").strip()
    s = re.sub(r"^\\\[|\\\]$", "", s).strip()
    s = re.sub(r"^\\\(|\\\)$", "", s).strip()
    return s


def parseable(ans):
    s = clean(ans)
    if not s:
        return None
    if HAVE_LATEX:
        try:
            parse_latex(s)
            return "latex"
        except Exception:
            pass
    try:
        sympify(s)
        return "sympify"
    except Exception:
        pass
    return None


def main():
    ds = load_dataset("ethz-spylab/realmath")
    print("splits:", {k: len(v) for k, v in ds.items()})
    split = list(ds.keys())[0]
    data = ds[split]
    print("split used:", split)
    print("columns:", data.column_names)
    print("rows:", len(data))
    print("latex parser available:", HAVE_LATEX)

    print("\n=== 5 samples ===")
    for i in range(min(5, len(data))):
        row = data[i]
        for k, v in row.items():
            sv = str(v)
            print(f"{k}: {sv[:300]}")
        print("-" * 60)

    ans_field = "answer" if "answer" in data.column_names else None
    if not ans_field:
        print("\nno 'answer' column; columns are:", data.column_names)
        return

    buckets = {"latex": 0, "sympify": 0, "unparseable": 0, "empty": 0}
    examples = {"latex": [], "sympify": [], "unparseable": []}
    for row in data:
        r = parseable(row[ans_field])
        if r is None:
            key = "empty" if not clean(row[ans_field]) else "unparseable"
            buckets[key] += 1
            if key == "unparseable" and len(examples["unparseable"]) < 8:
                examples["unparseable"].append(clean(row[ans_field])[:120])
        else:
            buckets[r] += 1
            if len(examples[r]) < 8:
                examples[r].append(clean(row[ans_field])[:120])

    total = len(data)
    print("\n=== answer parseability ===")
    for k, v in buckets.items():
        print(f"{k}: {v} ({100*v/total:.1f}%)")

    for k in ("latex", "sympify", "unparseable"):
        print(f"\n--- {k} examples ---")
        for e in examples[k]:
            print(repr(e))


if __name__ == "__main__":
    main()
