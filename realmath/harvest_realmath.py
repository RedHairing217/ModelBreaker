import re
import json
import argparse
import collections

import requests

import verifier as V

BAND = (0.125, 0.875)
JUNK = ["mathrm", "mathbb", "mathcal", "mathbf", "mathsf", "operatorname",
        "Bigl", "Bigr", "bigl", "bigr", "widetilde", "widehat", "cdots",
        "ldots", "efsub", "displaystyle", "text", "boldsymbol", "mathfrak",
        "mathscr"]


def in_band(p, band=BAND):
    return p is not None and band[0] <= p <= band[1]


def truth_garbage(truth):
    s = str(truth)
    return any(j in s for j in JUNK)


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_boxed(text):
    i = text.rfind("\\boxed")
    if i == -1:
        return None
    j = text.find("{", i)
    if j == -1:
        return None
    depth = 0
    out = []
    for c in text[j:]:
        if c == "{":
            depth += 1
            if depth == 1:
                continue
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
    return "".join(out).strip()


def extract_candidate(text):
    b = extract_boxed(text)
    if b:
        return b
    m = re.findall(r"\$([^$]+)\$", text)
    if m:
        return m[-1].strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def call_qwen(question, url, model, temperature, max_tokens, think, timeout):
    system = "Solve the problem. State only the final answer inside \\boxed{}."
    user = question if think else question + " /no_think"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="realmath_verifiable.jsonl")
    ap.add_argument("--output", default="reports/realmath_band.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--keep-garbage", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:1234/v1/chat/completions")
    ap.add_argument("--model", default="qwen/qwen3-8b")
    ap.add_argument("--band-lo", type=float, default=BAND[0])
    ap.add_argument("--band-hi", type=float, default=BAND[1])
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--misdir-thresh", type=float, default=0.5)
    args = ap.parse_args()
    band = (args.band_lo, args.band_hi)
    k = args.k
    early_ok = band[0] <= 1.0 / k + 1e-9 and band[1] >= (k - 1.0) / k - 1e-9

    rows = [json.loads(l) for l in open(args.input)]
    rows = rows[args.start:]
    if args.limit:
        rows = rows[:args.limit]

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out = open(args.output, "w")
    corpus_path = args.output.replace(".jsonl", "_corpus.jsonl")
    corpus = open(corpus_path, "w")
    remainder_path = args.output.replace(".jsonl", "_remainder.jsonl")
    remainder = open(remainder_path, "w")

    summary = collections.Counter()
    total_calls = 0
    sampled_problems = 0
    for idx, row in enumerate(rows):
        tier, truth = V.classify(row["answer"])
        if tier not in ("number", "tuple", "expr") or truth is None:
            summary["skip_unverifiable"] += 1
            continue
        if not args.keep_garbage and truth_garbage(truth):
            summary["skip_garbage_truth"] += 1
            continue

        correct = wrong = degenerate = used = 0
        wrongs = collections.Counter()
        samples = []
        early = False
        for _ in range(k):
            used += 1
            try:
                resp = call_qwen(row["question"], args.url, args.model,
                                 args.temperature, args.max_tokens,
                                 args.think, args.timeout)
            except Exception as e:
                degenerate += 1
                samples.append(f"<error:{e}>")
                continue
            cand = extract_candidate(strip_think(resp))
            if not cand:
                degenerate += 1
                if len(samples) < 4:
                    samples.append("<no-answer>")
                continue
            try:
                ok = V.verify(cand, truth, tier)
            except Exception:
                ok = False
            if ok:
                correct += 1
            else:
                wrong += 1
                wrongs[cand[:80]] += 1
            if len(samples) < 4:
                samples.append(cand[:80])
            if early_ok and correct >= 1 and wrong >= 1:
                early = True
                break

        total_calls += used
        sampled_problems += 1

        if early:
            label = "band"
            pass_at_k = None
        else:
            pass_at_k = correct / k
            degen_frac = degenerate / k
            if degen_frac >= 0.5:
                label = "degenerate"
            elif in_band(pass_at_k, band):
                label = "band"
            elif pass_at_k > band[1]:
                label = "too_easy"
            else:
                label = "too_hard"
        summary[label] += 1

        modal_wrong, modal_n = (wrongs.most_common(1)[0] if wrongs else ("", 0))
        top_share = modal_n / used if used else 0
        if early:
            label = "band"
            pass_at_k = None
        else:
            pass_at_k = correct / k
            degen_frac = degenerate / k
            if degen_frac >= 0.5:
                label = "degenerate"
            elif in_band(pass_at_k, band):
                label = "band"
            elif pass_at_k > band[1]:
                label = "too_easy"
            elif top_share >= args.misdir_thresh:
                label = "misdirection"
            else:
                label = "collapse"
        if label == "too_easy" and correct == k and degenerate == 0:
            label = "solved"
        keep = label in ("band", "misdirection")
        summary[label] += 1

        rec = {
            "link": row["link"],
            "question": row["question"],
            "answer": row["answer"],
            "tier": tier,
            "truth": str(truth),
            "k": k,
            "samples_used": used,
            "correct": correct,
            "wrong": wrong,
            "degenerate": degenerate,
            "pass_at_k": pass_at_k,
            "modal_wrong": modal_wrong,
            "top_wrong_share": top_share,
            "distinct_wrong": len(wrongs),
            "wrong_dist": dict(wrongs),
            "label": label,
            "keep": keep,
            "samples": samples,
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if keep:
            corpus.write(json.dumps(rec) + "\n")
            corpus.flush()
        elif label != "solved":
            remainder.write(json.dumps(row) + "\n")
            remainder.flush()
        pstr = "band*" if pass_at_k is None else f"{pass_at_k:.2f}"
        print(f"[{idx+1}/{len(rows)}] {label:11s} pass={pstr} used={used}/{k} "
              f"{tier} :: {row['question'][:46]}")

    out.close()
    corpus.close()
    remainder.close()
    print("\n=== summary ===")
    for key in ("band", "misdirection", "too_easy", "collapse", "degenerate",
                "solved", "skip_garbage_truth", "skip_unverifiable"):
        print(f"{key}: {summary.get(key, 0)}")
    kept = summary.get("band", 0) + summary.get("misdirection", 0)
    rem = summary.get("too_easy", 0) + summary.get("collapse", 0) + summary.get("degenerate", 0)
    print(f"KEPT (band + misdirection): {kept} -> {corpus_path}")
    print(f"REMAINDER (for higher-k stage): {rem} -> {remainder_path}")
    if sampled_problems:
        avg = total_calls / sampled_problems
        naive = sampled_problems * k
        print(f"calls: {total_calls} vs {naive} naive "
              f"({100*(1-total_calls/naive):.0f}% saved), avg {avg:.1f}/problem")


if __name__ == "__main__":
    main()
