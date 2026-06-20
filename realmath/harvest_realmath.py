import re
import json
import time
import argparse
import collections

import requests

import verifier as V

BAND = (0.125, 0.875)


def in_band(p, band=BAND):
    return p is not None and band[0] <= p <= band[1]


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
    ap.add_argument("--url", default="http://127.0.0.1:1234/v1/chat/completions")
    ap.add_argument("--model", default="qwen/qwen3-8b")
    ap.add_argument("--band-lo", type=float, default=BAND[0])
    ap.add_argument("--band-hi", type=float, default=BAND[1])
    ap.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args()
    band = (args.band_lo, args.band_hi)

    rows = [json.loads(l) for l in open(args.input)]
    rows = rows[args.start:]
    if args.limit:
        rows = rows[:args.limit]

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out = open(args.output, "w")

    summary = collections.Counter()
    for idx, row in enumerate(rows):
        tier, truth = V.classify(row["answer"])
        if tier not in ("number", "tuple", "expr") or truth is None:
            summary["skip_unverifiable"] += 1
            continue

        correct = wrong = degenerate = 0
        wrongs = collections.Counter()
        samples = []
        for _ in range(args.k):
            try:
                resp = call_qwen(row["question"], args.url, args.model,
                                 args.temperature, args.max_tokens,
                                 args.think, args.timeout)
            except Exception as e:
                degenerate += 1
                samples.append(f"<error:{e}>")
                continue
            body = strip_think(resp)
            cand = extract_candidate(body)
            if not cand:
                degenerate += 1
                samples.append("<no-answer>")
                continue
            ok = False
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

        k = args.k
        pass_at_k = correct / k
        degen_frac = degenerate / k
        modal_wrong, modal_n = (wrongs.most_common(1)[0] if wrongs else ("", 0))

        if degen_frac >= 0.5:
            label = "degenerate"
        elif in_band(pass_at_k, band):
            label = "band"
        elif pass_at_k > band[1]:
            label = "too_easy"
        else:
            label = "too_hard"
        summary[label] += 1

        rec = {
            "link": row["link"],
            "question": row["question"],
            "answer": row["answer"],
            "tier": tier,
            "truth": str(truth),
            "k": k,
            "correct": correct,
            "wrong": wrong,
            "degenerate": degenerate,
            "pass_at_k": pass_at_k,
            "degenerate_fraction": degen_frac,
            "modal_wrong": modal_wrong,
            "modal_wrong_n": modal_n,
            "top_wrong_share": modal_n / k,
            "label": label,
            "keep": label == "band",
            "samples": samples,
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"[{idx+1}/{len(rows)}] {label:11s} pass={pass_at_k:.2f} "
              f"deg={degen_frac:.2f} {tier} :: {row['question'][:50]}")

    out.close()
    print("\n=== summary ===")
    for k in ("band", "too_easy", "too_hard", "degenerate", "skip_unverifiable"):
        print(f"{k}: {summary.get(k, 0)}")


if __name__ == "__main__":
    main()
