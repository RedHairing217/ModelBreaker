"""Band harvest over the code-output prediction family.

Programs are graded against their executed output, so this keeps problems the model solves
inconsistently: pass@k inside a correctness band, the new-capability analog of the calculus
band. It also records the top wrong-value share per problem, so a program the model misreads
the same way every trial surfaces as confident-wrong rather than scatter without a separate run.
"""

import argparse
from statistics import mean

from shared.bands import BAND, in_band as band_contains

from shared.codeout import (DEFAULTS, SHAPES, build_code_output_problem, classify_code_output,
                            extract_output, run_snippet, sample_code_output)
from shared.harvest import run_problem_krate, write_report
from shared.lmstudio_client import DEFAULT_BASE_URL, build_client
from shared.reporting import OK, print_header, print_status


def self_test():
    code = "total = 0\nfor i in range(1, 5):\n    total += i * 3\nprint(total)"
    assert run_snippet(code, 5) == "30"
    assert extract_output("The program prints 30.") == 30
    assert extract_output("<think>maybe 7</think>\nOutput:\n42") == 42
    assert extract_output("no number here") is None
    pool = sample_code_output(5, True, 256, 0, 3, 5)
    assert len(pool) == 5 and all(isinstance(p.answer, int) for p in pool)
    assert all(p.detail["shape"] in SHAPES for p in pool)
    print_status(OK, "code-output self-test passed: executor, extractor, and sampler agree")


def main():
    parser = argparse.ArgumentParser(description="Band harvest over code-output prediction problems")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="qwen/qwen3-8b")
    parser.add_argument("--pool", type=int, default=40)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--band-lo", type=float, default=BAND[0])
    parser.add_argument("--band-hi", type=float, default=BAND[1])
    parser.add_argument("--capture-threshold", type=float, default=0.5,
                        help="Also keep problems whose single most common wrong answer reaches this share, the confident-wrong cases the band would discard")
    parser.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
    parser.add_argument("--max-size", type=int, default=DEFAULTS["max_size"])
    parser.add_argument("--exec-timeout", type=int, default=DEFAULTS["exec_timeout"])
    parser.add_argument("--think", action="store_true",
                        help="Use thinking mode with loop detection; default is no-think")
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="reports/code_output.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    thinking_mode = "think" if args.think else "no_think"
    detect_loops = thinking_mode == "think"
    print_header(f"Code-output harvest  ({args.model}, k={args.trials}, {thinking_mode})")

    client = build_client(args.base_url, timeout=args.timeout)
    pool = sample_code_output(args.pool, not args.think, args.max_tokens,
                              args.seed, args.max_size, args.exec_timeout)
    print_status(OK, f"built {len(pool)} executable code-output problems")

    kept, rejected = [], []
    shares = []
    kept_concentrated = kept_band = 0
    for problem in pool:
        result = run_problem_krate(client, args.model, problem, args.trials,
                                   args.temperature, classify_code_output, thinking_mode,
                                   detect_loops=detect_loops)
        cvc = result.committed_value_counts
        correct_key = str(problem.answer)
        wrong = {value: n for value, n in cvc.items() if value != correct_key}
        if wrong:
            top_wrong_value, top_wrong_n = max(wrong.items(), key=lambda kv: kv[1])
            top_wrong_share = top_wrong_n / result.trials
        else:
            top_wrong_value, top_wrong_share = None, 0.0
        shares.append(top_wrong_share)
        result.detail = {**result.detail,
                         "top_wrong_share": round(top_wrong_share, 3),
                         "top_wrong_value": top_wrong_value}
        in_band = band_contains(result.pass_at_k, (args.band_lo, args.band_hi))
        concentrated = top_wrong_share >= args.capture_threshold
        keep = concentrated or in_band
        kept_concentrated += concentrated
        kept_band += in_band
        result.in_band = keep
        result.status = "kept" if keep else "rejected"
        (kept if keep else rejected).append(result)
        flag = "KEEP" if keep else "    "
        print_status(OK if keep else "[..]",
                     f"{flag} {problem.label}  pass@k={result.pass_at_k:.2f} "
                     f"top_wrong={top_wrong_share:.2f} correct={result.correct}/{result.trials}")

    summary = {
        "family": "code_output",
        "pool_evaluated": len(pool),
        "kept": len(kept),
        "kept_concentrated": kept_concentrated,
        "kept_band": kept_band,
        "rejected": len(rejected),
        "band": [args.band_lo, args.band_hi],
        "capture_threshold": args.capture_threshold,
        "mean_top_wrong_share": round(mean(shares), 3) if shares else 0.0,
        "complete": True,
    }
    metadata = {
        "model": args.model,
        "base_url": args.base_url,
        "family": "code_output",
        "pool_target": args.pool,
        "trials": args.trials,
        "temperature": args.temperature,
        "band": [args.band_lo, args.band_hi],
        "capture_threshold": args.capture_threshold,
        "max_tokens": args.max_tokens,
        "max_size": args.max_size,
        "thinking_mode": thinking_mode,
        "seed": args.seed,
    }
    write_report(kept, summary, args.output, metadata, rejected=rejected)
    print_status(OK, f"kept {len(kept)} of {len(pool)}; mean top-wrong share "
                     f"{summary['mean_top_wrong_share']}; wrote {args.output}")


if __name__ == "__main__":
    main()
