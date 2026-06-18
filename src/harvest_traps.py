"""Confident-wrong search over systematic-error task families.

Unlike the calculus harvester, this does not prescreen for a correctness band: a real
trap is always-fail, which the band logic would discard. It runs k trials per problem
and keeps those whose committed answers concentrate on the predicted naive value, the
trap-capture rate. A high rate is the confident-wrong signal, the contrast to scatter.
"""

import argparse
import sys
from dataclasses import asdict
from statistics import mean

from shared.harvest import run_problem_krate, write_report
from shared.lmstudio_client import DEFAULT_BASE_URL, build_client
from shared.reporting import OK, print_header, print_status
from shared.traps import (DEFAULTS, FAMILIES, average_speed_answers, classify_trap,
                          extract_integer, novel_operator_answers)


def self_test():
    assert average_speed_answers(30, 60) == (40, 45)
    assert average_speed_answers(20, 30) == (24, 25)
    assert average_speed_answers(15, 30) is None
    assert average_speed_answers(40, 40) is None
    assert novel_operator_answers([1, 2, 3]) == (1, -3)
    assert novel_operator_answers([1, 2, 3, 4]) == (0, -10)
    assert extract_integer("<think>50 maybe</think> The answer is 45.") == 45
    assert extract_integer("1,344") == 1344
    print_status(OK, "trap self-test passed: families and extractor agree")


def main():
    parser = argparse.ArgumentParser(description="Confident-wrong search over trap task families")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="qwen/qwen3-8b")
    parser.add_argument("--family", default="average_speed", choices=sorted(FAMILIES))
    parser.add_argument("--pool", type=int, default=30)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--capture-threshold", type=float, default=0.5,
                        help="Keep problems whose single most common wrong answer reaches at least this share of trials")
    parser.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
    parser.add_argument("--speed-min", type=int, default=DEFAULTS["speed_min"])
    parser.add_argument("--speed-max", type=int, default=DEFAULTS["speed_max"])
    parser.add_argument("--think", action="store_true",
                        help="Use thinking mode; default is no-think, where the traps bite hardest")
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="reports/trap_harvest.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    thinking_mode = "think" if args.think else "no_think"
    detect_loops = thinking_mode == "think"
    print_header(f"Trap harvest: {args.family}  ({args.model}, k={args.trials}, {thinking_mode})")

    client = build_client(args.base_url, timeout=args.timeout)
    pool = FAMILIES[args.family](args.pool, not args.think, args.max_tokens,
                                 args.speed_min, args.speed_max, args.seed)
    print_status(OK, f"built {len(pool)} {args.family} problems")

    kept, rejected = [], []
    shares = []
    for index, problem in enumerate(pool, start=1):
        result = run_problem_krate(client, args.model, problem, args.trials,
                                   args.temperature, classify_trap, thinking_mode,
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
        extra = {"top_wrong_share": round(top_wrong_share, 3), "top_wrong_value": top_wrong_value}
        if "naive" in result.detail:
            extra["naive_capture"] = round(cvc.get(str(result.detail["naive"]), 0) / result.trials, 3)
        result.detail = {**result.detail, **extra}
        result.in_band = top_wrong_share >= args.capture_threshold
        result.status = "kept" if result.in_band else "rejected_low_concentration"
        (kept if result.in_band else rejected).append(result)
        flag = "KEEP" if result.in_band else "    "
        print_status(OK if result.in_band else "[..]",
                     f"{flag} {problem.label}  top_wrong={top_wrong_share:.2f} ({top_wrong_value}) "
                     f"correct={result.correct}/{result.trials} answer={problem.answer}")

    summary = {
        "family": args.family,
        "pool_evaluated": len(pool),
        "kept": len(kept),
        "rejected_low_concentration": len(rejected),
        "concentration_threshold": args.capture_threshold,
        "mean_top_wrong_share": round(mean(shares), 3) if shares else 0.0,
        "complete": True,
    }
    metadata = {
        "model": args.model,
        "base_url": args.base_url,
        "family": args.family,
        "pool_target": args.pool,
        "trials": args.trials,
        "temperature": args.temperature,
        "capture_threshold": args.capture_threshold,
        "max_tokens": args.max_tokens,
        "thinking_mode": thinking_mode,
        "seed": args.seed,
    }
    if args.family == "average_speed":
        metadata["speed_range"] = [args.speed_min, args.speed_max]
    write_report(kept, summary, args.output, metadata, rejected=rejected)
    print_status(OK, f"kept {len(kept)} of {len(pool)}; mean top-wrong share "
                     f"{summary['mean_top_wrong_share']}; wrote {args.output}")


if __name__ == "__main__":
    main()
