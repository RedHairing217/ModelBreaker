"""Per-problem pass@k harvesting: the framework's core ZPD primitive.

The unit of analysis is one problem, not a sweep level. Every trial is sorted
into one of three buckets so band membership can be trusted:

    correct        finish_reason stop, committed an answer, it matches the oracle
    wrong_complete finish_reason stop, committed an answer, it does not match
    degenerate     no committed answer: truncation, empty, unparseable, timeout

A mid-band rate is only a genuine zone-of-proximal-development signal when it
comes from the correct vs wrong_complete split. Rates driven by degenerate
trials are a budget artifact wearing a costume, so a problem is rejected when
its degenerate fraction exceeds a small threshold.

The engine is task-agnostic. A caller supplies HarvestProblem records (each
carrying its own messages and oracle) and a classify(finish_reason, text,
problem) -> bucket function. shared.calculus is the first such caller.

Usage:
    from shared.harvest import harvest, write_report
    kept, summary = harvest(client, model, problems, classify=classify_fn, ...)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import json
import os
import time
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from statistics import median

from shared.bands import in_band
from shared.lmstudio_client import ProbeResult, probe

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
CORRECT = "correct"
WRONG_COMPLETE = "wrong_complete"
DEGENERATE = "degenerate"
BUCKETS = (CORRECT, WRONG_COMPLETE, DEGENERATE)

# ─────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────
@dataclass
class HarvestProblem:
    """One candidate problem: how to ask it, how to grade it, what it is."""
    label: str
    messages: list
    answer: object
    key: tuple
    max_tokens: int
    detail: dict = field(default_factory=dict)


@dataclass
class ProblemResult:
    """A problem's k-rate outcome with the three-way trial breakdown."""
    label: str
    trials: int
    pass_at_k: float
    correct: int
    wrong_complete: int
    degenerate: int
    degenerate_fraction: float
    latency_median: float
    tokens_median: float
    thinking_mode: str
    degenerate_breakdown: dict = field(default_factory=dict)
    committed_value_counts: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)
    in_band: bool = field(default=False)
    status: str = field(default="")
    aborted: bool = field(default=False)

# ─────────────────────────────────────────────────────────────
# Candidate pool
# ─────────────────────────────────────────────────────────────
def dedupe(problems):
    """Drop problems sharing a canonical key, keeping first occurrence."""
    seen = {}
    for problem in problems:
        seen.setdefault(problem.key, problem)
    return list(seen.values())

# ─────────────────────────────────────────────────────────────
# Loop detection (think mode only)
# ─────────────────────────────────────────────────────────────
def is_looping(text, window=2000, ratio_threshold=0.15):
    """True when the trailing window is highly compressible, the signature of a loop.

    A coherent self-doubt loop or a token-salad collapse repeats heavily, so zlib packs
    the recent window to a small fraction of its size. Ordinary reasoning prose sits well
    above the threshold, so this fires on degeneration and not on normal text.
    """
    raw = text[-window:].encode("utf-8", "ignore")
    if len(raw) < window:
        return False
    return len(zlib.compress(raw)) / len(raw) < ratio_threshold


def generate_with_loop_detection(client, model, messages, *, temperature, max_tokens,
                                 min_chars=4000, check_every=800, window=2000,
                                 ratio_threshold=0.15, **params):
    """Stream a generation and abort as soon as the reasoning channel starts looping.

    Returns a ProbeResult so it is interchangeable with probe(). On a detected loop the
    finish_reason is set to "loop" (a degenerate outcome) and the stream is closed, so a
    runaway trace costs the tokens up to detection rather than the full cap. Completion
    tokens are estimated from streamed length, since usage is not reliably sent on a
    stream; the committed answer is whatever reached the content channel.
    """
    started = time.monotonic()
    try:
        stream = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, stream=True, **params)
    except Exception as exception:
        return ProbeResult(category="harvest", name="", passed=False,
                           latency_seconds=round(time.monotonic() - started, 2),
                           finish_reason=None,
                           error=f"{type(exception).__name__}: {exception}")
    reasoning_parts = []
    content_parts = []
    reasoning_len = 0
    last_check = 0
    finish_reason = None
    looped = False
    try:
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            reasoning_delta = getattr(delta, "reasoning_content", None) if delta else None
            content_delta = getattr(delta, "content", None) if delta else None
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
                reasoning_len += len(reasoning_delta)
            if content_delta:
                content_parts.append(content_delta)
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            if reasoning_len >= min_chars and reasoning_len - last_check >= check_every:
                last_check = reasoning_len
                if is_looping("".join(reasoning_parts), window, ratio_threshold):
                    looped = True
                    break
    finally:
        try:
            stream.close()
        except Exception:
            pass
    content = "".join(content_parts)
    full_chars = reasoning_len + len(content)
    return ProbeResult(
        category="harvest", name="", passed=True,
        latency_seconds=round(time.monotonic() - started, 2),
        finish_reason="loop" if looped else (finish_reason or "stop"),
        response_chars=len(content),
        completion_tokens=max(1, full_chars // 4),
        response_text=content,
    )


# ─────────────────────────────────────────────────────────────
# Per-problem k-rate
# ─────────────────────────────────────────────────────────────
def degenerate_reason(finish_reason, text):
    """Why a degenerate trial committed no answer: loop, truncated, empty, unparsed, or error."""
    if finish_reason == "loop":
        return "loop"
    if finish_reason == "length":
        return "truncated"
    if finish_reason in (None, "error"):
        return "error_or_timeout"
    if not (text or "").strip():
        return "empty"
    return "unparsed"


def run_problem_krate(client, model, problem, trials, temperature, classify, thinking_mode,
                      max_degenerate=None, extra_params=None, detect_loops=False,
                      loop_params=None):
    """Run one problem k times through probe and return its classified rate.

    classify(finish_reason, text, problem) returns (bucket, committed_value); the
    committed value is the model's parsed answer or None when it committed nothing.

    max_degenerate, if set, aborts the run once the degenerate count exceeds it: at
    that point the degenerate fraction can no longer come in under the gate no matter
    what the remaining trials do, so they would be wasted budget. Aborted runs are
    flagged and their rates are computed over the trials actually executed.

    extra_params are forwarded to each generation (e.g. frequency_penalty, stop) for
    inference-config experiments; they are recorded in the report metadata, not here.

    detect_loops routes generation through the streaming loop detector instead of probe;
    it is meant for think passes only, leaving no-think to run unimpeded through probe.
    loop_params overrides the detector knobs (ratio_threshold, min_chars, ...).
    """
    counts = {bucket: 0 for bucket in BUCKETS}
    degen_breakdown = {}
    value_counts = {}
    latencies = []
    tokens = []
    executed = 0
    aborted = False
    for _ in range(trials):
        if detect_loops:
            result = generate_with_loop_detection(
                client, model, problem.messages,
                temperature=temperature, max_tokens=problem.max_tokens,
                **(loop_params or {}), **(extra_params or {}))
        else:
            result = probe(client, model, "harvest", problem.label, problem.messages,
                           temperature=temperature, max_tokens=problem.max_tokens,
                           **(extra_params or {}))
        executed += 1
        bucket, committed = classify(result.finish_reason, result.response_text, problem)
        counts[bucket] += 1
        if bucket == DEGENERATE:
            reason = degenerate_reason(result.finish_reason, result.response_text)
            degen_breakdown[reason] = degen_breakdown.get(reason, 0) + 1
        elif committed is not None:
            key = str(committed)
            value_counts[key] = value_counts.get(key, 0) + 1
        if result.latency_seconds is not None:
            latencies.append(result.latency_seconds)
        if result.completion_tokens is not None:
            tokens.append(result.completion_tokens)
        if max_degenerate is not None and counts[DEGENERATE] > max_degenerate:
            aborted = True
            break
    return ProblemResult(
        label=problem.label,
        trials=executed,
        pass_at_k=round(counts[CORRECT] / executed, 4),
        correct=counts[CORRECT],
        wrong_complete=counts[WRONG_COMPLETE],
        degenerate=counts[DEGENERATE],
        degenerate_fraction=round(counts[DEGENERATE] / executed, 4),
        latency_median=round(median(latencies), 2) if latencies else 0.0,
        tokens_median=round(median(tokens), 1) if tokens else 0.0,
        thinking_mode=thinking_mode,
        degenerate_breakdown=degen_breakdown,
        committed_value_counts=value_counts,
        detail=dict(problem.detail),
        aborted=aborted,
    )

# ─────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────
def prescreen(client, model, problems, classify, *, prescreen_trials, temperature,
              thinking_mode, on_progress=None, progress_every=10, extra_params=None,
              detect_loops=False, loop_params=None):
    """Cheap low-k pass; keep only problems that are neither always-pass nor always-fail.

    Requires prescreen_trials >= 2: the survivor test 0 < correct < prescreen_trials
    is unsatisfiable at k=1, so a lower value drops everything.
    """
    if prescreen_trials < 2:
        raise ValueError("prescreen_trials must be >= 2; at k=1 no problem can survive")
    survivors = []
    always_pass = 0
    always_fail = 0
    started = time.monotonic()
    total = len(problems)
    for index, problem in enumerate(problems, start=1):
        screen = run_problem_krate(client, model, problem, prescreen_trials,
                                   temperature, classify, thinking_mode,
                                   extra_params=extra_params, detect_loops=detect_loops,
                                   loop_params=loop_params)
        if screen.correct == 0:
            always_fail += 1
        elif screen.correct == prescreen_trials:
            always_pass += 1
        else:
            survivors.append(problem)
        if on_progress is not None and (index % progress_every == 0 or index == total):
            on_progress(index, total, len(survivors), time.monotonic() - started)
    return survivors, {"always_pass": always_pass, "always_fail": always_fail}


def _summary(pool, survivors, kept, rejected_out_of_band, rejected_degenerate,
             complete, prescreen_stats=None, early_aborted=0):
    """Assemble the run summary; complete flags whether the full pass finished."""
    stats = prescreen_stats or {"always_pass": 0, "always_fail": 0}
    return {
        "pool_evaluated": pool,
        "prescreen_survivors": survivors,
        "prescreen_always_pass": stats["always_pass"],
        "prescreen_always_fail": stats["always_fail"],
        "kept": kept,
        "rejected_out_of_band": rejected_out_of_band,
        "rejected_degenerate": rejected_degenerate,
        "early_aborted": early_aborted,
        "complete": complete,
    }


def harvest(client, model, problems, *, classify, trials, prescreen_trials,
            band, degenerate_threshold, temperature, thinking_mode,
            on_keep=None, on_progress=None, checkpoint=None, progress_every=10,
            extra_params=None, detect_loops=False, loop_params=None):
    """Dedupe, prescreen, run full k, keep clean in-band problems, retain the rest.

    Every problem that reaches the full pass is recorded: kept ones in the returned
    kept list, the others in rejected, each tagged with a status. checkpoint(kept,
    rejected, summary), if given, fires after prescreen and after every full-pass
    problem so an interrupted run leaves a recoverable report.

    extra_params are forwarded to every generation (e.g. frequency_penalty, stop).
    detect_loops routes every generation through the streaming loop detector (think
    passes only); no-think callers leave it False and run through probe.
    """
    pool = dedupe(problems)
    survivors, prescreen_stats = prescreen(
        client, model, pool, classify,
        prescreen_trials=prescreen_trials, temperature=temperature,
        thinking_mode=thinking_mode, on_progress=on_progress, progress_every=progress_every,
        extra_params=extra_params, detect_loops=detect_loops, loop_params=loop_params)
    max_degenerate = int(degenerate_threshold * trials)
    kept = []
    rejected = []

    def summarize(complete):
        return _summary(len(pool), len(survivors), len(kept),
                        sum(1 for r in rejected if r.status == "rejected_out_of_band"),
                        sum(1 for r in rejected if r.status == "rejected_degenerate"),
                        complete, prescreen_stats,
                        early_aborted=sum(1 for r in rejected if r.aborted))

    if checkpoint is not None:
        checkpoint(kept, rejected, summarize(complete=False))
    for problem in survivors:
        result = run_problem_krate(client, model, problem, trials, temperature, classify,
                                   thinking_mode, max_degenerate=max_degenerate,
                                   extra_params=extra_params, detect_loops=detect_loops,
                                   loop_params=loop_params)
        if result.degenerate_fraction > degenerate_threshold:
            result.status = "rejected_degenerate"
            rejected.append(result)
        elif in_band(result.pass_at_k, band):
            result.in_band = True
            result.status = "kept"
            kept.append(result)
            if on_keep is not None:
                on_keep(result)
        else:
            result.status = "rejected_out_of_band"
            rejected.append(result)
        if checkpoint is not None:
            checkpoint(kept, rejected, summarize(complete=False))
    summary = summarize(complete=True)
    return kept, rejected, summary

# ─────────────────────────────────────────────────────────────
# JSON report
# ─────────────────────────────────────────────────────────────
def write_report(kept, summary, path, metadata, rejected=None):
    """Write the harvest report: metadata, summary, kept results, and rejected problems."""
    report = {
        "metadata": {**metadata, "generated_at": datetime.now(timezone.utc).isoformat()},
        "summary": summary,
        "results": [asdict(result) for result in kept],
    }
    if rejected is not None:
        report["rejected"] = [asdict(result) for result in rejected]
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as report_file:
        json.dump(report, report_file, indent=2)
