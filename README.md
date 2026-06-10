# ModelBreaker

Functional robustness testing for a locally served LLM. The battery sends edge-case, malformed, format-adherence, and long-context inputs to an LM Studio server and records how the model handles each, distinguishing graceful degradation from hard failure.

## Requirements

- Python 3.10+
- An LM Studio server running with a model loaded (Developer tab, default `http://localhost:1234/v1`)

Install the one dependency:

    pip install -r requirements.txt

## Layout

    .
    ├── reports/                 generated JSON reports
    └── src/
        ├── inspect_server.py    check the server and list loaded model ids
        ├── run_robustness.py    run the test battery and write a report
        ├── run_sweep.py         distractor sweep to locate the ZPD
        ├── run_arith.py         arithmetic sweep to locate the ZPD
        ├── run_algebra.py       coefficient-extraction sweep to locate the ZPD
        ├── dashboard.py         live browser monitor for a battery run
        ├── chat_dashboard.py    manual chat dashboard, no battery
        └── shared/
            ├── cases.py             test-case definitions and validators
            ├── sweep.py             distractor-needle sweep and K-rate runner
            ├── arithmetic.py        chained-arithmetic sweep family
            ├── algebra.py           polynomial coefficient-extraction family
            ├── lmstudio_client.py   client and single-probe primitive
            └── reporting.py         console table and JSON report writer

## Usage

Run every script from the repository root.

Confirm the server is up and find the model id:

    python src/inspect_server.py

Run the full battery:

    python src/run_robustness.py --model qwen/qwen3-8b --context-tokens 32768

Run a single category and choose an output path:

    python src/run_robustness.py --category long_context --output reports/long_context.json

All flags have defaults; `--help` lists them on either script.

## Dashboard

For a live view, run the dashboard; it launches the battery from CLI flags and streams progress to the page:

    python src/dashboard.py
    python src/dashboard.py --category long_context --max-tokens 2048

It opens `http://localhost:7850` (does not clash with HammerTime's 7841/7842 or LM Studio's 1234). The page is read-only: results fill the table case by case over Server-Sent Events, with a progress ring, a pass/fail summary, and a per-category chart. Each finished run also writes `reports/dashboard_run.json`.

For a read-only view of the prompts and responses, the battery dashboard also shows a scrollable transcript. For manual exploration, run the chat dashboard instead:

    python src/chat_dashboard.py

It opens `http://localhost:7851`, holds a multi-turn session, and has a dropdown to load any automated battery prompt into the input box. It runs no battery.

## Sweep (zone of proximal development)

The sweep looks for prompts the model can do only sometimes. It plants one active vault code among N revoked decoys of the same shape and asks for the active one, then runs each decoy count K times at non-zero temperature and reports the success rate per count:

    python src/run_sweep.py
    python src/run_sweep.py --trials 10 --distractors 0 4 8 16 32 --no-think

Counts whose rate lands inside the band (default 0.3 to 0.7) are flagged as ZPD candidates: not trivial, not impossible. A saturated sweep (everything near 0 or 1) means the knob is too easy or too hard and the difficulty needs another lever.

A second sweep uses chained arithmetic, evaluated strictly left to right, with operation count as the knob:

    python src/run_arith.py
    python src/run_arith.py --steps 2 4 6 8 10 12 --no-think

Same K-rate and band reporting. The step count is the difficulty lever; longer chains push the model from reliable to unreliable. The arithmetic builder also takes `--ops addsub` to drop multiplication and drive difficulty through chain length alone.

A third sweep extracts a polynomial coefficient. It expands a product of K linear factors and asks for the coefficient of x^2, with the factor count as the knob:

    python src/run_algebra.py
    python src/dashboard.py --no-battery --no-arith --no-distractor --algebra-factors 3 4 5 6 7 8

The answer is a single integer, so the model always commits, and ground truth comes from sympy. This is the first family designed to fail through wrong answers rather than blanks.

## Experiment log

The battery and sweeps were run against Qwen3-8B (4-bit MLX) served locally through LM Studio. Each run isolates one variable. Rate is the fraction of trials whose validated answer was correct.

### Robustness battery

**Hypothesis.** My hypothesis was that edge, malformed, and format inputs would surface handling failures distinct from capability failures.

**Result.** 18 of 19 cases passed. The four initially recorded failures (strict JSON and three needle depths) were not capability failures. Every case terminated on `length`, which indicates the response was truncated before an answer was emitted, and raising the cap recovered all four. The model retrieved the planted needle at depths 0.1, 0.5, and 0.9 once given room, with no lost in the middle effect. The `overflow` case does not reject cleanly; it stalls until the client timeout.

**Conclusion.** My conclusion is that single-shot pass or fail against Qwen3 is dominated by its reasoning verbosity. With thinking enabled the model spends its entire output budget reasoning, so an undersized `max_tokens` manufactures failures that resemble incapacity. Valid robustness testing requires a budget sized to the reasoning trace, not to the answer.

### Distractor sweep

**Hypothesis.** My hypothesis was that planting one active code among N revoked decoys of identical shape, under an active code rule, would degrade retrieval as N grows.

**Dataset.** Decoy counts 0, 2, 4, 8, 16, 32. Eight trials each, temperature 0.7.

**Result.** Rate held at **1.00** across every count, with an isolated **0.00** at N=2 that reproduces across runs and appears to be a layout artifact of a single decoy on each side of the active line rather than a difficulty effect.

**Conclusion.** Retrieval discrimination under this rule sits below the model's threshold and is not a viable difficulty axis at these context lengths. Shelved.

### Arithmetic sweep

**Hypothesis.** My hypothesis was that a chained expression evaluated strictly left to right, with operation count as the difficulty knob, would cross from reliable to unreliable as the chain lengthens.

**Changes.** The validator was tightened from "target appears anywhere in the reply" to "final integer in the reply equals the target", which removed false positives where an echoed operand coincided with the answer. A sample response is captured per variant. The token cap, instance count, and client timeout were each exposed as ceilings appeared.

**Result.** The crossover was repeatedly masked by resource ceilings rather than by capability, and lifting each ceiling relocated the apparent failure rather than removing it.

| `max_tokens` | steps=12 rate | Failure signature |
| --- | --- | --- |
| 512 | 0.00 | tokens pinned at cap, empty content |
| 1024 | 0.00 | tokens pinned at cap, empty content |
| 4096 | 0.50 | partial truncation near cap |
| 8192 | **0.875** | mostly completing |

At 8192 single-instance the rate was non-monotonic across step counts (12 at 0.875, 14 at 0.50, 16 at 0.50, 18 at 0.875), which indicates that operation count is not the true difficulty driver. Difficulty tracks the specific expression, dominated by the number of multiplications and the magnitude of the running value. Multi-instance sampling (five expressions per level) then surfaced expressions whose reasoning exceeds the **120s** client timeout. Raising the timeout to **360s** converted those timeouts back into cap-truncated blanks at steps=12, where `med_tok` returned 8191 with an empty sample, while steps=14 and 16 resolved around 4000 tokens with committed answers.

**Conclusion.** My conclusion is that the model's hard failure mode at this difficulty is a non-answer, not a wrong answer. Multiplication inflates both the magnitude of intermediate values and the length of the reasoning trace, and beyond a threshold the model spirals into second-guessing reasoning that exhausts the token budget or the wall-clock before it commits. This is a reasoning-capacity limit distinct from an arbitrary token ceiling: the ceiling can be raised and the collapse simply moves with it. The mechanism for overwhelming the model's reasoning capacity is reproducible.

### Operator restriction

**Hypothesis.** My hypothesis was that removing multiplication and driving difficulty through chain length alone would bound the reasoning trace and force the model to commit, converting blanks into wrong but present totals.

**Changes.** An addition and subtraction only operator set (`--ops addsub`), swept over 20, 30, and 40 term chains.

**Result.** Rate held at 0.00 across all three lengths, with `med_tok` pinned at the cap and empty samples. The model narrates one line per term regardless of operator, so reasoning length still scales with the chain and overflows before an answer is emitted.

**Conclusion.** My conclusion is that removing multiplication does not bound the reasoning trace, because the model's narration cost is set by the number of operations, not their type. With thinking enabled the addition and subtraction task is binary in the same way the mixed task was: a completed narration is accurate, and an incomplete one is blank. There is no corruption band in a task whose per-step reasoning is mechanically correct.

### Current direction

**Hypothesis.** A corruption band, where the model commits to a definite but incorrect answer, requires a task whose answer is short, so output length never forces a blank, and whose reasoning steps each carry real error probability, so mistakes accumulate into a confident wrong answer. My hypothesis is that addition and subtraction failed to corrupt because its per-step reasoning is error-free, and that a task with error-prone steps and a one-token answer will expose the band without suppressing the model's reasoning.

**Planned change.** Polynomial coefficient extraction. The product of K linear factors has a single integer coefficient of x^2, so the model always commits, while combining cross-terms across many factors is error-prone and should produce wrong coefficients before the expansion narration grows long enough to truncate. If the corruption band proves narrow before truncation, the calculus variant (a derivative at a point, evaluated to an integer) concentrates more difficulty into an equally short answer.

### Persistent findings

The reasoning trace, not the answer, sets the token budget for a thinking-enabled model; size the cap to the trace. A substring-match validator produces false positives that scale with reply length, so the final answer must be graded in isolation. A single fixed instance per level confounds difficulty with instance luck; multiple instances must be sampled and averaged. The model's high-difficulty failure mode is the blank rather than the wrong answer, and resource ceilings relocate that collapse rather than removing it. Narration cost is set by the number of reasoning steps, not their difficulty, so a task with mechanically correct steps fails only by truncation; a corruption band requires per-step error probability paired with a short answer.

### Pending experiments

| Experiment | Status |
| --- | --- |
| Addition and subtraction length sweep | done, no corruption band, narration still truncates |
| Polynomial coefficient extraction | running |
| Derivative at a point | pending, if the algebra band is narrow |
| Perplexity sweet spot characterisation | pending |
| Distractor discrimination scaling | shelved, saturated |
| Robustness battery with thinking disabled at source | pending |

## Categories

- `edge` empty, whitespace, single-character, oversized, and Unicode-stress inputs
- `malformed` chat-template token injection, special tokens, contradictory and truncated instructions, nested delimiters
- `format` strict JSON output with a real parse check
- `long_context` needle retrieval at several depths plus a deliberate context overflow

## Report format

Each run writes a JSON file with three top-level keys: `metadata` (model, category, context length, timestamp), `summary` (counts and latency stats), and `results` (one record per case with pass state, latency, finish reason, response size, and a note or error).

## Notes

Long-context token counts are approximated at four characters per token, since the script does not load the model tokenizer. Set `--context-tokens` to the context length you loaded the model with in LM Studio so the depth and overflow cases line up with the real window.
