# ModelBreaker

Functional robustness and zone-of-proximal-development (ZPD) testing for a locally served LLM. A battery probes how the model handles edge, malformed, format-adherence, and long-context inputs, and a family of sweeps pushes procedurally generated tasks across a difficulty knob to find the band where the model is right only sometimes. All runs target Qwen3-8B (4-bit MLX) served through LM Studio at `http://127.0.0.1:1234/v1`. The through-line developed in the results below is that this inconsistency comes in distinct modes, scatter, non-termination, and confident-wrong, set by task structure rather than by difficulty alone, and that the mode matters more than the rate.

Setup, full CLI usage, the dashboards, the input categories, and the report format are documented in `README_legacy.md`. This file focuses on the experiment log and the current results.

## Current experimental results

### Per-problem harvesting

The sweeps below score a whole factor-count level as a single rate. A level rate hides which individual problems sit in the band and blends genuine wrong-answer variability with truncation, which the log kept having to caveat after the fact. The harvester changes the unit of analysis to the single problem. Each candidate runs k times, every trial is classified as `correct`, `wrong_complete`, or `degenerate` (truncation, empty, unparseable, timeout, or a detector-cut loop), and a problem is kept only when its in-band rate comes from the correct vs wrong-complete split with a near-zero degenerate fraction. The distinction the project kept rediscovering, a blank non-answer masquerading as a wrong answer, is now a first-class filter rather than a footnote. Sampling reuses the calculus construction so harvested problems sit on the same surface as the sweep. The classification and per-problem k-rate live in `src/shared/harvest.py`; calculus is the first task wired to it through `src/harvest_calculus.py`. The harness prescreens at low k to discard always-pass and always-fail problems, checkpoints its report after prescreen and after every kept problem so an interrupted run stays recoverable, and prints a prescreen progress heartbeat.

### Thinking vs no-thinking: the band exists in one mode, not the other

A paired harvest of the same 40-problem pool (seed 0, k=3 prescreen, k=16, 0.1 degenerate gate) in both inference modes produced opposite outcomes. The seed makes the pool identical across modes, so the difference is the mode, not the problem set.

No-think kept 9 of 40. The band spans 5 to 8 factors, with pass@16 from 0.31 to 0.69; eight of the nine had a 0.0 degenerate fraction and the ninth 0.06, so the band is clean. Without a reasoning trace the model handles small problems reliably and then commits definite but wrong answers on the long arithmetic of larger ones. (The earlier pool-12 pilot showed only 6 and 8 factors; the fuller run widens the band down to 5.)

Think kept 0 of 40. All 12 prescreen survivors exceeded the degenerate threshold, and none landed in band with a clean split. At the 4096 cap the mid-range failures are blanks: the problems that are neither always-right nor always-wrong get there by exhausting the token budget mid-reasoning, not by committing wrong answers. This qualifies the earlier K=4 level-sweep result (rate 0.667, judged genuine at k=3 with `med_tok` just under the cap): three trials did not resolve the truncation tail that sixteen expose.

Raising the cap to 16384 to see past the truncation exposed the deeper mechanism, and it is not truncation. The think failure appears to be non-termination of the reasoning phase. On a band-zone problem the model reaches an answer and then keeps verifying instead of committing, either looping a coherent self-doubt phrase or, past roughly fifteen thousand tokens, collapsing into token salad; in both cases it emits empty content and stops only at the cap or the client timeout, near six minutes a trial. A frequency-penalty pass at 0.4 broke the coherent loop but produced a different non-terminating degeneration, and the model's internal answer was unstable along the trace, one value then another then noise, so reading the reasoning channel as the answer is unsound. Three configurations now collapse the same way, 4k by truncation, 16k by looping, 16k-with-penalty by salad, which makes the reading robust: stopping is itself a token decision, and on an uncertain problem the keep-verifying continuation appears to outweigh it, so think has no harvestable correctness band for these problems at this model size.

The measurement consequence is sharper than mere absence. A would-be-correct run and a would-be-wrong run both terminate in the same non-answer, so the band no-think exposes is not merely missing in think but unmeasurable there: uncertainty surfaces as failure-to-terminate rather than as a rate. The confident-wrong signal the project is after therefore lives in no-think, where the model commits. Per-generation cost tracks this, roughly 13 to 40 seconds in no-think against minutes in think once a problem loops.

Retaining all evaluated problems, flagged earlier as the next harness change, is now done: the report carries a `rejected` array holding every deep-pass problem with its status and degenerate breakdown, which is what makes the think collapse above inspectable rather than only countable.

### Streaming loop detector

Bounding the think cost meant catching the loop mid-generation rather than waiting out the cap. A streaming path in `shared/harvest.py` watches the reasoning channel and aborts when a trailing window's zlib compression ratio drops below a threshold (default 0.15), the signature of a repetition loop or a salad collapse. It is gated to think passes, so no-think runs unimpeded through the normal probe and the shared client is untouched. A pool-8 smoke at the 16384 cap validated it: the `loop` bucket populates, cut trials record around 3200 to 3700 tokens against the 16k cap, roughly two minutes rather than six, and coherent-but-long reasoning is not false-flagged. One gap remains, a trace whose repetition period is longer than the window can run to the cap undetected; a per-trial wall-clock backstop is the pending fix. The smoke also reconfirmed the band's absence: both prescreen survivors went fully degenerate at k=16, so their prescreen commits were stochastic flukes.

### Degradation as a signal

Because think collapses to non-answers, its informative axis is not correctness but the degradation itself. The per-problem degenerate fraction is a non-termination rate with the same binomial footing as pass@k, and `tokens_median` is a free proxy for the degree of degradation. Both are config-dependent rather than problem-intrinsic, so they compare only at a fixed cap and penalty, and the degenerate breakdown must be kept so a budget truncation is not read as a genuine loop. The pipeline this implies is a termination-frontier sweep, holding the configuration fixed and measuring mean degradation per difficulty level; it is the think-mode analog of the no-think band and arguably the more deployment-relevant signal, since an input that makes a reasoning model hang is a production failure in its own right. One harness now serves both axes: no-think read for correctness and the confident-wrong split, think read for degradation and the termination frontier.

### No-think confident-wrong search: errors scatter, they do not commit

Lowering the no-think band floor to 0.0 to 0.4 (same pool and seed, k=16) was meant to harvest the reliably-wrong end and read `committed_value_counts` for a dominant wrong value. It found none. One problem landed in band (n07_p3, pass 0.25), with 21 always-pass and 14 above-band, and zero degeneracy throughout. The shape matters more than the count: in every problem that reached the deep pass the correct answer is the modal commit and the wrong trials scatter. n07_p3 spreads twelve wrong answers across eleven distinct values; n07_p1 commits the correct -42700 nine times and then seven different near-misses; the tightest case, n05_p5 at point 5, clusters at 9240, 9244, and 9248, plus or minus four around the truth. No problem in the run has a wrong value that rivals the correct one.

So no-think does not commit a stable wrong belief on this task. It runs the right method, holds the right magnitude, and slips on the large multiplication differently each trial, which produces near-miss scatter rather than a confident wrong answer. One methodological boundary applies: the survivor prescreen discards always-fail problems, which are exactly where a reliably-wrong value would live if it existed, so this rules out confident-wrong in the sometimes-right band and identifies the mechanism as scatter, but it does not test the always-fail set. The mechanism suggests those scatter more widely rather than concentrate, but that is untested.

### Failure modes are set by task structure

The runs above describe three different ways a model can be inconsistent, and the working conclusion is that the mode is determined by where the task puts its difficulty, not by how hard the task is. When the difficulty sits in arithmetic executed at temperature, the errors are stochastic and scatter into near-misses, as no-think does on the derivative task. When the difficulty sits in the reasoning phase under uncertainty, the model fails to terminate, as think does. A third mode, a stable dominant wrong answer, requires a systematic error, the same wrong method or misconception producing the same wrong value every trial, which an arithmetic-variance task cannot generate at any difficulty. Confident-wrong is therefore a property of task structure, expected in families whose errors are deterministic such as intuition traps or method-selection problems with light arithmetic, and absent from this one.

The practical point for collecting inconsistently-solved problems is that the mode matters more than the rate, because a pass@16 of 0.5 means something different in each one. Scatter is self-flagging, since repeated sampling disagrees and signals low reliability. Confident-wrong is silent, since repeated sampling agrees on a wrong answer and defeats consistency-based confidence. Non-termination yields no answer at all and reads as a latency and cost failure. A band of inconsistently-solved problems is therefore only as useful as the mode label attached to it, and characterizing the mode, more than any single rate, is the transferable result here.

### Confident-wrong, located in letter counting

The third leg was then searched directly, with the harness generalized to select on top wrong-value share, the single most common wrong commit over k, so a stable wrong answer is detected without predicting it. Two math-style traps came up empty: the famous average-speed harmonic trap did not bite at all (100 percent correct, trained through), and a non-canonical novel-operator trap was harder but its errors scattered rather than concentrating. Confident-wrong appeared in letter counting, the strawberry class, where the error is structural rather than reasoned. Asked how many times a letter appears in a real word, the model commits one value and repeats it, and dispersion collapses to a median of one distinct answer per problem against the math families' spread of up to eleven. At a larger pool of 120 the same pattern holds at volume: 47 problems have a stable wrong modal commit, 38 on at least 80 percent of trials and many at 16 of 16. The error is almost always exactly one off the truth, with the direction fixed per word and split roughly evenly across words (23 undercount, 24 overcount), so it is a per-word off-by-one that survives resampling rather than a uniform bias or random scatter: one a in "calendar", two s's in "success", three b's in "blueberry". This is the consistency-defeating case in practice, sixteen agreeing samples that are all wrong, and it confirms the taxonomy: confident-wrong lives where the error is structural and is largely absent from verifiable math where errors are stochastic. A later larger sweep softens that boundary slightly rather than breaking it: in calculus the value 0 behaves as a weak attractor, one problem committing 0 on 10 of 16 trials, so confident-wrong is rare in arithmetic rather than strictly absent, while the novel-operator family at pool 80 confirmed scatter and proved a strong band generator at 55 of 80 mid-band.

### Aggregated corpus

A small aggregator consolidates the kept problems from every report into one JSONL, each record labeled by failure mode and thinking mode, with the prompt reconstructed from the stored detail and deduped on problem-by-mode. The current corpus holds 68 inconsistently-solved problems: 55 confident-wrong, of which 52 are counting off-by-ones plus the two novel-operator cases and the single calculus zero-attractor, and 13 scatter, the calculus band problems, with one legacy record that carries no reconstructed prompt. That mode-labeled set is the data-curation output the project is meant to produce, the form an eval or training-data pipeline would consume, where the label records not just that the model is unreliable on a problem but how.

## Experiment log

The battery and sweeps were run against Qwen3-8B (4-bit MLX) served locally through LM Studio. Each run isolates one variable. Rate is the fraction of trials whose validated answer was correct.

### Robustness battery

Edge, malformed, and format inputs were expected to surface handling failures distinct from capability failures. 18 of 19 cases passed; the four initial failures (strict JSON and three needle depths) all terminated on `length`, meaning the response was truncated before an answer was emitted, and raising the cap recovered all four. Needle retrieval was clean at depths 0.1, 0.5, and 0.9 with no lost-in-the-middle effect, while the overflow case stalls to the client timeout rather than rejecting cleanly. Single-shot pass or fail against Qwen3 is therefore dominated by reasoning verbosity: an undersized `max_tokens` manufactures failures that look like incapacity, so a valid test sizes the budget to the reasoning trace, not the answer.

### Distractor sweep

Planting one active code among N revoked decoys of identical shape was expected to degrade retrieval as N grows. Across decoy counts 0, 2, 4, 8, 16, 32 (eight trials each, temperature 0.7) the rate held at 1.00, with an isolated reproducible 0.00 at N=2 that appears to be a layout artifact of one decoy on each side of the active line rather than a difficulty effect. Retrieval discrimination under this rule sits below the model's threshold and is not a viable difficulty axis at these context lengths. Shelved.

### Arithmetic sweep

A chained expression evaluated strictly left to right, with operation count as the knob, was expected to cross from reliable to unreliable as the chain lengthens. The validator was first tightened from substring match to "final integer equals target" to remove false positives. The crossover was then repeatedly masked by resource ceilings rather than capability, and lifting each ceiling relocated the apparent failure instead of removing it.

| `max_tokens` | steps=12 rate | Failure signature |
| --- | --- | --- |
| 512 | 0.00 | tokens pinned at cap, empty content |
| 1024 | 0.00 | tokens pinned at cap, empty content |
| 4096 | 0.50 | partial truncation near cap |
| 8192 | **0.875** | mostly completing |

At 8192 single-instance the rate was non-monotonic across step counts (0.875, 0.50, 0.50, 0.875 at 12, 14, 16, 18), so operation count is not the true difficulty driver; difficulty tracks the specific expression, dominated by the number of multiplications and the magnitude of the running value. Multi-instance sampling then surfaced expressions exceeding the 120s timeout, and raising it to 360s converted those into cap-truncated blanks. The hard failure mode is a non-answer, not a wrong answer: beyond a threshold the model spirals into second-guessing that exhausts the token budget or the wall-clock before it commits, and raising the ceiling simply moves the collapse.

### Operator restriction

Removing multiplication (`--ops addsub`, swept over 20, 30, 40 term chains) was expected to bound the reasoning trace and force committed but wrong totals. Rate held at 0.00 across all three lengths, `med_tok` pinned at the cap with empty samples. The model narrates one line per term regardless of operator, so reasoning length still scales with chain length and overflows before an answer. Narration cost is set by the number of steps, not their type, and a task whose per-step reasoning is mechanically correct has no corruption band: it fails only by truncation.

### Algebra sweep

Polynomial coefficient extraction (coefficient of x^2 from a product of K linear factors, ground truth from sympy, five instances, three trials) was expected to expose a band of committed wrong coefficients. At a 4096 cap the rate was 0.93 at K=3 and 0.53 at K=4, the latter mixing wrong coefficients with truncation. Raising the cap to 8192 resolved K=4 to 0.89 (completing, with at least one finished wrong coefficient), while K=5 and above truncated with no answer. Committed wrong answers do occur, but expansion narration grows fast enough that truncation overtakes the band: the window where the model commits and errs sits near K=5, behind a truncation wall the cap cannot reach.

### Calculus sweep

A derivative at a point was expected to concentrate error-prone product-rule reasoning into a shorter narration than a full expansion, opening the band before truncation. With f(x) a product of K linear factors, f'(x) evaluated at an integer point, ground truth from sympy, three instances and three trials at a 4096 cap: K=3 held at 0.89, and **K=4 landed in band at 0.667** with `med_tok` 3604 under the cap and non-empty answers, so its failures are genuine wrong derivatives rather than truncation. K=5 and above collapse to 0.00 at the cap. This is the corruption band the project set out to find: reliable at K=3, committed but incorrect at K=4, collapsed into truncation at K=5. Differentiation concentrates more genuine difficulty per output token than expansion, so the band opens one step before the truncation wall rather than behind it.

### Persistent findings

The reasoning trace, not the answer, sets the token budget for a thinking-enabled model, so size the cap to the trace. A substring-match validator produces false positives that scale with reply length, so the final answer must be graded in isolation. A single fixed instance per level confounds difficulty with instance luck, so multiple instances must be sampled and averaged. The model's high-difficulty failure mode is the blank rather than the wrong answer, and resource ceilings relocate that collapse rather than removing it. Narration cost is set by the number of reasoning steps, not their difficulty, so a corruption band requires per-step error probability paired with a short answer. Among the families tested, differentiation at a point concentrates the most error-prone reasoning per output token, which is why it reaches the band one step before the truncation wall while expansion reaches it one step behind.

### Pending experiments

| Experiment | Status |
| --- | --- |
| Addition and subtraction length sweep | done, no corruption band, narration still truncates |
| Polynomial coefficient extraction | done, band hidden behind truncation near K=5 |
| Derivative at a point (level sweep) | done, ★ corruption band located at K=4, rate 0.67 |
| Per-problem calculus harvester | built, `shared/harvest.py` + `src/harvest_calculus.py` |
| Think vs no-think band comparison, pool 40 | done, ★ no-think bands at 5-8 factors (9 of 40 kept); think kept 0, all 12 survivors degenerate |
| Retain rejected problems in the report for inspection | done, `rejected` array with status and degenerate breakdown |
| Think non-termination at 16k cap | done, ★ failure is looping, not truncation; band unmeasurable in think |
| Think frequency-penalty pass | done, negative, breaks the coherent loop but non-termination persists |
| No-think confident-wrong search (band floor to 0.0) | done, negative, errors scatter rather than commit a stable wrong value |
| Streaming loop detector (think only) | built and validated, compression-ratio cut in `shared/harvest.py` |
| Per-trial wall-clock backstop for long-period loop escapees | pending |
| Termination-frontier / degradation sweep | proposed, the think-mode analog of the band |
| Generalized confident-wrong metric (top wrong-value share) | done, family-agnostic, detects a stable wrong answer without predicting it |
| Average-speed and novel-operator traps | done, both negative, the systematic-error surface is trained down or scatters on math |
| Letter-count family (the strawberry class) | done, ★ confident-wrong located, a stable per-word off-by-one (47 of 120 at pool 120), dispersion ~1 |
| Corpus aggregator across reports | done, consolidates kept problems into one JSONL labeled by failure mode and thinking mode |
| Code-output prediction (execute to verify) | proposed, a new-capability band generator |
| Distractor discrimination scaling | shelved, saturated |
| Robustness battery with thinking disabled at source | pending |
