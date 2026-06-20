# ModelBreaker: Experiment Log

Per-experiment hypothesis and result records. All runs target Qwen3-8B (4-bit MLX) through LM Studio. Each isolates one variable. Rate is the fraction of trials whose validated answer is correct, and outcomes use the four-way taxonomy defined in the README: solved, band, misdirection, collapse.

## Experiment summary

| Experiment | Knob | Outcome |
|------------|------|---------|
| Robustness battery | Input type | 18 of 19 pass; failures were truncation |
| Distractor sweep | Decoy count | Saturated, shelved |
| Arithmetic sweep | Term count and magnitude | Collapse by non-termination |
| Operator restriction | Chain length | Collapse only, no band |
| Algebra sweep | Factor count | Band behind a collapse wall near K=5 |
| Calculus level-sweep | Factor count | ★ Band at K=4, rate 0.667 |
| Calculus harvester, no-think | Factor count | Band at 5 to 8 factors, 9 of 40 kept |
| Calculus harvester, think | Factor count and cap | 0 kept, collapse by non-termination |
| No-think confident-wrong search | Factor count | 1 kept, calculus collapses by scatter |
| Letter-count trap | Word and letter | ★ Misdirection, stable off-by-one |
| Novel-operator trap | Operand count | Band generator, errors scatter |
| Average-speed trap | Speed pair | Awaiting first run |
| arXiv theorems (RealMath) | k sampling | ★ 116 curated, 70 band and 46 misdirection |

---

## Robustness battery

**Hypothesis.** Edge, malformed, and format inputs would surface handling failures distinct from capability failures.

**Dataset.** 19 cases across edge, malformed, strict-format, and long-context categories.

**Result.** 18 of 19 passed. The four initial failures all terminated on `length`, meaning truncation before an answer was emitted, and raising the cap recovered every one. Needle retrieval was clean at depths 0.1, 0.5, and 0.9 with no lost-in-the-middle effect. The overflow case stalls to the client timeout rather than rejecting.

**Conclusion.** Single-shot pass or fail against Qwen3 is dominated by reasoning verbosity, so a valid test sizes the token budget to the reasoning trace, not the answer.

---

## Distractor sweep

**Hypothesis.** Planting one active code among N revoked decoys of identical shape would degrade retrieval as N grows.

**Dataset.** Decoy counts 0, 2, 4, 8, 16, 32. Eight trials each, temperature 0.7.

**Result.** Rate held at 1.00 across every count, with an isolated reproducible 0.00 at N=2 that appears to be a layout artefact of one decoy on each side of the active line rather than a difficulty effect.

**Conclusion.** Retrieval discrimination under this rule sits below the model's threshold and is not a viable difficulty axis at these context lengths. Shelved.

---

## Arithmetic sweep

**Hypothesis.** A chained expression evaluated strictly left to right, with operation count as the knob, would cross from reliable to unreliable as the chain lengthens.

**Change.** The validator was tightened from substring match to "final integer equals target", which removed false positives where an echoed operand coincided with the answer.

**Result.** The crossover was repeatedly masked by resource ceilings rather than capability, and lifting each ceiling relocated the failure rather than removing it.

| `max_tokens` | steps=12 rate | Failure signature |
|--------------|---------------|-------------------|
| 512 | 0.00 | tokens pinned at cap, empty content |
| 1024 | 0.00 | tokens pinned at cap, empty content |
| 4096 | 0.50 | partial truncation near cap |
| 8192 | 0.875 | mostly completing |

At 8192 the rate was non-monotonic across step counts (0.875, 0.50, 0.50, 0.875 at 12, 14, 16, 18), so operation count is not the true driver. Difficulty tracks the specific expression, dominated by the number of multiplications and the magnitude of the running value.

**Conclusion.** The hard failure is collapse by non-termination, not misdirection. Beyond a threshold the model spirals into second-guessing that exhausts the budget before it commits, and raising the ceiling moves the collapse.

---

## Operator restriction

**Hypothesis.** Removing multiplication and driving difficulty through chain length alone would bound the reasoning trace and convert blanks into wrong but present totals.

**Change.** An addition and subtraction only set (`--ops addsub`), swept over 20, 30, and 40 term chains.

**Result.** Rate held at 0.00 across all three lengths, `med_tok` pinned at the cap with empty samples.

**Conclusion.** Narration cost is set by the number of reasoning steps, not their type, so a task whose per-step reasoning is mechanically correct produces only collapse and never misdirection.

---

## Algebra sweep

**Hypothesis.** Polynomial coefficient extraction, with a single integer answer and error-prone cross-term combination, would expose a band of committed wrong coefficients.

**Dataset.** Coefficient of x^2 from a product of K linear factors, ground truth from sympy, five instances per count, three trials.

**Result.** At a 4096 cap, 0.93 at K=3 and 0.53 at K=4, the latter mixing wrong coefficients with truncation. Raising the cap to 8192 lifted K=4 to 0.89 (+0.36, completing, with at least one finished wrong coefficient), while K=5 and above truncated with no answer.

**Conclusion.** Committed wrong answers do occur, but expansion narration grows fast enough that collapse overtakes the band near K=5, behind a wall the cap cannot reach.

---

## Calculus level-sweep

**Hypothesis.** A derivative at a point would concentrate error-prone product-rule reasoning into a shorter narration than a full expansion, opening the band before truncation.

**Dataset.** f(x) a product of K linear factors, f'(x) at an integer point, single integer answer, ground truth from sympy, three instances, three trials, 4096 cap.

**Result.** K=3 held at 0.89, and K=4 landed in band at 0.667 (-0.22 from K=3) with `med_tok` 3604 under the cap and non-empty answers, so its failures are genuine wrong derivatives. K=5 and above collapse to 0.00.

**Conclusion.** ★ The reliable, band, collapse progression the project set out to find: reliable at K=3, band at K=4, collapse at K=5. Differentiation concentrates more error-prone reasoning per output token than expansion, so the band opens one step before the collapse wall rather than behind it.

---

## Calculus harvester, think against no-think

**Hypothesis.** The band sits at a configuration rather than a fixed difficulty, so the two inference modes would break the model in different places.

**Change.** Per-problem harvesting at k=16 on a seeded pool of 40, run in both modes so the candidate set is identical and any shift is attributable to the mode.

**Result.** No-think kept 9 of 40, band at 5 to 8 factors, near-zero degenerate fraction, wrong answers scattered. Think kept 0, every hard problem ending in collapse by non-termination. Per-generation cost was roughly 25 to 33 seconds no-think against about 77 seconds think.

**Conclusion.** The two modes are different reasoning surfaces and must never be pooled. Think exhausts its budget reasoning on small problems and collapses on large ones, while no-think commits clean wrong totals.

---

## No-think confident-wrong search

**Hypothesis.** Dropping the band floor toward zero would harvest the reliably-wrong end of the calculus surface and surface a dominant wrong value, the misdirection signal.

**Change.** Band floor lowered to 0.0 to 0.4, with committed values recorded per trial.

**Result.** 1 kept. Wrong answers scattered with no dominant value, so calculus produces collapse by scatter rather than misdirection.

**Conclusion.** Arithmetic error does not concentrate, so misdirection needs a task whose wrong answer is an attractor. This sent the search to the trap families.

---

## Systematic-error trap families

**Hypothesis.** A task with a single intuitive wrong answer would produce misdirection where free arithmetic only collapses by scatter.

**Change.** Trap families that each carry both the correct value and the predicted attractor, so the harness measures a capture rate directly. Average-speed (correct is the harmonic mean, attractor the arithmetic mean), letter-count (counting a target letter in a word), and a novel-operator family (a defined right-associative operator whose attractor is the left-fold).

**Result.** ★ Letter counting located misdirection: at a pool of 120, 47 problems committed a stable off-by-one, almost always plus or minus one, with the direction word-specific. The novel-operator family was a strong band generator, with most errors scattering rather than landing on the predicted fold. Average-speed is awaiting its first full run.

**Conclusion.** Misdirection is a property of task structure. A near-miss intuitive answer such as the counting off-by-one produces a stable wrong commit, while free arithmetic of equal hardness only collapses by scatter.

---

## Graduate problems from arXiv (RealMath)

**Hypothesis.** Single-answer theorems from research mathematics would supply graduate-level problems that land in the band or trigger misdirection, extending the corpus past what generated tasks reach.

**Change.** The RealMath pipeline (ethz-spylab) retrieves and extracts single-answer theorems, with its OpenAI extraction ported to a drop-in Anthropic shim so the calls run on Claude. A deterministic verifier replaces RealMath's LLM judge: answers are classified into number, tuple, and single-expression tiers and graded by symbolic equality with a numeric-sampling fallback, while piecewise, relational, and prose answers are rejected. The band prescreen runs each problem through Qwen with a boxed-answer prompt, with early-exit, one correct and one wrong guarantees the band, and a garbage-truth prefilter. A k-cascade harvests band hits at k=8, removes them, then re-examines only the consistent remainder at higher k, dropping full-pass problems.

**Dataset.** Fresh arXiv math.NT, April 2026, was scraped first to limit contamination, but single-answer theorems are rare and the LaTeX extraction kept only a handful of papers per fifty, so the fresh yield was too low to brute-force. The verifiable pool came instead from RealMath's published set, 633 Math_arXiv problems, with contamination handled by the prescreen rather than by freshness.

**Result.** k=8 kept 156, 97 band and 59 misdirection. Raising to k=12 returned 1 band in 35 re-examined, so the cascade was stopped. Curation dropped 40 ground-truth artefacts (26 relational answers, 9 bare symbols, 3 junk parses, 2 kind-mismatch misdirections), leaving 116, of which 70 are band and 46 are misdirection. The misdirection cases are stable systematic errors, such as `2*n` answered as `n` and `M_{n-1}` answered as `M_n`.

**Conclusion.** ★ Curated arXiv theorems extend the corpus with graduate-level band and misdirection examples generated tasks cannot reach, and k=8 is the practical sampling resolution here, since the marginal band yield collapses past it. The deterministic verifier bounds the corpus, since relational and piecewise answers pose verification difficulties.

---

## Persistent findings

- The reasoning trace determined the token budget for a thinking-enabled model, so size the cap to the trace.
- A substring-match validator produces false positives that scale with reply length, so the final answer must be graded in isolation.
- A single fixed instance per level confounds difficulty with instance luck, so multiple instances must be sampled.
- Narration cost is set by the number of reasoning steps, not their difficulty, so a band needs per-step error probability paired with a short answer.
- Among consistent failures, misdirection and collapse are worth separating: one is a reproducible systematic error worth recording, the other a give-up worth dropping, and misdirection appears only where the wrong answer is an attractor.
- Sampling has a practical resolution, since on the arXiv set the band yield collapses past k=8. 
- For curated problems the deterministic verifier bounds the corpus, because relational and piecewise answers pose verificaiton issues.
- A band prescreen is robust to training contamination in the safe direction, because a memorised problem passes consistently and is discarded as solved.

---

## Pending experiments

| Experiment | Status |
|------------|--------|
| Derivative at a point, level sweep | done, ★ corruption band at K=4, rate 0.667 |
| Calculus harvester, think against no-think | done, no-think band 5 to 8 factors, think collapses |
| No-think confident-wrong search | done, calculus collapses by scatter |
| Letter-count trap | done, ★ misdirection located, stable off-by-one |
| Novel-operator trap | done, band generator, errors scatter |
| Average-speed trap | awaiting first run |
| arXiv single-answer theorems (RealMath) | done, ★ 116 curated, 70 band and 46 misdirection |
| Verifiable-corpus k-cascade | done, marginal band yield collapses past k=8 |
| Distractor discrimination scaling | shelved, saturated |
