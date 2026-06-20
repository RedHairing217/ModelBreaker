# ModelBreaker

A data-centric LLM reliability experiment. The model and its serving are fixed, and task structure is the variable: generated and curated tasks are pushed across a difficulty knob to isolate the problems a locally served Qwen3-8B (4-bit MLX) solves only sometimes, and every kept problem is labelled by failure mode.

---

## What This Project Is

ModelBreaker holds the model architecture and serving fixed (Qwen3-8B, 4-bit MLX, via LM Studio at `http://127.0.0.1:1234/v1`) and treats task structure as the independent variable. Each experiment changes one thing, the construction of the task or the inference mode, and measures how the model's reliability moves. The aim is not an accuracy number but a record of where and how a small reasoning model becomes unreliable, captured as a corpus of problems labelled by failure mode.

I built the pipeline from an initial skeleton I uploaded as the starting structure, then extended it run by run. Across sessions I use a deliberate handoff: at the close of a session the assistant emits a compact skeleton, the module interfaces, file layout, and current state, and a fresh session reads that skeleton as its starting context and builds on it. Each session inherits the prior architecture rather than re-deriving it.

---

## Structure

**Generated families.** Procedural tasks (chained arithmetic, polynomial coefficient extraction, derivative at a point, and intuition traps) with sympy ground truth, swept across a difficulty knob. The harvester scores each problem over k trials rather than scoring a whole difficulty level as one rate, so individual band members are visible and wrong answers are separated from truncation.

**Curated families.** Single-answer theorems mined from research mathematics through the RealMath pipeline, graded by a deterministic verifier, and run through the same band prescreen. These reach a graduate difficulty the generated families cannot.

---

## Failure-mode taxonomy

After k trials a problem is one of four outcomes, and the project keeps two of them.

**Solved** passed every trial, and is discarded as uninformative. **Band** passed some trials and failed others, the inconsistency the project targets, and is kept. **Misdirection** failed every trial but committed the same wrong value each time, a stable systematic error, and is kept. **Collapse** failed every trial without converging, either scattering a different wrong value each time or never committing through truncation or a non-terminating reasoning loop, and is discarded.

Misdirection and collapse are the two consistent-failure outcomes, and separating them is deliberate: misdirection is a reproducible error worth recording, collapse is a give-up worth dropping. The earlier confident-wrong mode is misdirection that still sits in the band; scatter and non-termination are the two mechanisms of collapse.

---

## Results

| Legend | |
|--------|--------|
| ★ = record | **Bold** = band or misdirection located |

| Family | Difficulty knob | Outcome |
|--------|-----------------|---------|
| Robustness battery | Input type | 18 of 19 pass; failures were truncation, not incapacity |
| Distractor sweep | Decoy count | Saturated, model robust; shelved |
| Arithmetic sweep | Term count and magnitude | Collapse by non-termination, no band |
| Algebra sweep | Factor count | Band hidden behind a collapse wall near K=5 |
| Calculus level-sweep | Factor count | ★ **Band at K=4, rate 0.667** |
| Calculus harvester, no-think | Factor count | **Band at 5 to 8 factors**, errors scatter |
| Calculus harvester, think | Factor count and cap | Collapse by non-termination, no band |
| Letter-count trap | Word and letter | ★ **Misdirection, stable off-by-one** |
| Novel-operator trap | Operand count | **Band generator**, errors scatter |
| Average-speed trap | Speed pair | Awaiting first run |
| arXiv theorems (RealMath) | k sampling | ★ **116 curated, 70 band and 46 misdirection** |

## Key Results

The corpus is the deliverable. The generated sweep located the reliable, band, collapse progression in a single family: derivative at a point is reliable at K=3 (rate **0.89**), lands in band at K=4 (rate **0.667**), and collapses to truncation at K=5. The trap families then showed that a stable wrong commit, misdirection, appears only where the wrong answer is an attractor: letter counting produced a stable off-by-one in 47 of 120 problems, while calculus of equal difficulty only scattered.

The curated arXiv set added **116** graduate-level problems, **70 band** and **46 misdirection**, after a deterministic verifier and a curation pass. The misdirection cases are clean systematic errors, such as `2*n` answered as `n` and the recurrence `M_{n-1}` answered as `M_n`. Together the families cover the full taxonomy: a model that scatters, a model that stalls, and a model that is reliably misled, kept apart rather than pooled into one error count.

---

## What I Learned

**The reasoning trace determines the token budget.** With thinking enabled the model spends its output budget reasoning, so an undersized cap manufactures failures that resemble incapacity. A valid test sizes the cap to the trace. Utlimately, I was unable to find a band of problems with thinking enabled that produced an inconsistent band before collapsed, so this experiment is targeted for a model with thinking disabled.

**The hard failure is more consistently a model collapse than a wrong answer for low complexity problems.** On long arithmetic the model collapses by non-termination, looping or truncating before it commits. Raising the token ceiling relocates that collapse rather than removing it, because narration cost is set by the number of reasoning steps, not their type.

**Solving the failure mode for misdirection found a band of inconsistency.** Free arithmetic of a given difficulty collapses by scatter, dispersing wrong values with no dominant answer. Only a task whose wrong answer is a near-miss intuition, such as the off-by-one in letter counting, produces a stable wrong commit. Misdirection is therefore a property of task structure. Solving the failure mode to induce misdirection without collapse consequently produced a band of inconsistency.

**Misdirection and collapse are worth separating.** Both are consistent failures, but one is a reproducible systematic error worth recording and the other is non-convergence worth discarding. 
Counting errors without this split loses the distinction between misunderstanding and failure to process. 

This distinction is valuable because it produces a clear four layer gradient of model capability:

1. **Solved:** Always correct 
2. **Band:** Sometimes correct 
3. **Misdirection:** Misunderstands the problem and/or easily directed to a false but consistent conclusion 
4.  **Collapse:** Unable to process the problem

**Sampling has a practical resolution.** On the arXiv set a k-cascade harvested most band members at k=8, and raising to k=12 returned one band problem in 35 re-examined. Past k=8 the marginal yield collapses, so additional trials buy almost nothing for this set.

**The deterministic verifier bounds the curated corpus.** Relational and piecewise answers are common in research mathematics and cannot be graded by symbolic equality, so the answer-bearing fraction, sets the ceiling on yield. Curation dropped 40 ground-truth artefacts from 156 to leave 116.

**A band prescreen is robust to training contamination.** A memorised problem passes consistently and is discarded as solved, so contamination removes problems from the corpus rather than corrupting it. This is why a published dataset could replace a fresh scrape once the fresh yield proved too low.

---

## Methodology

Each experiment follows a consistent structure:
1. Identify the failure mode or bottleneck under test through trial classification, false-answer analysis, or the prior run.
2. Form a falsifiable hypothesis about which task structure or inference mode will move the targeted outcome.
3. Change one variable at a time.
4. Harvest each problem over k trials, classify every trial correct, wrong_complete, or degenerate, and assign the problem an outcome.
5. Document the result, the conclusion, and what changes for the next run.

The pass band is centralised in `src/shared/bands.py`, and the per-problem classification in `src/shared/harvest.py`, so every family shares one definition of band membership.

---

## Technical Stack

- **Model:** Qwen3-8B, 4-bit MLX quantisation, served locally
- **Serving:** LM Studio at `http://127.0.0.1:1234/v1`, both thinking and no-thinking modes
- **Hardware:** Apple Silicon, MPS backend
- **Ground truth:** sympy symbolic equality with a numeric-sampling fallback; LaTeX answers parsed through antlr
- **Curated extraction:** the RealMath pipeline, ported from the OpenAI client to a drop-in Anthropic shim so the calls run on Claude
- **Pipeline:** per-problem harvester, band prescreen with early-exit, k-cascade, deterministic verifier, and a curation pass
- **Corpus:** the generated and trap families plus 116 curated arXiv problems, every kept problem labelled by failure mode

---

## Repository

Full run history, per-experiment analysis, and implementation detail are in separate documents:

- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md): experiment summary, per-experiment hypothesis and result records, persistent findings, pending table
- [`PROBLEMS.md`](PROBLEMS.md): task families, construction, difficulty knob, and headline result
- [`README_legacy.md`](README_legacy.md): setup, CLI usage, the dashboards, input categories, and report format
