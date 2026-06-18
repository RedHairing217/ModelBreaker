# Problem catalogue

The task families ModelBreaker has run against Qwen3-8B (4-bit MLX, LM Studio), how each is
constructed, the difficulty knob it varies, and the result it produced. Results are framed by
the failure-mode taxonomy in the README: a truncation artifact (a blank caused by the token
cap, not by incapacity), scatter (stochastic near-misses), non-termination (the reasoning phase
never commits), and confident-wrong (a stable dominant wrong value).

| Family | Construction | Difficulty knob | Headline result |
| --- | --- | --- | --- |
| Robustness battery | Edge, malformed, strict-format, and needle and long-context inputs | Input type | 18 of 19 pass; the failures were token-cap truncation, not incapacity |
| Distractor sweep | Task plus irrelevant clauses | Distractor count | Saturated at tested levels, model robust; shelved |
| Arithmetic sweep | Multi-term integer arithmetic | Term count and magnitude | Hard failure mode is the blank non-answer, not the wrong answer |
| Algebra sweep | Generated algebra across a difficulty range | Difficulty | Ran in the sweep family; no distinct band was singled out |
| Calculus level-sweep | f'(x0) for a product of linear factors, scored per factor-count level | Factor count | Band located near K=4 (rate 0.667), later shown truncation-contaminated at higher k |
| Calculus harvester, no-think | Same construction, scored per problem at k=16 | Factor count | Band at 5 to 8 factors, 9 of 40 kept, near-zero degeneracy; wrong answers scatter |
| Calculus harvester, think | Same, thinking enabled | Factor count and token cap | Kept 0; failure is non-termination, band unmeasurable in this mode |
| No-think confident-wrong search | Calculus, band floor dropped to 0.0 to 0.4 | Factor count | 1 kept; wrong answers scatter, no dominant wrong value |
| Average-speed trap | Round trip at speeds a and b; arithmetic mean is the attractor | Speed pair | Did not bite; 100 percent correct, the famous harmonic trap is trained through |
| Novel-operator trap | Custom right-to-left operator; left-fold is the attractor | Term count | 80 percent correct; errors scatter, no concentration on the predicted fold |
| Letter-count (the strawberry class) | Count a recurring letter in a real word | Word and letter | Confident-wrong located; at pool 120, 47 commit a stable wrong count, almost always an off-by-one with word-specific direction, dispersion ~1 value |

## Robustness battery

Edge, malformed, strict-format, and needle and long-context inputs, run once each to separate
handling failures from capability failures. 18 of 19 passed. The four initial failures (strict
JSON and three needle depths) all terminated on `length`, meaning the answer was truncated before
it was emitted, and raising the cap recovered all four. Needle retrieval was clean at depths 0.1,
0.5, and 0.9 with no lost-in-the-middle effect, while the overflow case stalls to the client
timeout rather than rejecting cleanly. The lesson the rest of the project inherits: single-shot
pass or fail is dominated by reasoning verbosity, so an undersized `max_tokens` manufactures
failures that look like incapacity.

## Distractor sweep

The task with irrelevant clauses appended, swept by distractor count. The discrimination
saturated at the levels tested, the model stayed robust, and the sweep was shelved as
uninformative for band-finding.

## Arithmetic and algebra sweeps

Procedurally generated integer arithmetic and algebra across a difficulty knob. The durable
finding came from arithmetic: the hard failure mode is the blank non-answer, a degenerate trial
that commits nothing, not a committed wrong answer. This is the distinction, a blank masquerading
as a wrong answer, that the per-problem harvester later promoted to a first-class filter. The
algebra sweep ran in the same family without a separately notable band result.

## Calculus level-sweep

f'(x0) for a product of linear factors (roots integer, answer integer), scored as one rate per
factor-count level. It located a corruption band near K=4 at rate 0.667, judged genuine at k=3
with median tokens just under the cap. Later work qualified this: three trials did not resolve the
truncation tail that sixteen expose, so the level rate blended genuine variability with
truncation. This motivated changing the unit of analysis from the level to the single problem.

## Calculus per-problem harvester

The same calculus construction, but each candidate runs k=16 and is classified as correct,
wrong-complete, or degenerate, with a problem kept only when its in-band rate comes from the
correct versus wrong split at near-zero degeneracy. A paired pool-40 run (seed 0) split sharply by
mode. No-think kept 9 of 40, a clean band across 5 to 8 factors at pass@16 0.31 to 0.69, the model
committing definite wrong totals on the long arithmetic. Think kept 0: at the 4096 cap the
mid-range failures were blanks, and raising the cap to 16384 exposed the real mechanism, which is
non-termination of the reasoning phase rather than truncation. A streaming compression-ratio loop
detector now bounds the think cost, cutting loops near 3200 to 3700 tokens instead of running to
the cap.

## No-think confident-wrong search

The calculus harvester with the band floor dropped to 0.0 to 0.4 to harvest the reliably-wrong
end. It returned a clean negative. One problem landed in band, and across every deep-pass problem
the correct answer was the modal commit while the wrong trials scattered, the tightest case
clustering within plus or minus four of the truth. No-think does not commit a stable wrong belief
on this task: it holds the method and magnitude and slips on the large arithmetic differently each
trial. A larger pool-200 sweep, run partially, refined this: errors are predominantly scatter, but
the value 0 acts as a weak systematic attractor on some hard problems, one committing 0 on 10 of 16
trials against a six-figure answer. So confident-wrong is rare in arithmetic rather than strictly
absent, surfacing as a zero default. This contrast is what motivated the systematic-error trap family.

## Systematic-error trap families

These families move the difficulty out of arithmetic and into method or perception, so that a
wrong but deterministic response yields the same wrong value every trial. Each problem carries
both the correct answer and, where one can be predicted, the naive attractor; the harness selects
on top wrong-value share, the single most common wrong commit over k, which detects a stable wrong
answer without needing the attractor predicted in advance.

The average-speed trap (round trip at speeds a and b, harmonic mean correct, arithmetic mean the
attractor) did not bite at all: 100 percent correct, zero capture. The harmonic trap is textbook,
so the model has been trained through it, and the problems were trivially easy besides. The
novel-operator trap (a custom operation evaluated right to left, where the habitual left-to-right
fold is the attractor) was the fair non-canonical test. At pool 80 it was harder, 82 percent
correct, but the errors scattered rather than landing on the predicted fold: a median of two to
three distinct values per problem, with only 2 of 80 concentrating and neither on the predicted
fold. It is, though, a strong band generator: 55 of 80 problems are mid-band as difficulty scales
with term count. Two negatives for confident-wrong, the second a fair one, plus a useful
scatter-band source.

Letter-count is where confident-wrong appeared. Counting a recurring letter in a real word is
trivially verifiable and genuinely non-math, and the error is structural, driven by tokenization
rather than a reasoning gotcha that training can sand down. The model commits one value and repeats
it: dispersion collapses to a median of one distinct answer per problem against the math families'
spread of up to eleven. At a pool of 120, 47 problems have a stable wrong modal commit, 38 of them
on at least 80 percent of trials and many at 16 of 16. The error is almost always exactly one off
the truth, and the direction is fixed per word but splits roughly evenly across words (23 undercount
against 24 overcount among the kept set), so it is a per-word off-by-one that survives resampling,
not a uniform directional bias and not random scatter: "calendar" counted as one a, "success" as two
s's, "blueberry" as three b's, "community" as one m. This is the consistency-defeating mode: sixteen
samples agree and the agreement is wrong, which is exactly what a self-consistency confidence check
would miss. It completes the taxonomy with a real instance of confident-wrong, located where the
error is structural and absent from verifiable math where errors are stochastic.

## Aggregated corpus

The aggregator consolidates the kept problems from all of the above into one JSONL labeled by
failure mode and thinking mode. The current corpus is 68 inconsistently-solved problems: 55
confident-wrong (52 counting off-by-ones, 2 novel-operator, 1 calculus zero-attractor) and 13
scatter (calculus band problems). It is the single place the harvested problems live, each carrying
not just a pass rate but the mode of failure.
