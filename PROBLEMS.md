# Problem families

The task families ModelBreaker uses to locate and label inconsistency, with the construction, the difficulty knob, and the headline result for each. Families are scored per problem through the harvester unless noted. The aggregator collects every kept problem into a labelled corpus tagged by failure mode, 68 problems at the current count.

| Family | Construction | Difficulty knob | Headline result |
| --- | --- | --- | --- |
| Robustness battery | Edge, malformed, strict-format, needle and long-context inputs | Input type | 18 of 19 pass; the failures were token-cap truncation, not incapacity |
| Distractor sweep | One active code among N revoked decoys of identical shape | Decoy count | Saturated at tested levels, model robust; shelved |
| Arithmetic sweep | Multi-term integer arithmetic, strict left to right | Term count and magnitude | Hard failure is the blank, not the wrong answer |
| Algebra sweep | Coefficient of x^2 from a product of K linear factors | Factor count | Band hidden behind a truncation wall near K=5 |
| Calculus level-sweep | f'(x0) for a product of linear factors, scored per level | Factor count | ★ Band located at K=4, rate 0.667 |
| Calculus harvester, no-think | Same construction, scored per problem at k=16 | Factor count | Band at 5 to 8 factors, 9 of 40 kept, errors scatter |
| Calculus harvester, think | Same, thinking enabled | Factor count and cap | 0 kept; failure is non-termination |
| No-think confident-wrong search | Calculus, band floor 0.0 to 0.4 | Factor count | 1 kept; no dominant wrong value |
| Letter-count trap | Count a target letter in a word | Word and letter | ★ Confident-wrong, stable off-by-one |
| Novel-operator trap | A defined right-associative operator | Operand count | Strong band generator, errors scatter |
| Average-speed trap | Round trip at speeds a and b | Speed pair | Arithmetic-mean attractor; awaiting first run |

## Construction notes

**Robustness battery.** Edge, malformed, strict-format, and long-context inputs scored single-shot. It measures input handling rather than capability, and its apparent failures resolved to token-cap truncation once the budget was sized to the reasoning trace.

**Distractor sweep.** One active vault code is planted among N revoked decoys of the same shape under an active-code rule, with the decoy count as the knob. Retrieval held at 1.00 across the tested range, so discrimination under this rule sits below the model's threshold. Shelved as a difficulty axis.

**Arithmetic sweep.** A chained integer expression is evaluated strictly left to right, with the operation count as the knob. Difficulty tracks the specific expression, dominated by multiplications and the magnitude of the running value, rather than the raw step count, and the model's hard failure is a truncated blank rather than a committed wrong total.

**Algebra sweep.** A product of K linear factors is expanded and the coefficient of x^2 requested, ground truth from sympy. Committed wrong coefficients occur, but the expansion narration grows fast enough that truncation overtakes the band near K=5, behind a wall the token cap cannot reach.

**Calculus level-sweep and harvester.** f(x) is a product of K linear factors; the model computes f'(x), evaluates it at an integer point, and returns one integer, ground truth from sympy. The single committed integer makes this the first generated family to fail by wrong answers rather than blanks. The level-sweep located the band at K=4 (rate 0.667). The per-problem harvester then showed the band is mode-dependent: no-think keeps a band at 5 to 8 factors with scattered wrong answers and near-zero degeneracy, while think keeps nothing because hard problems fail by non-termination. Dropping the band floor toward zero confirmed that calculus error scatters rather than concentrating on one wrong value.

**Systematic-error trap families.** Each trap carries both the correct value and the predicted intuitive wrong answer, so the harness measures a trap-capture rate directly, the share of trials that commit the naive value. They were built to test whether confident-wrong is reachable where free arithmetic only scatters. Letter counting confirmed it: a stable off-by-one, almost always plus or minus one, with the direction word-specific. The novel-operator family, a defined right-associative operator whose intuitive answer is the left-fold, instead generates bands, with most errors scattering rather than landing on the fold. The average-speed trap, where the harmonic mean is correct and the arithmetic mean is the attractor, is awaiting its first run.

## Corpus

The aggregator deduplicates kept problems across runs and tags each with its failure mode, so the corpus records not just which problems break the model but how. Letter counting supplies the confident-wrong examples, the calculus harvester and novel-operator family supply the inconsistent-band examples, and scatter and non-termination cases are recorded but separated from the committed-error set. The labelling is the deliverable: a model that scatters, a model that stalls, and a model that is reliably misled are three different failures, and the corpus keeps them apart.
