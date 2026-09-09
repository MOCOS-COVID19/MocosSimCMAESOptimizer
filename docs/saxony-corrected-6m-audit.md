# Audit of `saxony-corrected-tournament/baseline` at six months

Audit date: 2026-09-09  
Source snapshot: <https://github.com/marcinb-pwr/saxony-corrected-tournament>

## Executive finding

The reported best score, `0.3712726844`, is not the configured negative-binomial
training loss and is not a full-horizon goodness-of-fit percentage. Because
`rank_on_validation=true`, it is the unweighted mean of three relative mean
absolute errors (RMAEs) over days 153--180. Candidate 5/23 has validation RMAEs
of `0.300306` for detections, `0.300199` for deaths, and `0.513314` for
age-5--14 detections; their mean is exactly `0.371273`.
Only these three dimensions appeared because the legacy validation scorer
hard-coded that tuple; the other age-stratified series were calculated as
training diagnostics but were not included in candidate selection. The Phase 1
and Phase 2 configurations replace that legacy set with explicit weighted total,
age-stratified detection, and age-stratified death dimensions.

The same candidate is a poor calibration: training-window (days 1--152) RMAE
is `0.823826` for detections and `0.819735` for deaths, while training-window
cumulative relative errors are `0.823769` and `0.810301`. Its independent
full-stage predicted/observed totals
(`41,416/190,633` detections and `2,014/8,347` deaths) are consistent with
those errors. Selection has optimized the last 28 days at the expense of the
preceding 152 days.

## Error evolution

All 192 completed candidates (24 candidates in each of eight iterations) were
read from `optimizer_history.json`. The table reports the minimum and median
selection score and the training-window cumulative errors of the candidate with
the minimum selection score in each iteration.

| Iteration | Best candidate | Selection min | Selection median | Training detection cumulative error | Training death cumulative error | Training NB NLL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 0.506936 | 0.688963 | 0.886240 | 0.910377 | 4276.103 |
| 2 | 4  | 0.472611 | 0.677808 | 0.874080 | 0.883605 | 3527.416 |
| 3 | 13 | 0.433197 | 0.603552 | 0.816684 | 0.850204 | 2162.055 |
| 4 | 1  | 0.429114 | 0.582075 | 0.853453 | 0.864865 | 2856.561 |
| 5 | 23 | **0.371273** | **0.453194** | 0.823769 | 0.810301 | 2386.254 |
| 6 | 15 | 0.403249 | 0.623248 | 0.800905 | 0.820755 | 2203.767 |
| 7 | 18 | 0.400417 | 0.490888 | 0.850779 | 0.865630 | 2722.664 |
| 8 | 15 | 0.407346 | 0.486624 | 0.851436 | 0.868180 | 2838.649 |

The selection objective improved through iteration 5, then plateaued. The
training-window errors did not converge to an acceptable level. Across all
candidates, Pearson correlations between selection score and detection/death
cumulative errors are `-0.771` and `-0.753`: lower validation scores generally
correspond to *worse* cumulative fit. The best mean cumulative-error candidate
(iteration 6, candidate 11, mean `0.496899`) has a disastrous validation score
of `6.160599`. This is direct evidence of an objective conflict, not merely an
insufficient CMA step size.

The configured negative-binomial training NLL also disagrees with selection:
its correlation with the validation score is `-0.227`. It is computed and
stored, but does not rank candidates when `rank_on_validation` is enabled.

## Scalar evolution and sigma

| Iteration | Best school | Best class | Best age coupling | Population median school | Population median class | Population median age coupling |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0500 | 0.1838 | 0.5987 | 0.0500 | 0.2456 | 0.5996 |
| 2 | 0.0500 | 0.1780 | 0.5939 | 0.0570 | 0.2178 | 0.5991 |
| 3 | 0.0544 | 0.1841 | 0.5940 | 0.0500 | 0.2168 | 0.5975 |
| 4 | 0.0733 | 0.1796 | 0.5934 | 0.0609 | 0.2004 | 0.5939 |
| 5 | 0.0724 | 0.1615 | **0.5900** | 0.0641 | 0.1687 | 0.5934 |
| 6 | 0.0500 | 0.1593 | 0.6009 | 0.0610 | 0.1522 | 0.5953 |
| 7 | 0.0668 | 0.1487 | 0.6008 | 0.0636 | 0.1586 | 0.5954 |
| 8 | 0.0597 | 0.1800 | 0.5961 | 0.0639 | 0.1591 | 0.5981 |

`class` moved materially downward, `school` remained near its lower bound, and
the selected age-coupling value hit its lower bound in iteration 5 but did not
remain there. Univariate correlations of each scalar with validation and
cumulative errors are all weak (absolute value at most `0.09`). These are not
causal sensitivity estimates because all twelve coordinates moved together.

Configured sigma is already the implementation ceiling (`0.12`). At iteration
8 the normalized coordinate-wise sigmas for `(age coupling, class, school)`
were `(0.1069, 0.1200, 0.0611)`; several temporal coordinates also remained at
`0.12`. Raising only the JSON value cannot increase those coordinates while
the global `CMA_SIGMA_MAX` remains `0.12`, and wider exploration cannot repair
an objective that rewards the wrong temporal trade-off.

## Recommended experiment design

A scalar-first pilot is worthwhile as a **conditioning experiment**, not as a
claim that the scalars can be uniquely estimated independently of temporal
modulation. `age_coupling_param` should not simply be removed: it changes
mixing across ages and is scientifically relevant, while the age-specific
residuals are large. Fixing it arbitrarily would transfer its error into the
three modulation vectors.

Use the following sequence:

1. **Repair selection first.** Rank on a declared composite containing
   training-window detection/death cumulative error plus rolling-window shape
   error. Report every component and reject candidates whose total ratios fall
   outside a declared calibration band (a starting diagnostic band of
   `0.5--2.0`, not a final scientific threshold).
2. **Run a small scalar profile.** Freeze all modulation vectors at a few
   plausible reference trajectories. Evaluate a space-filling design over
   `school`, `class`, and `age_coupling_param`, with common random seeds. Use
   total detections/deaths, age-stratified errors, and the composite objective;
   do not use cumulative error alone.
3. **Retain uncertainty, do not point-freeze prematurely.** Carry the best
   several scalar triples (or a covariance/range) into short vector-only runs.
   If vector winners differ substantially by scalar triple, the parameters are
   confounded and must remain jointly optimized.
4. **Optimize temporal vectors conditionally.** Freeze scalars only for this
   comparison and fit infection, mild-detection, and tracing modulation. Use a
   reduced representation or tail-only buckets and common random numbers.
5. **Joint refinement.** Unfreeze the three scalars with a smaller marginal
   scale and jointly refine all retained solutions. Validate on multiple seeds
   and select using the same composite loss.

This staged/profiled design is preferable to either removing
`age_coupling_param` or increasing sigma globally. It reduces confounding and
dimension early, but the final joint refinement prevents bias from an
incorrect conditional scalar estimate.

## Artifact caveats

The published `final_best_candidate.json` is the original seed-like candidate,
not iteration 5 candidate 23; this is explained by the blocked-stage
bookkeeping defect fixed after this run. Candidate identity and metrics should
therefore be taken from `optimizer_history.json` and the candidate directories,
not that final file. The published validation replicates also used days
333--360 and all have non-finite negative-binomial scores, so they do not
validate the six-month winner and should not be used to choose parameters.

## Reproduction

The audit cloned the source snapshot and joined each history row to
`baseline/real_sims/horizon_6m/iter_<n>/cand_<nn>/config.json`. Per-iteration
minima and medians used finite `score` rows. Correlations are ordinary Pearson
correlations across all 192 candidates; they are descriptive and confounded,
not causal effects.
