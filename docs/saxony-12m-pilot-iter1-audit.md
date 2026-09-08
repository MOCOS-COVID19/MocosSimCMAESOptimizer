# Audit of `saxony-12m-pilot-iter1`

Audit date: 2026-09-08  
Result repository: <https://github.com/marcinb-pwr/saxony-12m-pilot-iter1>

## Executive finding

The run does not demonstrate that CMA-ES converges slowly. It demonstrates that
the experiment stopped before CMA-ES had enough generations to converge, while
a bookkeeping defect taught CMA from temporal coordinates that were not sent to
the simulator. The 12-month stage has 39 coordinates, a population of 24, and
only five generations. Twenty-four or fewer candidates completed each
generation. That is an exploratory budget, not a convergence budget.

The best 12-month score did improve from 1720.326 in generation 1 to 1540.602
in generation 5 (10.4%), and the population median improved from 1965.530 to
1804.596 (8.2%). This is continuing progress rather than a plateau. The best
candidate arrived in the final generation, so terminating there was especially
premature.

More importantly, the production loop applied historical temporal-prefix locks
to the candidate configuration but retained the pre-lock sampled vector for
ranking, archive output, and the CMA update. At horizon extension, CMA therefore
updated its distribution as if locked historical values had been evaluated.
The stored path norms and covariance condition numbers are consequently not
trustworthy diagnostics of the simulator response. This audit fixes that defect
by rebuilding the ranked vector from the effective post-transition config.

## Evidence from the artifacts

| Stage | Dimensions | Population | Generations | First-generation min | Final-generation min | Final best |
|---|---:|---:|---:|---:|---:|---:|
| 3 months | 12 | 16 | 4 | 108.289 | 105.249 | 100.990 (generation 2) |
| 6 months | 21 | 16 | 4 | 486.769 | 356.862 | 356.862 |
| 9 months | 30 | 24 | 4 | 593.998 | 593.970 | 593.970 |
| 12 months | 39 | 24 | 5 | 1720.326 | 1540.602 | 1540.602 |

Additional findings:

1. **The scores of different horizons are not comparable.** Negative log
   likelihood grows with the number of observations. The rise from 101 at three
   months to 1541 at twelve months is not evidence of deterioration. Report
   score per observed week/series, and compare generations only within a stage.
2. **The extension quality gate was effectively disabled.** Every stage used a
   threshold of 1,000,000, although observed scores were 101–1541. All gates
   passed irrespective of scientific fit.
3. **Only the `baseline` policy ran.** The leaderboard contains one row, so it
   provides no evidence that baseline beats `wide`, `narrow`, or
   `temporal_escape`.
4. **The final validation failed for all seeds.** Seeds 42, 43, and 44 have null
   scores and `sim_failed=true`. The simulator processes exited with code zero
   and wrote daily files, so this should be treated as an output-validation or
   scoring integration defect until the stored `output_error` explains it—not
   as a stochastic model failure. No out-of-sample or cross-seed conclusion can
   be drawn from this run.
5. **The nominal holdout is not a holdout from optimization.** Candidate
   artifacts report a rolling 28-day window inside each simulated/fitted
   horizon. It is diagnostic data already included in the full likelihood, not
   an independently withheld selection set.
6. **Bounds are active.** The three-month winner includes values at 0 and 1,
   while temporal values are also restricted to a 0.15 adjacent-bucket change.
   This suggests either an optimum outside the declared bounds, excessive
   confounding, or projection bias. Bound-hit rates need to be reported per
   coordinate before interpreting CMA step sizes.

## Recommended next run

Do not immediately replace CMA-ES. First make the experiment capable of giving
CMA-ES a fair, measurable test:

1. Apply this audit's effective-vector fix and discard transferred CMA state
   produced by the affected run. Candidate outputs remain useful for analysis,
   but their stored sampled/evaluated vectors and covariance are unsafe for
   resume.
2. Re-run 12 months from the best valid nine-month configuration, with the
   historical prefix fixed and an **active search space containing only the
   three scalar parameters and nine new temporal values**. Avoid carrying 27
   locked temporal coordinates in CMA's dimension.
3. Use at least 20 generations for the corrected pilot, with a plateau rule
   such as no material improvement in the best and median score for five
   generations. Five generations cannot establish a plateau.
4. Spend the first fixed budget on a small policy tournament: independent
   baseline, temporal-escape, and one restart run, using identical simulator
   seeds and candidate budgets. Select on rolling-origin validation loss, not
   training likelihood.
5. Evaluate each candidate initially with one common-random-number seed. Re-run
   candidates near the selection cutoff with two additional seeds. Rank by mean
   loss plus a standard-error penalty. This spends replication budget where
   simulator noise can change selection.
6. Replace the 1,000,000 gate with stage-relative criteria: minimum archive
   size, validation improvement over a naive epidemic baseline, acceptable
   bound-hit rate, and stability across seeds.
7. Repair final validation and require three finite replicate scores before the
   run can be called successful. Preserve a precise output error in the result
   artifact whenever an exit-zero adapter run is rejected.

## If corrected CMA-ES is still too expensive

The most promising alternative is not another black-box optimizer over all 39
coordinates. Reduce the temporal representation first:

- fit infection modulation with a low-dimensional spline or change points;
- derive detection/tracing modulation from known testing-policy covariates where
  possible;
- profile or condition on the scalar transmission parameters;
- then compare restart CMA-ES with Bayesian optimization or a surrogate-assisted
  evolutionary strategy under the same simulator-call budget.

For a stochastic, expensive simulator, a surrogate-assisted optimizer can be
more sample-efficient, but only after fixing validation, parameter
identifiability, and simulator-noise measurement. Switching optimizers before
those fixes would make a faster but still untrustworthy search.

## Acceptance criteria for the corrective pilot

- ranked `evaluated_vector` values exactly match every effective candidate
  configuration after locks and transition enforcement;
- no affected state from this first run is resumed;
- all validation replicates produce finite scores or a machine-readable output
  error;
- every compared policy receives the same number of simulator calls and seeds;
- convergence plots show best, median, interquartile range, bound-hit rate, and
  score per observation;
- stopping occurs by a declared plateau/budget rule, not merely because the
  configured generation count is exhausted.
