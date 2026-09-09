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

## Parameter-evolution audit

The 6-, 9-, and 12-month portion of `optimizer_history.json` contains 280
candidate rows. Five 12-month rows have no finite score, leaving 275 scored
candidate configurations (64, 96, and 115 by stage). Because history contains
candidate identity and scores but not parameter snapshots, this analysis joins
each row to `real_sims/<stage>/iter_<n>/cand_<nn>/config.json`. The repository's
`build_parameter_evolution_audit.py` utility makes that join explicit and
produces an interactive report rather than implying that values came directly
from the history JSON.

Population medians moved as follows from the first to final generation:

| Stage | `school` | `class` | `age_coupling_param` |
|---|---:|---:|---:|
| 6 months | 0.0568 → 0.0637 | 0.2482 → 0.3121 | 0.5985 → 0.5993 |
| 9 months | 0.1309 → 0.1956 | 0.3069 → 0.3204 | 0.6050 → 0.6116 |
| 12 months | 0.2009 → 0.2321 | 0.2836 → 0.2483 | 0.6114 → 0.6082 |

The strongest systematic scalar movement is therefore in `school`, while age
coupling stays in a narrow band. `class` changes direction in the 12-month
stage. The best candidates do not simply equal the final population medians:
their `(school, class, age_coupling_param)` values are respectively
`(0.2602, 0.3070, 0.6172)`, `(0.1959, 0.2927, 0.6117)`, and
`(0.2572, 0.2607, 0.6054)` for 6, 9, and 12 months. This is consistent with an
actively moving, weakly identified population—not evidence of scalar
convergence.

All three modulation configs retain fifteen monthly values. The handoff is
wrong for every modulation vector. The six-month winner is iteration 4,
candidate 9, but buckets 4–6 in the first nine-month configuration do not match
that winner. The nine-month winner is iteration 4, candidate 20, but buckets
7–9 in the first twelve-month configuration do not match it either. In each
case the values reverted to an older trajectory. Thus the later stages did not
start from all improvements fitted by their immediate predecessor.

The root cause is distinct from the sampled/evaluated-vector defect: stage
handoff selected the first diversity-archive entry as the prefix seed, while
the persisted `historical_trajectory` was the CMA population mean. Neither is
necessarily the best scored effective candidate. The corrective implementation
now makes the immediate predecessor's best effective configuration and vector
the sole owner of locked historical coordinates. Survivor archive entries are
still useful as protected diversity candidates, but cannot define the locked
prefix. Fresh-process loading verifies matching best-config hashes and rejects
a trajectory that differs from the best vector. Existing state from this run
does not contain that new proof and must not be resumed.

Use the report's handoff-integrity table to see the exact mismatching bucket
indices, its vector selector to inspect infection, mild-detection, and tracing
separately, and its stage filters to avoid joining lines across horizon changes.

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
3. Use maximum budgets of 10, 15, and 20 generations for the corrected 6-, 9-,
   and 12-month stages, with no material improvement in both best and median
   validation score for three generations as the plateau rule. Require at least
   six completed generations before that rule is eligible to stop a stage.
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

### Initial-sigma conclusion

The affected run cannot identify an optimal sigma: its CMA updates used vectors
that did not match the evaluated configurations, and the run contains only one
initial-sigma policy. There is therefore no valid counterfactual showing that
`0.12` beats a narrower scale.

For an unchanged `4/4/4/5`-generation exploratory rerun, keep configured sigma
at `0.12` for 3, 6, 9, and 12 months. With so few updates, reducing the initial
scale would spend more of the run expanding a distribution that already lacks a
convergence budget. This recommendation is conditional on monitoring projection
and bound hits; `0.12` was the pilot setting, not a measured optimum. The
implementation now permits `0.20`, while this historical pilot remains at
`0.12` for reproducibility.

For the corrected 20-or-more-generation experiment, compare `0.06`, `0.09`, and
`0.12` from identical predecessor state and simulator seeds. Rank the policies
on rolling-origin validation, use the same total candidate count, and record
actual marginal deviations (`sigma[i] * sqrt(covariance[i,i])`). Do not assign
different values merely from horizon length: later stages inherit coordinate
uncertainty, so their configured scalar is not the complete sampling scale.

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
