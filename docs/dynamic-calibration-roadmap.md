# Audit and roadmap for dynamic calibration of Saxony 2020–2022 data

## 1. Goal and definition of done

The production goal should be a **reproducible calibration process** that automatically:

1. validates its inputs and runs the real MocosSim simulator;
2. learns a distribution over parameters rather than only a single “best” vector;
3. extends the time horizon without refitting the historical prefix;
4. responds to stalled improvement, simulator noise, and epidemic regime changes;
5. selects a model using data that was not used for training;
6. finishes with multi-seed validation and reports uncertainty and identifiability.

“Dynamic” should not mean only that some parameters have monthly values. It
should mean that the controller chooses the next experiment from accumulated
results, monitors convergence, and can resume safely.

## 2. What the repository already provides

### Strong foundations

- A staged CMA-ES loop samples and clips a population, ranks it, and updates the
  mean, per-coordinate `sigma`, covariance, and evolution paths.
- Scalar and temporal parameters are mapped by name. A longer-horizon stage
  transfers common coordinates, creates new intervals, and enforces a maximum
  change to the historical prefix.
- Scoring covers detections, hospitalizations, deaths, and student detections,
  including weekly, age-stratified, and cumulative forms. Day identities are
  preserved when observations are missing.
- Per-bucket errors are recorded for temporal parameters and used to unlock
  poorly fitted periods more strongly after resume.
- The pipeline has a diverse survivor archive, stage transfer, random
  immigrants, RNG persistence, atomic commits, and artifact-integrity checks.
- Local and Slurm execution paths and final validation replicates exist. The
  posterior is, however, a diagonal-quadratic approximation fitted to the CMA
  archive, not a posterior predictive distribution of the epidemic model.

### Data and configuration status

- The main detection and death series contain 907 days, age-stratified series
  contain 900 days, hospitalizations contain 1,170 days, and student
  observations are sparse (51 entries from day 72 through day 590). A shared
  calendar, explicit availability masks, and a policy for differing end dates
  are therefore required.
- The direct optimizer configuration contains only a three-month stage; the
  pipeline configuration contains only three- and twelve-month stages. Neither
  covers the stated 2020–2022 period (approximately 30 months in the supplied
  series).
- Paths to Julia, the launcher, seed configuration, and output directory are
  machine-specific absolute paths. The referenced `config13.json` is absent
  from the repository (other example seeds are present), so the default
  configuration cannot run on a fresh checkout without external files.
- The README correctly describes the verified path as fixture-backed and the
  production simulator/Slurm execution as deferred. Orchestration contracts
  are consequently better tested than the scientific validity of calibration.

## 3. Most important gaps

### P0 — blockers for a credible experiment

1. **No portable, verified end-to-end run.** Preflight validates schemas and
   paths, but the repository does not provide a runnable Saxony profile or a
   test against the real adapter. Establish one small, deterministically
   reproducible local run first, followed by its Slurm equivalent.
2. **Validation is not a true holdout.** `validation_score_from_daily` reports
   the last 28 days of the same horizon used by the objective. Final replicates
   change the random seed but still do not test temporal generalization. This
   risks selecting parameters fitted to the entire observed trajectory.
3. **The observation model is underspecified.** Reported cases, deaths, and
   hospitalizations have different delays, dispersion, and completeness. The
   current weighted sum of RMAE and cumulative terms counts some information
   more than once and does not model observation noise. The `0/1/0.5` weights
   are heuristic.
4. **Stages do not cover the full period.** A 900-day study requires stages up
   to 30 months, or an explicit shorter study cutoff. The current result cannot
   answer a question about all of 2020–2022.
5. **There is no scientific quality gate.** File integrity and archive quality
   do not establish whether the model reproduces waves, peaks, totals, and age
   structure out of sample. Posterior predictive criteria and comparison with
   a simple baseline are required.

### P1 — automatic adaptation

1. The `determine` policy chooses a strategy solely from horizon length, using
   thresholds at 8 and 10 months. It is not a result-driven policy tournament.
2. There is no plateau-based early stop, simulation budget, diversity-collapse
   restart, or result-driven population-size adjustment.
3. A single candidate evaluation can be noisy. Replicates currently run only
   for the final candidate; CMA should adaptively repeat candidates whose
   confidence intervals overlap near the selection boundary.
4. Monthly buckets are rigid. Variants, restrictions, school holidays, and
   reporting changes need not align with 30-day boundaries. Regularization and
   a comparison of weekly/monthly buckets with change points are needed.
5. NUTS operates on a local diagonal surrogate. It can propose candidates, but
   its samples should not yet be interpreted as calibrated simulator-parameter
   uncertainty.

### P2 — operations and research auditability

- There is no single experiment manifest containing simulator and optimizer
  revisions, data identities, configuration, environment, and seeds.
- Contract tests against a versioned, small real JLD2 sample and a local–Slurm
  parity test are missing.
- Existing plots focus on optimization progress. A scientific report should
  cover train/validation/test results, predictive intervals, temporal and
  age-specific residuals, peak coverage, and parameter stability across seeds.

## 4. Recommended target architecture

```text
versioned data + event calendar
        │
        ▼
preflight → experiment plan → candidate generator
                                │ CMA / restart / policy bandit
                                ▼
MocosSim adapter → adaptive replicates → observation model
                                │
                                ▼
training objective + temporal validation + uncertainty
        │
        ├─ plateau? restart / switch policy / add replicates
        ├─ quality gate? extend the horizon
        └─ final: frozen test + posterior predictive report
```

Keep three responsibilities separate:

1. **process simulator** — transmission and interventions;
2. **observation model** — detection, delay, day-of-week effects, and dispersion;
3. **experiment controller** — CMA, replicates, budget, stop/restart, and transfer.

Otherwise transmission parameters will compensate for reporting artifacts, and
an improved score will not have an unambiguous epidemiological interpretation.

## 5. Work plan and acceptance criteria

### Milestone 1 — working production baseline (P0)

- Add a portable `optimizer_config.saxony.example.json` using relative paths or
  environment variables, and document required external data and launcher
  versions.
- Add stages such as 3, 6, 12, 18, 24, and 30 months. Derive the final day from
  the intersection of the selected study period and required metrics, rather
  than from the longest file.
- Add a `preflight --no-launch` command and a time-bounded smoke test of the
  real adapter.
- Record SHA-256 identities for every input and the optimizer and launcher
  commits.

**Acceptance:** given the documented external inputs, a fresh checkout runs one
iteration with two candidates, produces real JLD2 output, and records an
equivalent manifest for local and Slurm execution.

### Milestone 2 — sound data protocol (P0)

- Build a canonical `date/day/metric/value/source/status` table, checking for
  duplicates, gaps, negative values, revisions, and timezone assumptions.
- Define the calendar date represented by day 1 and document every Saxony source
  transformation. Remove duplicate student-series names or declare one alias.
- Use rolling-origin training and validation, plus a **frozen test set** that is
  invisible to optimization and weight selection.

**Acceptance:** a data-quality report is generated before simulation; test data
cannot enter a score or CMA decision; every plotted observation has a date and
source.

### Milestone 3 — observation model and objective (P0)

- Replace the duplicated heuristic error sum with a likelihood, for example a
  Negative Binomial count model with metric-specific dispersion and delay
  convolution. Model day-of-week effects and reporting-definition changes
  explicitly.
- Preserve multiple metrics in reporting; rank candidates with a predeclared
  likelihood or a Pareto rule with an explicit final selection rule. Tune any
  weights only on validation data.
- Add synthetic recovery tests: known parameters should be recovered within
  uncertainty, and shifting a series by one week should worsen the score.

**Acceptance:** parameters are recovered on synthetic data, valid masks never
produce NaN/Inf, and validation performance beats a seasonal or last-week
baseline.

### Milestone 4 — adaptive controller (P1)

- Use validation trends, effective candidate diversity, covariance condition
  number, and bound-hit rates to implement plateau stopping, IPOP/BIPOP
  restarts, population adjustment, and policy switching.
- Run policies as a bandit or successive-halving experiment under a shared
  budget. The leaderboard must compare them on the same stage and seeds rather
  than aggregate one policy across different horizons.
- Add sequential replicates for candidates close to the selection cutoff. Rank
  on expected validation loss with an uncertainty penalty.
- Extend the horizon only after validation quality, cross-seed stability, and
  minimum archive diversity pass together.

**Acceptance:** under a fixed budget and across independent runs, the controller
achieves median validation loss no worse than the current CMA, terminates
plateaus automatically, and records an explicit reason for every decision.

### Milestone 5 — uncertainty and final report (P1/P2)

- Treat the current NUTS surrogate as a proposal mechanism. Validate uncertainty
  with posterior predictive simulations; if needed, use a full-covariance
  emulator or likelihood-free inference such as SNPE/ABC.
- Run independent seeds for the selected population and report the median plus
  50%, 90%, and 95% intervals instead of only the best trajectory.
- Generate an experiment card containing the time range, data, metrics,
  fitted/frozen parameters, budget, failures, coverage, residuals, and known
  limitations.

**Acceptance:** the frozen test runs exactly once after model selection; the
report includes coverage and sensitivity analysis and is reproducible from its
manifest.

## 6. Minimum next experiment for these data

1. Establish the calendar date corresponding to `day=1` and the common study
   cutoff.
2. Require detections, deaths, and hospitalizations. Treat student observations
   as a sparse auxiliary target only where they are available.
3. Reserve the final 8–12 weeks as test data and the preceding 8 weeks for
   rolling validation.
4. Run the 3- and 6-month stages locally with 2–3 seeds and examine delays and
   identifiability before running 12/18/24/30-month stages.
5. Compare the current weighted RMAE, a Negative Binomial likelihood, and a
   simple baseline.
6. Do not interpret fitted parameters epidemiologically until synthetic
   recovery and posterior predictive checks establish identifiability.

## 7. Prioritized backlog

| Order | Task | Priority | Dependency |
|---:|---|:---:|---|
| 1 | Portable profile and real smoke test | P0 | launcher and complete inputs |
| 2 | Canonical calendar and data-quality report | P0 | study-period definition |
| 3 | Temporal train/validation/frozen-test split | P0 | task 2 |
| 4 | Likelihood/observation model and synthetic test | P0 | tasks 1–3 |
| 5 | Stages covering the full 2020–2022 horizon | P0 | tasks 1–4 |
| 6 | Adaptive replicates and noise-robust ranking | P1 | task 4 |
| 7 | Plateau/restart/budget and a real policy tournament | P1 | task 6 |
| 8 | Posterior predictive checks and scientific report | P1 | tasks 4–7 |
| 9 | Slurm parity, cost monitoring, and operations runbook | P2 | stable pipeline |

Tasks 1–4 deliver the most value first. Increasing population size, iteration
count, or NUTS complexity before removing validation leakage and defining the
observation model would increase cost without increasing credibility.
