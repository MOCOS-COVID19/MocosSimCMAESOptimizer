# Methodology

## Overview

This optimizer fits simulator parameters to ground-truth (GT) time series using a CMA-ES-like search loop over:

- scalar parameters
- temporal parameters represented as arrays of bucketed values

The current production path is centered on:

- one active optimization stage at a time
- external simulation via Slurm array jobs
- partial-completion iteration cutoffs
- resume-time temporal de-locking to avoid inherited local minima

---

## Objective function

The active score is a weighted sum of per-metric errors computed from simulation output vs GT:

- `daily_detections`
- `daily_hospitalizations`
- `daily_deaths`
- `daily_student_detections`
- cumulative variants for selected metrics

The combined score is lower-is-better.

### Student detections

`daily_student_detections` is treated specially:

- GT is interpreted as weekly-style known infected counts
- comparison uses 7-day averaged / windowed handling
- cumulative student error can be disabled via:
  - `daily_student_detections_cumulative = 0.0`

---

## Search algorithm

The optimizer uses a CMA-style population search:

1. sample a population from current mean/covariance/sigma
2. clip to parameter bounds
3. run simulation for each candidate
4. compute candidate scores
5. rank candidates
6. update mean/covariance/sigma from the best subset

The optimizer persists stage state so later reruns can resume from previous search state.

---

## Resume behavior

When reusable optimizer state exists, the optimizer warm-starts from:

- prior mean
- prior covariance
- prior sigma

This is useful, but can trap temporal parameters near older minima. To mitigate that, the current methodology includes two temporal unlock phases.

---

## Temporal unlock: Phase 1

Implemented in `build_state_from_reusable(...)`.

Purpose:
- keep scalar warm-starts stable
- loosen temporal parameters on resume

Behavior:
- temporal means get random jitter
- temporal covariance diagonals are inflated
- later temporal buckets get stronger unlocking than earlier buckets

This helps escape older temporal minima inherited from shorter-horizon or repeatedly resumed runs.

---

## Temporal unlock: Phase 2

Phase 2 makes resume-time unlock depend on actual GT mismatch.

### Bucket error generation

During candidate scoring, the optimizer now writes bucket-level GT-vs-sim errors into `metrics.json`:

```json
"bucket_errors": {
  "infection_modulation.params.interval_values": [...],
  "mild_detection_modulation.params.interval_values": [...],
  "tracing_modulation.params.interval_values": [...]
}
```

These are computed using:

- `temporal_bucket_day_ranges(...)`
- GT series
- candidate simulation output

### Resume-time use

When reusable state is loaded, the optimizer reads the latest iteration top candidates and uses the best candidate’s bucket errors to unlock temporal buckets:

- larger bucket error => stronger perturbation
- smaller bucket error => weaker perturbation

This makes the optimizer less sticky exactly where weekly GT fit is poor.

---

## Slurm execution model

The primary production path uses Slurm array jobs.

For each optimizer iteration:

1. candidate configs are written to candidate directories
2. a Slurm array is submitted
3. each task runs simulation and writes output artifacts
4. the main optimizer waits for enough completed candidates
5. remaining long-tail candidates may be skipped/canceled
6. the optimizer updates CMA state and advances to the next iteration

---

## Partial-completion iteration policy

To avoid waiting for the slowest tasks, the optimizer uses:

- `min_completion_fraction`
- `finish_iter_delay`

### Meaning

If population size is `N`, then:

```text
threshold = ceil(N * min_completion_fraction)
```

Once at least `threshold` candidates complete:

1. start a grace timer
2. wait `finish_iter_delay` seconds
3. mark leftovers as skipped
4. cancel remaining Slurm array tasks
5. continue to the next iteration

### Current recommended values

```json
"min_completion_fraction": 0.5,
"finish_iter_delay": 15
```

This means:

- with population size `50`
- threshold is `25`
- once `25` candidates finish, wait `15` more seconds
- then skip/cancel leftovers and proceed

---

## Candidate statuses

Candidates may end up with statuses such as:

- `completed`
- `skipped`
- `failed`

Skipped candidates are treated as terminal and are not used in ranking.

---

## Iteration logging

Per-candidate iteration records are appended to `iter_metrics.jsonl`.

Important fields include:

- `stage`
- `iteration`
- `candidate`
- `score`
- `status`
- `threshold_reached`
- `iteration_truncated`
- `completed_count`
- `failed_count`
- `pending_count`

### Interpretation

- `threshold_reached = true`
  - minimum completion fraction was reached
- `iteration_truncated = true`
  - the optimizer stopped waiting, skipped leftovers, and moved on

This is iteration-level context attached to candidate rows for observability.

---

## Top-K candidate persistence

The optimizer keeps top candidates in two scopes:

### Per iteration

Saved to:

```text
real_sims/<stage>/iter_<n>/top_candidates.json
```

These include:

- score
- config
- metrics
- rank within iteration

### Per stage

Saved to:

```text
<output_dir>/<stage>_top_candidates.json
```

and also embedded in:

```text
<output_dir>/<stage>_summary.json
```

This supports shortlist analysis and future ensemble-style workflows.

---

## Resume bookkeeping

The optimizer infers resume state from persisted stage artifacts.

It uses:

- `stage_state.json`
- iteration directories
- candidate-list artifacts
- top-candidate artifacts when available

The goal is to continue from the latest completed iteration rather than restart from scratch.

---

## Logging and diagnostics

The optimizer logs:

- stage start
- iteration start
- Slurm submission
- wait result
- CMA update
- iteration finish
- stage finish

Critical JSON writes are wrapped to report which artifact failed if persistence errors occur.

---

## Practical operating notes

### If the optimizer is too sticky

Likely causes:

- heavy resume-state reuse
- temporal covariance too narrow
- inherited weekly shapes from older runs

Mitigation already implemented:

- Phase 1 temporal unlock
- Phase 2 GT bucket-error-driven unlock

### If iterations are too slow

Adjust:

- `min_completion_fraction`
- `finish_iter_delay`

and verify that:

- skipped candidates are being canceled
- plotting is not on the critical path for candidate completion

---

## Current recommended objective block

```json
"objective": {
  "weights": {
    "daily_detections": 1.0,
    "daily_hospitalizations": 0.0,
    "daily_deaths": 1.0,
    "daily_student_detections": 0.2,
    "daily_detections_cumulative": 1.0,
    "daily_hospitalizations_cumulative": 0.0,
    "daily_deaths_cumulative": 1.0,
    "daily_student_detections_cumulative": 0.0
  },
  "top_k": 5,
  "min_completion_fraction": 0.5,
  "finish_iter_delay": 15
}
```

Adjust weights as needed for your fitting priorities.
