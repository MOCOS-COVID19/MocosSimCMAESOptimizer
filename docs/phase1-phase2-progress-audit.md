# Phase 1 / phase 2 optimization-progress audit

Audit date: 2026-09-09. Evidence was read from the published result repositories
[`saxony-corrected-phase1-scalars`](https://github.com/marcinb-pwr/saxony-corrected-phase1-scalars)
and [`saxony-corrected-phase2-vectors`](https://github.com/marcinb-pwr/saxony-corrected-phase2-vectors).

## Verdict

Phase 1 **does** retain its lowest selection-score candidate. Across its 96
history rows the minimum is `0.5512370210471056`, candidate 9 of iteration 6.
That value equals the stage summary's `best_score`, and the candidate's config
is structurally equal to both `phase1_scalar_6m_best_candidate.json` and
`final_best_candidate.json`. Its fitted scalars are school `0.05`, class
`0.12550451424694703`, and age coupling `0.5938804583997165`. The running best
also falls monotonically from `0.6132914672847019` in iteration 1 through
`0.5512370210471056` in iteration 6 and remains there in iterations 7 and 8.

The overall two-phase procedure nevertheless does **not** preserve or prove an
improvement over that incumbent:

* Phase 2 reports a best selection score of `0.6358084276875294`, 15.34% worse
  than phase 1's `0.5512370210471056`. Its final negative-binomial validation
  objective is `2276.5423367135854`, versus `621.587450092104` for phase 1.
* Phase 2 never evaluates its input seed as a named incumbent. A non-elitist
  CMA population is sampled immediately, so neither its history nor its final
  result establishes that the phase-1 candidate was compared under phase 2's
  objective.
* Published phase-2 configs claim only monthly coordinates 4--6 were active,
  but all top configs overwrite coordinates 1--3 with copies of 4--6. For the
  winning infection vector, for example, the first six values are
  `[0.6519145582, 0.7180173525, 0.6378631247, 0.6519145582, 0.7180173525,
  0.6378631247]`, rather than preserving phase 1's prefix
  `[0.8547299977, 0.8535603259, 0.7409015498]`. Thus the published phase-2 run
  does not satisfy its own declared coordinate identity.
* The orchestration had a concrete archive-transfer bug: the Slurm preparation
  and local paths constructed an archive-derived `cand_cfg` but submitted the
  unrelated `candidate_cfg`. Archive transfer therefore existed in metadata
  while the simulator received the sampled config.

The score around `0.55` is the composite **selection** loss (40% validation
mean error, 30% cumulative detections error, 30% cumulative deaths error), not
negative-binomial loss. That is why it differs by three orders of magnitude
from `validation_replicates.json`. Reports must label these separately; they
cannot be compared as if they were the same objective.

## Fix implemented

1. Reserve the final population slot for the effective incumbent on every
   iteration, including iteration 1. A separately launched phase 2 therefore
   evaluates its phase-1 seed and can only replace it with a lower score under
   the declared phase-2 selection loss.
2. Mark that row as `stage_incumbent` in candidate provenance, so the evidence
   is directly auditable.
3. Correct both misspelled archive-transfer branches so the config built from
   the archive vector is the config actually checked, saved, and simulated.
4. Raise the normalized sigma ceiling and both phase initial scales from
   `0.12` to `0.20`. This makes the requested wider initial exploration real;
   changing only JSON would otherwise have been silently clamped to `0.12`.

Existing published results should not be retroactively relabelled. Re-run both
phases into fresh output directories and require, before accepting phase 2:
(a) a `stage_incumbent` row in iteration 1, (b) its config prefix equals the
phase-1 final prefix, and (c) phase 2's final selection score is no greater than
that incumbent row's score. Compare negative-binomial validation scores only to
negative-binomial validation scores.
