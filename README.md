# MocosSimCMAESOptimizer

This package implements the staged CMA-ES/NUTS orchestration pipeline. The reliable path is currently fixture-backed: it exercises stage transitions, archive handoff, scoring, persistence, and resume semantics without launching a live simulator.

## Completed reliable pipeline

- **Canonical transitions:** stage changes use name-based parameter transitions and explicit effective-coordinate delta reports; positional coincidence is not used as identity.
- **Paired-index scoring:** observed and simulated series are paired by their original indices before scoring, preserving alignment when rows are filtered or rejected.
- **Reliable archive:** survivor archives adapt around 30–50 entries (subject to the configured bounds) and are admitted only when the archive quality gate passes. The immediate predecessor archive and its transfer manifest are consumed by the next stage.
- **Commit and resume integrity:** production and fixture commits have distinct schemas and complete artifact key/hash manifests. Validation rejects missing, extra, tampered, or fixture-shaped production artifacts; interruption/resume restores the RNG stream, population, archive, and next iteration deterministically.
- **Readiness:** the four-stage, 24-month pipeline readiness check is fixture-only and verifies orchestration without starting an external simulation.

## Julia 1.7 fixture checks

Use the repository's Julia 1.7 binary and project environment:

```sh
JULIA=/Users/marcinbodych/Workspace/saxocov/julia-1.7.0/bin/julia
$JULIA --project=. tests/scoring_contract.jl
$JULIA --project=. tests/transition_contract.jl
$JULIA --project=. tests/archive_two_stage_contract.jl
$JULIA --project=. tests/orchestration_commit_idempotence.jl
$JULIA --project=. tests/transition_orchestration_fixtures.jl
$JULIA --project=. tests/readiness_no_launch_fixture.jl
```

The readiness fixture is intentionally no-launch: it must not invoke `advanced_cli`.

## Configuration and local runs

`pipeline_config.json` describes the staged pipeline; `optimizer_config.json` configures a direct optimizer run.

Install the local dependencies before running either entry point:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

For a configured local optimizer run:

```sh
julia --project=. run_optimizer.jl [path/to/optimizer_config.json]
```

Results are written below the configured output directory. Do not interpret a fixture run as evidence of simulator validity.

## Dynamic calibration roadmap

For a detailed repository audit and prioritized implementation plan for
automatic, dynamic calibration of the Saxony 2020--2022 data, including the
P0--P2 gaps, target architecture, milestones, acceptance criteria, and backlog,
see [`docs/dynamic-calibration-roadmap.md`](docs/dynamic-calibration-roadmap.md).

Implementation of its first five backlog items has started with a portable
Saxony profile, canonical data-quality protocol, leakage-resistant temporal
split, Negative-Binomial observation likelihood, and full 3–30 month stage
sequence. See [`docs/production-baseline.md`](docs/production-baseline.md) for
required external inputs and the no-launch preflight procedure.
The same guide documents the bounded two-candidate production smoke test,
provenance manifest, and local/Slurm parity check.

## Parameter-evolution audit

`drawing-utilities/build_parameter_evolution_audit.py` joins the candidate
identity and score in `optimizer_history.json` to each candidate's effective
`config.json`, then writes a self-contained interactive HTML report. The report
has 6-, 9-, and 12-month stage filters, independent scalar toggles for
`school`, `class`, and `age_coupling_param`, and a selector for the infection,
mild-detection, and tracing modulation vectors.
It also audits each 6m→9m and 9m→12m handoff against the preceding stage's
actual best scored candidate and lists any reverted prefix buckets.

```sh
python3 drawing-utilities/build_parameter_evolution_audit.py \
  --history /path/to/optimizer_history.json \
  --config-root /path/to/real_sims \
  --output /path/to/parameter_evolution_audit.html
```

The candidate configs are required because the history artifact itself records
scores and candidate coordinates, but does not embed parameter snapshots.

The focused [six-month corrected-run audit](docs/saxony-corrected-6m-audit.md)
explains why the reported `0.371273` validation score coexists with severe
full-horizon underprediction and proposes a scalar-profile, conditional-vector,
then joint-refinement experiment instead of a global sigma increase.

## Two-phase corrected calibration

Run the first two calibration phases as separate Slurm jobs. Phase 2 points to
Phase 1's final candidate and therefore intentionally fails preflight until
Phase 1 has completed:

```sh
sbatch scripts/run_cmaes.slurm optimizer_config.saxony.phase1-scalars.json
# Wait for runs/saxony-corrected-phase1-scalars/final_best_candidate.json.
sbatch scripts/run_cmaes.slurm optimizer_config.saxony.phase2-vectors.json
```

Both configurations rank by a normalized composite of 40% rolling validation
RMAE, 30% training-window cumulative detection error, and 30% training-window
cumulative death error. Phase 1 freezes modulation vectors and fits the three
scalars; Phase 2 consumes that result, freezes the scalars, and fits modulation.
The rolling validation term covers total detections/deaths plus detections and
deaths for all six age groups. Totals receive 25% each; each family of age-group
metrics receives 25%, distributed by the configured population shares.

Both phases start with normalized sigma `0.20` (20% of each
parameter range); the optimizer ceiling is also `0.20`, so this value is not
silently clamped back to the former `0.12` limit. Incumbent preservation keeps
the wider initial search from discarding the best configuration already found.

The three-month alternative is checked in as
`optimizer_config.saxony.phase1-scalars-alternative.json`. Submit it explicitly:

```sh
sbatch scripts/run_cmaes.slurm optimizer_config.saxony.phase1-scalars-alternative.json
```

The configured initial sigma only initializes a new CMA state. A run resumed
in an existing output directory correctly restores its adaptive sigma from
`stage_state.json`; changing the JSON does not reset that state. Use a fresh
output directory to start at `0.20`. Startup logs and newly written CMA
artifacts report both `configured_initial_sigma` and the executable's
`sigma_limits`, making an old checkout (with the former `0.12` ceiling) visible.

## Explicitly deferred

The following are **DEFERRED** and are not claimed by the reliable pipeline:

- real `advanced_cli` simulation execution;
- Slurm dispatch or cluster runs;
- validation replicates;
- multi-seed runs.
