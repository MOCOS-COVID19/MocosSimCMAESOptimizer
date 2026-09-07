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

## Explicitly deferred

The following are **DEFERRED** and are not claimed by the reliable pipeline:

- real `advanced_cli` simulation execution;
- Slurm dispatch or cluster runs;
- validation replicates;
- multi-seed runs.
