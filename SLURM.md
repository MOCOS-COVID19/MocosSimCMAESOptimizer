# Slurm: parallel candidate scoring for CMA-ES

The recommended first production run is the 344-candidate (5.39%) profile in
`optimizer_config.saxony.12m-pilot.json`. Complete the real local and Slurm smoke
checks in `docs/production-baseline.md` before starting it. This profile stops at
12 months, uses 9 months as an intermediate transfer stage, limits both longer
stages to 24 candidates per generation, and leaves the 18/24/30-month stages
outside the initial budget.

## Required environment

All paths must be absolute and visible with the same names on the controller and
compute nodes:

```bash
export JULIA_BIN=/path/to/julia
export MOCOSSIM_LAUNCHER_DIR=/path/to/MocosSimLauncher
export MOCOSSIM_ADVANCED_CLI="$MOCOSSIM_LAUNCHER_DIR/advanced_cli.jl"
export MOCOSSIM_SEED_CONFIG=/path/to/saxony-seed.json
```

## Submit the budgeted profile

```bash
sbatch scripts/run_cmaes.slurm optimizer_config.saxony.12m-pilot.json
```

The config argument is optional; the wrapper defaults to the 12-month pilot
profile. It resolves the repository from its own location, instantiates both
Julia environments, runs the no-launch preflight, and only then starts the
optimizer controller. Do not reuse an output directory from a smoke or another
scientific run.

## Array dispatch

For every iteration the controller writes candidate directories and a
`candidate_list.txt`, submits one array task per candidate, and waits for
terminal artifacts. Each task calls `scripts/score_candidates.sh` with the Julia
binary, launcher project, `advanced_cli.jl`, ground truth directory, and adapter
timeout from the selected optimizer config.

Array resources are currently fixed in `submit_slurm_array` at 4 CPUs, 20 GB,
and 75 minutes, with an adapter deadline of 55 minutes in the pilot profile.
Confirm from smoke accounting that these limits fit the 12-month horizon before
starting the pilot. The controller itself requests one CPU.

The launcher and optimizer environments are instantiated once in the controller
wrapper. Array tasks do not run `Pkg.instantiate` and do not mutate a shared
Python virtual environment. Per-candidate plots are opt-in with
`MOCOSSIM_PLOT_CANDIDATES=1`; install their Python dependencies before submission
if plots are required.
