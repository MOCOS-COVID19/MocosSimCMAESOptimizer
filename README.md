## Parameter evolution plots

To plot how selected parameters evolve across a search tree:

```bash
uv run -- python drawing-utilities/plot_param_evolution.py \
  --search-dir runs/wcss3m/real_sims \
  --out-dir runs/wcss3m/real_sims/plots
```
# MocosSimCMAESOptimizer

This package runs a synthetic CMA-ES optimizer against a configurable epidemic simulation seed and records the best parameter combinations per stage.

## Requirements

- Julia (the project uses the versions declared in `Project.toml`; run `julia --project=. -e 'using Pkg; Pkg.instantiate()'` to install dependencies).
- A seed configuration file referencing the simulation state (e.g., `seed/config2.json`).
- Writable output directory for storing optimizer artifacts.

## Configuration

1. Update `optimizer_config.json` to point at the desired seed file, output directory, and CMA-ES parameters.
2. Define:
   - `scalar_bounds` and `temporal_bounds` to constrain parameter search ranges.
   - `stages` that specify the number of fit months, iterations, population size, and starting sigma.
   - `objective` weights plus `recent_days`/`early_reject_multiplier` to shape scoring.

## Running the optimizer

```sh
julia --project=. run_optimizer.jl [path/to/optimizer_config.json]
```

- If no config path is provided, `run_optimizer.jl` loads the default `optimizer_config.json` in the repository root.
- Results (best candidates, history, summaries) are written into the configured `output_dir` (e.g., `runs/default/`).

## Inspecting results

- Each stage emits `<stage>_best_candidate.json` with the optimized configuration for that stage.
- The optimizer also drops `optimizer_history.json`, `stage_summary.json`, `final_best_candidate.json`, and `<stage>_summary.json` under `output_dir`.
- Per-stage runtime logs are written under `output_dir/real_sims/<stage>/` as `stage_state.json` and `iter_metrics.jsonl`.

## Testing

```sh
julia --project=. tests/smoke_test.jl
```

This script runs the optimizer against the default configuration and prints the summary dictionary.

## Running on Slurm

Use the thin wrapper, which enables per-iteration Slurm array dispatch inside `run_optimizer.jl`:

```sh
sbatch scripts/run_cmaes.slurm
```

If invoking manually on a Slurm node:
```sh
/home/mbodych/1.7.0-school_class/julia-1.7.0/bin/julia --project=. run_optimizer.jl --slurm
```

If a run stops mid-stage, re-running the same command resumes from the latest `real_sims/<stage>/stage_state.json` and `iter_metrics.jsonl` artifacts.
