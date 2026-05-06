# Slurm: parallel candidate scoring for CMA-ES

This repo’s CMA-ES can score candidates using real simulations. To parallelize scoring on Slurm:

## 1) Prepare candidates per iteration
The optimizer writes configs under `runs/default/real_sims/<stage>/iter_<k>/cand_<i>/config.json`. If you generate them yourself, follow the same layout.

## 2) Single-submit flow (recommended)

Run the thin wrapper; it calls `run_optimizer.jl --slurm`, which internally submits per-iteration arrays and waits:
```bash
sbatch scripts/run_cmaes.slurm
```

Manual invocation on a Slurm node:
```bash
/home/mbodych/1.7.0-school_class/julia-1.7.0/bin/julia --project=. run_optimizer.jl --slurm
```

## 3) How array dispatch works (inside `run_optimizer.jl --slurm`)
- For each iteration, candidates are written to `runs/default/real_sims/<stage>/iter_<k>/cand_##/config.json`.
- A `candidate_list.txt` is created for that iteration.
- A Slurm array is submitted via `scripts/score_candidates.sh <candidate_list.txt> <julia_bin> <project_dir> <advanced_cli.jl> <gt_dir>`.
- The optimizer waits for the array to finish (polls `squeue`) before scoring and proceeding to the next iteration.
- Each array task runs the sim (`advanced_cli.jl`), writes `output_daily.jld2`/`summary.jld2`, and generates `gt_vs_sim.png` in the candidate dir.

## 4) Helpers
- `scripts/score_candidates.sh`: runs one candidate (sim + plot); accepts either a root dir or a candidate list file; used by the optimizer’s array dispatch.
- Plots are written per candidate; you can collect them from `runs/default/real_sims/**/gt_vs_sim.png` (the wrapper also copies them into `runs/default/plots/candidates/` if desired).
