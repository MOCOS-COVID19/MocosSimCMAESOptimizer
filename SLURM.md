# Slurm: parallel candidate scoring for CMA-ES

This repo’s CMA-ES can score candidates using real simulations. To parallelize scoring on Slurm:

## 1) Prepare candidates per iteration
The optimizer writes configs under `runs/default/real_sims/<stage>/iter_<k>/cand_<i>/config.json`. If you generate them yourself, follow the same layout.

## 2) Submit array
Use the helper script:
```bash
N=$(find runs/default/real_sims/stage_01/iter_3 -maxdepth 1 -name 'cand_*' -type d | wc -l)
sbatch --array=0-$((N-1)) scripts/score_candidates.sh \
  runs/default/real_sims/stage_01/iter_3 \
  /path/to/julia-1.7.0/bin/julia \
  /path/to/MocosSimLauncher \
  /path/to/MocosSimLauncher/advanced_cli.jl \
  ./gt
```

Inside each array task, the script:
- Runs the simulation: `julia --project=<MocosSimLauncher> --threads=4 advanced_cli.jl <config> --output-daily output_daily.jld2 --output-summary summary.jld2`.
- Generates per-candidate plot: `drawing-utilities/plot_gt_vs_sim.py` (uses `./gt`).

## 3) Wait and collect
- Monitor: `squeue -u $USER -n cmaes-score` (if you name the job accordingly).
- After completion, consume outputs under the candidate dirs: `output_daily.jld2`, `summary.jld2`, `gt_vs_sim.png`. Integrate the RMSE back into CMA-ES history as needed.

## Notes
- Adjust threads/timeout/partition in `scripts/score_candidates.sh`/sbatch flags to fit your cluster.
- `./gt` is the default GT source; change the fifth arg to point elsewhere if needed.
