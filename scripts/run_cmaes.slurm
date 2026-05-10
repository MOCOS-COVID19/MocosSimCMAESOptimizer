#!/bin/bash
# Slurm job: run CMA-ES with real simulation scoring, then plot all candidates.
# Submit with: sbatch scripts/run_cmaes.slurm

#SBATCH -J cmaes-real
#SBATCH -o logs/cmaes-real-%j.out
#SBATCH -e logs/cmaes-real-%j.err
#SBATCH -c 4
#SBATCH -t 24:00:00

set -euo pipefail

echo "[INFO] Instantiating MocosSimLauncher environment (fresh each job)"
/home/mbodych/1.7.0-school_class/julia-1.7.0/bin/julia --project=/home/mbodych/1.7.0-school_class/MocosSimLauncher/ -e 'using Pkg; Pkg.instantiate()'

JULIA_BIN="/home/mbodych/1.7.0-school_class/julia-1.7.0/bin/julia"
REPO="/home/mbodych/1.7.0-school_class/MocosSimCMAESOptimizer"
GT_DIR="$REPO/gt"

mkdir -p "$REPO/logs"
cd "$REPO"
"$JULIA_BIN" --project=. -e 'using Pkg; Pkg.instantiate();'

echo "[INFO] Running CMA-ES with srun (no nested sbatch), enabling internal per-iteration arrays"
srun --cpu-bind=cores "$JULIA_BIN" --project=. run_optimizer.jl --slurm

echo "[INFO] Done"
