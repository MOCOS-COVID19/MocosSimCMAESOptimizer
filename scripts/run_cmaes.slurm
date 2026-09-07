#!/bin/bash
# Slurm controller job for the budgeted Saxony CMA-ES production profile.
# Submit with: sbatch scripts/run_cmaes.slurm [CONFIG]

#SBATCH -J cmaes-real
#SBATCH -o cmaes-real-%j.out
#SBATCH -e cmaes-real-%j.err
#SBATCH -c 1
#SBATCH -t 24:00:00

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="${1:-optimizer_config.saxony.12m-pilot.json}"
: "${JULIA_BIN:?export JULIA_BIN before submitting}"
: "${MOCOSSIM_LAUNCHER_DIR:?export MOCOSSIM_LAUNCHER_DIR before submitting}"
: "${MOCOSSIM_ADVANCED_CLI:?export MOCOSSIM_ADVANCED_CLI before submitting}"
: "${MOCOSSIM_SEED_CONFIG:?export MOCOSSIM_SEED_CONFIG before submitting}"
for command in sbatch squeue scancel srun timeout; do
  command -v "$command" >/dev/null || {
    echo "[ERROR] Required command not found: $command" >&2
    exit 2
  }
done

cd "$REPO"
[[ "$CONFIG" = /* ]] || CONFIG="$REPO/$CONFIG"

export JULIA_PKG_PRECOMPILE_AUTO=0

echo "[INFO] Instantiating MocosSimLauncher environment"
"$JULIA_BIN" --project="$MOCOSSIM_LAUNCHER_DIR" -e \
  'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'

echo "[INFO] Instantiating optimizer environment"
"$JULIA_BIN" --project="$REPO" -e \
  'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'

echo "[INFO] Validating $CONFIG without launching simulations"
"$JULIA_BIN" --project="$REPO" run_optimizer.jl --preflight "$CONFIG"

echo "[INFO] Running CMA-ES with internal per-iteration Slurm arrays"
srun --cpu-bind=cores "$JULIA_BIN" --project="$REPO" \
  run_optimizer.jl --slurm "$CONFIG"

echo "[INFO] Done"
