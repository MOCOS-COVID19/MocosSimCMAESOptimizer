#!/bin/bash
# Slurm controller job for the budgeted Saxony CMA-ES production profile.
# Submit with: sbatch scripts/run_cmaes.slurm [CONFIG]

#SBATCH -J cmaes-real
#SBATCH -o cmaes-real-%j.out
#SBATCH -e cmaes-real-%j.err
# The controller performs the final validation replicates in-process after the
# candidate arrays finish.  Match the four-thread, 20 GiB resources used by
# each candidate array task so a 12-month validation run is not constrained by
# the controller's smaller default allocation.
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH -t 24:00:00

set -euo pipefail

if [[ -n "${REPO:-}" ]]; then
  REPO=$(cd "$REPO" && pwd)
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SUBMIT_DIR=$(cd "$SLURM_SUBMIT_DIR" && pwd)
  if [[ -f "$SUBMIT_DIR/Project.toml" ]]; then
    REPO="$SUBMIT_DIR"
  elif [[ -f "$SUBMIT_DIR/../Project.toml" ]]; then
    REPO=$(cd "$SUBMIT_DIR/.." && pwd)
  else
    echo "[ERROR] Cannot locate optimizer Project.toml from SLURM_SUBMIT_DIR=$SUBMIT_DIR" >&2
    exit 2
  fi
else
  REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
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
echo "[INFO] Optimizer checkout: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
[[ "$CONFIG" = /* ]] || CONFIG="$REPO/$CONFIG"
[[ -f "$REPO/Project.toml" ]] || {
  echo "[ERROR] Optimizer project not found under REPO=$REPO" >&2
  exit 2
}
[[ -f "$MOCOSSIM_LAUNCHER_DIR/Project.toml" ]] || {
  echo "[ERROR] Launcher project not found under MOCOSSIM_LAUNCHER_DIR=$MOCOSSIM_LAUNCHER_DIR" >&2
  exit 2
}

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
