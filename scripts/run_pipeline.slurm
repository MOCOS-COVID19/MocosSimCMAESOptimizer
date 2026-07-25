#!/usr/bin/env bash
#SBATCH -J cma-nuts-pipeline
#SBATCH -o logs/cma-nuts-pipeline-%j.out
#SBATCH -e logs/cma-nuts-pipeline-%j.err
#SBATCH -c 4
#SBATCH -t 72:00:00

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
JULIA_BIN="${JULIA_BIN:-/home/mbodych/1.7.0-school_class/julia-1.7.0/bin/julia}"
PIPELINE_CONFIG="${1:-$REPO/pipeline_config.json}"

mkdir -p "$REPO/logs"
cd "$REPO"

export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_PRECOMPILE_TASKS=1

echo "[INFO] Instantiating MocosSimLauncher environment"
"$JULIA_BIN" --project="$REPO/../MocosSimLauncher" -e \
  'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'

echo "[INFO] Instantiating optimizer environment"
"$JULIA_BIN" --project="$REPO" -e \
  'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'

echo "[INFO] Running short CMA-ES -> NUTS -> long CMA-ES pipeline"
srun --cpu-bind=cores "$JULIA_BIN" --project="$REPO" \
  "$REPO/scripts/run_pipeline.jl" "$PIPELINE_CONFIG"

echo "[INFO] Pipeline completed"
