#!/usr/bin/env bash
# Slurm array helper: run one candidate simulation + optional GT-vs-sim plot.
# Usage (example):
#   sbatch -t 06:00:00 -c 4 --array=0-$((N-1)) scripts/score_candidates.sh \
#     runs/default/real_sims/stage_01/iter_3 \
#     /path/to/julia-1.7.0/bin/julia \
#     /path/to/MocosSimLauncher \
#     /path/to/MocosSimLauncher/advanced_cli.jl \
#     ./gt
#
# Or using a candidate list file (one directory per line):
#   sbatch -t 06:00:00 -c 4 --array=0-$((N-1)) scripts/score_candidates.sh \
#     /path/to/candidate_list.txt \
#     /path/to/julia-1.7.0/bin/julia \
#     /path/to/MocosSimLauncher \
#     /path/to/MocosSimLauncher/advanced_cli.jl \
#     ./gt
#
# Positional args:
#   $1 = CAND_ROOT (iter directory containing cand_XX subdirs)
#   $2 = JULIA_BIN
#   $3 = PROJECT_DIR (MocosSimLauncher project)
#   $4 = ADVANCED_CLI (advanced_cli.jl)
#   $5 = GT_DIR (directory with daily_* CSVs)

set -euo pipefail

TARGET="$1"
JULIA_BIN="$2"
PROJECT_DIR="$3"
ADVANCED_CLI="$4"
GT_DIR="$5"

IDX=${SLURM_ARRAY_TASK_ID:-0}

if [ -f "$TARGET" ]; then
  # list file mode
  CAND_DIR=$(sed -n "$((IDX + 1))p" "$TARGET")
else
  # root directory mode
  CAND_DIR=$(printf "%s/cand_%02d" "$TARGET" "$IDX")
fi

CFG="$CAND_DIR/config.json"
OUT_DAILY="$CAND_DIR/output_daily.jld2"
OUT_SUMMARY="$CAND_DIR/summary.jld2"

echo "[$(date)] candidate=$CAND_DIR"

echo "[$(date)] Instantiating MocosSimLauncher for this task"
"$JULIA_BIN" --project="$PROJECT_DIR" -e 'using Pkg; Pkg.instantiate()'

echo "[$(date)] Preparing uv environment for plotting"
if command -v uv >/dev/null 2>&1; then
  uv venv .venv >/dev/null 2>&1 || true
  uv pip install matplotlib numpy h5py >/dev/null 2>&1 || true
fi

"$JULIA_BIN" --project="$PROJECT_DIR" --threads=4 "$ADVANCED_CLI" "$CFG" \
  --output-daily "$OUT_DAILY" --output-summary "$OUT_SUMMARY"

# Optional plotting per candidate (non-fatal on failure)
if command -v uv >/dev/null 2>&1; then
  uv run -- python drawing-utilities/plot_gt_vs_sim.py \
    --output-dir "$CAND_DIR" \
    --gt-dir "$GT_DIR" \
    --daily "$OUT_DAILY" \
    --out "$CAND_DIR/gt_vs_sim.png" || true
else
  python3 drawing-utilities/plot_gt_vs_sim.py \
    --output-dir "$CAND_DIR" \
    --gt-dir "$GT_DIR" \
    --daily "$OUT_DAILY" \
    --out "$CAND_DIR/gt_vs_sim.png" || true
fi

echo "[$(date)] done candidate=$CAND_DIR"
