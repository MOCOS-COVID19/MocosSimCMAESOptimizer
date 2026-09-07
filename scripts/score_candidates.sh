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
#   $6 = adapter timeout in seconds (optional; defaults to 3600)

set -euo pipefail

TARGET="$1"
JULIA_BIN="$2"
PROJECT_DIR="$3"
ADVANCED_CLI="$4"
GT_DIR="$5"
TIMEOUT_SECONDS="${6:-3600}"

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
DONE_OK="$CAND_DIR/done.ok"
FAILED_OK="$CAND_DIR/failed.ok"

rm -f "$DONE_OK" "$FAILED_OK"

if [ -z "$CAND_DIR" ] || [ ! -s "$CFG" ]; then
  echo "[ERROR] Missing candidate config: $CFG" >&2
  if [ -n "$CAND_DIR" ]; then
    mkdir -p "$CAND_DIR"
    touch "$FAILED_OK"
  fi
  exit 2
fi

echo "[$(date)] candidate=$CAND_DIR"

export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_PRECOMPILE_TASKS=1

STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
COMMAND=("$JULIA_BIN" "--project=$PROJECT_DIR" "--compiled-modules=no" "--threads=4"
  "$ADVANCED_CLI" "$CFG" "--output-daily" "$OUT_DAILY" "--output-summary" "$OUT_SUMMARY")
set +e
timeout --signal=TERM --kill-after=30 "$TIMEOUT_SECONDS" "${COMMAND[@]}" \
  >"$CAND_DIR/adapter.stdout.log" 2>"$CAND_DIR/adapter.stderr.log"
EXIT_CODE=$?
set -e
FINISHED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
export STARTED_AT FINISHED_AT EXIT_CODE TIMEOUT_SECONDS CAND_DIR
python3 - "${COMMAND[@]}" <<'PY'
import json, os, sys
root = os.environ["CAND_DIR"]
exit_code = int(os.environ["EXIT_CODE"])
payload = {
    "schema_version": "adapter-invocation-v1",
    "command": sys.argv[1:],
    "working_directory": root,
    "started_at": os.environ["STARTED_AT"],
    "finished_at": os.environ["FINISHED_AT"],
    "timeout_seconds": float(os.environ["TIMEOUT_SECONDS"]),
    "timed_out": exit_code in (124, 137),
    "exit_code": exit_code,
    "success": exit_code == 0,
    "stdout": os.path.join(root, "adapter.stdout.log"),
    "stderr": os.path.join(root, "adapter.stderr.log"),
}
with open(os.path.join(root, "adapter_invocation.json"), "w") as stream:
    json.dump(payload, stream, indent=2)
PY
if [ "$EXIT_CODE" -ne 0 ]; then
  touch "$FAILED_OK"
  exit "$EXIT_CODE"
fi

if [ ! -s "$OUT_DAILY" ]; then
  echo "[ERROR] Adapter succeeded but did not write $OUT_DAILY" >&2
  touch "$FAILED_OK"
  exit 3
fi

# Optional plotting per candidate (non-fatal on failure)
if [ "${MOCOSSIM_PLOT_CANDIDATES:-0}" = "1" ]; then
  python3 drawing-utilities/plot_gt_vs_sim.py \
    --output-dir "$CAND_DIR" \
    --gt-dir "$GT_DIR" \
    --daily "$OUT_DAILY" \
    --out "$CAND_DIR/gt_vs_sim.png" || true
fi

touch "$DONE_OK"

echo "[$(date)] done candidate=$CAND_DIR"
