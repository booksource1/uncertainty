#!/usr/bin/env bash
set -euo pipefail

# Wait for the 4 expected outputs of the small320 run, then run analysis.
#
# Usage:
#   bash uncertainty_research/scripts/wait_and_analyze_small320_4gpu.sh --run_id <RUN_ID> [--out_root ...] [--tag ...]

OUT_ROOT="/home/payneli/project/nwm/uncertainty_research/results"
TAG="small320"
RUN_ID=""
SLEEP_SECS=60

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_id) RUN_ID="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --sleep) SLEEP_SECS="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  echo "Missing --run_id" >&2
  exit 1
fi

need=(
  "${OUT_ROOT}/${RUN_ID}_none_0_${TAG}/shard_0/early_exit_generic_shard0.csv"
  "${OUT_ROOT}/${RUN_ID}_noise_5_${TAG}/shard_0/early_exit_generic_shard0.csv"
  "${OUT_ROOT}/${RUN_ID}_blur_5_${TAG}/shard_0/early_exit_generic_shard0.csv"
  "${OUT_ROOT}/${RUN_ID}_go_stanford_0_${TAG}/shard_0/early_exit_generic_shard0.csv"
)

echo "Waiting for outputs (sleep=${SLEEP_SECS}s):"
for f in "${need[@]}"; do
  echo "  - $f"
done

while true; do
  missing=0
  for f in "${need[@]}"; do
    if [[ ! -f "$f" ]]; then
      missing=$((missing+1))
    fi
  done
  if [[ "$missing" -eq 0 ]]; then
    break
  fi
  echo "[wait] missing=${missing}/4 ... $(date)" >&2
  sleep "$SLEEP_SECS"
done

echo "[wait] all outputs present. Running analysis..."
cd /home/payneli/project/nwm
/home/payneli/anaconda3/envs/wm/bin/python -u uncertainty_research/scripts/analyze_small320_4gpu.py \
  --out_root "${OUT_ROOT}" \
  --run_id "${RUN_ID}" \
  --tag "${TAG}"

echo "[wait] analysis done."


