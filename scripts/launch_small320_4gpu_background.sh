#!/usr/bin/env bash
set -euo pipefail

# Launch the requested small-scale experiment in background on 4 GPUs:
# - GPU0: RECON none
# - GPU1: RECON noise severity=5
# - GPU2: RECON blur severity=5
# - GPU3: go_stanford (full index), none
#
# All jobs run concurrently (nohup + &), and each writes:
# - logs:  <OUT_ROOT>/_logs/<RUN_ID>/*.log
# - done:  <OUT_ROOT>/<RUN_ID>_<label>_<sev>_<tag>/shard_0/DONE
#
# Usage:
#   export MODEL_PATH=...
#   export VAE_PATH=...
#   bash uncertainty_research/scripts/launch_small320_4gpu_background.sh
#
# Optional overrides via env:
#   RUN_ID=... OUT_ROOT=... TAG=... NUM_SAMPLES=320 BATCH_SIZE=32 NUM_SEEDS=5 PROBE_STEPS=...

cd /home/payneli/project/nwm

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "Missing env var: MODEL_PATH" >&2
  exit 1
fi
if [[ -z "${VAE_PATH:-}" ]]; then
  echo "Missing env var: VAE_PATH" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_small320}"
OUT_ROOT="${OUT_ROOT:-/home/payneli/project/nwm/uncertainty_research/results}"
TAG="${TAG:-small320}"

NUM_SAMPLES="${NUM_SAMPLES:-320}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_SEEDS="${NUM_SEEDS:-5}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-50}"
PROBE_STEPS="${PROBE_STEPS:-2,5,10,20,30,40,50}"
NUM_WORKERS="${NUM_WORKERS:-6}"
BASE_SEED="${BASE_SEED:-1234}"

LOG_DIR="${OUT_ROOT}/_logs/${RUN_ID}"
mkdir -p "${LOG_DIR}"

echo "RUN_ID=${RUN_ID}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "LOG_DIR=${LOG_DIR}"

launch_one () {
  local gpu_id="$1"
  shift
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"

  # Run in background. Use env CUDA_VISIBLE_DEVICES to isolate GPU per job.
  # Note: we avoid constructing a bash -lc string to prevent argument parsing bugs.
  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" \
    ./uncertainty_research/scripts/run_launch_experiment.sh \
      --run_id "${RUN_ID}" \
      --tag "${TAG}" \
      --out_root "${OUT_ROOT}" \
      --diffusion_steps "${DIFFUSION_STEPS}" \
      --probe_steps "${PROBE_STEPS}" \
      --perceptual_probe_steps "${PROBE_STEPS}" \
      --num_samples "${NUM_SAMPLES}" \
      --num_seeds "${NUM_SEEDS}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --num_shards 1 \
      --shard_id 0 \
      --base_seed "${BASE_SEED}" \
      --log_every_samples 64 \
      "$@" \
      >"${log_file}" 2>&1 &

  echo "launched gpu=${gpu_id} name=${name} log=${log_file}"
}

# GPU0: RECON none
launch_one 0 "gpu0_recon_none" \
  --dataset recon --split test \
  --perturbation none

# GPU1: RECON noise severity=5
launch_one 1 "gpu1_recon_noise5" \
  --dataset recon --split test \
  --perturbation noise --severity 5

# GPU2: RECON blur severity=5
launch_one 2 "gpu2_recon_blur5" \
  --dataset recon --split test \
  --perturbation blur --severity 5

# GPU3: go_stanford full, none
launch_one 3 "gpu3_go_stanford" \
  --dataset go_stanford --split test --go_eval_type full \
  --perturbation none --severity 0

echo
echo "All jobs launched in background."
echo "Tail logs:"
echo "  tail -f ${LOG_DIR}/*.log"
echo
echo "When finished, run analysis:"
echo "  /home/payneli/anaconda3/envs/wm/bin/python -u uncertainty_research/scripts/analyze_small320_4gpu.py --out_root \"${OUT_ROOT}\" --run_id \"${RUN_ID}\" --tag \"${TAG}\""


