#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="/home/payneli/data/nwm_cdit_xl/checkpoints/cdit_xl_ego4d_200000.pth.tar"
export VAE_PATH="/home/payneli/data/sd-vae-ft-ema"
cd /home/payneli/project/nwm

# Launch the requested FULL (all samples) experiment in background on 4 GPUs:
# - GPU0: RECON none
# - GPU1: RECON noise severity=1
# - GPU2: RECON noise severity=3
# - GPU3: go_stanford (full index), none
#
# All jobs run concurrently (nohup + &), and each writes:
# - logs:  <OUT_ROOT>/_logs/<RUN_ID>/*.log
# - results: <OUT_ROOT>/<RUN_ID>_<label>_<sev>_<tag>/shard_0/early_exit_generic_shard0.csv
#
# Usage:
#   export MODEL_PATH=...
#   export VAE_PATH=...
#   bash uncertainty_research/scripts/launch_full0_4gpu_background.sh
#
# Optional overrides via env:
#   RUN_ID=... OUT_ROOT=... TAG=full0 NUM_SAMPLES=-1 BATCH_SIZE=32 NUM_SEEDS=5 PROBE_STEPS=...

cd /home/payneli/project/nwm

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "Missing env var: MODEL_PATH" >&2
  exit 1
fi
if [[ -z "${VAE_PATH:-}" ]]; then
  echo "Missing env var: VAE_PATH" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_full0}"
OUT_ROOT="${OUT_ROOT:-/home/payneli/project/nwm/uncertainty_research/results}"
TAG="${TAG:-full0}"

# Full dataset by default (-1 means all)
NUM_SAMPLES="${NUM_SAMPLES:--1}"

# As requested
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
echo "${RUN_ID}" > /tmp/full0_run_id.txt
echo "Saved RUN_ID to: /tmp/full0_run_id.txt"

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
      --log_every_samples 256 \
      "$@" \
      >"${log_file}" 2>&1 &

  echo "launched gpu=${gpu_id} name=${name} log=${log_file}"
}

# NOTE: Use arrays to avoid bash line-continuation footguns (missing trailing '\', or '\ ').

# GPU0: RECON  noise severity=5, num_shards=4
args0=(--dataset recon --split test --perturbation noise --severity 4 --num_shards 4 --shard_id 0)
launch_one 0 "gpu0_recon_noise4" "${args0[@]}"

# GPU1: RECON  noise severity=5, num_shards=4
args1=(--dataset recon --split test --perturbation noise --severity 4 --num_shards 4 --shard_id 1)
launch_one 1 "gpu1_recon_noise4" "${args1[@]}"

# GPU2: RECON  noise severity=5, num_shards=4
args2=(--dataset recon --split test --perturbation noise --severity 4 --num_shards 4 --shard_id 2)
launch_one 2 "gpu2_recon_noise4" "${args2[@]}"

# GPU3: RECON  noise severity=5, num_shards=4
args3=(--dataset recon --split test --perturbation noise --severity 4 --num_shards 4 --shard_id 3)
launch_one 3 "gpu3_recon_noise4" "${args3[@]}"

echo
echo "All jobs launched in background."
echo "Tail logs:"
echo "  tail -f ${LOG_DIR}/*.log"
echo
echo "Quick status check (copy/paste):"
echo "  cd /home/payneli/project/nwm && RUN_ID=\$(cat /tmp/full0_run_id.txt) && echo \"RUN_ID=\$RUN_ID\" && sleep 300 && for d in none_0 noise_1 noise_3 go_stanford_0; do f=\"${OUT_ROOT}/\${RUN_ID}_\${d}_${TAG}/shard_0/early_exit_generic_shard0.csv\"; if [[ -f \"\$f\" ]]; then echo \"DONE \$f\"; else echo \"PENDING \$f\"; fi; done && echo \"--- tail logs ---\" && tail -n 5 ${OUT_ROOT}/_logs/\$RUN_ID/*.log"


