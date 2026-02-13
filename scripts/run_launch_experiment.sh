#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="/home/payneli/data/nwm_cdit_xl/checkpoints/cdit_xl_ego4d_200000.pth.tar"
export VAE_PATH="/home/payneli/data/sd-vae-ft-ema"
cd /home/payneli/project/nwm

# Unified launcher for Early-Exit (probe steps) experiments ONLY:
# - early-exit probes at selected diffusion steps -> run_early_exit_generic.py
#
# - recon/go_stanford datasets
# - go_stanford full index via traj_names: --go_eval_type full
# - deterministic sampling: --deterministic_per_seed
# - deterministic perturbations: --deterministic_perturb
#
# Usage examples:
# Early-exit on go_stanford full (traj index), photometric sev=3
#   MODEL_PATH=... VAE_PATH=... ./uncertainty_research/scripts/run_launch_experiment.sh \
#     --dataset go_stanford --split test --go_eval_type full \
#     --num_samples 2000 --num_seeds 5 --batch_size 32 \
#     --perturbation photometric --severity 3 \
#     --diffusion_steps 50 --probe_steps 2,10,20,30,40,50 --perceptual_probe_steps 2,10,50

DATASET="recon"            # recon | go_stanford
SPLIT="test"
GO_EVAL_TYPE="full"        # rollout | time | full (only for go_stanford)

# Comma-separated lists are supported:
#   --perturbation none
#   --perturbation noise --severity 1,3,5
#   --perturbation none,noise --severity 1,3,5  (none will always run with severity=0)
PERT="none"                # none | noise | blur | blackout | photometric (or comma list)
SEV="0"                    # comma list, used only when perturbation != none

DIFFUSION_STEPS="50"
PROBE_STEPS="2,10,20,30,40,50"
PERC_STEPS=""              # empty => all probe steps

NUM_SAMPLES="200"
NUM_SEEDS="5"
BATCH_SIZE="16"
NUM_WORKERS="6"
PIN_MEMORY="1"

NUM_SHARDS="1"
SHARD_ID="0"

BASE_SEED="1234"

NO_LPIPS="0"
NO_DREAMSIM="0"

LOG_EVERY_SAMPLES="256"

DET_PER_SEED="1"
DET_PERT="1"

OUT_ROOT="/home/payneli/project/nwm/uncertainty_research/results"
TAG=""
RUN_ID=""                  # if set, all shards can share the same output root

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --go_eval_type) GO_EVAL_TYPE="$2"; shift 2 ;;
    --perturbation) PERT="$2"; shift 2 ;;
    --severity) SEV="$2"; shift 2 ;;
    --diffusion_steps) DIFFUSION_STEPS="$2"; shift 2 ;;
    --probe_steps) PROBE_STEPS="$2"; shift 2 ;;
    --perceptual_probe_steps) PERC_STEPS="$2"; shift 2 ;;
    --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
    --num_seeds) NUM_SEEDS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --pin_memory) PIN_MEMORY="$2"; shift 2 ;;
    --num_shards) NUM_SHARDS="$2"; shift 2 ;;
    --shard_id) SHARD_ID="$2"; shift 2 ;;
    --base_seed) BASE_SEED="$2"; shift 2 ;;
    --no_lpips) NO_LPIPS="1"; shift 1 ;;
    --no_dreamsim) NO_DREAMSIM="1"; shift 1 ;;
    --log_every_samples) LOG_EVERY_SAMPLES="$2"; shift 2 ;;
    --deterministic_per_seed) DET_PER_SEED="1"; shift 1 ;;
    --no_deterministic_per_seed) DET_PER_SEED="0"; shift 1 ;;
    --deterministic_perturb) DET_PERT="1"; shift 1 ;;
    --no_deterministic_perturb) DET_PERT="0"; shift 1 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --run_id) RUN_ID="$2"; shift 2 ;;
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

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "Missing env var: MODEL_PATH" >&2
  exit 1
fi
if [[ -z "${VAE_PATH:-}" ]]; then
  echo "Missing env var: VAE_PATH" >&2
  exit 1
fi

if [[ -n "$RUN_ID" ]]; then
  TS="$RUN_ID"
else
  # Default: minute precision to reduce chance shards split into different folders.
  TS=$(date +%Y%m%d_%H%M)
fi

# Ensure probe_steps always includes the final step (= diffusion_steps)
if [[ -z "${PROBE_STEPS}" ]]; then
  PROBE_STEPS="${DIFFUSION_STEPS}"
else
  if [[ ",${PROBE_STEPS}," != *",${DIFFUSION_STEPS},"* ]]; then
    PROBE_STEPS="${PROBE_STEPS},${DIFFUSION_STEPS}"
  fi
fi

TAG_SUFFIX=""
if [[ -n "$TAG" ]]; then
  TAG_SUFFIX="_${TAG}"
fi

PY="/home/payneli/anaconda3/envs/wm/bin/python"
COMMON_DET=()
if [[ "$DET_PER_SEED" == "1" ]]; then COMMON_DET+=(--deterministic_per_seed); fi
if [[ "$DET_PERT" == "1" ]]; then COMMON_DET+=(--deterministic_perturb); fi

COMMON_METRICS=()
if [[ "$NO_LPIPS" == "1" ]]; then COMMON_METRICS+=(--no_lpips); fi
if [[ "$NO_DREAMSIM" == "1" ]]; then COMMON_METRICS+=(--no_dreamsim); fi

PIN_ARGS=()
if [[ "$PIN_MEMORY" == "1" ]]; then PIN_ARGS+=(--pin_memory); fi

cd /home/payneli/project/nwm

# Split perturbations / severities (comma-separated)
IFS=',' read -r -a PERT_LIST <<< "${PERT}"
IFS=',' read -r -a SEV_LIST <<< "${SEV}"
if [[ "${#PERT_LIST[@]}" -eq 0 ]]; then
  echo "Empty --perturbation" >&2
  exit 1
fi

for p_raw in "${PERT_LIST[@]}"; do
  p="$(echo "$p_raw" | xargs)"
  if [[ -z "$p" ]]; then
    continue
  fi
  if [[ "$p" == "none" ]]; then
    # none always runs with severity=0, regardless of --severity
    label="none"
    sev_out="0"
    if [[ "$DATASET" == "go_stanford" ]]; then
      label="go_stanford"
    fi
    run_dir="${OUT_ROOT}/${TS}_${label}_${sev_out}${TAG_SUFFIX}/shard_${SHARD_ID}"
    mkdir -p "$run_dir"
    echo "[run] dataset=${DATASET} perturbation=none severity=0 out=${run_dir}"
    "$PY" -u uncertainty_research/scripts/run_early_exit_generic.py \
      --model_path "$MODEL_PATH" \
      --vae_path "$VAE_PATH" \
      --device "cuda:0" \
      --dataset_name "$DATASET" \
      --split "$SPLIT" \
      --go_eval_type "$GO_EVAL_TYPE" \
      --perturbation "none" \
      --severity 0 \
      --diffusion_steps "$DIFFUSION_STEPS" \
      --probe_steps "$PROBE_STEPS" \
      --perceptual_probe_steps "$PERC_STEPS" \
      --num_seeds "$NUM_SEEDS" \
      --num_samples "$NUM_SAMPLES" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      "${PIN_ARGS[@]}" \
      --base_seed "$BASE_SEED" \
      --num_shards "$NUM_SHARDS" \
      --shard_id "$SHARD_ID" \
      --log_every_samples "$LOG_EVERY_SAMPLES" \
      --output_dir "$run_dir" \
      "${COMMON_DET[@]}" \
      "${COMMON_METRICS[@]}"
  else
    # perturbations with severities
    if [[ "${#SEV_LIST[@]}" -eq 0 ]]; then
      echo "Missing --severity for perturbation=$p" >&2
      exit 1
    fi
    for s_raw in "${SEV_LIST[@]}"; do
      s="$(echo "$s_raw" | xargs)"
      if [[ -z "$s" ]]; then
        continue
      fi
      label="$p"
      sev_out="$s"
      if [[ "$DATASET" == "go_stanford" ]]; then
        # As requested, go_stanford runs are grouped under "{ts}_go_stanford_{severity}/...".
        # If user also applies a perturbation on go_stanford, keep it in the label to avoid overwriting.
        if [[ "$p" == "none" ]]; then
          label="go_stanford"
        else
          label="go_stanford_${p}"
        fi
      fi
      run_dir="${OUT_ROOT}/${TS}_${label}_${sev_out}${TAG_SUFFIX}/shard_${SHARD_ID}"
      mkdir -p "$run_dir"
      echo "[run] dataset=${DATASET} perturbation=${p} severity=${s} out=${run_dir}"
      "$PY" -u uncertainty_research/scripts/run_early_exit_generic.py \
        --model_path "$MODEL_PATH" \
        --vae_path "$VAE_PATH" \
        --device "cuda:0" \
        --dataset_name "$DATASET" \
        --split "$SPLIT" \
        --go_eval_type "$GO_EVAL_TYPE" \
        --perturbation "$p" \
        --severity "$s" \
        --diffusion_steps "$DIFFUSION_STEPS" \
        --probe_steps "$PROBE_STEPS" \
        --perceptual_probe_steps "$PERC_STEPS" \
        --num_seeds "$NUM_SEEDS" \
        --num_samples "$NUM_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        "${PIN_ARGS[@]}" \
        --base_seed "$BASE_SEED" \
        --num_shards "$NUM_SHARDS" \
        --shard_id "$SHARD_ID" \
        --log_every_samples "$LOG_EVERY_SAMPLES" \
        --output_dir "$run_dir" \
        "${COMMON_DET[@]}" \
        "${COMMON_METRICS[@]}"
    done
  fi
done

echo "DONE. Results in: ${OUT_ROOT}/${TS}_*${TAG_SUFFIX}/shard_${SHARD_ID}/"


