#!/usr/bin/env bash
set -euo pipefail

# Smoke validation for the unified launcher:
# - early-exit (recon, N=8)
# - early-exit (go_stanford full, N=8)
#
# Requires:
#   export MODEL_PATH=...
#   export VAE_PATH=...

cd /home/payneli/project/nwm

if [[ -z "${MODEL_PATH:-}" || -z "${VAE_PATH:-}" ]]; then
  echo "Please export MODEL_PATH and VAE_PATH first." >&2
  exit 1
fi

echo "[SMOKE] full diffusion on recon..."
./uncertainty_research/scripts/run_launch_experiment.sh \
  --out_root "/home/payneli/project/nwm/uncertainty_research/results/_smoke_launch" \
  --dataset recon \
  --split test \
  --num_samples 8 \
  --num_seeds 2 \
  --batch_size 8 \
  --perturbation noise \
  --severity 1 \
  --diffusion_steps 50 \
  --probe_steps 2,50 \
  --perceptual_probe_steps 2,50 \
  --log_every_samples 8 \
  --tag smoke

echo "[SMOKE] early-exit on go_stanford (full traj index)..."
./uncertainty_research/scripts/run_launch_experiment.sh \
  --out_root "/home/payneli/project/nwm/uncertainty_research/results/_smoke_launch" \
  --dataset go_stanford \
  --split test \
  --go_eval_type full \
  --num_samples 8 \
  --num_seeds 2 \
  --batch_size 8 \
  --perturbation none \
  --severity 0 \
  --diffusion_steps 50 \
  --probe_steps 2,10,50 \
  --perceptual_probe_steps 2,50 \
  --log_every_samples 8 \
  --tag smoke

echo "[SMOKE] done."



