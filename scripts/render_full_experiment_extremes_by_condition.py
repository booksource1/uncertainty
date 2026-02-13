import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "uncertainty_research/src"))
sys.path.append(PROJECT_ROOT)

from data.dataset import get_recon_dataset
from wrappers.nwm_wrapper import NWMWrapper
from tools.perturbations import PerturbationManager, denormalize
from metrics.metrics import UncertaintyMetrics


def safe_name(s: str) -> str:
    return str(s).replace("/", "_").replace(" ", "_")


def to_m11_from_imnet(x_imnet: torch.Tensor) -> torch.Tensor:
    return denormalize(x_imnet).clamp(0, 1) * 2.0 - 1.0


@torch.no_grad()
def render_one(
    model: NWMWrapper,
    metrics: UncertaintyMetrics,
    perturb: PerturbationManager,
    dataset,
    sample_idx: int,
    group: str,
    perturbation: str,
    severity: int,
    num_seeds: int,
    base_seed: int,
    deterministic_perturb_seed: int,
    out_dir: str,
):
    item = dataset[int(sample_idx)]
    obs = item[1].unsqueeze(0).to(model.device)          # [1,T,3,H,W] imnet
    gt_seq = item[2].to(model.device)                    # [len,3,H,W] imnet
    gt0 = gt_seq[0]                                      # [3,H,W] imnet
    delta = item[-1].unsqueeze(0).to(model.device)       # [1,len,3]
    action = delta[:, 0, :]                              # [1,3]

    # Save raw context and gt
    os.makedirs(out_dir, exist_ok=True)
    obs_01 = denormalize(obs.squeeze(0)).clamp(0, 1)      # [T,3,H,W]
    vutils.save_image(vutils.make_grid(obs_01, nrow=obs_01.shape[0]), os.path.join(out_dir, "raw_context.png"))
    for t in range(obs_01.shape[0]):
        vutils.save_image(obs_01[t], os.path.join(out_dir, f"raw_context_t{t}.png"))
    vutils.save_image(denormalize(gt0).clamp(0, 1), os.path.join(out_dir, "gt_next.png"))
    with open(os.path.join(out_dir, "action.json"), "w") as f:
        json.dump({"action_delta0": action.squeeze(0).detach().cpu().float().tolist()}, f, indent=2)

    obs_in = obs
    if perturbation != "none" and severity > 0:
        if deterministic_perturb_seed > 0:
            seed = int(deterministic_perturb_seed) + int(sample_idx)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        obs_in = perturb.apply_perturbation(obs, perturbation, severity)
        obs_in_01 = denormalize(obs_in.squeeze(0)).clamp(0, 1)
        vutils.save_image(vutils.make_grid(obs_in_01, nrow=obs_in_01.shape[0]), os.path.join(out_dir, "perturbed_context.png"))
        for t in range(obs_in_01.shape[0]):
            vutils.save_image(obs_in_01[t], os.path.join(out_dir, f"perturbed_context_t{t}.png"))

    # predict S seeds (final output only)
    x_cond_bs = model.encode_x_cond(obs_in)  # [1,num_cond,4,lat,lat]
    B = 1
    S = num_seeds
    x_cond = x_cond_bs.unsqueeze(1).repeat(1, S, 1, 1, 1, 1).reshape(B * S, *x_cond_bs.shape[1:])
    action_rep = action.repeat_interleave(S, dim=0)

    init_noise = torch.empty((B * S, 4, model.latent_size, model.latent_size), device=model.device)
    key = f"{group}_{perturbation}_{severity}"
    key_hash = abs(hash(key)) % 10000
    for s_idx in range(S):
        seed_val = int(base_seed) + int(sample_idx) * 100000 + int(key_hash) * 100 + int(s_idx)
        g = torch.Generator(device=model.device).manual_seed(seed_val)
        init_noise[s_idx] = torch.randn((4, model.latent_size, model.latent_size), device=model.device, generator=g)

    preds = model.predict_from_x_cond(x_cond=x_cond, action=action_rep, init_noise=init_noise)  # [S,3,H,W] in [-1,1]
    preds = torch.clamp(preds, -1, 1)
    preds_01 = (preds + 1.0) / 2.0
    vutils.save_image(vutils.make_grid(preds_01, nrow=S), os.path.join(out_dir, "preds_grid.png"))

    # grid with gt in front
    gt0_m11 = to_m11_from_imnet(gt0)
    gt0_01 = (gt0_m11 + 1.0) / 2.0
    grid2 = vutils.make_grid(torch.cat([gt0_01.unsqueeze(0), preds_01], dim=0), nrow=S + 1)
    vutils.save_image(grid2, os.path.join(out_dir, "gt_plus_preds_grid.png"))

    # also write computed metrics (divergence + gt error)
    div = metrics.compute_diversity(preds)
    gt_err = metrics.compute_gt_error(preds, gt0_m11)
    stats = {
        "pixel_variance": float(metrics.compute_pixel_variance(preds)),
        "lpips": float(div.get("lpips", 0.0)),
        "dreamsim": float(div.get("dreamsim", 0.0)),
        "lpips_gt": float(gt_err.get("lpips_gt", 0.0)),
        "dreamsim_gt": float(gt_err.get("dreamsim_gt", 0.0)),
    }
    with open(os.path.join(out_dir, "computed_metrics.json"), "w") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser("Render full-diffusion experiment extremes by (group, perturbation, severity)")
    parser.add_argument("--results_csv", type=str, required=True, help="final_results.csv from run_experiment")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="/home/payneli/data/sd-vae-ft-ema")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=1234)
    parser.add_argument("--deterministic_perturb_seed", type=int, default=999)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    # filter recon only (this script is for the perturbation experiment)
    if "dataset" in df.columns:
        df = df[df["dataset"] == "recon"]

    # metrics to select maxima by (these are divergence metrics, not gt error)
    select_metrics = ["pixel_variance", "lpips", "dreamsim"]

    # group conditions
    cond_cols = ["group", "perturbation", "severity"]
    df["severity"] = df["severity"].astype(int)
    for c in cond_cols:
        df[c] = df[c].astype(str)

    selections = []
    for (g, p, s), sub in df.groupby(cond_cols):
        for m in select_metrics:
            if m not in sub.columns:
                continue
            row = sub.loc[sub[m].astype(float).idxmax()]
            selections.append(
                {
                    "group": g,
                    "perturbation": p,
                    "severity": int(s),
                    "metric": m,
                    "sample_idx": int(row["sample_idx"]),
                    "value": float(row[m]),
                }
            )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "selected_extremes_by_condition.json"), "w") as f:
        json.dump(selections, f, indent=2, ensure_ascii=False)

    model = NWMWrapper(args.model_path, args.vae_path, device=args.device, diffusion_steps=args.diffusion_steps)
    img_size = model.config.get("image_size", 128)
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    dataset = get_recon_dataset(split="test", image_size=img_size)
    metrics = UncertaintyMetrics(args.device, use_lpips=True, use_dreamsim=True)
    perturb = PerturbationManager(args.device)

    # render
    for sel in selections:
        d = os.path.join(
            args.out_dir,
            f"group={safe_name(sel['group'])}",
            f"pert={safe_name(sel['perturbation'])}_sev={sel['severity']}",
            f"max_{sel['metric']}_idx{sel['sample_idx']}",
        )
        render_one(
            model=model,
            metrics=metrics,
            perturb=perturb,
            dataset=dataset,
            sample_idx=int(sel["sample_idx"]),
            group=sel["group"],
            perturbation=sel["perturbation"],
            severity=int(sel["severity"]),
            num_seeds=args.num_seeds,
            base_seed=args.base_seed,
            deterministic_perturb_seed=args.deterministic_perturb_seed,
            out_dir=d,
        )

    print(f"Saved renders to: {args.out_dir}")


if __name__ == "__main__":
    main()




