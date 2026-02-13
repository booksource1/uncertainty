import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "uncertainty_research/src"))
sys.path.append(PROJECT_ROOT)

from data.dataset import get_recon_dataset, get_go_stanford_dataset
from wrappers.nwm_wrapper import NWMWrapper
from tools.perturbations import PerturbationManager, denormalize


def parse_int_list(s: str) -> list[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_merged(csv_glob: str) -> pd.DataFrame:
    paths = sorted(glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"No csvs matched: {csv_glob}")
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        d["source_csv"] = p
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def group_filter(df: pd.DataFrame, group: str) -> pd.DataFrame:
    if group == "A":
        return df[df["scenario"] == "A_ID"]
    if group == "B":
        return df[df["scenario"].isin(["B_noise5", "B_blur5"])]
    if group == "C":
        return df[df["scenario"] == "C_semantic"]
    raise ValueError(f"Unknown group: {group}")


def select_group_extremes(df: pd.DataFrame, group: str, probe_step: int) -> list[dict]:
    """
    Select the max sample for each metric within a group at a probe_step.
    Returns list of dicts containing selection info.
    """
    sub = group_filter(df, group)
    sub = sub[sub["probe_step"] == int(probe_step)]
    if len(sub) == 0:
        raise ValueError(f"No rows for group={group} at probe_step={probe_step}")

    selected = []
    for metric in ["latent_var", "pixel_var", "lpips", "dreamsim"]:
        if metric not in sub.columns:
            continue
        if float(sub[metric].fillna(0).abs().sum()) == 0:
            continue
        row = sub.loc[sub[metric].astype(float).idxmax()]
        selected.append(
            {
                "group": group,
                "metric": metric,
                "probe_step": int(probe_step),
                "scenario": str(row["scenario"]),
                "dataset": str(row["dataset"]),
                "perturbation": str(row.get("perturbation", "none")),
                "severity": int(row.get("severity", 0)),
                "sample_idx": int(row["sample_idx"]),
                "value": float(row[metric]),
            }
        )
    return selected


def save_context_and_action(out_dir: str, obs: torch.Tensor, action: torch.Tensor, meta: dict, prefix: str):
    """
    obs: [1,T,3,H,W] normalized
    action: [1,3]
    """
    os.makedirs(out_dir, exist_ok=True)
    # context images to grid
    obs_den = denormalize(obs.squeeze(0)).clamp(0, 1)  # [T,3,H,W]
    grid = vutils.make_grid(obs_den, nrow=obs_den.shape[0], padding=2)
    vutils.save_image(grid, os.path.join(out_dir, f"{prefix}_context.png"))

    # also save per-frame
    for t in range(obs_den.shape[0]):
        vutils.save_image(obs_den[t], os.path.join(out_dir, f"{prefix}_context_t{t}.png"))

    # save action vector
    action_list = action.squeeze(0).detach().cpu().float().tolist()
    meta2 = dict(meta)
    meta2["action_delta0"] = action_list
    with open(os.path.join(out_dir, f"{prefix}_meta.json"), "w") as f:
        json.dump(meta2, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, f"{prefix}_action.txt"), "w") as f:
        f.write("action_delta0: " + " ".join([f"{x:.6f}" for x in action_list]) + "\n")


@torch.no_grad()
def render_x0_grids(
    model: NWMWrapper,
    obs: torch.Tensor,         # [1,T,3,H,W] normalized
    action: torch.Tensor,      # [1,3]
    probe_steps: list[int],
    num_seeds: int,
    base_seed: int,
    sample_idx: int,
    out_dir: str,
    prefix: str,
):
    x_cond_bs = model.encode_x_cond(obs)  # [1,num_cond,4,latent,latent]
    B = 1
    S = num_seeds
    x_cond = x_cond_bs.unsqueeze(1).repeat(1, S, 1, 1, 1, 1).reshape(B * S, *x_cond_bs.shape[1:])
    action_rep = action.repeat_interleave(S, dim=0)

    init_noise = torch.empty((B * S, 4, model.latent_size, model.latent_size), device=model.device)
    for s_idx in range(S):
        seed_val = int(base_seed) + int(sample_idx) * 100000 + int(s_idx)
        g = torch.Generator(device=model.device).manual_seed(seed_val)
        init_noise[s_idx] = torch.randn((4, model.latent_size, model.latent_size), device=model.device, generator=g)

    _, probes = model.sample_with_probes_from_x_cond(
        x_cond=x_cond,
        action=action_rep,
        init_noise=init_noise,
        probe_steps=probe_steps,
        stop_at_max_probe=False,
        progress=False,
    )

    os.makedirs(out_dir, exist_ok=True)
    for step in probe_steps:
        if step not in probes:
            continue
        lat = probes[step]  # [S,4,latent,latent]
        imgs = model.vae.decode(lat / 0.18215).sample
        imgs = torch.clamp(imgs, -1.0, 1.0)
        imgs01 = (imgs + 1.0) / 2.0
        grid = vutils.make_grid(imgs01, nrow=S, padding=2)
        vutils.save_image(grid, os.path.join(out_dir, f"{prefix}_x0_step{step}.png"))


def main():
    parser = argparse.ArgumentParser("Render A/B/C extreme samples (max metrics) with context+action and x0 grids")
    parser.add_argument("--results_csv_glob", type=str, required=True)
    parser.add_argument("--select_probe_step", type=int, default=50, help="Which probe_step to select maxima from")
    parser.add_argument("--render_probe_steps", type=str, default="2,10,20,30,40,50")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=1234)

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="/home/payneli/data/sd-vae-ft-ema")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--diffusion_steps", type=int, default=50)

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--go_eval_type", type=str, default="time", choices=["rollout", "time"])
    parser.add_argument("--deterministic_perturb_seed", type=int, default=0, help="If >0, seed perturbations as seed = deterministic_perturb_seed + sample_idx")
    args = parser.parse_args()

    df = load_merged(args.results_csv_glob)
    probe_steps = parse_int_list(args.render_probe_steps)

    selections = []
    for g in ["A", "B", "C"]:
        selections.extend(select_group_extremes(df, g, args.select_probe_step))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "selected_extremes_abc.json"), "w") as f:
        json.dump(selections, f, indent=2, ensure_ascii=False)
    print("Selected extremes:", selections)

    model = NWMWrapper(
        args.model_path,
        args.vae_path,
        device=args.device,
        diffusion_steps=args.diffusion_steps,
    )
    img_size = model.config.get("image_size", 128)
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    ds_recon = get_recon_dataset(split="test", image_size=img_size)
    ds_go = get_go_stanford_dataset(split="test", image_size=img_size, eval_type=args.go_eval_type)
    perturb = PerturbationManager(device=args.device)

    for sel in selections:
        group = sel["group"]
        scenario = sel["scenario"]
        sample_idx = int(sel["sample_idx"])
        perturbation = sel.get("perturbation", "none")
        severity = int(sel.get("severity", 0))

        # load sample
        if group in ["A", "B"]:
            item = ds_recon[sample_idx]
        else:
            item = ds_go[sample_idx]

        obs = item[1].unsqueeze(0).to(args.device)   # [1,T,3,H,W]
        delta = item[-1].unsqueeze(0).to(args.device)  # [1,len,3]
        action = delta[:, 0, :]  # [1,3]

        # extract index_to_data meta if available
        meta = dict(sel)
        ds = ds_recon if group in ["A", "B"] else ds_go
        if hasattr(ds, "index_to_data"):
            try:
                meta["index_to_data"] = ds.index_to_data[sample_idx]
            except Exception:
                pass

        # save raw context+action
        prefix = f"group{group}_{scenario}_max_{sel['metric']}_idx{sample_idx}"
        sample_dir = os.path.join(args.out_dir, prefix)
        save_context_and_action(sample_dir, obs, action, meta, prefix="raw")

        # apply perturbation for B and save perturbed context (and use it for x0 probes)
        obs_for_model = obs
        if group == "B" and perturbation != "none" and severity > 0:
            if args.deterministic_perturb_seed > 0:
                seed = int(args.deterministic_perturb_seed) + int(sample_idx)
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            obs_pert = perturb.apply_perturbation(obs, perturbation, severity)
            save_context_and_action(sample_dir, obs_pert, action, meta, prefix="perturbed")
            obs_for_model = obs_pert

        # render x0 grids from (possibly perturbed) context
        render_x0_grids(
            model=model,
            obs=obs_for_model,
            action=action,
            probe_steps=probe_steps,
            num_seeds=args.num_seeds,
            base_seed=args.base_seed,
            sample_idx=sample_idx,
            out_dir=sample_dir,
            prefix="x0",
        )

    print(f"Saved A/B/C extreme visualizations to: {args.out_dir}")


if __name__ == "__main__":
    main()




