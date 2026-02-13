import argparse
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

from data.dataset import get_go_stanford_dataset
from wrappers.nwm_wrapper import NWMWrapper
from tools.perturbations import PerturbationManager


def parse_int_list(s: str) -> list[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_merged(csv_glob: str) -> pd.DataFrame:
    paths = sorted(glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"No csvs matched: {csv_glob}")
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def select_extremes(df: pd.DataFrame, scenario: str, probe_step: int) -> dict[str, int]:
    """
    Returns sample_idx for max of each metric at given probe_step.
    """
    sub = df[(df["scenario"] == scenario) & (df["probe_step"] == probe_step)]
    if len(sub) == 0:
        raise ValueError(f"No rows for scenario={scenario}, probe_step={probe_step}")

    out = {}
    for metric in ["latent_var", "pixel_var", "lpips", "dreamsim"]:
        if metric not in sub.columns:
            continue
        # skip empty metric columns
        if float(sub[metric].fillna(0).abs().sum()) == 0:
            continue
        row = sub.loc[sub[metric].astype(float).idxmax()]
        out[metric] = int(row["sample_idx"])
    return out


@torch.no_grad()
def render_sample(
    model: NWMWrapper,
    dataset,
    sample_global_idx: int,
    num_seeds: int,
    diffusion_steps: int,
    probe_steps: list[int],
    base_seed: int,
    out_dir: str,
    title_prefix: str,
):
    item = dataset[sample_global_idx]
    obs = item[1].unsqueeze(0).to(model.device)  # [1,T,3,H,W]
    delta = item[-1].unsqueeze(0).to(model.device)  # [1,len,3]
    action = delta[:, 0, :]  # [1,3]

    x_cond_bs = model.encode_x_cond(obs)  # [1,num_cond,4,latent,latent]
    B = 1
    S = num_seeds
    x_cond = x_cond_bs.unsqueeze(1).repeat(1, S, 1, 1, 1, 1).reshape(B * S, *x_cond_bs.shape[1:])
    action_rep = action.repeat_interleave(S, dim=0)

    init_noise = torch.empty((B * S, 4, model.latent_size, model.latent_size), device=model.device)
    for s_idx in range(S):
        seed_val = int(base_seed) + int(sample_global_idx) * 100000 + int(s_idx)
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
    # For each probe step: decode S images and save a grid (rows=seeds)
    for step in probe_steps:
        if step not in probes:
            continue
        lat = probes[step]  # [S,4,latent,latent]
        imgs = model.vae.decode(lat / 0.18215).sample
        imgs = torch.clamp(imgs, -1.0, 1.0)
        # to [0,1] for saving
        imgs01 = (imgs + 1.0) / 2.0
        grid = vutils.make_grid(imgs01, nrow=S, padding=2)
        path = os.path.join(out_dir, f"{title_prefix}_idx{sample_global_idx}_step{step}.png")
        vutils.save_image(grid, path)


def main():
    parser = argparse.ArgumentParser("Render x0 probe visualizations for extreme samples")
    parser.add_argument("--results_csv_glob", type=str, required=True, help="e.g. /path/to/results/early_exit_results_shard*.csv")
    parser.add_argument("--scenario", type=str, default="C_semantic")
    parser.add_argument("--select_probe_step", type=int, default=50)
    parser.add_argument("--render_probe_steps", type=str, default="2,10,20,30,40,50")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="/home/payneli/data/sd-vae-ft-ema")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=1234)

    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    df = load_merged(args.results_csv_glob)
    extremes = select_extremes(df, args.scenario, args.select_probe_step)
    print("Selected extremes:", extremes)

    model = NWMWrapper(
        args.model_path,
        args.vae_path,
        device=args.device,
        diffusion_steps=args.diffusion_steps,
    )
    img_size = model.config.get("image_size", 128)
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    # always use go_stanford time for rendering, since the experiment does
    dataset = get_go_stanford_dataset(split="test", image_size=img_size, eval_type="time")

    probe_steps = parse_int_list(args.render_probe_steps)
    for metric, sample_idx in extremes.items():
        render_sample(
            model=model,
            dataset=dataset,
            sample_global_idx=sample_idx,
            num_seeds=args.num_seeds,
            diffusion_steps=args.diffusion_steps,
            probe_steps=probe_steps,
            base_seed=args.base_seed,
            out_dir=args.out_dir,
            title_prefix=f"{args.scenario}_max_{metric}",
        )

    print(f"Saved visualizations to: {args.out_dir}")


if __name__ == "__main__":
    main()




