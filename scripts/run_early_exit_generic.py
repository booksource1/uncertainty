import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "uncertainty_research/src"))
sys.path.append(PROJECT_ROOT)

from data.dataset import get_recon_dataset, get_go_stanford_dataset  # noqa: E402
from wrappers.nwm_wrapper import NWMWrapper  # noqa: E402
from tools.perturbations import PerturbationManager, denormalize  # noqa: E402
from metrics.metrics import UncertaintyMetrics  # noqa: E402


def parse_int_list(s: str) -> list[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def latent_mean_variance(latents: torch.Tensor) -> float:
    """
    latents: [S, 4, H, W] float
    return: mean variance over all dims
    """
    latents = latents.float()
    var_map = torch.var(latents, dim=0, unbiased=False)
    return var_map.mean().item()


@dataclass
class Scenario:
    name: str
    dataset_name: str
    split: str
    go_eval_type: str | None
    perturbation: str
    severity: int


def main():
    parser = argparse.ArgumentParser("Generic Early-Exit runner (single scenario, configurable dataset/perturbation)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="/home/payneli/data/sd-vae-ft-ema")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--dataset_name", type=str, required=True, choices=["recon", "go_stanford"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--go_eval_type", type=str, default="full", choices=["rollout", "time", "full"])

    parser.add_argument("--perturbation", type=str, default="none", choices=["none", "noise", "blur", "blackout", "photometric"])
    parser.add_argument("--severity", type=int, default=0)

    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--probe_steps", type=str, default="2,10,20,30,40,50")
    parser.add_argument("--perceptual_probe_steps", type=str, default="")
    parser.add_argument("--num_seeds", type=int, default=5)

    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to evaluate (stride subsample). -1 means all.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument("--base_seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no_dreamsim", action="store_true")
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--log_every_samples", type=int, default=64)

    # Multi-GPU sharding
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)

    # Determinism switches (recommended)
    parser.add_argument("--deterministic_per_seed", action="store_true")
    parser.add_argument("--deterministic_perturb", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Best-effort deterministic kernels (does not guarantee full determinism for all ops,
    # but helps make CUDA behavior stable across runs).
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.base_seed)
    np.random.seed(args.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.base_seed)

    probe_steps = parse_int_list(args.probe_steps)
    perc_steps = set(parse_int_list(args.perceptual_probe_steps))
    if len(perc_steps) == 0:
        perc_steps = set(probe_steps)

    model = NWMWrapper(args.model_path, args.vae_path, device=args.device, diffusion_steps=args.diffusion_steps)
    img_size = model.config.get("image_size", 128)
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    metrics = UncertaintyMetrics(args.device, use_lpips=(not args.no_lpips), use_dreamsim=(not args.no_dreamsim))
    perturb_mgr = PerturbationManager(args.device)

    sc = Scenario(
        name=f"{args.dataset_name}_pert={args.perturbation}_sev={args.severity}",
        dataset_name=args.dataset_name,
        split=args.split,
        go_eval_type=(args.go_eval_type if args.dataset_name == "go_stanford" else None),
        perturbation=args.perturbation,
        severity=int(args.severity),
    )

    if sc.dataset_name == "recon":
        ds = get_recon_dataset(split=sc.split, image_size=img_size)
    else:
        ds = get_go_stanford_dataset(split=sc.split, image_size=img_size, eval_type=sc.go_eval_type or "full")

    if args.num_samples == -1:
        n = len(ds)
    else:
        n = min(int(args.num_samples), len(ds))
    if n <= 0:
        raise ValueError("--num_samples must be -1 or a positive int")

    stride = max(1, len(ds) // n)
    indices_all = list(range(0, len(ds), stride))[:n]
    indices = [ix for k, ix in enumerate(indices_all) if (k % args.num_shards) == args.shard_id]
    if len(indices) == 0:
        print(f"[warn] shard {args.shard_id}/{args.num_shards} has 0 samples; exiting.")
        return

    subset = Subset(ds, indices)

    def thin_collate(batch):
        idxs = torch.stack([b[0] for b in batch], dim=0)  # [B,1] or [B]
        obs = torch.stack([b[1] for b in batch], dim=0)  # [B,T,3,H,W]
        gt0 = torch.stack([b[2][0] for b in batch], dim=0)  # [B,3,H,W] (ImageNet normalized)
        delta = torch.stack([b[-1] for b in batch], dim=0)  # [B,len,3]
        return idxs, obs, gt0, delta

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        collate_fn=thin_collate,
    )

    rows = []
    pbar = tqdm(total=len(indices), desc=f"{sc.name} shard {args.shard_id}/{args.num_shards}", unit="sample", dynamic_ncols=True)
    processed = 0

    for batch in loader:
        idxs = batch[0].to(args.device)
        if idxs.ndim == 2:
            idxs_flat = idxs[:, 0].long()
        else:
            idxs_flat = idxs.long()
        obs = batch[1].to(args.device)
        gt0_imnet = batch[2].to(args.device)
        delta = batch[3].to(args.device)
        action = delta[:, 0, :] if (delta.ndim >= 3 and delta.shape[-1] == 3) else torch.zeros((obs.shape[0], 3), device=args.device)

        gt0_m11 = denormalize(gt0_imnet).clamp(0, 1) * 2.0 - 1.0

        # perturbation
        if sc.perturbation != "none" and sc.severity > 0:
            if args.deterministic_perturb:
                pert_id_map = {"noise": 1, "blur": 2, "blackout": 3, "photometric": 4}
                pid = int(pert_id_map.get(sc.perturbation, 99))
                pert_seeds = (args.base_seed + idxs_flat.long() * 100000 + pid * 100 + int(sc.severity)).to("cpu")
                obs_in = perturb_mgr.apply_perturbation(obs, sc.perturbation, sc.severity, seeds=pert_seeds)
            else:
                obs_in = perturb_mgr.apply_perturbation(obs, sc.perturbation, sc.severity)
        else:
            obs_in = obs

        x_cond_bs = model.encode_x_cond(obs_in)  # [B, num_cond, 4, h, w]
        B = x_cond_bs.shape[0]
        S = int(args.num_seeds)

        x_cond = x_cond_bs.unsqueeze(1).repeat(1, S, 1, 1, 1, 1).reshape(B * S, *x_cond_bs.shape[1:])
        action_rep = action.repeat_interleave(S, dim=0)

        init_noise = torch.empty((B * S, 4, model.latent_size, model.latent_size), device=args.device)
        noise_sequence = None
        if args.deterministic_per_seed:
            noise_sequence = torch.empty(
                (model.diffusion.num_timesteps, B * S, 4, model.latent_size, model.latent_size),
                device=args.device,
            )

        for b in range(B):
            sid = int(idxs_flat[b].item())
            for s_idx in range(S):
                seed_val = int(args.base_seed) + sid * 100000 + 0 * 100 + int(s_idx)
                g = torch.Generator(device=args.device).manual_seed(seed_val)
                init_noise[b * S + s_idx] = torch.randn((4, model.latent_size, model.latent_size), device=args.device, generator=g)
                if noise_sequence is not None:
                    g2 = torch.Generator(device=args.device).manual_seed(seed_val + 99991)
                    noise_sequence[:, b * S + s_idx] = torch.randn(
                        (model.diffusion.num_timesteps, 4, model.latent_size, model.latent_size),
                        device=args.device,
                        generator=g2,
                    )

        latent_final, probes = model.sample_with_probes_from_x_cond(
            x_cond=x_cond,
            action=action_rep,
            init_noise=init_noise,
            probe_steps=probe_steps,
            noise_sequence=noise_sequence,
            stop_at_max_probe=False,
            progress=False,
        )

        probes = {k: v.reshape(B, S, *v.shape[1:]) for k, v in probes.items()}

        for b in range(B):
            for step in probe_steps:
                if step not in probes:
                    continue
                lat_s = probes[step][b]  # [S,4,h,w]
                lv = latent_mean_variance(lat_s)

                out = {
                    "scenario": sc.name,
                    "dataset": sc.dataset_name,
                    "perturbation": sc.perturbation,
                    "severity": int(sc.severity),
                    "sample_idx": int(idxs_flat[b].item()),
                    "probe_step": int(step),
                    "latent_var": float(lv),
                    "pixel_var": 0.0,
                    "lpips": 0.0,
                    "dreamsim": 0.0,
                    "lpips_gt": 0.0,
                    "dreamsim_gt": 0.0,
                    "mse_gt": 0.0,
                    "psnr_gt": 0.0,
                    "ssim_gt": 0.0,
                }

                if step in perc_steps:
                    with torch.no_grad():
                        imgs = model.vae.decode(lat_s / 0.18215).sample
                        imgs = torch.clip(imgs, -1.0, 1.0)
                    out["pixel_var"] = float(metrics.compute_pixel_variance(imgs))
                    div = metrics.compute_diversity(imgs)
                    out["lpips"] = float(div.get("lpips", 0.0))
                    out["dreamsim"] = float(div.get("dreamsim", 0.0))
                    gt_err = metrics.compute_gt_error(imgs, gt0_m11[b])
                    out["lpips_gt"] = float(gt_err.get("lpips_gt", 0.0))
                    out["dreamsim_gt"] = float(gt_err.get("dreamsim_gt", 0.0))
                    out["mse_gt"] = float(gt_err.get("mse_gt", 0.0))
                    out["psnr_gt"] = float(gt_err.get("psnr_gt", 0.0))
                    out["ssim_gt"] = float(gt_err.get("ssim_gt", 0.0))

                rows.append(out)

        processed += B
        pbar.update(B)
        if args.log_every_samples > 0 and (processed % args.log_every_samples) < B:
            print(f"[progress] samples={processed}/{len(indices)} rows={len(rows)}", flush=True)

    pbar.close()
    out_csv = os.path.join(args.output_dir, f"early_exit_generic_shard{args.shard_id}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()



