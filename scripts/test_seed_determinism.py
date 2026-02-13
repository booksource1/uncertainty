import argparse
import os
import sys
import json
from typing import Tuple

import numpy as np
import torch

# Add project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "uncertainty_research/src"))
sys.path.append(PROJECT_ROOT)

from data.dataset import get_recon_dataset, get_go_stanford_dataset  # noqa: E402
from wrappers.nwm_wrapper import NWMWrapper  # noqa: E402
from tools.perturbations import PerturbationManager  # noqa: E402


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_once(
    model: NWMWrapper,
    obs: torch.Tensor,
    action: torch.Tensor,
    seed: int,
    reseed_global: bool,
    deterministic_per_seed: bool,
) -> torch.Tensor:
    """
    Run one prediction. Returns pred image in [-1,1], shape [1,3,H,W]
    """
    if reseed_global:
        seed_everything(seed)

    # encode once (deterministic_vae=True in wrapper)
    x_cond = model.encode_x_cond(obs)  # [1, num_cond, 4, h, w]

    # init_noise is always controlled by a per-run generator
    g = torch.Generator(device=model.device).manual_seed(int(seed))
    init_noise = torch.randn(
        (1, 4, model.latent_size, model.latent_size),
        device=model.device,
        generator=g,
    )
    noise_sequence = None
    if deterministic_per_seed:
        g2 = torch.Generator(device=model.device).manual_seed(int(seed) + 99991)
        noise_sequence = torch.randn(
            (model.diffusion.num_timesteps, 1, 4, model.latent_size, model.latent_size),
            device=model.device,
            generator=g2,
        )

    pred = model.predict_from_x_cond(x_cond=x_cond, action=action, init_noise=init_noise, noise_sequence=noise_sequence)
    return pred.detach().float().cpu()


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    d = (a - b).abs()
    return float(d.max().item()), float(d.mean().item())


def main():
    parser = argparse.ArgumentParser("Seed determinism smoke test")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="/home/payneli/data/sd-vae-ft-ema")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--diffusion_steps", type=int, default=50)

    parser.add_argument("--dataset_name", type=str, default="recon", choices=["recon", "go_stanford"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--go_eval_type", type=str, default="time", choices=["rollout", "time"])
    parser.add_argument("--sample_index", type=int, default=0, help="Index in dataset after loading (not sample_idx field).")

    parser.add_argument("--seed", type=int, default=1234, help="Seed under test")
    parser.add_argument("--seed_alt", type=int, default=1235, help="Different seed for contrast")
    parser.add_argument("--reseed_global", action="store_true", help="Reset torch/numpy RNG before each run")
    parser.add_argument("--deterministic_per_seed", action="store_true", help="Use per-step noise_sequence so results are independent of global RNG.")
    parser.add_argument("--apply_perturbation", type=str, default="none", help="none|noise|blur|blackout|photometric")
    parser.add_argument("--severity", type=int, default=0)

    parser.add_argument("--out_dir", type=str, default="", help="If set, write a small json report here.")
    args = parser.parse_args()

    # Stronger determinism settings (best-effort; some ops may still be nondeterministic on some GPUs)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = NWMWrapper(
        model_path=args.model_path,
        vae_path=args.vae_path,
        device=args.device,
        diffusion_steps=args.diffusion_steps,
    )

    img_size = model.config.get("image_size", 128)
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    if args.dataset_name == "recon":
        ds = get_recon_dataset(split=args.split, image_size=img_size)
    else:
        ds = get_go_stanford_dataset(split=args.split, image_size=img_size, eval_type=args.go_eval_type)

    idx = int(args.sample_index)
    item = ds[idx]
    # EvalDataset returns (idx, obs_image, pred_image, delta)
    obs = item[1].unsqueeze(0).to(args.device)  # [1,T,3,H,W]
    delta = item[-1].unsqueeze(0).to(args.device)  # [1,len,3]
    action = delta[:, 0, :]  # [1,3]

    if args.apply_perturbation != "none" and args.severity > 0:
        perturb = PerturbationManager(args.device)
        obs_in = perturb.apply_perturbation(obs, args.apply_perturbation, int(args.severity))
    else:
        obs_in = obs

    # Run twice with the same seed
    p1 = run_once(model, obs_in, action, seed=args.seed, reseed_global=args.reseed_global, deterministic_per_seed=args.deterministic_per_seed)
    p2 = run_once(model, obs_in, action, seed=args.seed, reseed_global=args.reseed_global, deterministic_per_seed=args.deterministic_per_seed)
    same_max, same_mean = diff_stats(p1, p2)

    # Run with a different seed for contrast
    p3 = run_once(model, obs_in, action, seed=args.seed_alt, reseed_global=args.reseed_global, deterministic_per_seed=args.deterministic_per_seed)
    alt_max, alt_mean = diff_stats(p1, p3)

    report = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "go_eval_type": args.go_eval_type,
        "sample_index": idx,
        "diffusion_steps": int(args.diffusion_steps),
        "seed": int(args.seed),
        "seed_alt": int(args.seed_alt),
        "reseed_global": bool(args.reseed_global),
        "deterministic_per_seed": bool(args.deterministic_per_seed),
        "perturbation": args.apply_perturbation,
        "severity": int(args.severity),
        "same_seed_max_abs_diff": same_max,
        "same_seed_mean_abs_diff": same_mean,
        "diff_seed_max_abs_diff": alt_max,
        "diff_seed_mean_abs_diff": alt_mean,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "seed_determinism_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


