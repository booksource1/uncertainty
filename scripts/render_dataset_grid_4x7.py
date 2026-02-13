import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

from PIL import Image


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# Some dataset utilities expect to open config files via relative paths from repo root.
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "uncertainty_research/src"))
sys.path.append(PROJECT_ROOT)

from data.dataset import get_go_stanford_dataset, get_recon_dataset  # noqa: E402
from tools.perturbations import PerturbationManager, denormalize  # noqa: E402


@dataclass(frozen=True)
class RowSpec:
    label: str
    dataset: str  # recon | go_stanford
    perturbation: str  # none | noise
    severity: int


def _to_rgb01(x_chw: torch.Tensor) -> np.ndarray:
    """x_chw: [3,H,W] in [0,1]."""
    x = x_chw.detach().float().cpu().clamp(0, 1).numpy()
    return np.transpose(x, (1, 2, 0))


def _pick_frame(obs_tchw: torch.Tensor, frame_idx: int) -> torch.Tensor:
    """obs_tchw: [T,3,H,W] normalized."""
    T = int(obs_tchw.shape[0])
    idx = int(frame_idx)
    if idx < 0:
        idx = T + idx
    idx = max(0, min(T - 1, idx))
    return obs_tchw[idx]  # [3,H,W]


def _noise_seeds_for_samples(sample_indices: list[int], *, base_seed: int, severity: int) -> torch.Tensor:
    """
    Match repo convention used in scripts/analysis/render_extremes_by_condition.py:
        s = base_seed + sample_idx*100000 + pid*100 + severity
    where pid(noise)=1.
    """
    pid = 1
    seeds = [int(base_seed) + int(si) * 100000 + int(pid) * 100 + int(severity) for si in sample_indices]
    return torch.tensor(seeds, dtype=torch.long)


def render_grid(
    *,
    out_path: str,
    recon_indices: list[int],
    stanford_indices: list[int],
    seed: int,
    frame_idx: int,
    image_size: tuple[int, int],
    go_eval_type: str,
    dpi: int,
) -> None:
    # ---- Load datasets ----
    ds_recon = get_recon_dataset(split="test", image_size=image_size)
    ds_go = get_go_stanford_dataset(split="test", image_size=image_size, eval_type=go_eval_type)

    # ---- Define rows (7 image rows) ----
    rows: list[RowSpec] = [
        RowSpec(label="ID", dataset="recon", perturbation="none", severity=0),
        RowSpec(label="OOD-Noise1", dataset="recon", perturbation="noise", severity=1),
        RowSpec(label="OOD-Noise2", dataset="recon", perturbation="noise", severity=2),
        RowSpec(label="OOD-Noise3", dataset="recon", perturbation="noise", severity=3),
        RowSpec(label="OOD-Noise4", dataset="recon", perturbation="noise", severity=4),
        RowSpec(label="OOD-Noise5", dataset="recon", perturbation="noise", severity=5),
        RowSpec(label="OOD-Stanford", dataset="go_stanford", perturbation="none", severity=0),
    ]

    # ---- Collect base images for the N columns ----
    ncols = 6
    if len(recon_indices) != ncols or len(stanford_indices) != ncols:
        raise ValueError(f"recon_indices 和 stanford_indices 都必须是长度为 {ncols} 的列表。")

    recon_frames = []
    for idx in recon_indices:
        item = ds_recon[int(idx)]
        obs = item[1]  # [T,3,H,W] normalized
        recon_frames.append(_pick_frame(obs, frame_idx=frame_idx))
    recon_frames_bchw = torch.stack(recon_frames, dim=0)  # [N,3,H,W]

    go_frames = []
    for idx in stanford_indices:
        item = ds_go[int(idx)]
        obs = item[1]  # [T,3,H,W] normalized
        go_frames.append(_pick_frame(obs, frame_idx=frame_idx))
    go_frames_bchw = torch.stack(go_frames, dim=0)  # [N,3,H,W]

    perturb = PerturbationManager(device="cpu")

    # ---- Build per-row images (each row -> list of 4 np images) ----
    per_row_imgs: list[list[np.ndarray]] = []
    for r in rows:
        if r.dataset == "recon":
            base = recon_frames_bchw
            sample_indices = recon_indices
        else:
            base = go_frames_bchw
            sample_indices = stanford_indices

        if r.perturbation == "none" or int(r.severity) == 0:
            out = base
        elif r.perturbation == "noise":
            seeds = _noise_seeds_for_samples(sample_indices, base_seed=seed, severity=int(r.severity))
            out = perturb.apply_perturbation(base, "noise", int(r.severity), seeds=seeds)
        else:
            raise ValueError(f"Unsupported perturbation in this script: {r.perturbation}")

        out01 = denormalize(out).clamp(0, 1)  # [N,3,H,W] in [0,1]
        per_row_imgs.append([_to_rgb01(out01[c]) for c in range(ncols)])

    # ---- Compose a single PNG via PIL (avoid matplotlib binary deps) ----
    nrows = len(rows)  # 7
    H, W = int(image_size[0]), int(image_size[1])

    # Spacing tuned for a cleaner look (no text labels).
    outer_margin = max(8, int(0.06 * min(H, W)))
    gap_x = max(6, int(0.04 * W))
    gap_y = max(6, int(0.05 * H))

    canvas_w = outer_margin * 2 + ncols * W + (ncols - 1) * gap_x
    canvas_h = outer_margin * 2 + nrows * H + (nrows - 1) * gap_y

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    y = outer_margin
    for r_idx in range(nrows):
        x = outer_margin
        for c in range(ncols):
            arr = (per_row_imgs[r_idx][c] * 255.0).round().clip(0, 255).astype(np.uint8)
            im = Image.fromarray(arr, mode="RGB")
            if im.size != (W, H):
                im = im.resize((W, H), resample=Image.BILINEAR)
            canvas.paste(im, (x, y))
            x += W + gap_x
        y += H + gap_y

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser("Render a dataset visualization grid: ID + Noise1..5 + Stanford")
    parser.add_argument("--out_path", type=str, required=True, help="Output png path")

    parser.add_argument("--seed", type=int, default=1234, help="Seed for deterministic sampling/noise")
    parser.add_argument("--frame_idx", type=int, default=-1, help="Which frame from context [T,3,H,W] to visualize (default: last)")
    parser.add_argument("--go_eval_type", type=str, default="time", choices=["rollout", "time", "full"])

    parser.add_argument("--image_size", type=str, default="128,128", help="e.g. 128,128")
    parser.add_argument("--dpi", type=int, default=220)

    parser.add_argument(
        "--recon_indices",
        type=str,
        default="",
        help="Optional: comma list of 6 fixed indices from recon dataset (after loading). If empty, sample randomly.",
    )
    parser.add_argument(
        "--stanford_indices",
        type=str,
        default="",
        help="Optional: comma list of 6 fixed indices from go_stanford dataset (after loading). If empty, sample randomly.",
    )

    args = parser.parse_args()

    try:
        H, W = [int(x.strip()) for x in str(args.image_size).split(",")]
        image_size = (H, W)
    except Exception as e:
        raise ValueError(f"Invalid --image_size '{args.image_size}', expected 'H,W'") from e

    rng = np.random.default_rng(int(args.seed))

    # Load datasets once to know lengths for sampling
    ds_recon = get_recon_dataset(split="test", image_size=image_size)
    ds_go = get_go_stanford_dataset(split="test", image_size=image_size, eval_type=args.go_eval_type)

    def _parse_indices(s: str) -> list[int]:
        if not s:
            return []
        xs = [int(x.strip()) for x in s.split(",") if x.strip()]
        return xs

    ncols = 6
    recon_indices = _parse_indices(args.recon_indices)
    if not recon_indices:
        recon_indices = rng.choice(len(ds_recon), size=ncols, replace=False).tolist()

    stanford_indices = _parse_indices(args.stanford_indices)
    if not stanford_indices:
        stanford_indices = rng.choice(len(ds_go), size=ncols, replace=False).tolist()

    if len(recon_indices) != ncols or len(stanford_indices) != ncols:
        raise ValueError(f"--recon_indices/--stanford_indices 若提供，必须各自恰好 {ncols} 个。")

    render_grid(
        out_path=os.path.abspath(args.out_path),
        recon_indices=[int(x) for x in recon_indices],
        stanford_indices=[int(x) for x in stanford_indices],
        seed=int(args.seed),
        frame_idx=int(args.frame_idx),
        image_size=image_size,
        go_eval_type=str(args.go_eval_type),
        dpi=int(args.dpi),
    )

    print(
        f"[OK] saved grid to {os.path.abspath(args.out_path)}\n"
        f"  recon_indices={recon_indices}\n"
        f"  stanford_indices={stanford_indices}\n"
        f"  frame_idx={int(args.frame_idx)} go_eval_type={args.go_eval_type} image_size={image_size} seed={int(args.seed)}"
    )


if __name__ == "__main__":
    main()

