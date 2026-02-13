import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
# Some dataset utilities expect to open config files via relative paths from repo root.
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "uncertainty_research/src"))
sys.path.append(PROJECT_ROOT)

from data.dataset import get_recon_dataset, get_go_stanford_dataset  # noqa: E402
from wrappers.nwm_wrapper import NWMWrapper  # noqa: E402
from tools.perturbations import PerturbationManager, denormalize  # noqa: E402


@dataclass(frozen=True)
class Condition:
    dataset: str
    perturbation: str
    severity: int

    def key(self) -> str:
        return f"{self.dataset}_{self.perturbation}_{int(self.severity)}"


def parse_int_list(s: str) -> list[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_all_concat(out_root: str, run_id: str, tag: str) -> pd.DataFrame:
    analysis_root = os.path.join(out_root, f"{run_id}_analysis_{tag}")
    path = os.path.join(analysis_root, "all_concat.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing all_concat.csv at: {path}. Please run big_table.py first.")
    return pd.read_csv(path)


def _cond_filter(df: pd.DataFrame, cond: Condition) -> pd.DataFrame:
    return df[
        (df["dataset"].astype(str) == str(cond.dataset))
        & (df["perturbation"].astype(str) == str(cond.perturbation))
        & (df["severity"].astype(int) == int(cond.severity))
    ]


def _ensure_probe_steps(df: pd.DataFrame, probe_steps: list[int]) -> list[int]:
    if probe_steps:
        return sorted({int(x) for x in probe_steps})
    return sorted({int(x) for x in df["probe_step"].unique()})


def _pid_for_pert(pert: str) -> int:
    return {"noise": 1, "blur": 2, "blackout": 3, "photometric": 4}.get(str(pert), 99)


def apply_condition_perturbation(
    obs: torch.Tensor,
    *,
    cond: Condition,
    device: str,
    base_seed: int,
    sample_idx: int,
    deterministic_perturb: bool,
) -> torch.Tensor:
    if cond.perturbation == "none" or int(cond.severity) == 0:
        return obs
    perturb = PerturbationManager(device=device)
    if deterministic_perturb:
        pid = _pid_for_pert(cond.perturbation)
        s = int(base_seed) + int(sample_idx) * 100000 + int(pid) * 100 + int(cond.severity)
        seeds = torch.tensor([s], dtype=torch.long)  # [1]
        return perturb.apply_perturbation(obs, cond.perturbation, int(cond.severity), seeds=seeds)
    return perturb.apply_perturbation(obs, cond.perturbation, int(cond.severity))


@torch.no_grad()
def render_one_sample_grid(
    *,
    model: NWMWrapper,
    obs: torch.Tensor,   # [1,T,3,H,W] normalized
    gt: torch.Tensor,    # [1,3,H,W] normalized
    action: torch.Tensor,  # [1,3]
    probe_steps: list[int],
    num_seeds: int,
    base_seed: int,
    sample_idx: int,
    out_dir: str,
    prefix: str,
    deterministic_per_seed: bool,
    summary_layout: str = "horizontal",  # "vertical" (old) or "horizontal" (transpose)
    summary_no_text: bool = False,
    summary_compact: bool = True,
    step_gt: dict[int, dict] | None = None,
    label_margin_px: int = 260,  # unused in matplotlib rendering (kept for backwards-compat)
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Prepare context + GT images (0..1)
    obs_den = denormalize(obs.squeeze(0)).clamp(0, 1)  # [T,3,H,W]
    gt_den = denormalize(gt.squeeze(0)).clamp(0, 1)    # [3,H,W]

    # Encode
    x_cond_bs = model.encode_x_cond(obs)  # [1,num_cond,4,latent,latent]
    B = 1
    S = int(num_seeds)
    x_cond = x_cond_bs.unsqueeze(1).repeat(1, S, 1, 1, 1, 1).reshape(B * S, *x_cond_bs.shape[1:])
    action_rep = action.repeat_interleave(S, dim=0)

    # Deterministic init_noise and per-step noise_sequence (to match experiment logic)
    init_noise = torch.empty((B * S, 4, model.latent_size, model.latent_size), device=model.device)
    noise_sequence = None
    if deterministic_per_seed:
        noise_sequence = torch.empty((model.diffusion.num_timesteps, B * S, 4, model.latent_size, model.latent_size), device=model.device)

    for s_idx in range(S):
        seed_val = int(base_seed) + int(sample_idx) * 100000 + 0 * 100 + int(s_idx)
        g = torch.Generator(device=model.device).manual_seed(seed_val)
        init_noise[s_idx] = torch.randn((4, model.latent_size, model.latent_size), device=model.device, generator=g)
        if noise_sequence is not None:
            g2 = torch.Generator(device=model.device).manual_seed(seed_val + 99991)
            noise_sequence[:, s_idx] = torch.randn(
                (model.diffusion.num_timesteps, 4, model.latent_size, model.latent_size),
                device=model.device,
                generator=g2,
            )

    _, probes = model.sample_with_probes_from_x_cond(
        x_cond=x_cond,
        action=action_rep,
        init_noise=init_noise,
        probe_steps=probe_steps,
        noise_sequence=noise_sequence,
        stop_at_max_probe=False,
        progress=False,
    )

    # Build per-step rows: each row is a seed grid for that step
    step_labels: list[str] = []
    step_imgs: dict[int, torch.Tensor] = {}  # step -> [S,3,H,W] in 0..1
    for step in probe_steps:
        if step not in probes:
            continue
        lat = probes[step]  # [S,4,latent,latent]
        imgs = model.vae.decode(lat / 0.18215).sample
        imgs = torch.clamp(imgs, -1.0, 1.0)
        imgs01 = (imgs + 1.0) / 2.0
        step_imgs[int(step)] = imgs01.detach().cpu()

        # save per-step image too (quick grid, for debugging)
        # Always save per-step as a horizontal strip: 1 row x S columns (your request).
        row_grid = vutils.make_grid(imgs01, nrow=S, padding=2)  # [3,H,W]
        vutils.save_image(row_grid, os.path.join(out_dir, f"{prefix}_x0_step{step}.png"))

        # label text for this row (include GT errors if available)
        gt_txt = ""
        if step_gt is not None and int(step) in step_gt:
            g = step_gt[int(step)]
            def _fmt(k: str) -> str:
                v = g.get(k, None)
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    return "NA"
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return "NA"
            # Requested: gt_lpips gt_dreamsim gt_psnr gt_ssim
            gt_txt = f" | GT_LPIPS={_fmt('lpips_gt')} GT_DreamSim={_fmt('dreamsim_gt')} GT_PSNR={_fmt('psnr_gt')} GT_SSIM={_fmt('ssim_gt')}"
        step_labels.append(f"probe_step={int(step)}{gt_txt}")

    if not step_imgs:
        return

    # --- Matplotlib summary figure (prettier than raw tensor concatenation) ---
    def _to_img(x: torch.Tensor) -> np.ndarray:
        # x: [3,H,W] in 0..1
        return np.transpose(x.detach().cpu().clamp(0, 1).numpy(), (1, 2, 0))

    context_size = int(obs_den.shape[0])
    steps_in_order = [s for s in probe_steps if int(s) in step_imgs]
    layout = str(summary_layout).lower().strip()

    if layout not in ("vertical", "horizontal"):
        raise ValueError(f"summary_layout must be 'vertical' or 'horizontal', got: {summary_layout!r}")

    if layout == "vertical":
        # --- Original layout (rows = steps, cols = seeds; first row is context+GT) ---
        ncols = max(int(num_seeds), context_size + 1)
        nrows = 1 + len(steps_in_order)

        # Use a dedicated left "label column" for nicer alignment unless no-text mode.
        has_label_col = not bool(summary_no_text)
        total_cols = (ncols + 1) if has_label_col else ncols

        fig_w = max(10.0, 2.0 * total_cols)
        fig_h = max(6.0, 2.0 * nrows)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=180)
        gs = fig.add_gridspec(
            nrows=nrows,
            ncols=total_cols,
            width_ratios=([1.9] + [1.0] * ncols) if has_label_col else ([1.0] * ncols),
            wspace=0.01 if summary_compact else 0.03,
            hspace=0.01 if summary_compact else 0.06,
            left=0.005 if summary_compact else 0.03,
            right=0.995,
            top=0.995,
            bottom=0.005 if summary_compact else 0.03,
        )

        def _clean_ax(ax):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.axis("off")

        # Row 0 label cell
        if has_label_col:
            ax_lbl0 = fig.add_subplot(gs[0, 0])
            _clean_ax(ax_lbl0)
            if not summary_no_text:
                ax_lbl0.text(
                    0.0,
                    0.5,
                    "context + ground truth",
                    ha="left",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    clip_on=True,
                )

        # Row 0 images: context_1..context_T + ground_truth (and blanks if extra)
        for c in range(ncols):
            ax = fig.add_subplot(gs[0, c + (1 if has_label_col else 0)])
            _clean_ax(ax)
            if c < context_size:
                ax.imshow(_to_img(obs_den[c]))
            elif c == context_size:
                ax.imshow(_to_img(gt_den))
            else:
                ax.imshow(np.zeros_like(_to_img(gt_den)))

        # Subsequent rows: label cell + seed images
        for i, step in enumerate(steps_in_order):
            r = i + 1

            # label cell
            if has_label_col:
                ax_lbl = fig.add_subplot(gs[r, 0])
                _clean_ax(ax_lbl)
                if not summary_no_text:
                    lbl = step_labels[i] if i < len(step_labels) else f"probe_step={int(step)}"
                    # split long label into multiple lines for readability
                    lbl = lbl.replace(" | ", "\n")
                    lbl = lbl.replace(" GT_", "\nGT_")
                    ax_lbl.text(0.0, 0.5, lbl, ha="left", va="center", fontsize=10, clip_on=True)

            # images
            imgs01 = step_imgs[int(step)]  # [S,3,H,W]
            for c in range(ncols):
                ax = fig.add_subplot(gs[r, c + (1 if has_label_col else 0)])
                _clean_ax(ax)
                if c < imgs01.shape[0]:
                    ax.imshow(_to_img(imgs01[c]))
                else:
                    ax.imshow(np.zeros_like(_to_img(gt_den)))

    else:
        # --- New horizontal layout (transpose): columns = old rows (context+GT, each probe_step) ---
        # This matches your request: original 1st row -> 1st column, 2nd row -> 2nd column, ...
        # rows are "slots": max(S seeds, T context frames + GT)
        nrows_img = max(int(num_seeds), context_size + 1)
        ncols_img = 1 + len(steps_in_order)  # context+GT + each probe_step

        # If no-text, drop the label row entirely for a tighter collage.
        has_label_row = not bool(summary_no_text)
        nrows_total = nrows_img + (1 if has_label_row else 0)

        fig_w = max(10.0, 2.0 * ncols_img)
        fig_h = max(6.0, 2.0 * nrows_total)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=180)
        gs = fig.add_gridspec(
            nrows=nrows_total,
            ncols=ncols_img,
            # Give the label row more room; also reduce overlap risk with clipping below.
            height_ratios=([0.6] + [1.0] * nrows_img) if has_label_row else ([1.0] * nrows_img),
            wspace=0.01 if summary_compact else 0.03,
            hspace=0.01 if summary_compact else 0.08,
            left=0.005 if summary_compact else 0.03,
            right=0.995,
            top=0.995,
            bottom=0.005 if summary_compact else 0.03,
        )

        def _clean_ax(ax):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.axis("off")

        # Column labels (top row) - skipped in no-text mode.
        row_offset = 1 if has_label_row else 0
        if has_label_row:
            col_labels = ["context + ground truth"] + [step_labels[i] if i < len(step_labels) else f"probe_step={int(s)}" for i, s in enumerate(steps_in_order)]
            for c, lbl in enumerate(col_labels):
                ax = fig.add_subplot(gs[0, c])
                _clean_ax(ax)
                lbl = str(lbl).replace(" | ", "\n").replace(" GT_", "\nGT_")
                ax.text(
                    0.5,
                    0.5,
                    lbl,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold" if c == 0 else "normal",
                    clip_on=True,
                    wrap=True,
                )

        # Column 0: context frames (rows 0..T-1) + GT at row T
        for r in range(nrows_img):
            ax = fig.add_subplot(gs[r + row_offset, 0])
            _clean_ax(ax)
            if r < context_size:
                ax.imshow(_to_img(obs_den[r]))
            elif r == context_size:
                ax.imshow(_to_img(gt_den))
            else:
                ax.imshow(np.zeros_like(_to_img(gt_den)))

        # Other columns: each probe_step, rows are seeds (0..S-1)
        for c, step in enumerate(steps_in_order, start=1):
            imgs01 = step_imgs[int(step)]  # [S,3,H,W]
            for r in range(nrows_img):
                ax = fig.add_subplot(gs[r + row_offset, c])
                _clean_ax(ax)
                if r < imgs01.shape[0]:
                    ax.imshow(_to_img(imgs01[r]))
                else:
                    ax.imshow(np.zeros_like(_to_img(gt_den)))

    # Save two copies to avoid IDE image caching issues
    out_path = os.path.join(out_dir, f"{prefix}_summary.png")
    out_path_v2 = os.path.join(out_dir, f"{prefix}_summary_v2.png")
    pad = 0.0 if summary_compact else 0.05
    fig.savefig(out_path, bbox_inches="tight", pad_inches=pad, facecolor="white")
    fig.savefig(out_path_v2, bbox_inches="tight", pad_inches=pad, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Render max/min extreme samples per condition and per variance metric (final step selection).")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="/home/payneli/data/sd-vae-ft-ema")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--go_eval_type", type=str, default="full", choices=["rollout", "time", "full"])

    parser.add_argument("--select_step", type=int, default=50, help="Select max/min at this probe_step (typically final).")
    parser.add_argument("--probe_steps", type=str, default="2,5,10,20,30,40,50", help="Steps to render rows for.")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=1234)
    parser.add_argument(
        "--summary_layout",
        type=str,
        default="horizontal",
        choices=["horizontal", "vertical"],
        help="Layout for x0_summary*.png. 'horizontal' transposes the old layout (rows->cols).",
    )
    parser.add_argument("--summary_no_text", action="store_true", help="Remove ALL text from x0_summary*.png (labels/titles/GT metrics).")
    parser.add_argument("--summary_compact", action="store_true", help="Make x0_summary*.png more compact (smaller gaps/margins, no padding).")

    parser.add_argument("--deterministic_per_seed", action="store_true", help="Use noise_sequence like the experiment for full determinism.")
    parser.add_argument("--deterministic_perturb", action="store_true", help="Deterministic perturbations per sample.")

    parser.add_argument(
        "--conditions",
        type=str,
        default="auto",
        help="Comma list of dataset:perturbation:severity. Use 'auto' to discover from all_concat.csv at select_step.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="pixel_var,latent_var,lpips,dreamsim",
        help="Comma list of variance metrics to select extremes from.",
    )
    parser.add_argument("--extremes", type=str, default="max,min", help="Comma list: max,min")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of (condition,metric,extreme) cases to render (for smoke).")
    args = parser.parse_args()

    df = load_all_concat(args.out_root, args.run_id, args.tag)
    probe_steps = parse_int_list(args.probe_steps)
    if int(args.select_step) not in probe_steps:
        probe_steps = sorted(set(probe_steps + [int(args.select_step)]))

    # conditions to process
    conds: list[Condition] = []
    if args.conditions.strip().lower() == "auto":
        sub = df[df["probe_step"].astype(int) == int(args.select_step)]
        keys = sub[["dataset", "perturbation", "severity"]].drop_duplicates().to_dict("records")
        for r in keys:
            conds.append(Condition(dataset=str(r["dataset"]), perturbation=str(r["perturbation"]), severity=int(r["severity"])))
        # stable order
        conds.sort(key=lambda c: (c.dataset, c.perturbation, int(c.severity)))
    else:
        for part in args.conditions.split(","):
            part = part.strip()
            if not part:
                continue
            ds, pt, sv = part.split(":")
            conds.append(Condition(dataset=ds, perturbation=pt, severity=int(sv)))

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    extremes = [e.strip() for e in args.extremes.split(",") if e.strip()]

    out_dir = os.path.join(args.out_root, f"{args.run_id}_analysis_{args.tag}", "extremes_finalstep")
    os.makedirs(out_dir, exist_ok=True)

    # Init model + datasets
    model = NWMWrapper(args.model_path, args.vae_path, device=args.device, diffusion_steps=args.diffusion_steps)
    img_size = model.config.get("image_size", 128)
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    ds_recon = get_recon_dataset(split="test", image_size=img_size)
    ds_go = get_go_stanford_dataset(split="test", image_size=img_size, eval_type=args.go_eval_type)

    plan = []
    for cond in conds:
        sub = _cond_filter(df, cond)
        sub = sub[sub["probe_step"].astype(int) == int(args.select_step)]
        if sub.empty:
            continue
        for met in metrics:
            if met not in sub.columns:
                continue
            svals = sub[met].astype(float)
            if svals.isna().all():
                continue
            if "max" in extremes:
                idx = int(svals.idxmax())
                plan.append((cond, met, "max", df.loc[idx]))
            if "min" in extremes:
                idx = int(svals.idxmin())
                plan.append((cond, met, "min", df.loc[idx]))

    # Apply limit
    if int(args.limit) > 0:
        plan = plan[: int(args.limit)]

    # save selection plan
    plan_json = []
    for cond, met, ex, row in plan:
        plan_json.append(
            {
                "condition": cond.key(),
                "metric": met,
                "extreme": ex,
                "select_step": int(args.select_step),
                "sample_idx": int(row["sample_idx"]),
                "value": float(row[met]),
            }
        )
    with open(os.path.join(out_dir, "selected_maxmin.json"), "w") as f:
        json.dump(plan_json, f, indent=2, ensure_ascii=False)

    # Render
    pbar = tqdm(plan, desc="render extremes", dynamic_ncols=True)
    for cond, met, ex, row in pbar:
        sample_idx = int(row["sample_idx"])
        pbar.set_postfix({"cond": cond.key(), "metric": met, "ext": ex, "idx": sample_idx})

        # load sample
        if cond.dataset == "recon":
            item = ds_recon[sample_idx]
        else:
            item = ds_go[sample_idx]

        obs = item[1].unsqueeze(0).to(args.device)   # [1,T,3,H,W]
        # Ground truth next frame (first pred frame)
        gt = item[2][0].unsqueeze(0).to(args.device)  # [1,3,H,W]
        delta = item[-1].unsqueeze(0).to(args.device)  # [1,len,3]
        action = delta[:, 0, :]  # [1,3]

        # apply perturbation for this condition
        obs_in = apply_condition_perturbation(
            obs,
            cond=cond,
            device=args.device,
            base_seed=int(args.base_seed),
            sample_idx=sample_idx,
            deterministic_perturb=bool(args.deterministic_perturb),
        )

        # folder layout: metric/{condition}/{max|min}_idx{}/
        metric_dir = os.path.join(out_dir, f"metric={met}")
        cond_dir = os.path.join(metric_dir, f"cond={cond.key()}")
        sample_dir = os.path.join(cond_dir, f"{ex}_idx{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)

        meta = {
            "dataset": cond.dataset,
            "perturbation": cond.perturbation,
            "severity": int(cond.severity),
            "metric": met,
            "extreme": ex,
            "select_step": int(args.select_step),
            "value": float(row[met]),
            "sample_idx": sample_idx,
        }
        with open(os.path.join(sample_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # Save action
        with open(os.path.join(sample_dir, "action.txt"), "w") as f:
            f.write("action_delta0: " + " ".join([f"{x:.6f}" for x in action.squeeze(0).detach().cpu().float().tolist()]) + "\n")

        # Save context (raw and perturbed) as grids
        obs_den = denormalize(obs.squeeze(0)).clamp(0, 1)
        obs_in_den = denormalize(obs_in.squeeze(0)).clamp(0, 1)
        vutils.save_image(vutils.make_grid(obs_den, nrow=obs_den.shape[0], padding=2), os.path.join(sample_dir, "context_raw.png"))
        vutils.save_image(vutils.make_grid(obs_in_den, nrow=obs_in_den.shape[0], padding=2), os.path.join(sample_dir, "context_input.png"))

        # Gather per-step GT errors from all_concat for this exact condition + sample_idx.
        step_gt = {}
        for st in probe_steps:
            mrow = df[
                (df["dataset"].astype(str) == cond.dataset)
                & (df["perturbation"].astype(str) == cond.perturbation)
                & (df["severity"].astype(int) == int(cond.severity))
                & (df["sample_idx"].astype(int) == int(sample_idx))
                & (df["probe_step"].astype(int) == int(st))
            ]
            if len(mrow) == 0:
                continue
            r0 = mrow.iloc[0].to_dict()
            step_gt[int(st)] = {
                "lpips_gt": r0.get("lpips_gt", None),
                "dreamsim_gt": r0.get("dreamsim_gt", None),
                "psnr_gt": r0.get("psnr_gt", None),
                "ssim_gt": r0.get("ssim_gt", None),
            }

        render_one_sample_grid(
            model=model,
            obs=obs_in,
            gt=gt,
            action=action,
            probe_steps=probe_steps,
            num_seeds=int(args.num_seeds),
            base_seed=int(args.base_seed),
            sample_idx=sample_idx,
            out_dir=sample_dir,
            prefix="x0",
            deterministic_per_seed=bool(args.deterministic_per_seed),
            summary_layout=str(args.summary_layout),
            summary_no_text=bool(args.summary_no_text),
            summary_compact=bool(args.summary_compact),
            step_gt=step_gt,
        )

    print(f"Saved extremes to: {out_dir}")


if __name__ == "__main__":
    main()


