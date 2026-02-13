import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Fast AUROC for binary labels without sklearn.
    Uses the Mann–Whitney U / rank statistic formulation.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    assert y_true.ndim == 1 and y_score.ndim == 1 and y_true.shape[0] == y_score.shape[0]
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # handle ties by average rank
    uniq, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for k, c in enumerate(counts):
            if c <= 1:
                continue
            idx = np.where(inv == k)[0]
            ranks[idx] = ranks[idx].mean()

    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


@dataclass
class RunSpec:
    name: str
    label: str  # for grouping
    sev: int


def find_csv(out_root: str, run_id: str, label: str, sev: int, tag: str) -> str:
    pat = os.path.join(out_root, f"{run_id}_{label}_{sev}_{tag}", "shard_0", "early_exit_generic_shard0.csv")
    hits = glob.glob(pat)
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly 1 csv at: {pat} (found {len(hits)})")
    return hits[0]


def main():
    ap = argparse.ArgumentParser("Analyze small320 4-gpu run (no sklearn)")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    args = ap.parse_args()

    specs = [
        RunSpec(name="recon_none", label="none", sev=0),
        RunSpec(name="recon_noise5", label="noise", sev=5),
        RunSpec(name="recon_blur5", label="blur", sev=5),
        RunSpec(name="go_stanford", label="go_stanford", sev=0),
    ]

    frames = []
    for s in specs:
        csv_path = find_csv(args.out_root, args.run_id, s.label, s.sev, args.tag)
        df = pd.read_csv(csv_path)
        df["condition"] = s.name
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    analysis_dir = os.path.join(args.out_root, f"{args.run_id}_analysis_{args.tag}")
    os.makedirs(analysis_dir, exist_ok=True)

    # Save raw concatenated
    all_csv = os.path.join(analysis_dir, "all_concat.csv")
    all_df.to_csv(all_csv, index=False)

    # Aggregate means per condition/probe_step
    metric_cols = [
        "latent_var",
        "pixel_var",
        "lpips",
        "dreamsim",
        "lpips_gt",
        "dreamsim_gt",
        "mse_gt",
        "psnr_gt",
        "ssim_gt",
    ]
    metric_cols = [c for c in metric_cols if c in all_df.columns]
    agg = (
        all_df.groupby(["condition", "probe_step"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["condition", "probe_step"])
    )
    agg_csv = os.path.join(analysis_dir, "means_by_condition_probe.csv")
    agg.to_csv(agg_csv, index=False)

    # AUROC: ID=recon_none vs each OOD, computed per probe_step for key scores
    id_df = all_df[all_df["condition"] == "recon_none"].copy()
    ood_conditions = ["recon_noise5", "recon_blur5", "go_stanford"]
    score_cols = ["latent_var", "pixel_var", "lpips", "dreamsim", "mse_gt", "psnr_gt", "ssim_gt"]
    score_cols = [c for c in score_cols if c in all_df.columns]

    rows = []
    for ood in ood_conditions:
        ood_df = all_df[all_df["condition"] == ood].copy()
        for step in sorted(all_df["probe_step"].unique()):
            a = id_df[id_df["probe_step"] == step]
            b = ood_df[ood_df["probe_step"] == step]
            # If ID and OOD come from the same dataset (e.g., recon vs recon_noise/blur),
            # align by sample_idx for a paired comparison. If they are from different datasets
            # (e.g., recon vs go_stanford), do NOT intersect ids (would collapse N).
            same_dataset = False
            try:
                same_dataset = (str(a["dataset"].iloc[0]) == str(b["dataset"].iloc[0]))
            except Exception:
                same_dataset = False

            if same_dataset:
                common = np.intersect1d(a["sample_idx"].to_numpy(), b["sample_idx"].to_numpy())
                if common.size == 0:
                    continue
                a2 = a[a["sample_idx"].isin(common)]
                b2 = b[b["sample_idx"].isin(common)]
                n_eff = int(common.size)
            else:
                a2 = a
                b2 = b
                n_eff = int(min(len(a2), len(b2)))
            for sc in score_cols:
                xs = np.concatenate([a2[sc].to_numpy(), b2[sc].to_numpy()])
                ys = np.concatenate([np.zeros(len(a2), dtype=np.int64), np.ones(len(b2), dtype=np.int64)])
                # For PSNR/SSIM higher is "more ID-like", invert so higher score => more OOD
                score = xs
                if sc in ["psnr_gt", "ssim_gt"]:
                    score = -xs
                auc = roc_auc_binary(ys, score)
                rows.append({"ood": ood, "probe_step": int(step), "score": sc, "auroc": float(auc), "n": n_eff, "paired": bool(same_dataset)})

    auc_df = pd.DataFrame(rows).sort_values(["ood", "score", "probe_step"])
    auc_csv = os.path.join(analysis_dir, "auroc_by_probe.csv")
    auc_df.to_csv(auc_csv, index=False)

    # Minimal console summary at final step
    final_step = int(max(all_df["probe_step"].unique()))
    summary_lines = []
    summary_lines.append(f"Saved analysis to: {analysis_dir}")
    summary_lines.append(f"- all_concat.csv: {all_csv}")
    summary_lines.append(f"- means_by_condition_probe.csv: {agg_csv}")
    summary_lines.append(f"- auroc_by_probe.csv: {auc_csv}")
    if not auc_df.empty:
        best = auc_df[auc_df["probe_step"] == final_step].sort_values("auroc", ascending=False).head(12)
        summary_lines.append(f"Top AUROC @ probe_step={final_step}:")
        for _, r in best.iterrows():
            summary_lines.append(f"  ood={r['ood']} score={r['score']} auroc={r['auroc']:.4f} n={int(r['n'])}")
    print("\n".join(summary_lines), flush=True)


if __name__ == "__main__":
    main()


