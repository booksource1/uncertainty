import argparse
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def main():
    parser = argparse.ArgumentParser("Analyze early-exit experiment results")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    out_dir = os.path.abspath(args.out_dir or os.path.join(os.path.dirname(csv_path), "analysis"))
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid")

    # Convergence plot: metric vs probe_step, lines per scenario
    metrics = ["latent_var", "pixel_var", "lpips", "dreamsim"]
    for m in metrics:
        if m not in df.columns:
            continue
        # skip empty perceptual fields if not computed
        if df[m].abs().sum() == 0:
            continue
        plt.figure(figsize=(9, 5))
        agg = df.groupby(["scenario", "probe_step"])[m].agg(["mean", "std", "count"]).reset_index()
        agg["sem"] = agg["std"] / np.sqrt(np.maximum(agg["count"], 1))
        for sc, sub in agg.groupby("scenario"):
            sub = sub.sort_values("probe_step")
            plt.plot(sub["probe_step"], sub["mean"], marker="o", label=sc)
            plt.fill_between(sub["probe_step"], sub["mean"] - sub["sem"], sub["mean"] + sub["sem"], alpha=0.2)
        plt.title(f"{m}: convergence vs diffusion steps")
        plt.xlabel("probe_step (in spaced diffusion)")
        plt.ylabel(m)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"convergence_{m}.png"), dpi=200)
        plt.close()

    # Efficiency trade-off: AUROC vs probe_step (ID vs OOD)
    # Here we define ID = A_ID; OOD = B_noise5/B_blur5/C_semantic.
    id_df = df[df["scenario"] == "A_ID"]
    ood_df = df[df["scenario"] != "A_ID"]
    for m in ["latent_var", "pixel_var", "lpips", "dreamsim"]:
        if m not in df.columns or df[m].abs().sum() == 0:
            continue
        rows = []
        for step in sorted(df["probe_step"].unique()):
            a = id_df[id_df["probe_step"] == step][m].dropna().to_numpy()
            b = ood_df[ood_df["probe_step"] == step][m].dropna().to_numpy()
            y = np.concatenate([np.zeros_like(a), np.ones_like(b)])
            s = np.concatenate([a, b])
            rows.append({"metric": m, "probe_step": int(step), "auroc_id_vs_all_ood": auroc(y, s)})
        auc_df = pd.DataFrame(rows)
        auc_df.to_csv(os.path.join(out_dir, f"auroc_vs_step_{m}.csv"), index=False)
        plt.figure(figsize=(8, 4))
        plt.plot(auc_df["probe_step"], auc_df["auroc_id_vs_all_ood"], marker="o")
        plt.ylim(0.0, 1.0)
        plt.title(f"AUROC (ID vs all OOD) vs probe_step — {m}")
        plt.xlabel("probe_step")
        plt.ylabel("AUROC")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"auroc_vs_step_{m}.png"), dpi=200)
        plt.close()

    df.to_csv(os.path.join(out_dir, "early_exit_results_copy.csv"), index=False)
    print(f"Saved analysis to: {out_dir}")


if __name__ == "__main__":
    main()




