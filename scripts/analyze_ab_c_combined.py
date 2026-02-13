import argparse
import os
from glob import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def _load_shards(root: str) -> pd.DataFrame:
    csvs = sorted(glob(os.path.join(root, "shard_*", "final_results.csv")))
    if not csvs:
        raise FileNotFoundError(f"No shard_*/final_results.csv under: {root}")
    dfs = []
    for p in csvs:
        df = pd.read_csv(p)
        df["source_root"] = os.path.basename(root.rstrip("/"))
        df["source_csv"] = os.path.relpath(p, root)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _ensure_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    if "dataset" not in df.columns:
        df["dataset"] = dataset_name
    # normalize dtypes
    df["severity"] = df["severity"].astype(int)
    for c in ["pixel_variance", "lpips", "dreamsim"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute AUROC without sklearn (rank-based; equivalent to Mann–Whitney U / Wilcoxon rank-sum).
    y_true: 0/1 labels
    scores: higher => more likely positive
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if y_true.ndim != 1 or scores.ndim != 1 or y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must be 1D arrays of same length")

    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # ranks of scores (average ranks for ties)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # tie correction: average ranks for equal scores
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
    parser = argparse.ArgumentParser("Combine A/B (Recon) and C (go_stanford) results and analyze")
    parser.add_argument("--recon_root", type=str, required=True)
    parser.add_argument("--go_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    recon_root = os.path.abspath(args.recon_root)
    go_root = os.path.abspath(args.go_root)
    out_dir = os.path.abspath(
        args.out_dir
        or os.path.join(
            "/home/payneli/project/nwm/uncertainty_research/results",
            f"analysis_combined_{os.path.basename(recon_root)}__{os.path.basename(go_root)}",
        )
    )
    os.makedirs(out_dir, exist_ok=True)

    recon = _ensure_columns(_load_shards(recon_root), "recon")
    go = _ensure_columns(_load_shards(go_root), "go_stanford")

    # Harmonize group naming:
    # recon: group in {ID, OOD}
    # go: group == OOD_SEMANTIC (baseline, perturbation none)
    df = pd.concat([recon, go], ignore_index=True)
    df.to_csv(os.path.join(out_dir, "combined_all_results.csv"), index=False)

    # Define subsets
    id_df = df[(df["dataset"] == "recon") & (df["group"] == "ID") & (df["perturbation"] == "none") & (df["severity"] == 0)]
    near_ood = df[(df["dataset"] == "recon") & (df["group"] == "OOD")]
    sem_ood = df[(df["dataset"] == "go_stanford") & (df["group"].str.contains("OOD", na=False))]

    # AUROC: ID vs Semantic OOD
    metrics = ["pixel_variance", "lpips", "dreamsim"]
    rows = []
    for m in metrics:
        a = id_df[m].dropna().to_numpy()
        b = sem_ood[m].dropna().to_numpy()
        y = np.concatenate([np.zeros_like(a), np.ones_like(b)])
        s = np.concatenate([a, b])
        rows.append(
            {
                "task": "ID(recon) vs OOD_SEMANTIC(go_stanford)",
                "metric": m,
                "auroc": auroc(y, s),
                "n_id": int(len(a)),
                "n_ood": int(len(b)),
                "id_mean": float(np.mean(a)) if len(a) else float("nan"),
                "ood_mean": float(np.mean(b)) if len(b) else float("nan"),
            }
        )

    # AUROC: ID vs Near-OOD (by perturbation,severity)
    for m in metrics:
        for (p, sev), sub in near_ood.groupby(["perturbation", "severity"]):
            a = id_df[m].dropna().to_numpy()
            b = sub[m].dropna().to_numpy()
            y = np.concatenate([np.zeros_like(a), np.ones_like(b)])
            s = np.concatenate([a, b])
            rows.append(
                {
                    "task": f"ID(recon) vs OOD_NEAR(recon) {p}_s{sev}",
                    "metric": m,
                    "auroc": auroc(y, s),
                    "n_id": int(len(a)),
                    "n_ood": int(len(b)),
                    "id_mean": float(np.mean(a)) if len(a) else float("nan"),
                    "ood_mean": float(np.mean(b)) if len(b) else float("nan"),
                }
            )

    auroc_df = pd.DataFrame(rows).sort_values(["task", "metric"])
    auroc_df.to_csv(os.path.join(out_dir, "auroc.csv"), index=False)

    # Summary table by dataset/group/perturbation/severity
    summary = (
        df.groupby(["dataset", "group", "perturbation", "severity"])[metrics]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    summary.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(os.path.join(out_dir, "summary_by_condition.csv"), index=False)

    # Plots
    sns.set_theme(style="whitegrid")
    # Plot A/B/C combined distribution (ID vs semantic OOD vs each perturbation pooled)
    for m in metrics:
        plt.figure(figsize=(11, 5))
        plot_df = pd.concat(
            [
                id_df.assign(label="A_ID"),
                sem_ood.assign(label="C_SEMANTIC_OOD"),
                near_ood.assign(label="B_NEAR_OOD"),
            ],
            ignore_index=True,
        )
        order = ["A_ID", "B_NEAR_OOD", "C_SEMANTIC_OOD"]
        sns.boxplot(data=plot_df, x="label", y=m, order=order, showfliers=False)
        plt.title(f"{m}: A(ID) vs B(Near-OOD pooled) vs C(Semantic OOD)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"box_abc_{m}.png"), dpi=200)
        plt.close()

    # Plot C vs A as density
    for m in metrics:
        plt.figure(figsize=(10, 4))
        sns.kdeplot(id_df[m], label="A_ID(recon)", fill=True, alpha=0.3)
        sns.kdeplot(sem_ood[m], label="C_OOD_SEMANTIC(go_stanford)", fill=True, alpha=0.3)
        plt.title(f"{m}: KDE A vs C")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"kde_a_vs_c_{m}.png"), dpi=200)
        plt.close()

    print(f"Saved combined analysis to: {out_dir}")


if __name__ == "__main__":
    main()


