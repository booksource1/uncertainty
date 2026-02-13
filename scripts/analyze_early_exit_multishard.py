import argparse
import os
from glob import glob

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


def load_shards(results_root: str) -> pd.DataFrame:
    paths = sorted(glob(os.path.join(results_root, "shard_*", "early_exit_results_shard*.csv")))
    if not paths:
        raise FileNotFoundError(f"No shard csvs found under: {results_root}")
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        d["source_csv"] = os.path.relpath(p, results_root)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser("Analyze early-exit results across shards (A/B/C)")
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)
    out_dir = os.path.abspath(args.out_dir or os.path.join(results_root, "analysis"))
    os.makedirs(out_dir, exist_ok=True)

    df = load_shards(results_root)
    df.to_csv(os.path.join(out_dir, "combined_all_results.csv"), index=False)

    # sanity: ensure expected scenarios
    scenarios = ["A_ID", "B_noise5", "B_blur5", "C_semantic"]
    df["scenario"] = df["scenario"].astype(str)
    df["probe_step"] = df["probe_step"].astype(int)

    metrics = ["latent_var", "pixel_var", "lpips", "dreamsim"]
    sns.set_theme(style="whitegrid")

    # Convergence plots: mean ± SEM vs probe_step for each scenario
    for m in metrics:
        if m not in df.columns:
            continue
        if float(df[m].fillna(0).abs().sum()) == 0:
            continue
        agg = df.groupby(["scenario", "probe_step"])[m].agg(["mean", "std", "count"]).reset_index()
        agg["sem"] = agg["std"] / np.sqrt(np.maximum(agg["count"], 1))
        plt.figure(figsize=(9, 5))
        for sc in scenarios:
            sub = agg[agg["scenario"] == sc].sort_values("probe_step")
            if len(sub) == 0:
                continue
            plt.plot(sub["probe_step"], sub["mean"], marker="o", label=sc)
            plt.fill_between(sub["probe_step"], sub["mean"] - sub["sem"], sub["mean"] + sub["sem"], alpha=0.2)
        plt.title(f"{m}: uncertainty vs diffusion steps (x0 probes)")
        plt.xlabel("probe_step")
        plt.ylabel(m)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"convergence_{m}.png"), dpi=200)
        plt.close()

    # AUROC vs step:
    # - ID vs Near-OOD (B_noise5+B_blur5)
    # - ID vs Semantic-OOD (C_semantic)
    # - ID vs All-OOD (B + C)
    id_df = df[df["scenario"] == "A_ID"]
    near_df = df[df["scenario"].isin(["B_noise5", "B_blur5"])]
    sem_df = df[df["scenario"] == "C_semantic"]
    all_ood_df = df[df["scenario"] != "A_ID"]

    auc_rows = []
    for m in metrics:
        if m not in df.columns or float(df[m].fillna(0).abs().sum()) == 0:
            continue
        for step in sorted(df["probe_step"].unique()):
            def _auc(pos_df, neg_df, tag):
                a = neg_df[neg_df["probe_step"] == step][m].dropna().to_numpy()
                b = pos_df[pos_df["probe_step"] == step][m].dropna().to_numpy()
                y = np.concatenate([np.zeros_like(a), np.ones_like(b)])
                s = np.concatenate([a, b])
                return {
                    "metric": m,
                    "probe_step": int(step),
                    "task": tag,
                    "auroc": auroc(y, s),
                    "n_id": int(len(a)),
                    "n_ood": int(len(b)),
                    "id_mean": float(np.mean(a)) if len(a) else float("nan"),
                    "ood_mean": float(np.mean(b)) if len(b) else float("nan"),
                }

            auc_rows.append(_auc(near_df, id_df, "ID vs Near-OOD (noise5+blur5)"))
            auc_rows.append(_auc(sem_df, id_df, "ID vs Semantic-OOD (go_stanford)"))
            auc_rows.append(_auc(all_ood_df, id_df, "ID vs All-OOD (B+C)"))

    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(os.path.join(out_dir, "auroc_vs_step.csv"), index=False)

    for m in auc_df["metric"].unique():
        plt.figure(figsize=(9, 4))
        subm = auc_df[auc_df["metric"] == m]
        for task, sub in subm.groupby("task"):
            sub = sub.sort_values("probe_step")
            plt.plot(sub["probe_step"], sub["auroc"], marker="o", label=task)
        plt.ylim(0.0, 1.0)
        plt.title(f"AUROC vs probe_step — {m}")
        plt.xlabel("probe_step")
        plt.ylabel("AUROC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"auroc_vs_step_{m}.png"), dpi=200)
        plt.close()

    # Summary table
    summary = df.groupby(["scenario", "probe_step"])[metrics].agg(["mean", "std", "median", "count"]).reset_index()
    summary.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(os.path.join(out_dir, "summary_by_scenario_step.csv"), index=False)

    # pick extremes in C at final step (for documentation; rendering handled by separate script)
    final_step = int(max(df["probe_step"].unique()))
    cfinal = df[(df["scenario"] == "C_semantic") & (df["probe_step"] == final_step)].copy()
    extreme_rows = []
    for m in metrics:
        if float(cfinal[m].fillna(0).abs().sum()) == 0:
            continue
        row = cfinal.loc[cfinal[m].astype(float).idxmax()]
        extreme_rows.append({"metric": m, "probe_step": final_step, "sample_idx": int(row["sample_idx"]), "value": float(row[m])})
    pd.DataFrame(extreme_rows).to_csv(os.path.join(out_dir, "c_semantic_extremes.csv"), index=False)

    print(f"Saved analysis to: {out_dir}")


if __name__ == "__main__":
    main()




