import argparse
import os
from glob import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def load_shards(results_root: str) -> pd.DataFrame:
    shard_csvs = sorted(glob(os.path.join(results_root, "shard_*", "final_results.csv")))
    if not shard_csvs:
        # Support single-process outputs that write directly to <results_root>/final_results.csv
        single = os.path.join(results_root, "final_results.csv")
        if os.path.exists(single):
            df = pd.read_csv(single)
            df["shard_csv"] = "final_results.csv"
            return df
        raise FileNotFoundError(f"No shard CSVs found under: {results_root} (and no final_results.csv found)")
    dfs = []
    for p in shard_csvs:
        df = pd.read_csv(p)
        df["shard_csv"] = os.path.relpath(p, results_root)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    return out


def save_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser("Aggregate + analyze uncertainty_research results")
    parser.add_argument("--results_root", type=str, required=True, help="e.g. uncertainty_research/results_full_recon_YYYYMMDD_HHMMSS")
    parser.add_argument("--out_dir", type=str, default=None, help="output dir (default: <results_root>/analysis)")
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)
    out_dir = os.path.abspath(args.out_dir or os.path.join(results_root, "analysis"))
    os.makedirs(out_dir, exist_ok=True)

    df = load_shards(results_root)

    # Basic cleanup/types
    metric_cols = ["pixel_variance", "lpips", "dreamsim", "lpips_gt", "dreamsim_gt", "mse_gt", "psnr_gt", "ssim_gt"]
    for c in metric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["severity"] = df["severity"].astype(int)

    # Aggregate CSV
    save_table(df, os.path.join(out_dir, "all_results.csv"))

    # Sanity checks
    exp_rows_per_sample = 1 + (df.query("group=='OOD'")[["perturbation", "severity"]].drop_duplicates().shape[0])
    # count per sample_id
    per_sample = df.groupby("sample_idx").size().reset_index(name="rows")
    sanity = {
        "num_rows": int(len(df)),
        "num_unique_samples": int(df["sample_idx"].nunique()),
        "expected_rows_per_sample": int(exp_rows_per_sample),
        "rows_per_sample_min": int(per_sample["rows"].min()),
        "rows_per_sample_max": int(per_sample["rows"].max()),
    }
    with open(os.path.join(out_dir, "sanity.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(sanity, f, ensure_ascii=False, indent=2)

    # Summary stats by condition
    group_cols = ["group", "perturbation", "severity"]
    present_metrics = [c for c in ["pixel_variance", "lpips", "dreamsim", "lpips_gt", "dreamsim_gt", "mse_gt", "psnr_gt", "ssim_gt"] if c in df.columns]
    summary = (
        df.groupby(group_cols)[present_metrics]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    # flatten columns
    summary.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in summary.columns]
    save_table(summary, os.path.join(out_dir, "summary_by_condition.csv"))

    # Severity trend plots (mean +/- std)
    sns.set_theme(style="whitegrid")
    metrics = [m for m in ["pixel_variance", "lpips", "dreamsim", "lpips_gt", "dreamsim_gt", "mse_gt", "psnr_gt", "ssim_gt"] if m in df.columns]
    ood = df[df["group"] == "OOD"].copy()
    id_df = df[df["group"] == "ID"].copy()

    # Plot 1: distribution ID vs each perturbation (severity pooled)
    for m in metrics:
        plt.figure(figsize=(10, 5))
        tmp = df.copy()
        tmp["cond"] = np.where(tmp["group"] == "ID", "ID", tmp["perturbation"] + "_s" + tmp["severity"].astype(str))
        # keep readable: only show top-level by perturbation (pool severities) in this plot
        tmp2 = df.copy()
        tmp2["cond"] = np.where(tmp2["group"] == "ID", "ID", tmp2["perturbation"])
        order = ["ID"] + sorted([x for x in tmp2["cond"].unique() if x != "ID"])
        sns.boxplot(data=tmp2, x="cond", y=m, order=order, showfliers=False)
        plt.title(f"{m}: ID vs OOD (pooled severities)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"box_id_vs_ood_{m}.png"), dpi=200)
        plt.close()

    # Plot 2: severity curves per perturbation (mean with CI via bootstrapping approx by std/sqrt(n))
    for m in metrics:
        plt.figure(figsize=(9, 5))
        agg = (
            ood.groupby(["perturbation", "severity"])[m]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["sem"] = agg["std"] / np.sqrt(np.maximum(agg["count"], 1))
        for p, sub in agg.groupby("perturbation"):
            sub = sub.sort_values("severity")
            plt.plot(sub["severity"], sub["mean"], marker="o", label=p)
            plt.fill_between(sub["severity"], sub["mean"] - sub["sem"], sub["mean"] + sub["sem"], alpha=0.2)
        id_mean = id_df[m].mean()
        plt.axhline(id_mean, color="black", linestyle="--", linewidth=1, label="ID mean")
        plt.title(f"{m}: severity trend (OOD) with ID mean")
        plt.xlabel("severity")
        plt.ylabel(m)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"severity_trend_{m}.png"), dpi=200)
        plt.close()

    # Simple monotonicity check (Spearman) per perturbation
    try:
        from scipy.stats import spearmanr

        rows = []
        for m in metrics:
            for p in sorted(ood["perturbation"].unique()):
                sub = ood[ood["perturbation"] == p]
                rho, pval = spearmanr(sub["severity"], sub[m])
                rows.append({"metric": m, "perturbation": p, "spearman_rho": float(rho), "p_value": float(pval)})
        corr = pd.DataFrame(rows)
        save_table(corr, os.path.join(out_dir, "severity_spearman.csv"))
    except Exception:
        pass

    print(f"Saved analysis to: {out_dir}")


if __name__ == "__main__":
    main()


