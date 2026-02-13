from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common_io import InputSpec, default_input_glob, ensure_out_dir, load_sharded_csvs
from common_metrics import average_precision, fpr_at_tpr, roc_auc


@dataclass(frozen=True)
class OODRow:
    name: str
    dataset: str
    perturbation: str
    severity: int


METHOD_TO_COL = {
    "Pixel_Var": "pixel_var",
    "Latent_Var": "latent_var",
    "LPIPS_Var": "lpips",
    "DreamSim_Var": "dreamsim",
}


def _subset(df: pd.DataFrame, *, dataset: str, perturbation: str, severity: int) -> pd.DataFrame:
    return df[
        (df["dataset"].astype(str) == str(dataset))
        & (df["perturbation"].astype(str) == str(perturbation))
        & (df["severity"].astype(int) == int(severity))
    ].copy()


def _compute_metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
    return {
        "auroc": roc_auc(y_true, score),
        "aupr": average_precision(y_true, score),
        "fpr95": fpr_at_tpr(y_true, score, tpr_target=0.95),
    }


def main():
    ap = argparse.ArgumentParser("Big Table: AUROC/AUPR/FPR95 for OOD detection across steps & methods")
    ap.add_argument("--out_root", type=str, required=True, help="results root folder")
    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--input_glob", type=str, default="", help="Optional override glob for early_exit_generic_shard*.csv")
    ap.add_argument("--probe_steps", type=str, default="", help="Comma list; default uses all steps present")
    ap.add_argument("--out_subdir", type=str, default="big_table", help="Subdir under {run_id}_analysis_{tag}/")
    args = ap.parse_args()

    spec = InputSpec(out_root=args.out_root, run_id=args.run_id, tag=args.tag)
    input_glob = args.input_glob.strip() or default_input_glob(spec)
    df = load_sharded_csvs(input_glob)

    # Always save a raw concatenated copy for downstream analyses.
    analysis_root = os.path.join(args.out_root, f"{args.run_id}_analysis_{args.tag}")
    os.makedirs(analysis_root, exist_ok=True)
    all_concat_path = os.path.join(analysis_root, "all_concat.csv")
    df.to_csv(all_concat_path, index=False)

    # Determine steps
    if args.probe_steps.strip():
        steps = [int(x.strip()) for x in args.probe_steps.split(",") if x.strip()]
    else:
        steps = sorted({int(x) for x in df["probe_step"].unique()})

    # ID definition: recon + none + severity 0
    id_df_all = _subset(df, dataset="recon", perturbation="none", severity=0)
    if id_df_all.empty:
        raise ValueError("No ID rows found for dataset=recon perturbation=none severity=0")

    # OOD rows: noise-k, blur-k from recon; stanford from go_stanford
    ood_rows: list[OODRow] = []
    # recon perturbations (auto-discover all, excluding ID's none/0)
    recon = df[df["dataset"].astype(str) == "recon"].copy()
    perts = sorted({str(x) for x in recon["perturbation"].unique()})
    for pert in perts:
        if str(pert).lower() == "none":
            continue
        sub = recon[recon["perturbation"].astype(str) == str(pert)]
        for sev in sorted({int(x) for x in sub["severity"].unique()}):
            # Keep a stable, readable label for legends/tables.
            name = f"{str(pert).capitalize()}-{int(sev)}"
            ood_rows.append(OODRow(name=name, dataset="recon", perturbation=str(pert), severity=int(sev)))
    # stanford
    if (df["dataset"].astype(str) == "go_stanford").any():
        # go_stanford is expected to have perturbation=none severity=0
        ood_rows.append(OODRow(name="Stanford", dataset="go_stanford", perturbation="none", severity=0))

    out_dir = ensure_out_dir(args.out_root, args.run_id, args.tag, args.out_subdir)

    long_rows = []
    for row in ood_rows:
        ood_df_all = _subset(df, dataset=row.dataset, perturbation=row.perturbation, severity=row.severity)
        if ood_df_all.empty:
            continue

        for step in steps:
            id_df = id_df_all[id_df_all["probe_step"].astype(int) == int(step)]
            ood_df = ood_df_all[ood_df_all["probe_step"].astype(int) == int(step)]
            if id_df.empty or ood_df.empty:
                continue

            # Always use full data (no balancing / truncation).
            id_use, ood_use = id_df, ood_df

            for method, col in METHOD_TO_COL.items():
                if col not in df.columns:
                    continue
                xs = np.concatenate([id_use[col].to_numpy(), ood_use[col].to_numpy()])
                ys = np.concatenate([np.zeros(len(id_use), dtype=np.int64), np.ones(len(ood_use), dtype=np.int64)])
                m = _compute_metrics(ys, xs)
                long_rows.append(
                    {
                        "ood": row.name,
                        "dataset": row.dataset,
                        "perturbation": row.perturbation,
                        "severity": int(row.severity),
                        "probe_step": int(step),
                        "method": method,
                        "score_col": col,
                        "n_id": int(len(id_use)),
                        "n_ood": int(len(ood_use)),
                        **m,
                    }
                )

    long_df = pd.DataFrame(long_rows)
    long_path = os.path.join(out_dir, "big_table_long.csv")
    long_df.to_csv(long_path, index=False)

    # Wide tables: one file per metric
    for metric in ["auroc", "aupr", "fpr95"]:
        wide = (
            long_df.pivot_table(
                index="ood",
                columns=["probe_step", "method"],
                values=metric,
                aggfunc="mean",
            )
            .sort_index()
        )
        wide.columns = [f"{m}@step{int(s)}" for (s, m) in wide.columns.to_list()]
        wide_path = os.path.join(out_dir, f"big_table_{metric}.csv")
        wide.to_csv(wide_path)

    # Also produce a compact AUROC-only "paper view" at final step (max step)
    if not long_df.empty:
        final_step = int(max(steps))
        paper = long_df[long_df["probe_step"] == final_step].pivot_table(index="ood", columns="method", values="auroc", aggfunc="mean")
        paper = paper.reindex(columns=list(METHOD_TO_COL.keys()))
        paper_path = os.path.join(out_dir, f"big_table_auroc_step{final_step}.csv")
        paper.to_csv(paper_path)

    print(f"Saved Big Table outputs to: {out_dir}")
    print(f"- all_concat.csv: {all_concat_path}")
    print(f"- {os.path.basename(long_path)}")
    print(f"- big_table_auroc.csv / big_table_aupr.csv / big_table_fpr95.csv")
    if not long_df.empty:
        print(f"- big_table_auroc_step{int(max(steps))}.csv")


if __name__ == "__main__":
    main()


