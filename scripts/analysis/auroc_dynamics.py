from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common_io import ensure_out_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser("Discriminability dynamics: AUROC vs diffusion/probe step (reverse x-axis)")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--metric", type=str, default="auroc", choices=["auroc", "aupr", "fpr95"])
    ap.add_argument("--steps", type=str, default="", help="Comma list; default uses steps present")
    ap.add_argument(
        "--methods",
        type=str,
        default="Pixel_Var,Latent_Var,LPIPS_Var,DreamSim_Var",
        help="Comma list from big_table_long.csv 'method' column",
    )
    ap.add_argument("--invert_x", action="store_true", help="If set, show x-axis as 50→2 (reverse). Default is forward.")
    ap.add_argument("--grid", action="store_true", help="If set, draw all methods into one multi-panel figure (2x2).")
    ap.add_argument("--title", type=str, default="", help="Optional title override")
    args = ap.parse_args()

    analysis_root = os.path.join(args.out_root, f"{args.run_id}_analysis_{args.tag}")
    big_long = os.path.join(analysis_root, "big_table", "big_table_long.csv")
    if not os.path.exists(big_long):
        raise FileNotFoundError(
            f"Missing {big_long}. Please run big_table.py first to generate big_table_long.csv."
        )
    df = pd.read_csv(big_long)
    if args.metric not in df.columns:
        raise ValueError(f"metric {args.metric} not found in {big_long}")

    if args.steps.strip():
        steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    else:
        steps = sorted({int(x) for x in df["probe_step"].unique()})

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    out_dir = ensure_out_dir(args.out_root, args.run_id, args.tag, "auroc_dynamics")

    # Determine OOD rows present (exclude ID row; big_table_long already excludes ID)
    oods = sorted({str(x) for x in df["ood"].unique()})

    # Save the raw/aggregated table used for plotting
    plot_df = df[
        (df["method"].astype(str).isin(methods))
        & (df["ood"].astype(str).isin(oods))
        & (df["probe_step"].astype(int).isin(steps))
    ].copy()
    plot_df["probe_step"] = plot_df["probe_step"].astype(int)
    plot_df["method"] = plot_df["method"].astype(str)
    plot_df["ood"] = plot_df["ood"].astype(str)

    # Aggregate defensively (in case multiple shards/files introduce duplicates)
    plot_df = (
        plot_df.groupby(["method", "ood", "probe_step"], as_index=False)[args.metric]
        .mean(numeric_only=True)
        .sort_values(["method", "ood", "probe_step"])
    )

    long_csv = os.path.join(out_dir, f"{args.metric}_vs_step_long.csv")
    plot_df.to_csv(long_csv, index=False)

    # Wide view for quick inspection: index=(ood, probe_step), columns=method
    wide = plot_df.pivot_table(index=["ood", "probe_step"], columns="method", values=args.metric, aggfunc="mean").reset_index()
    wide_csv = os.path.join(out_dir, f"{args.metric}_vs_step_wide.csv")
    wide.to_csv(wide_csv, index=False)
    print(f"Saved plot data: {long_csv}")
    print(f"Saved plot data: {wide_csv}")

    def _plot_one(ax, method: str):
        subm = plot_df[plot_df["method"].astype(str) == method].copy()
        if subm.empty:
            ax.set_title(f"{method} (no data)")
            ax.axis("off")
            return
        for ood in oods:
            s = subm[subm["ood"].astype(str) == ood].copy()
            if s.empty:
                continue
            g = s.sort_values("probe_step")
            if g.empty:
                continue
            ax.plot(
                g["probe_step"].astype(int),
                g[args.metric].astype(float),
                marker="o",
                linewidth=2,
                label=ood,
            )
        ax.set_title(method)
        ax.set_xlabel("probe_step")
        ax.set_ylabel(args.metric.upper())
        ax.grid(alpha=0.25)
        if args.invert_x:
            ax.invert_xaxis()

    if args.grid:
        # 2x2 grid for up to 4 methods; if more, extend rows.
        n = len(methods)
        ncols = 2
        nrows = int((n + ncols - 1) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
        if args.title.strip():
            fig.suptitle(args.title.strip(), fontsize=14)
        else:
            fig.suptitle(f"{args.metric.upper()} vs Probe Step", fontsize=14)

        for i, method in enumerate(methods):
            r, c = divmod(i, ncols)
            _plot_one(axes[r][c], method)

        # hide unused
        for j in range(n, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")

        # single legend (take from first non-empty axis)
        handles, labels = [], []
        for ax in axes.ravel():
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        out_path = os.path.join(out_dir, f"{args.metric}_vs_step_grid.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_path}", flush=True)
        return

    # per-method figures
    for method in methods:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        _plot_one(ax, method)
        if args.title.strip():
            plt.title(args.title.strip())
        else:
            plt.title(f"{args.metric.upper()} vs Probe Step — {method}")
        plt.xlabel("Probe step (denoise iteration)")
        plt.ylabel(args.metric.upper())
        plt.legend()
        plt.tight_layout()
        fname = f"{args.metric}_vs_step_{method}.png".replace("/", "_")
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()


