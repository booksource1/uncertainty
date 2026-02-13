from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common_io import InputSpec, default_input_glob, ensure_out_dir, load_sharded_csvs  # noqa: E402


@dataclass(frozen=True)
class Group:
    key: str
    label: str
    dataset: str
    perturbation: str
    severity: int


def _resolve_df(out_root: str, run_id: str, tag: str, input_glob: str) -> pd.DataFrame:
    analysis_root = os.path.join(out_root, f"{run_id}_analysis_{tag}")
    all_concat = os.path.join(analysis_root, "all_concat.csv")
    if os.path.exists(all_concat):
        return pd.read_csv(all_concat)
    spec = InputSpec(out_root=out_root, run_id=run_id, tag=tag)
    g = input_glob.strip() or default_input_glob(spec)
    return load_sharded_csvs(g)


def _label_for(dataset: str, perturbation: str, severity: int) -> str:
    if dataset == "recon" and perturbation == "none" and int(severity) == 0:
        return "ID"
    if dataset == "go_stanford":
        return "Stanford"
    if perturbation in ["noise", "blur", "blackout", "photometric"]:
        return f"{perturbation.capitalize()}-{int(severity)}"
    return f"{dataset}:{perturbation}-{int(severity)}"


def _discover_groups(df: pd.DataFrame) -> list[Group]:
    keys = (
        df[["dataset", "perturbation", "severity"]]
        .drop_duplicates()
        .assign(severity=lambda x: x["severity"].astype(int))
        .sort_values(["dataset", "perturbation", "severity"])
        .to_dict("records")
    )
    out: list[Group] = []
    for r in keys:
        ds = str(r["dataset"])
        pt = str(r["perturbation"])
        sv = int(r["severity"])
        key = f"{ds}|{pt}|{sv}"
        out.append(Group(key=key, label=_label_for(ds, pt, sv), dataset=ds, perturbation=pt, severity=sv))
    # Put ID first if present
    out.sort(key=lambda g: (0 if g.label == "ID" else 1, g.label))
    return out


def _minmax_norm(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    xmin = float(np.nanmin(x.to_numpy()))
    xmax = float(np.nanmax(x.to_numpy()))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) < eps:
        return x * 0.0
    return (x - xmin) / (xmax - xmin)


DISPLAY_NAME = {
    "lpips": "lpips_var",
    "dreamsim": "dreamsim_var",
    "pixel_var": "pixel_var",
    "latent_var": "latent_var",
}


def _disp(col: str) -> str:
    return DISPLAY_NAME.get(col, col)


def main():
    ap = argparse.ArgumentParser("Raw variance dynamics: normalized score vs probe_step (multi-panel)")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--input_glob", type=str, default="", help="Override glob if not using all_concat.csv")
    ap.add_argument("--steps", type=str, default="", help="Comma list; default uses all steps present")
    ap.add_argument(
        "--metrics",
        type=str,
        default="lpips,pixel_var,latent_var,dreamsim",
        help="Comma list of score columns to plot",
    )
    ap.add_argument("--normalize", type=str, default="minmax", choices=["minmax", "none"], help="Normalization per metric")
    ap.add_argument("--title", type=str, default="", help="Optional title override")
    args = ap.parse_args()

    df = _resolve_df(args.out_root, args.run_id, args.tag, args.input_glob)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Missing metrics columns: {missing}. Available: {list(df.columns)}")

    if args.steps.strip():
        steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    else:
        steps = sorted({int(x) for x in df["probe_step"].unique()})

    groups = _discover_groups(df)

    # Aggregate mean score per group per step
    rows = []
    for g in groups:
        sub = df[
            (df["dataset"].astype(str) == g.dataset)
            & (df["perturbation"].astype(str) == g.perturbation)
            & (df["severity"].astype(int) == int(g.severity))
            & (df["probe_step"].astype(int).isin(steps))
        ]
        if sub.empty:
            continue
        for step in steps:
            ssub = sub[sub["probe_step"].astype(int) == step]
            if ssub.empty:
                continue
            out = {"group": g.label, "dataset": g.dataset, "perturbation": g.perturbation, "severity": int(g.severity), "probe_step": int(step)}
            for m in metrics:
                out[m] = float(ssub[m].astype(float).mean())
            rows.append(out)

    agg = pd.DataFrame(rows)
    if agg.empty:
        raise RuntimeError("No aggregated rows; check input data.")

    # Normalize per metric across ALL groups/steps (so curves are comparable per panel)
    plot_df = agg.copy()
    if args.normalize == "minmax":
        for m in metrics:
            plot_df[m] = _minmax_norm(plot_df[m])

    out_dir = ensure_out_dir(args.out_root, args.run_id, args.tag, "raw_dynamics")
    plot_csv = os.path.join(out_dir, "raw_dynamics_mean.csv")
    plot_df.to_csv(plot_csv, index=False)

    # 2x2 grid for 4 metrics; if more, expand rows
    n = len(metrics)
    ncols = 2
    nrows = int((n + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    if args.title.strip():
        fig.suptitle(args.title.strip(), fontsize=14)
    else:
        fig.suptitle(f"Raw Variance Dynamics (normalized={args.normalize})", fontsize=14)

    # Deterministic color cycle
    color_map = {
        "ID": "black",
        "Noise-5": "green",
        "Blur-5": "purple",
        "Stanford": "red",
    }

    for i, m in enumerate(metrics):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        for grp in sorted(plot_df["group"].unique(), key=lambda x: (0 if x == "ID" else 1, x)):
            s = plot_df[plot_df["group"] == grp].sort_values("probe_step")
            if s.empty:
                continue
            ax.plot(
                s["probe_step"].astype(int),
                s[m].astype(float),
                marker="o",
                linewidth=2,
                label=str(grp),
                color=color_map.get(str(grp), None),
            )
        ax.set_title(_disp(m))
        ax.set_xlabel("probe_step")
        ax.set_ylabel("Normalized variance" if args.normalize != "none" else "Raw variance")
        ax.grid(alpha=0.25)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    # Single legend
    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_png = os.path.join(out_dir, "raw_variance_vs_step_grid.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {plot_csv}")


if __name__ == "__main__":
    main()


