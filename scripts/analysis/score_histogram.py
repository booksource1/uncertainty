from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common_io import InputSpec, default_input_glob, ensure_out_dir, load_sharded_csvs  # noqa: E402


@dataclass(frozen=True)
class GroupSpec:
    label: str
    dataset: str
    perturbation: str
    severity: int
    color: str


def _subset(df: pd.DataFrame, *, dataset: str, perturbation: str, severity: int, probe_step: int) -> pd.DataFrame:
    return df[
        (df["dataset"].astype(str) == str(dataset))
        & (df["perturbation"].astype(str) == str(perturbation))
        & (df["severity"].astype(int) == int(severity))
        & (df["probe_step"].astype(int) == int(probe_step))
    ].copy()

def _auto_color(label: str) -> str:
    s = str(label).lower()
    if s == "id" or s.startswith("id "):
        return "black"
    if "stanford" in s:
        return "red"
    if "noise" in s:
        return "green"
    if "blur" in s:
        return "purple"
    if "blackout" in s:
        return "orange"
    if "photo" in s:
        return "brown"
    return "gray"


def _label_for(dataset: str, perturbation: str, severity: int) -> str:
    if dataset == "recon" and perturbation == "none" and int(severity) == 0:
        return "ID"
    if dataset == "go_stanford":
        return "Stanford"
    return f"{str(perturbation).capitalize()}-{int(severity)}"


def _discover_groups(df: pd.DataFrame) -> list[GroupSpec]:
    keys = (
        df[["dataset", "perturbation", "severity"]]
        .drop_duplicates()
        .assign(severity=lambda x: x["severity"].astype(int))
        .sort_values(["dataset", "perturbation", "severity"])
        .to_dict("records")
    )
    groups: list[GroupSpec] = []
    for r in keys:
        ds = str(r["dataset"])
        pt = str(r["perturbation"])
        sv = int(r["severity"])
        # Treat recon/none/0 as ID; ignore any other recon/none/* (if ever present).
        if ds == "recon" and pt == "none" and sv != 0:
            continue
        label = _label_for(ds, pt, sv)
        groups.append(GroupSpec(label=label, dataset=ds, perturbation=pt, severity=sv, color=_auto_color(label)))
    groups.sort(key=lambda g: (0 if g.label == "ID" else 1, g.label))
    return groups

def _safe_name(s: str) -> str:
    s = str(s)
    out = []
    for ch in s:
        if ch.isalnum() or ch in ["-", "_", "."]:
            out.append(ch)
        elif ch in [" ", "/", ":", "|"]:
            out.append("_")
        else:
            out.append("_")
    # collapse multiple underscores
    name = "".join(out)
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_") or "group"


DISPLAY_NAME = {
    # In our tables, lpips/dreamsim are diversity scores (pairwise mean distance) and are used as "Var"-style detectors.
    "lpips": "lpips_var",
    "dreamsim": "dreamsim_var",
    "pixel_var": "pixel_var",
    "latent_var": "latent_var",
}


def _disp(col: str) -> str:
    return DISPLAY_NAME.get(col, col)


def _resolve_source_df(out_root: str, run_id: str, tag: str, input_glob: str) -> pd.DataFrame:
    analysis_root = os.path.join(out_root, f"{run_id}_analysis_{tag}")
    all_concat = os.path.join(analysis_root, "all_concat.csv")
    if os.path.exists(all_concat):
        return pd.read_csv(all_concat)
    spec = InputSpec(out_root=out_root, run_id=run_id, tag=tag)
    g = input_glob.strip() or default_input_glob(spec)
    return load_sharded_csvs(g)


def main():
    ap = argparse.ArgumentParser("Score Distribution Histogram")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--input_glob", type=str, default="", help="Override glob if not using all_concat.csv")

    ap.add_argument("--probe_step", type=int, default=2)
    ap.add_argument("--all_steps", action="store_true", help="If set, generate plots for all probe_step values present in the data.")
    ap.add_argument("--score_col", type=str, default="lpips", help="Single score column (legacy).")
    ap.add_argument("--score_cols", type=str, default="", help="Comma list for multi-panel plot, e.g. 'lpips,pixel_var,latent_var,dreamsim'")
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--density", action="store_true", help="Plot density instead of raw counts")
    ap.add_argument(
        "--style",
        type=str,
        default="filled",
        choices=["step", "filled"],
        help="Histogram style. 'step' avoids color blending by drawing outlines only.",
    )
    ap.add_argument(
        "--layout",
        type=str,
        default="overlay",
        choices=["overlay", "compare2", "id_vs_ood"],
        help=(
            "overlay: overlay all groups in each panel (may blend colors). "
            "compare2: per-metric rows, 2 columns (ID vs each selected OOD), overlaying only 2 groups per panel. "
            "id_vs_ood: generate MANY figures; each figure compares ID vs ONE OOD (4 rows by metrics)."
        ),
    )
    ap.add_argument(
        "--compare_oods",
        type=str,
        default="stanford,noise5",
        help="For layout=compare2: comma list of OODs. Supported: stanford,noise5,blur5",
    )
    ap.add_argument("--title", type=str, default="", help="Optional plot title override")
    ap.add_argument("--xlim", type=str, default="", help="Optional x-limits: 'min,max'")
    ap.add_argument(
        "--auto_groups",
        action="store_true",
        help="If set, auto-discover all (dataset, perturbation, severity) groups instead of using the fixed paper-style groups.",
    )
    ap.add_argument(
        "--ood_filters",
        type=str,
        default="",
        help=(
            "For layout=id_vs_ood: optional comma list of OOD filters as dataset:perturbation:severity "
            "(e.g. 'recon:noise:1,go_stanford:none:0'). If empty, use all discovered groups except ID."
        ),
    )

    args = ap.parse_args()

    df = _resolve_source_df(args.out_root, args.run_id, args.tag, args.input_glob)
    if args.score_cols.strip():
        score_cols = [c.strip() for c in args.score_cols.split(",") if c.strip()]
    else:
        score_cols = [args.score_col.strip()]
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing score cols: {missing}. Available: {list(df.columns)}")

    # If the user asks for the canonical 4 metrics, enforce a clean 2x2 order:
    # Row1: lpips, pixel_var; Row2: latent_var, dreamsim
    canonical = {"lpips", "pixel_var", "latent_var", "dreamsim"}
    if len(score_cols) == 4 and set(score_cols) == canonical:
        score_cols = ["lpips", "pixel_var", "latent_var", "dreamsim"]

    # Fixed groups for the paper-style view (default), or auto-discover from data.
    id_group = GroupSpec(label="ID (Recon / None)", dataset="recon", perturbation="none", severity=0, color="blue")
    if args.auto_groups:
        groups = _discover_groups(df)
    else:
        groups = [
            id_group,
            GroupSpec(label="OOD (Stanford)", dataset="go_stanford", perturbation="none", severity=0, color="red"),
            GroupSpec(label="OOD (Noise-5)", dataset="recon", perturbation="noise", severity=5, color="green"),
            GroupSpec(label="OOD (Blur-5)", dataset="recon", perturbation="blur", severity=5, color="purple"),
        ]

    out_dir = ensure_out_dir(args.out_root, args.run_id, args.tag, "score_histograms")

    def _hist_kwargs():
        hk = {"bins": int(args.bins), "density": bool(args.density)}
        if args.style == "step":
            hk.update({"histtype": "step", "linewidth": 2.0, "alpha": 1.0})
        else:
            hk.update({"alpha": float(args.alpha), "edgecolor": "black", "linewidth": 0.4})
        return hk

    # New layout: per-metric rows, 2 columns (ID vs selected OODs)
    if args.layout == "compare2":
        # Resolve which OODs to compare
        ood_map = {
            "stanford": GroupSpec(label="OOD (Stanford)", dataset="go_stanford", perturbation="none", severity=0, color="red"),
            "noise5": GroupSpec(label="OOD (Noise-5)", dataset="recon", perturbation="noise", severity=5, color="green"),
            "blur5": GroupSpec(label="OOD (Blur-5)", dataset="recon", perturbation="blur", severity=5, color="purple"),
        }
        ood_keys = [x.strip().lower() for x in args.compare_oods.split(",") if x.strip()]
        oods = []
        for k in ood_keys:
            if k not in ood_map:
                raise ValueError(f"Unknown compare_oods item '{k}'. Supported: {list(ood_map.keys())}")
            oods.append(ood_map[k])
        if len(oods) != 2:
            raise ValueError("layout=compare2 requires exactly 2 OODs in --compare_oods (e.g. 'stanford,noise5')")

        def _save_compare2(step: int) -> str:
            nrows = len(score_cols)
            ncols = 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.6 * nrows), squeeze=False)

            if args.title.strip():
                fig.suptitle(args.title.strip(), fontsize=14)
            else:
                fig.suptitle(f"Score Distributions (probe_step={int(step)})", fontsize=14)

            hk = _hist_kwargs()
            for r, score_col in enumerate(score_cols):
                for c, ood in enumerate(oods):
                    ax = axes[r][c]
                    id_sub = _subset(df, dataset=id_group.dataset, perturbation=id_group.perturbation, severity=id_group.severity, probe_step=step)
                    ood_sub = _subset(df, dataset=ood.dataset, perturbation=ood.perturbation, severity=ood.severity, probe_step=step)
                    id_scores = id_sub[score_col].astype(float).to_numpy()
                    ood_scores = ood_sub[score_col].astype(float).to_numpy()
                    id_scores = id_scores[np.isfinite(id_scores)]
                    ood_scores = ood_scores[np.isfinite(ood_scores)]

                    ax.hist(id_scores, label=f"{id_group.label}", color=id_group.color, **hk)
                    ax.hist(ood_scores, label=f"{ood.label}", color=ood.color, **hk)
                    if c == 0:
                        ax.set_ylabel(f"{_disp(score_col)}\n" + ("Density" if args.density else "Frequency"))
                    ax.set_title(f"{ood.label}")
                    ax.set_xlabel(_disp(score_col))
                    ax.grid(alpha=0.2)
                    ax.legend(fontsize=8)

            fig.tight_layout(rect=[0, 0.02, 1, 0.95])
            safe_cols = "_".join([_disp(c) for c in score_cols])
            fig_path = os.path.join(out_dir, f"hist_compare2_{safe_cols}_step{int(step)}.png")
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            return fig_path

        if args.all_steps:
            steps_all = sorted({int(x) for x in df["probe_step"].unique()})
            for step in steps_all:
                fp = _save_compare2(step)
                print(f"Saved compare2 histogram to: {fp}", flush=True)
            return
        else:
            fp = _save_compare2(int(args.probe_step))
            print(f"Saved compare2 histogram to: {fp}", flush=True)
            return

    # New layout: MANY figures, each compares ID vs ONE OOD (rows=metrics, cols=1)
    if args.layout == "id_vs_ood":
        id_sub_all = df[
            (df["dataset"].astype(str) == id_group.dataset)
            & (df["perturbation"].astype(str) == id_group.perturbation)
            & (df["severity"].astype(int) == int(id_group.severity))
        ].copy()
        if id_sub_all.empty:
            raise RuntimeError("No ID rows matched recon/none/0; cannot run layout=id_vs_ood.")

        # Resolve which OOD groups to plot
        if args.ood_filters.strip():
            ood_specs = []
            for item in [x.strip() for x in args.ood_filters.split(",") if x.strip()]:
                ds, pt, sv = item.split(":")
                ood_specs.append((str(ds), str(pt), int(sv)))
            ood_groups: list[GroupSpec] = []
            for (ds, pt, sv) in ood_specs:
                if ds == "recon" and pt == "none" and int(sv) == 0:
                    continue
                label = _label_for(ds, pt, sv)
                ood_groups.append(GroupSpec(label=label, dataset=ds, perturbation=pt, severity=int(sv), color=_auto_color(label)))
        else:
            ood_groups = [g for g in groups if g.label != "ID"]

        if not ood_groups:
            raise RuntimeError("layout=id_vs_ood found no OOD groups to plot.")

        subdir_root = os.path.join(out_dir, "id_vs_ood")
        os.makedirs(subdir_root, exist_ok=True)

        def _save_one(step: int, ood: GroupSpec) -> str | None:
            id_sub = _subset(df, dataset=id_group.dataset, perturbation=id_group.perturbation, severity=id_group.severity, probe_step=step)
            ood_sub = _subset(df, dataset=ood.dataset, perturbation=ood.perturbation, severity=ood.severity, probe_step=step)
            if id_sub.empty or ood_sub.empty:
                return None

            nrows = len(score_cols)
            fig, axes = plt.subplots(nrows, 1, figsize=(7.5, 3.2 * nrows), squeeze=False)
            if args.title.strip():
                fig.suptitle(args.title.strip(), fontsize=14)
            else:
                fig.suptitle(f"ID vs {ood.label} (probe_step={int(step)})", fontsize=14)

            hk = _hist_kwargs()
            for r, score_col in enumerate(score_cols):
                ax = axes[r][0]
                id_scores = id_sub[score_col].astype(float).to_numpy()
                ood_scores = ood_sub[score_col].astype(float).to_numpy()
                id_scores = id_scores[np.isfinite(id_scores)]
                ood_scores = ood_scores[np.isfinite(ood_scores)]
                ax.hist(id_scores, label="ID", color=id_group.color, **hk)
                ax.hist(ood_scores, label=str(ood.label), color=ood.color, **hk)
                ax.set_title(_disp(score_col))
                ax.set_xlabel(_disp(score_col))
                ax.set_ylabel("Density" if args.density else "Frequency")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=9)

            fig.tight_layout(rect=[0, 0.02, 1, 0.95])

            safe_cols = "_".join([_disp(c) for c in score_cols])
            step_dir = os.path.join(subdir_root, f"step{int(step)}")
            ood_dir = os.path.join(step_dir, _safe_name(str(ood.label)))
            os.makedirs(ood_dir, exist_ok=True)
            fig_path = os.path.join(ood_dir, f"hist_id_vs_{_safe_name(str(ood.label))}_{safe_cols}_step{int(step)}.png")
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            return fig_path

        if args.all_steps:
            steps_all = sorted({int(x) for x in df["probe_step"].unique()})
            for step in steps_all:
                for ood in ood_groups:
                    fp = _save_one(int(step), ood)
                    if fp:
                        print(f"Saved id_vs_ood histogram to: {fp}", flush=True)
            return
        else:
            step = int(args.probe_step)
            for ood in ood_groups:
                fp = _save_one(step, ood)
                if fp:
                    print(f"Saved id_vs_ood histogram to: {fp}", flush=True)
            return

    if len(score_cols) == 1:
        score_col = score_cols[0]
        fig_path = os.path.join(out_dir, f"hist_{_disp(score_col)}_step{int(args.probe_step)}.png")

        plt.figure(figsize=(10, 6))

        hist_kwargs = _hist_kwargs()

        plotted_any = False
        for g in groups:
            sub = _subset(df, dataset=g.dataset, perturbation=g.perturbation, severity=g.severity, probe_step=args.probe_step)
            if sub.empty:
                continue
            scores = sub[score_col].astype(float).to_numpy()
            scores = scores[np.isfinite(scores)]
            if scores.size == 0:
                continue
            plt.hist(scores, label=f"{g.label}", color=g.color, **hist_kwargs)
            plotted_any = True

        if not plotted_any:
            raise RuntimeError("No data matched the requested groups. Check run_id/tag/probe_step/score_col.")

        if args.title.strip():
            title = args.title.strip()
        else:
            title = f"Distribution of {_disp(score_col)} (probe_step={int(args.probe_step)})"
        plt.title(title)
        plt.xlabel(_disp(score_col))
        plt.ylabel("Density" if args.density else "Frequency")
        plt.legend()
        plt.grid(alpha=0.2)

        if args.xlim.strip():
            a, b = args.xlim.split(",")
            plt.xlim(float(a), float(b))

        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"Saved histogram to: {fig_path}", flush=True)
        return

    # Multi-panel plot
    n = len(score_cols)
    ncols = 2 if n <= 4 else 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    if args.title.strip():
        fig.suptitle(args.title.strip(), fontsize=14)
    else:
        fig.suptitle(f"Score Distributions (probe_step={int(args.probe_step)})", fontsize=14)

    for i, score_col in enumerate(score_cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        plotted_any = False

        hist_kwargs = _hist_kwargs()

        for g in groups:
            sub = _subset(df, dataset=g.dataset, perturbation=g.perturbation, severity=g.severity, probe_step=args.probe_step)
            if sub.empty:
                continue
            scores = sub[score_col].astype(float).to_numpy()
            scores = scores[np.isfinite(scores)]
            if scores.size == 0:
                continue
            ax.hist(scores, label=f"{g.label}", color=g.color, **hist_kwargs)
            plotted_any = True
        if not plotted_any:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(_disp(score_col))
        ax.set_xlabel(_disp(score_col))
        ax.set_ylabel("Density" if args.density else "Frequency")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    safe_cols = "_".join([_disp(c) for c in score_cols])
    fig_path = os.path.join(out_dir, f"hist_multi_{safe_cols}_step{int(args.probe_step)}.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved multi histogram to: {fig_path}", flush=True)


if __name__ == "__main__":
    main()


