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

try:
    from scipy.stats import kendalltau as _kendalltau_scipy
except Exception:  # pragma: no cover
    _kendalltau_scipy = None


def _resolve_df(out_root: str, run_id: str, tag: str, input_glob: str) -> pd.DataFrame:
    analysis_root = os.path.join(out_root, f"{run_id}_analysis_{tag}")
    all_concat = os.path.join(analysis_root, "all_concat.csv")
    if os.path.exists(all_concat):
        return pd.read_csv(all_concat)
    spec = InputSpec(out_root=out_root, run_id=run_id, tag=tag)
    g = input_glob.strip() or default_input_glob(spec)
    return load_sharded_csvs(g)


class _Fenwick:
    def __init__(self, n: int):
        self.n = int(n)
        self.bit = np.zeros(self.n + 1, dtype=np.int64)

    def add(self, i: int, delta: int = 1):
        i += 1
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def sum_prefix(self, i: int) -> int:
        """sum over [0..i] inclusive"""
        i += 1
        s = 0
        while i > 0:
            s += int(self.bit[i])
            i -= i & -i
        return int(s)


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """
    Kendall's tau-b (tie-adjusted) without scipy.
    Runs in O(n log n) using inversion counting.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be 1D arrays with same length")

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.shape[0])
    if n < 2:
        return float("nan")

    # Sort by x, then y
    order = np.lexsort((y, x))
    x = x[order]
    y = y[order]

    # Count ties in x: n1
    _, x_counts = np.unique(x, return_counts=True)
    n1 = int(np.sum(x_counts * (x_counts - 1) // 2))

    # Count ties in y: n2
    _, y_counts = np.unique(y, return_counts=True)
    n2 = int(np.sum(y_counts * (y_counts - 1) // 2))

    # Count ties in both x and y: n3
    _, xy_counts = np.unique(np.stack([x, y], axis=1), axis=0, return_counts=True)
    n3 = int(np.sum(xy_counts * (xy_counts - 1) // 2))

    n0 = n * (n - 1) // 2
    # Total comparable pairs (not tied in either), equals C + D
    n_concord_plus_discord = int(n0 - n1 - n2 + n3)
    if n_concord_plus_discord <= 0:
        return float("nan")

    # Compress y values to ranks 0..m-1
    y_vals, y_rank = np.unique(y, return_inverse=True)
    m = int(y_vals.shape[0])
    bit = _Fenwick(m)

    # Count discordant pairs (inversions in y) across x-groups (exclude pairs tied in x)
    D = 0
    seen = 0
    i = 0
    while i < n:
        j = i
        while j < n and x[j] == x[i]:
            j += 1

        # query discordance for this block against previous blocks only
        for k in range(i, j):
            r = int(y_rank[k])
            le = bit.sum_prefix(r)  # count of prior y <= current
            D += (seen - le)  # prior y > current

        # then add this block
        for k in range(i, j):
            bit.add(int(y_rank[k]), 1)
            seen += 1

        i = j

    D = int(D)
    C = int(n_concord_plus_discord - D)

    denom = np.sqrt((n0 - n1) * (n0 - n2))
    if denom <= 0:
        return float("nan")
    tau = (C - D) / denom
    return float(tau)

def kendall_tau_b_with_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Return (tau_b, pvalue). Prefer scipy.stats.kendalltau when available (tie-aware).
    """
    if _kendalltau_scipy is not None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 2:
            return float("nan"), float("nan")
        r = _kendalltau_scipy(x, y, variant="b", nan_policy="omit")
        return float(r.statistic), float(r.pvalue)
    # fallback: tau only, pvalue unknown
    return kendall_tau_b(x, y), float("nan")


@dataclass(frozen=True)
class Filter:
    dataset: str
    perturbation: str
    severity: int

def _parse_filter(s: str) -> Filter:
    try:
        ds, pt, sv = s.split(":")
        return Filter(dataset=ds, perturbation=pt, severity=int(sv))
    except Exception:
        raise ValueError(f"Invalid filter '{s}'. Expected dataset:perturbation:severity, e.g. recon:none:0")


def _filter_key(flt: Filter) -> str:
    # safe folder name
    ds = str(flt.dataset).replace("/", "_")
    pt = str(flt.perturbation).replace("/", "_")
    sv = int(flt.severity)
    return f"{ds}_{pt}_{sv}"


VAR_METRICS = {
    "Pixel_Var": "pixel_var",
    "Latent_Var": "latent_var",
    "LPIPS_Var": "lpips",
    "DreamSim_Var": "dreamsim",
}

GT_METRICS = {
    "GT_MSE": "mse_gt",
    "GT_PSNR": "psnr_gt",
    "GT_SSIM": "ssim_gt",
    "GT_LPIPS": "lpips_gt",
    "GT_DreamSim": "dreamsim_gt",
}


def main():
    ap = argparse.ArgumentParser("Calibration Analysis: Kendall's tau heatmaps vs GT errors")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--input_glob", type=str, default="")

    ap.add_argument("--steps", type=str, default="", help="Comma list; default uses all steps present")
    ap.add_argument(
        "--filter",
        type=str,
        default="recon:none:0",
        help="Which subset to compute correlations on. Format: dataset:perturbation:severity (default ID only).",
    )
    ap.add_argument(
        "--filters",
        type=str,
        default="",
        help="Comma list of filters to run in batch. Each is dataset:perturbation:severity. If set, overrides --filter.",
    )
    ap.add_argument(
        "--invert_gt_psnr",
        action="store_true",
        help="If set, use -GT_PSNR so larger means worse (error-like).",
    )
    ap.add_argument(
        "--invert_gt_ssim",
        action="store_true",
        help="If set, use -GT_SSIM so larger means worse (error-like).",
    )
    ap.add_argument(
        "--gt_metrics",
        type=str,
        default="GT_LPIPS,GT_DreamSim,GT_PSNR,GT_SSIM",
        help=(
            "Comma list of GT metrics to include as columns. "
            "Default excludes GT_MSE (per current analysis preference). "
            f"Supported: {list(GT_METRICS.keys())}"
        ),
    )
    ap.add_argument("--out_subdir", type=str, default="calibration_kendall")
    ap.add_argument("--title", type=str, default="")
    ap.add_argument(
        "--per_step_figs",
        action="store_true",
        help="If set, save one figure per step with two panels (tau heatmap + p-value heatmap).",
    )
    args = ap.parse_args()

    df = _resolve_df(args.out_root, args.run_id, args.tag, args.input_glob)

    # Decide which filters to run
    if args.filters.strip():
        filter_list = [s.strip() for s in args.filters.split(",") if s.strip()]
    else:
        filter_list = [args.filter.strip()]

    var_names = list(VAR_METRICS.keys())
    if args.gt_metrics.strip():
        gt_names = [x.strip() for x in args.gt_metrics.split(",") if x.strip()]
    else:
        gt_names = list(GT_METRICS.keys())
    unknown = [g for g in gt_names if g not in GT_METRICS]
    if unknown:
        raise ValueError(f"Unknown gt_metrics: {unknown}. Supported: {list(GT_METRICS.keys())}")

    for fstr in filter_list:
        flt = _parse_filter(fstr)
        sub = df[
            (df["dataset"].astype(str) == flt.dataset)
            & (df["perturbation"].astype(str) == flt.perturbation)
            & (df["severity"].astype(int) == int(flt.severity))
        ].copy()
        if sub.empty:
            print(f"[warn] No rows match filter={fstr}, skipping.")
            continue

        if args.steps.strip():
            steps_use = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
        else:
            steps_use = sorted({int(x) for x in sub["probe_step"].unique()})

        # Check columns for this subset
        for k, c in {**VAR_METRICS, **GT_METRICS}.items():
            if c not in sub.columns:
                raise ValueError(f"Missing required column {c} for {k}")

        out_dir = ensure_out_dir(args.out_root, args.run_id, args.tag, os.path.join(args.out_subdir, _filter_key(flt)))

        long_rows = []
        mats_tau: dict[int, np.ndarray] = {}
        mats_p: dict[int, np.ndarray] = {}

        for step in steps_use:
            ssub = sub[sub["probe_step"].astype(int) == int(step)].copy()
            if ssub.empty:
                continue

            mat_tau = np.zeros((len(var_names), len(gt_names)), dtype=np.float64)
            mat_p = np.zeros((len(var_names), len(gt_names)), dtype=np.float64)
            for i, vname in enumerate(var_names):
                vcol = VAR_METRICS[vname]
                x = ssub[vcol].astype(float).to_numpy()
                for j, gname in enumerate(gt_names):
                    gcol = GT_METRICS[gname]
                    y = ssub[gcol].astype(float).to_numpy()
                    if gname == "GT_PSNR" and args.invert_gt_psnr:
                        y = -y
                    if gname == "GT_SSIM" and args.invert_gt_ssim:
                        y = -y
                    tau, pval = kendall_tau_b_with_pvalue(x, y)
                    mat_tau[i, j] = tau
                    mat_p[i, j] = pval
                    long_rows.append(
                        {
                            "probe_step": int(step),
                            "var_metric": vname,
                            "gt_metric": gname,
                            "tau": float(tau),
                            "p_value": float(pval),
                            "n": int(np.isfinite(x).sum()),
                            "filter": fstr,
                            "invert_gt_psnr": bool(args.invert_gt_psnr),
                        }
                    )

            mats_tau[int(step)] = mat_tau
            mats_p[int(step)] = mat_p

        # Plot figures
        if args.per_step_figs:
            for step in steps_use:
                step_i = int(step)
                if step_i not in mats_tau:
                    continue
                mat_tau = mats_tau[step_i]
                mat_p = mats_p[step_i]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
                if args.title.strip():
                    fig.suptitle(args.title.strip(), fontsize=14)
                else:
                    fig.suptitle(f"Kendall τ & p-value (step={step_i}, filter={fstr})", fontsize=14)

                im1 = ax1.imshow(mat_tau, vmin=-1.0, vmax=1.0, cmap="coolwarm")
                ax1.set_title("Kendall's τ (tau-b)")
                ax1.set_xticks(range(len(gt_names)))
                ax1.set_xticklabels(gt_names, rotation=30, ha="right")
                ax1.set_yticks(range(len(var_names)))
                ax1.set_yticklabels(var_names)
                for i in range(mat_tau.shape[0]):
                    for j in range(mat_tau.shape[1]):
                        val = mat_tau[i, j]
                        txt = "nan" if not np.isfinite(val) else f"{val:.2f}"
                        ax1.text(j, i, txt, ha="center", va="center", fontsize=10, color="black")
                fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="τ")

                vmax_p = 0.05
                mat_p_clip = np.clip(mat_p, 0.0, vmax_p)
                im2 = ax2.imshow(mat_p_clip, vmin=0.0, vmax=vmax_p, cmap="viridis_r")
                ax2.set_title("p-value (Kendall's τ)")
                ax2.set_xticks(range(len(gt_names)))
                ax2.set_xticklabels(gt_names, rotation=30, ha="right")
                ax2.set_yticks(range(len(var_names)))
                ax2.set_yticklabels(var_names)
                for i in range(mat_p.shape[0]):
                    for j in range(mat_p.shape[1]):
                        val = mat_p[i, j]
                        if not np.isfinite(val):
                            txt = "nan"
                        else:
                            txt = f"{val:.1e}" if val < 1e-3 else f"{val:.3f}"
                        ax2.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
                fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="p (clipped at 0.05)")

                out_png = os.path.join(out_dir, f"kendall_step{step_i}_tau_p.png")
                fig.savefig(out_png, dpi=200)
                plt.close(fig)
                print(f"Saved: {out_png}")
        else:
            n = len(steps_use)
            ncols = 3 if n > 4 else 2
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.0 * nrows), squeeze=False, constrained_layout=True)
            if args.title.strip():
                fig.suptitle(args.title.strip(), fontsize=14)
            else:
                fig.suptitle(f"Kendall's τ vs GT (filter={fstr})", fontsize=14)

            last_im = None
            for idx, step in enumerate(steps_use):
                r, c = divmod(idx, ncols)
                ax = axes[r][c]
                step_i = int(step)
                if step_i not in mats_tau:
                    ax.axis("off")
                    continue
                mat_tau = mats_tau[step_i]
                last_im = ax.imshow(mat_tau, vmin=-1.0, vmax=1.0, cmap="coolwarm")
                ax.set_title(f"step={step_i}")
                ax.set_xticks(range(len(gt_names)))
                ax.set_xticklabels(gt_names, rotation=30, ha="right")
                ax.set_yticks(range(len(var_names)))
                ax.set_yticklabels(var_names)
                for i in range(mat_tau.shape[0]):
                    for j in range(mat_tau.shape[1]):
                        val = mat_tau[i, j]
                        txt = "nan" if not np.isfinite(val) else f"{val:.2f}"
                        ax.text(j, i, txt, ha="center", va="center", fontsize=10, color="black")

            for k in range(len(steps_use), nrows * ncols):
                r, c = divmod(k, ncols)
                axes[r][c].axis("off")

            if last_im is not None:
                fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8, label="Kendall's τ")
            out_png = os.path.join(out_dir, "kendall_tau_matrix_steps.png")
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
            print(f"Saved: {out_png}")

        long_df = pd.DataFrame(long_rows)
        long_csv = os.path.join(out_dir, "kendall_tau_long.csv")
        long_df.to_csv(long_csv, index=False)

        for step in steps_use:
            w_tau = long_df[long_df["probe_step"] == int(step)].pivot(index="var_metric", columns="gt_metric", values="tau")
            w_tau = w_tau.reindex(index=var_names, columns=gt_names)
            w_tau.to_csv(os.path.join(out_dir, f"kendall_tau_step{int(step)}.csv"))

            w_p = long_df[long_df["probe_step"] == int(step)].pivot(index="var_metric", columns="gt_metric", values="p_value")
            w_p = w_p.reindex(index=var_names, columns=gt_names)
            w_p.to_csv(os.path.join(out_dir, f"kendall_p_step{int(step)}.csv"))

        print(f"Saved: {long_csv}")


if __name__ == "__main__":
    main()


