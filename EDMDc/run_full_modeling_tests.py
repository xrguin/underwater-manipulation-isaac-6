"""Full held-out modeling-accuracy test: does history depth h improve K-step
prediction?

For each scenario (gripper only, gripper + cube), all within-regime (train and
test from the same physical configuration; no model mismatch):

  * fit HODMDc+ARX (ds = di = h) for h = 0..hmax on the training set
    (h=0 recovers plain 34-D EDMDc exactly; same dictionary, same ridge lam)
  * K-step prediction: from every valid start in every test trajectory,
    initialize with the MEASURED history, roll forward K steps feeding the
    RECORDED inputs, score the error at step K. Identical start sets across
    all h (shared warmup = hmax).
  * per-axis RMSE + a normalized aggregate (each axis scaled by the held-out
    std of that axis)
  * uncertainty at the trajectory level: per-trajectory RMSE -> mean +/- 95%
    t-CI over held-out trajectories (starts within a trajectory are
    correlated, so trajectory is the independent unit)
  * paired improvement vs h=0 per trajectory, Wilcoxon signed-rank
  * overfitting control: same protocol evaluated on the training trajectories
  * ridge sensitivity: lam sweep at K=30

Pure NumPy/matplotlib; does not boot Isaac.

Usage:
    python -m EDMDc.run_full_modeling_tests \\
        [--out-dir EDMDc/data/full_modeling_tests] [--hmax 6] [--K 10 30 60]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from . import _plot_style
from .edmdc import DICT_DIM, NU_DIM, NU_NAMES, lift
from .edmdc_delay import fit_delay, _ordered_traj_indices

_plot_style.apply()

PROJECT = Path(__file__).resolve().parents[1]

SCENARIOS = (
    ("with_gripper",
     PROJECT / "EDMDc/data/with_gripper/ardusub_train_k100_20260629_021456.npz",
     PROJECT / "EDMDc/data/with_gripper/ardusub_heldout_k20_20260629_021456.npz"),
    ("with_gripper_cube",
     PROJECT / "EDMDc/data/with_gripper_cube/ardusub_cube_train_k100_20260629_022955.npz",
     PROJECT / "EDMDc/data/with_gripper_cube/ardusub_cube_heldout_k20_20260629_022955.npz"),
)
SCEN_LABEL = {"with_gripper": "gripper only", "with_gripper_cube": "gripper + cube"}


def _load(path: Path):
    d = np.load(path, allow_pickle=True)
    return (d["X"], d["U"], d["X_next"], d["traj_idx"],
            d["step_idx"] if "step_idx" in d.files else None,
            float(d["dt"]) if "dt" in d.files else 1.0 / 60.0)


def kstep_errors_per_traj(A_list, B_list, h, X, U, traj_idx, step_idx,
                          K, warmup, stride):
    """Error at step K for every start, batched per trajectory.

    Mirrors edmdc_delay.rollout_delay semantics (propagates the predicted
    lifted state; recorded inputs at every offset). Returns
    {traj: (n_starts, 6) error} on the shared start grid.
    """
    out = {}
    for tr in np.unique(traj_idx):
        idxs = _ordered_traj_indices(traj_idx, step_idx, tr)
        L = len(idxs)
        starts = np.arange(warmup, L - K, stride)
        if starts.size == 0:
            continue
        psi_hist = [lift(X[idxs[starts - i]]) for i in range(h + 1)]
        psi_next = psi_hist[0]
        for t in range(K):
            psi_next = np.zeros((starts.size, DICT_DIM), dtype=np.float64)
            for i in range(h + 1):
                psi_next += psi_hist[i] @ A_list[i].T
            for j in range(h + 1):
                psi_next += U[idxs[starts + (t - j)]] @ B_list[j].T
            psi_hist.insert(0, psi_next)
            psi_hist.pop()
        nu_pred = psi_next[:, 1:1 + NU_DIM]
        out[int(tr)] = nu_pred - X[idxs[starts + K]]
    return out


def summarize(errs: dict[int, np.ndarray], axis_std: np.ndarray):
    """Pooled per-axis RMSE, per-traj normalized RMSE, pooled normalized RMSE."""
    pooled = np.vstack(list(errs.values()))
    fin = np.isfinite(pooled).all(axis=1)
    pooled = pooled[fin]
    per_axis = np.sqrt(np.mean(pooled ** 2, axis=0))
    traj_norm = {}
    for tr, e in errs.items():
        e = e[np.isfinite(e).all(axis=1)]
        if len(e):
            traj_norm[tr] = float(np.sqrt(np.mean((e / axis_std) ** 2)))
    pooled_norm = float(np.sqrt(np.mean((pooled / axis_std) ** 2)))
    return per_axis, traj_norm, pooled_norm


def t_ci(vals: np.ndarray, conf: float = 0.95):
    n = len(vals)
    m = float(np.mean(vals))
    if n < 2:
        return m, m, m
    half = stats.t.ppf(0.5 + conf / 2, n - 1) * np.std(vals, ddof=1) / np.sqrt(n)
    return m, m - half, m + half


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT / "EDMDc/data/full_modeling_tests")
    ap.add_argument("--hmax", type=int, default=6)
    ap.add_argument("--K", type=int, nargs="+", default=[10, 30, 60])
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--lam-sens", type=float, nargs="+", default=[1e-4, 1e-2],
                    help="Extra ridge values for the sensitivity sweep (K=30).")
    ap.add_argument("--heldout-stride", type=int, default=2)
    ap.add_argument("--train-stride", type=int, default=10)
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hs = list(range(args.hmax + 1))
    warmup = args.hmax
    K_sens = 30 if 30 in args.K else args.K[len(args.K) // 2]

    rmse_rows: list[dict] = []
    paired_rows: list[dict] = []
    summary_parts: list[str] = []

    for scen, train_path, heldout_path in SCENARIOS:
        label = SCEN_LABEL[scen]
        print(f"[{scen}] train={train_path.name} heldout={heldout_path.name}")
        Xt, Ut, Xnt, traj_t, step_t, dt = _load(train_path)
        Xh, Uh, _, traj_h, step_h, _ = _load(heldout_path)
        axis_std = np.std(Xh, axis=0)
        print(f"[{scen}] heldout axis std: "
              + ", ".join(f"{n}={s:.3f}" for n, s in zip(NU_NAMES, axis_std)))

        # ---- fit one model per (lam, h) ----
        lams = [args.lam] + list(args.lam_sens)
        models: dict[tuple[float, int], tuple] = {}
        for lam in lams:
            for h in hs:
                A_list, B_list, M = fit_delay(Xt, Ut, Xnt, traj_t, step_t,
                                              ds=h, di=h, lam=lam)
                models[(lam, h)] = (A_list, B_list)
            print(f"[{scen}] fitted h=0..{args.hmax} at lam={lam:g} ({M:,} windows)")

        # ---- main evaluation (lam = args.lam): heldout + train, all K ----
        # traj_norm_store[(split, K, h)] = {traj: normalized rmse}
        traj_norm_store: dict[tuple[str, int, int], dict[int, float]] = {}
        for split, (X, U, tr_idx, st_idx, stride) in {
            "heldout": (Xh, Uh, traj_h, step_h, args.heldout_stride),
            "train":   (Xt, Ut, traj_t, step_t, args.train_stride),
        }.items():
            for K in args.K:
                for h in hs:
                    A_list, B_list = models[(args.lam, h)]
                    errs = kstep_errors_per_traj(
                        A_list, B_list, h, X, U, tr_idx, st_idx,
                        K, warmup, stride)
                    per_axis, traj_norm, pooled_norm = summarize(errs, axis_std)
                    traj_norm_store[(split, K, h)] = traj_norm
                    n_starts = sum(len(e) for e in errs.values())
                    vals = np.array(list(traj_norm.values()))
                    m, lo, hi = t_ci(vals)
                    rmse_rows.append(dict(
                        scenario=scen, split=split, lam=args.lam, h=h, K=K,
                        axis="norm_all", rmse=pooled_norm,
                        traj_mean=m, ci_lo=lo, ci_hi=hi,
                        n_traj=len(vals), n_starts=n_starts))
                    for ax_i, ax_name in enumerate(NU_NAMES):
                        rmse_rows.append(dict(
                            scenario=scen, split=split, lam=args.lam, h=h, K=K,
                            axis=ax_name, rmse=float(per_axis[ax_i]),
                            traj_mean="", ci_lo="", ci_hi="",
                            n_traj=len(vals), n_starts=n_starts))
                print(f"[{scen}] {split:7s} K={K:3d} done")

        # ---- paired improvement vs h=0 (heldout, all K) ----
        for K in args.K:
            base = traj_norm_store[("heldout", K, 0)]
            for h in hs[1:]:
                cur = traj_norm_store[("heldout", K, h)]
                common = sorted(set(base) & set(cur))
                b = np.array([base[t] for t in common])
                c = np.array([cur[t] for t in common])
                impr = (b - c) / b * 100.0
                try:
                    p = float(stats.wilcoxon(b, c).pvalue)
                except ValueError:
                    p = float("nan")
                paired_rows.append(dict(
                    scenario=scen, lam=args.lam, K=K, h=h,
                    median_impr_pct=float(np.median(impr)),
                    mean_impr_pct=float(np.mean(impr)),
                    frac_traj_improved=float(np.mean(impr > 0)),
                    wilcoxon_p=p, n_traj=len(common)))

        # ---- lambda sensitivity (heldout, K_sens) ----
        lam_curves: dict[float, list[float]] = {}
        for lam in lams:
            curve = []
            for h in hs:
                if lam == args.lam:
                    vals = np.array(list(
                        traj_norm_store[("heldout", K_sens, h)].values()))
                    curve.append(float(np.mean(vals)))
                    continue
                A_list, B_list = models[(lam, h)]
                errs = kstep_errors_per_traj(
                    A_list, B_list, h, Xh, Uh, traj_h, step_h,
                    K_sens, warmup, args.heldout_stride)
                _, traj_norm, _ = summarize(errs, axis_std)
                curve.append(float(np.mean(list(traj_norm.values()))))
                rmse_rows.append(dict(
                    scenario=scen, split="heldout", lam=lam, h=h, K=K_sens,
                    axis="norm_all", rmse="", traj_mean=curve[-1],
                    ci_lo="", ci_hi="", n_traj=len(traj_norm), n_starts=""))
            lam_curves[lam] = curve
        print(f"[{scen}] lambda sensitivity done")

        # ================= plots =================
        # 1. normalized heldout RMSE vs h with CI bands
        fig, ax = plt.subplots(figsize=(_plot_style.FIG_W, 3.2))
        for K in args.K:
            ms, los, his = [], [], []
            for h in hs:
                vals = np.array(list(traj_norm_store[("heldout", K, h)].values()))
                m, lo, hi = t_ci(vals)
                ms.append(m); los.append(lo); his.append(hi)
            line, = ax.plot(hs, ms, "o-", label=f"K={K} ({K * dt:.2f}s)")
            ax.fill_between(hs, los, his, alpha=0.18, color=line.get_color())
        ax.set_xlabel("history length h  (h=0 = EDMDc)")
        ax.set_ylabel("held-out normalized RMSE")
        ax.set_title(f"{label}: K-step held-out error vs history "
                     f"(mean ± 95% CI over trajectories)")
        ax.set_xticks(hs)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"rmse_vs_h_{scen}.png", dpi=150)
        plt.close(fig)

        # 2. per-axis heldout RMSE vs h at K_sens
        fig, ax = plt.subplots(figsize=(_plot_style.FIG_W, 3.2))
        for ax_i, ax_name in enumerate(NU_NAMES):
            curve = []
            for h in hs:
                r = [row for row in rmse_rows
                     if row["scenario"] == scen and row["split"] == "heldout"
                     and row["lam"] == args.lam and row["h"] == h
                     and row["K"] == K_sens and row["axis"] == ax_name]
                curve.append(r[0]["rmse"])
            ax.plot(hs, curve, "o-", label=ax_name)
        ax.set_xlabel("history length h")
        ax.set_ylabel(f"held-out RMSE at K={K_sens}")
        ax.set_title(f"{label}: per-axis held-out error vs history")
        ax.set_xticks(hs)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"per_axis_rmse_{scen}.png", dpi=150)
        plt.close(fig)

        # 3. train vs heldout (overfitting control) at K_sens
        fig, ax = plt.subplots(figsize=(_plot_style.FIG_W, 3.2))
        for split, style in (("train", "s--"), ("heldout", "o-")):
            ms = [float(np.mean(list(traj_norm_store[(split, K_sens, h)].values())))
                  for h in hs]
            ax.plot(hs, ms, style, label=split)
        ax.set_xlabel("history length h")
        ax.set_ylabel(f"normalized RMSE at K={K_sens}")
        ax.set_title(f"{label}: train vs held-out (capacity control)")
        ax.set_xticks(hs)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"train_vs_heldout_{scen}.png", dpi=150)
        plt.close(fig)

        # 4. lambda sensitivity at K_sens
        fig, ax = plt.subplots(figsize=(_plot_style.FIG_W, 3.2))
        for lam, curve in lam_curves.items():
            ax.plot(hs, curve, "o-", label=f"lam={lam:g}")
        ax.set_xlabel("history length h")
        ax.set_ylabel(f"held-out normalized RMSE at K={K_sens}")
        ax.set_title(f"{label}: ridge sensitivity")
        ax.set_xticks(hs)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"lambda_sensitivity_{scen}.png", dpi=150)
        plt.close(fig)

        # ---- summary fragment ----
        summary_parts.append(f"## {label}\n")
        summary_parts.append("| K | h=0 RMSE | h=3 RMSE | median impr. | "
                             "frac traj improved | Wilcoxon p |")
        summary_parts.append("|---|---:|---:|---:|---:|---:|")
        for K in args.K:
            v0 = np.array(list(traj_norm_store[("heldout", K, 0)].values()))
            v3 = np.array(list(traj_norm_store[("heldout", K, 3)].values()))
            pr = [r for r in paired_rows
                  if r["scenario"] == scen and r["K"] == K and r["h"] == 3][0]
            summary_parts.append(
                f"| {K} | {np.mean(v0):.4f} | {np.mean(v3):.4f} "
                f"| {pr['median_impr_pct']:+.1f}% "
                f"| {pr['frac_traj_improved']:.2f} "
                f"| {pr['wilcoxon_p']:.2g} |")
        summary_parts.append("")

    # ================= CSVs + summary =================
    with (out_dir / "results_rmse.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rmse_rows[0].keys()))
        w.writeheader()
        w.writerows(rmse_rows)
    with (out_dir / "paired_improvement.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
        w.writeheader()
        w.writerows(paired_rows)

    header = [
        "# Full modeling-accuracy test: history depth vs K-step held-out error",
        "",
        "Within-regime (no model mismatch): each scenario trained and tested "
        "on its own physical configuration. h=0 is plain 34-D EDMDc; "
        "h>0 adds delayed lifted states + delayed inputs (HODMDc+ARX), same "
        "dictionary, same ridge. Error measured at step K of an open-loop "
        "rollout from measured history with recorded inputs; normalized RMSE "
        "scales each axis by its held-out std. CIs and Wilcoxon tests use "
        "held-out trajectories as the independent unit "
        "(n=20 per scenario).",
        "",
    ]
    (out_dir / "summary.md").write_text(
        "\n".join(header + summary_parts) + "\n")

    print(f"[done] wrote {out_dir}/results_rmse.csv, paired_improvement.csv, "
          f"summary.md and plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
