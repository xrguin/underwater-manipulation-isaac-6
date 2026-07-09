"""Plot EDMDc vs EDMDc_delay on held-out APRBS data.

Two figures:
  1) One held-out trajectory: ground truth nu vs the EDMDc and EDMDc_delay
     rollouts (recorded inputs), all six body-velocity axes.
  2) Mean K-step RMSE (averaged over the six nu axes and all (traj, start)
     pairs) as a function of the prediction horizon h, for both models.

Pure NumPy + matplotlib (Agg) -- does NOT boot Isaac, safe to run in-process.

Usage:
    python -m EDMDc.plot_delay_compare HELDOUT.npz \\
        EDMDc/model/ardusub_edmdc.npz EDMDc/model/edmdc_delay.npz \\
        [--traj 0] [--out-dir EDMDc/data/plots/with_gripper]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import _plot_style
from .edmdc import NU_NAMES, rollout
from .edmdc_delay import MODEL_KIND, load_delay_model, rollout_delay
from .validate_delay import kstep_rmse

_plot_style.apply()


def _ordered(traj_idx, step_idx, tr):
    idxs = np.where(traj_idx == tr)[0]
    if step_idx is not None:
        idxs = idxs[np.argsort(step_idx[idxs])]
    return idxs


def _load_any(model_path: Path):
    """Return ('plain', A, B) or ('delay', A_list, B_list, ds, di)."""
    m = np.load(model_path, allow_pickle=True)
    if "kind" in m.files and str(m["kind"]) == MODEL_KIND:
        A_list, B_list, ds, di = load_delay_model(model_path)
        return ("delay", A_list, B_list, ds, di)
    return ("plain", m["A"], m["B"])


def plot_trajectory(ax_list, X, U, idxs, dt, edmdc, delay, warmup,
                    truth_label="Actual (Gripper only)",
                    edmdc_color="C0", delay_color="C3"):
    """Overlay truth + both model rollouts for one trajectory on 6 axes."""
    p = warmup
    L = len(idxs)
    K = L - 1 - p
    tau = U[idxs[p:p + K]]
    truth = X[idxs[p:p + K + 1]]
    t = np.arange(K + 1) * dt

    # EDMDc rollout from the single start state.
    pred_e = rollout(edmdc[1], edmdc[2], X[idxs[p]], tau)

    # EDMDc_delay rollout with history.
    _, A_list, B_list, ds, di = delay
    nu_hist = X[idxs[p - np.arange(ds + 1)]]
    u_past = (U[idxs[p - 1 - np.arange(di)]] if di > 0
              else np.zeros((0, U.shape[1])))
    pred_d = rollout_delay(A_list, B_list, nu_hist, u_past, tau)

    for i, ax in enumerate(ax_list):
        ax.plot(t, truth[:, i], "k-", lw=1.1, label=truth_label)
        ax.plot(t, pred_e[:, i], color=edmdc_color, ls="--", lw=1.4, label="EDMDc")
        ax.plot(t, pred_d[:, i], color=delay_color, ls="-.", lw=1.4,
                label=f"EDMDc with $h={ds}$")
        ax.set_title(NU_NAMES[i])
        ax.set_xlabel("t [s]")
        ax.grid(alpha=0.3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("heldout", type=Path)
    ap.add_argument("edmdc", type=Path, help="plain EDMDc model .npz")
    ap.add_argument("delay", type=Path, help="edmdc_delay model .npz")
    ap.add_argument("--traj", type=int, default=0,
                    help="Which held-out trajectory index to plot (default 0).")
    ap.add_argument("--horizons", type=int, nargs="+",
                    default=[1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100],
                    help="Horizons h (steps) for the RMSE-vs-h sweep.")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("EDMDc/data/plots/with_gripper"))
    ap.add_argument("--truth-label", type=str, default=None,
                    help="Legend label for the ground-truth line. Default: "
                         "auto from the heldout npz `cube` flag — "
                         "'Actual (Gripped cube)' or 'Actual (Gripper only)'.")
    args = ap.parse_args()

    d = np.load(args.heldout)
    X = d["X"]; U = d["U"]
    traj_idx = d["traj_idx"]
    step_idx = d["step_idx"] if "step_idx" in d.files else None
    dt = float(d["dt"]) if "dt" in d.files else 1.0 / 60.0

    has_cube = bool(d["cube"]) if "cube" in d.files else False
    truth_label = (args.truth_label if args.truth_label is not None
                   else ("Actual (Gripped cube)" if has_cube
                         else "Actual (Gripper only)"))

    edmdc = _load_any(args.edmdc)
    delay = _load_any(args.delay)
    assert delay[0] == "delay", "third arg must be an edmdc_delay model"
    ds, di = delay[3], delay[4]
    warmup = max(ds, di)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: one trajectory, six axes ----
    tr = np.unique(traj_idx)[args.traj]
    idxs = _ordered(traj_idx, step_idx, tr)
    # Distinct color scheme per scenario so stacked rollouts are separable.
    edmdc_color, delay_color = (("C2", "C1") if has_cube else ("C0", "C3"))
    fig1, axes = plt.subplots(2, 3, figsize=(_plot_style.FIG_W, 3.9))
    plot_trajectory(axes.ravel(), X, U, idxs, dt, edmdc, delay, warmup,
                    truth_label=truth_label,
                    edmdc_color=edmdc_color, delay_color=delay_color)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                bbox_to_anchor=(0.5, -0.01), handlelength=2.4)
    fig1.tight_layout(rect=(0, 0.08, 1, 1))
    f1 = args.out_dir / "heldout_traj_rollout.png"
    fig1.savefig(f1, dpi=130)
    print(f"[plot] wrote {f1}")

    # ---- Figure 2: mean RMSE vs horizon h ----
    hs = sorted(args.horizons)
    rmse_e, rmse_d = [], []
    for K in hs:
        re = kstep_rmse(args.edmdc, X, U, traj_idx, step_idx, K, warmup)
        rd = kstep_rmse(args.delay, X, U, traj_idx, step_idx, K, warmup)
        rmse_e.append(np.mean(re) if re is not None else np.nan)
        rmse_d.append(np.mean(rd) if rd is not None else np.nan)
        print(f"[sweep] h={K:3d}  EDMDc={rmse_e[-1]:.4e}  "
              f"EDMDc_delay={rmse_d[-1]:.4e}")

    fig2, ax = plt.subplots(figsize=(_plot_style.FIG_W, 2.9))
    ax.plot(hs, rmse_e, "C0o--", label="EDMDc")
    ax.plot(hs, rmse_d, "C3s-.", label=f"EDMDc_delay (ds={ds}, di={di})")
    ax.set_xlabel("prediction horizon h [steps]")
    ax.set_ylabel("mean nu RMSE (6 axes, all starts)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    f2 = args.out_dir / "heldout_rmse_vs_horizon.png"
    fig2.savefig(f2, dpi=130)
    print(f"[plot] wrote {f2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
