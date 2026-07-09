"""Held-out RMSE vs history length h (delay depth) for HODMDc+ARX.

x-axis is the *history length* h = state-delay = input-delay of the
EDMDc_delay model (NOT the prediction horizon). h=0 recovers ordinary EDMDc.
For each h we fit a delay model on the training set and measure the mean
K-step held-out RMSE (averaged over the six nu axes and all valid starts).

All h are evaluated on the SAME (trajectory, start) set -- the shared warmup
equals the largest h -- so the comparison across history lengths is fair.

Pure NumPy + matplotlib (Agg); does not boot Isaac.

Usage:
    python -m EDMDc.plot_delay_vs_history TRAIN.npz HELDOUT.npz \\
        [--hmax 6] [--K 10 30 60] [--lam 1e-3] \\
        [--out-dir EDMDc/data/plots/with_gripper]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import _plot_style
from .edmdc import NU_NAMES
from .edmdc_delay import fit_delay, rollout_delay

_plot_style.apply()


def _ordered(traj_idx, step_idx, tr):
    idxs = np.where(traj_idx == tr)[0]
    if step_idx is not None:
        idxs = idxs[np.argsort(step_idx[idxs])]
    return idxs


def heldout_kstep_rmse(A_list, B_list, ds, di, X, U, traj_idx, step_idx,
                       K, warmup):
    """Mean-per-axis K-step RMSE for a delay model, starts offset by `warmup`."""
    err = []
    for tr in np.unique(traj_idx):
        idxs = _ordered(traj_idx, step_idx, tr)
        L = len(idxs)
        for p in range(warmup, L - K):
            tau = U[idxs[p:p + K]]
            truth = X[idxs[p + K]]
            nu_hist = X[idxs[p - np.arange(ds + 1)]]
            u_past = (U[idxs[p - 1 - np.arange(di)]] if di > 0
                      else np.zeros((0, U.shape[1])))
            pred = rollout_delay(A_list, B_list, nu_hist, u_past, tau)
            err.append((pred[K] - truth).reshape(1, -1))
    if not err:
        return np.full(len(NU_NAMES), np.nan)
    e = np.vstack(err)
    fin = np.isfinite(e).all(axis=1)
    if not fin.any():
        return np.full(len(NU_NAMES), np.inf)
    return np.sqrt(np.mean(e[fin] ** 2, axis=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train", type=Path)
    ap.add_argument("heldout", type=Path)
    ap.add_argument("--hmax", type=int, default=6,
                    help="Max history length h (inclusive). h ranges 0..hmax.")
    ap.add_argument("--K", type=int, nargs="+", default=[10, 30, 60],
                    help="Prediction horizon(s) in steps; one curve each.")
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("EDMDc/data/plots/with_gripper"))
    args = ap.parse_args()

    tr = np.load(args.train)
    Xt, Ut, Xnt = tr["X"], tr["U"], tr["X_next"]
    traj_t = tr["traj_idx"]
    step_t = tr["step_idx"] if "step_idx" in tr.files else None

    ho = np.load(args.heldout)
    Xh, Uh = ho["X"], ho["U"]
    traj_h = ho["traj_idx"]
    step_h = ho["step_idx"] if "step_idx" in ho.files else None
    dt = float(ho["dt"]) if "dt" in ho.files else 1.0 / 60.0

    hs = list(range(0, args.hmax + 1))
    warmup = args.hmax              # shared start set across all h

    # Fit one delay model per history length, cache (A_list, B_list).
    models = {}
    for h in hs:
        A_list, B_list, M = fit_delay(Xt, Ut, Xnt, traj_t, step_t,
                                      ds=h, di=h, lam=args.lam)
        models[h] = (A_list, B_list)
        print(f"[fit] h={h}: {M:,} windows")

    # RMSE vs h, one row per K.
    rmse = {K: [] for K in args.K}
    for K in args.K:
        for h in hs:
            A_list, B_list = models[h]
            r = heldout_kstep_rmse(A_list, B_list, h, h, Xh, Uh,
                                   traj_h, step_h, K, warmup)
            rmse[K].append(float(np.mean(r)))
        row = "  ".join(f"h={h}:{rmse[K][i]:.4e}" for i, h in enumerate(hs))
        print(f"[K={K:3d}] {row}")

    fig, ax = plt.subplots(figsize=(_plot_style.FIG_W, 3.0))
    for K in args.K:
        ax.plot(hs, rmse[K], "o-", label=f"horizon K={K} ({K*dt:.2f}s)")
    ax.set_xlabel("history length  h  (state-delay = input-delay; h=0 ⇒ EDMDc)")
    ax.set_ylabel("mean held-out nu RMSE")
    ax.set_xticks(hs)
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "heldout_rmse_vs_history.png"
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
