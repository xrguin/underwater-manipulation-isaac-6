"""K-step held-out accuracy analysis for the thruster-cube / per-motor-unit
EDMDc and ARX models (norm_inputs experiment).

Loads one normalized dataset, splits trajectories into train / test, fits EDMDc
(``edmdc.fit_edmdc``) and HODMDc+ARX (``edmdc_delay.fit_delay``) on the train
split, then evaluates open-loop K-step prediction RMSE on the held-out test
trajectories (the standard sysID metric; see CLAUDE.md "When investigating model
accuracy"). Reports per-axis RMSE at the final horizon and the RMSE-vs-horizon
curve, and writes a plot under ``EDMDc/norm_inputs/data/plots/``.

This is prediction-accuracy only (open-loop rollout), the honest test of model
fit independent of any controller.

Usage:
    python -m EDMDc.norm_inputs.validate_norm \\
        EDMDc/norm_inputs/data/norm_cube_train.npz --K 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from EDMDc.edmdc import NU_NAMES, fit_edmdc, rollout
from EDMDc.edmdc_delay import build_windows, fit_delay, rollout_delay


def _ordered(traj_idx, step_idx, tr):
    idx = np.where(traj_idx == tr)[0]
    return idx[np.argsort(step_idx[idx])]


def _split_trajs(traj_idx, frac_test, seed):
    trajs = np.unique(traj_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(trajs)
    n_test = max(1, int(round(len(trajs) * frac_test)))
    test = set(int(t) for t in trajs[:n_test])
    train = set(int(t) for t in trajs[n_test:])
    return train, test


def _subset(mask, *arrays):
    return [a[mask] for a in arrays]


def kstep_rmse_edmdc(A, B, X, U, traj_idx, step_idx, test_trajs, K, stride):
    """Per-horizon squared-error accumulation over all test windows. Returns
    (err2[K,6], count[K]) where err2[k] sums squared error at horizon k+1."""
    err2 = np.zeros((K, 6)); cnt = np.zeros(K)
    for tr in test_trajs:
        g = _ordered(traj_idx, step_idx, tr)
        Xt, Ut = X[g], U[g]
        L = len(g)
        for s in range(0, L - 1, stride):
            kk = min(K, L - 1 - s)
            if kk <= 0:
                continue
            nu_hat = rollout(A, B, Xt[s], Ut[s:s + kk])      # (kk+1, 6)
            for k in range(kk):
                err2[k] += (nu_hat[k + 1] - Xt[s + k + 1]) ** 2
                cnt[k] += 1
    return err2, cnt


def kstep_rmse_arx(A_list, B_list, X, U, traj_idx, step_idx, test_trajs, K, stride):
    ds = len(A_list) - 1
    di = len(B_list) - 1
    d0 = max(ds, di)
    err2 = np.zeros((K, 6)); cnt = np.zeros(K)
    for tr in test_trajs:
        g = _ordered(traj_idx, step_idx, tr)
        Xt, Ut = X[g], U[g]
        L = len(g)
        for s in range(d0, L - 1, stride):
            kk = min(K, L - 1 - s)
            if kk <= 0:
                continue
            nu_hist = np.stack([Xt[s - i] for i in range(ds + 1)])      # (ds+1,6)
            u_past = (np.stack([Ut[s - 1 - j] for j in range(di)])
                      if di > 0 else np.zeros((0, Ut.shape[1])))
            nu_hat = rollout_delay(A_list, B_list, nu_hist, u_past, Ut[s:s + kk])
            for k in range(kk):
                err2[k] += (nu_hat[k + 1] - Xt[s + k + 1]) ** 2
                cnt[k] += 1
    return err2, cnt


def _rmse(err2, cnt):
    return np.sqrt(err2 / np.maximum(cnt[:, None], 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--K", type=int, default=30, help="max prediction horizon (steps).")
    ap.add_argument("--frac-test", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=5, help="window start stride.")
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--state-delay", type=int, default=3)
    ap.add_argument("--input-delay", type=int, default=1)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    X, U, Xn = d["X"], d["U"], d["X_next"]
    traj_idx, step_idx = d["traj_idx"], d["step_idx"]
    units = str(d["input_units"]) if "input_units" in d.files else "unknown"
    print(f"[val] {args.npz.name}: {X.shape[0]:,} snapshots, units={units}, "
          f"dt={float(d['dt']):.4f}")

    train, test = _split_trajs(traj_idx, args.frac_test, args.split_seed)
    tr_mask = np.isin(traj_idx, list(train))
    Xtr, Utr, Xntr, tjtr, sjtr = _subset(tr_mask, X, U, Xn, traj_idx, step_idx)
    test_trajs = sorted(test)
    print(f"[val] split: {len(train)} train trajs ({tr_mask.sum():,} snaps), "
          f"{len(test)} test trajs")

    # --- Fit both models on the train split ---
    A, B = fit_edmdc(Xtr, Xntr, Utr, lam=args.lam)
    print(f"[val] EDMDc fit: A{A.shape} B{B.shape}, rho(A)="
          f"{np.max(np.abs(np.linalg.eigvals(A))):.3f}")
    A_list, B_list, M = fit_delay(Xtr, Utr, Xntr, tjtr, sjtr,
                                  ds=args.state_delay, di=args.input_delay, lam=args.lam)
    print(f"[val] ARX fit (ds={args.state_delay}, di={args.input_delay}): "
          f"{M:,} windows")

    # --- K-step held-out rollout ---
    K = args.K
    e_ed, c_ed = kstep_rmse_edmdc(A, B, X, U, traj_idx, step_idx, test_trajs, K, args.stride)
    e_ar, c_ar = kstep_rmse_arx(A_list, B_list, X, U, traj_idx, step_idx, test_trajs, K, args.stride)
    rmse_ed = _rmse(e_ed, c_ed)
    rmse_ar = _rmse(e_ar, c_ar)

    # Aggregate (RMS over all 6 axes) at the final horizon.
    agg_ed = np.sqrt(np.mean(rmse_ed[K - 1] ** 2))
    agg_ar = np.sqrt(np.mean(rmse_ar[K - 1] ** 2))
    print(f"\n[val] === K={K}-step held-out RMSE (per axis) ===")
    print(f"      {'axis':>4}  {'EDMDc':>10}  {'ARX':>10}  {'ARX gain':>9}")
    for i, n in enumerate(NU_NAMES):
        ge = rmse_ed[K - 1, i]; ga = rmse_ar[K - 1, i]
        gain = (1 - ga / ge) * 100 if ge > 0 else 0.0
        print(f"      {n:>4}  {ge:>10.4e}  {ga:>10.4e}  {gain:>+8.1f}%")
    print(f"      {'agg':>4}  {agg_ed:>10.4e}  {agg_ar:>10.4e}  "
          f"{(1-agg_ar/agg_ed)*100:>+8.1f}%")

    # --- Plot RMSE vs horizon ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = np.arange(1, K + 1)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
        for i, (n, ax) in enumerate(zip(NU_NAMES, axes.ravel())):
            ax.plot(ks, rmse_ed[:, i], label="EDMDc", lw=1.8)
            ax.plot(ks, rmse_ar[:, i], label=f"ARX(ds={args.state_delay},di={args.input_delay})",
                    lw=1.8, ls="--")
            ax.set_title(f"{n}"); ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
            if i >= 3:
                ax.set_xlabel("horizon k (steps)")
            if i % 3 == 0:
                ax.set_ylabel("K-step RMSE")
        fig.suptitle(f"Held-out K-step prediction RMSE — {args.npz.stem} "
                     f"(units={units})")
        fig.tight_layout()
        out_dir = Path(__file__).resolve().parent / "data" / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{args.npz.stem}_kstep_rmse.png"
        fig.savefig(out_png, dpi=130)
        print(f"\n[val] wrote plot {out_png}")
    except Exception as e:
        print(f"[val] plot skipped ({e})")

    # --- Save numeric results ---
    out_npz = args.npz.with_name(args.npz.stem + "_kstep_results.npz")
    np.savez(out_npz, rmse_edmdc=rmse_ed, rmse_arx=rmse_ar,
             K=K, nu_names=np.array(NU_NAMES), units=units,
             state_delay=args.state_delay, input_delay=args.input_delay)
    print(f"[val] wrote results {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
