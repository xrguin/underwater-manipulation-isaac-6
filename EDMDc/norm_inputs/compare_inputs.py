"""Compare K-step held-out prediction error: normalized [-1,1] inscribed-box
inputs (norm_inputs) vs the baseline Newton-wrench inscribed-box pipeline
(``collect_isaac``, "like we trained before"). EDMDc only and EDMDc + ARX.

For each dataset the script does the *same* methodology as ``validate_norm``:
train/test trajectory split, fit EDMDc (``fit_edmdc``) and HODMDc+ARX
(``fit_delay``) on the train split, K-step open-loop rollout on the held-out
test trajectories. The two datasets differ only in excitation + input units
(matched n / episode / seed / gripper), so this isolates the effect of the
input representation on model accuracy.

Because the two excitations visit different velocity ranges, raw RMSE is not
directly comparable across pipelines; we therefore also report NRMSE (RMSE
divided by the per-axis velocity std of that dataset's test set), which is
scale-free and comparable.

Usage:
    python -m EDMDc.norm_inputs.compare_inputs \\
        --norm     EDMDc/norm_inputs/data/norm_inscribed_train.npz \\
        --baseline EDMDc/norm_inputs/data/baseline_wrench_train.npz --K 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from EDMDc.edmdc import NU_NAMES, fit_edmdc
from EDMDc.edmdc_delay import fit_delay
from EDMDc.norm_inputs.validate_norm import (
    _ordered, _split_trajs, _subset, _rmse,
    kstep_rmse_edmdc, kstep_rmse_arx,
)


def _eval_dataset(npz, K, frac_test, split_seed, stride, lam, ds, di):
    d = np.load(npz, allow_pickle=True)
    X, U, Xn = d["X"], d["U"], d["X_next"]
    traj_idx, step_idx = d["traj_idx"], d["step_idx"]
    units = str(d["input_units"]) if "input_units" in d.files else "newton_wrench"

    train, test = _split_trajs(traj_idx, frac_test, split_seed)
    tr = np.isin(traj_idx, list(train))
    Xtr, Utr, Xntr, tjtr, sjtr = _subset(tr, X, U, Xn, traj_idx, step_idx)
    test_trajs = sorted(test)

    A, B = fit_edmdc(Xtr, Xntr, Utr, lam=lam)
    rhoA = float(np.max(np.abs(np.linalg.eigvals(A))))
    A_list, B_list, _ = fit_delay(Xtr, Utr, Xntr, tjtr, sjtr, ds=ds, di=di, lam=lam)

    e_ed, c_ed = kstep_rmse_edmdc(A, B, X, U, traj_idx, step_idx, test_trajs, K, stride)
    e_ar, c_ar = kstep_rmse_arx(A_list, B_list, X, U, traj_idx, step_idx, test_trajs, K, stride)
    rmse_ed = _rmse(e_ed, c_ed)[K - 1]            # per-axis RMSE at horizon K
    rmse_ar = _rmse(e_ar, c_ar)[K - 1]

    # Per-axis velocity std of the test set (for NRMSE).
    test_mask = np.isin(traj_idx, test_trajs)
    vstd = X[test_mask].std(axis=0)
    vstd[vstd == 0] = 1.0

    return dict(units=units, U_cols=U.shape[1], rhoA=rhoA, n_test=len(test_trajs),
                n_snap=X.shape[0], rmse_ed=rmse_ed, rmse_ar=rmse_ar,
                nrmse_ed=rmse_ed / vstd, nrmse_ar=rmse_ar / vstd, vstd=vstd)


def _agg(per_axis):
    return float(np.sqrt(np.mean(per_axis ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm", type=Path, required=True,
                    help="normalized [-1,1] / inscribed-unit dataset.")
    ap.add_argument("--baseline", type=Path, required=True,
                    help="Newton-wrench / inscribed-box dataset (collect_isaac).")
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--frac-test", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--state-delay", type=int, default=3)
    ap.add_argument("--input-delay", type=int, default=1)
    args = ap.parse_args()

    K = args.K
    print(f"[cmp] evaluating both datasets at K={K} "
          f"(ARX ds={args.state_delay}, di={args.input_delay})\n")
    res = {}
    for tag, npz in (("norm[-1,1]", args.norm), ("baseline-wrench", args.baseline)):
        r = _eval_dataset(npz, K, args.frac_test, args.split_seed, args.stride,
                          args.lam, args.state_delay, args.input_delay)
        res[tag] = r
        print(f"[cmp] {tag:>15}: {npz.name}  units={r['units']}  "
              f"U_dim={r['U_cols']}  rho(A)={r['rhoA']:.3f}  "
              f"snaps={r['n_snap']:,}  test_trajs={r['n_test']}")

    nc, bl = res["norm[-1,1]"], res["baseline-wrench"]

    def table(metric_ed, metric_ar, title):
        print(f"\n[cmp] === {title} (K={K}) ===")
        print(f"      {'axis':>4} | {'norm EDMDc':>11} {'base EDMDc':>11} | "
              f"{'norm ARX':>11} {'base ARX':>11}")
        for i, n in enumerate(NU_NAMES):
            print(f"      {n:>4} | {nc[metric_ed][i]:>11.4e} {bl[metric_ed][i]:>11.4e} | "
                  f"{nc[metric_ar][i]:>11.4e} {bl[metric_ar][i]:>11.4e}")
        print(f"      {'agg':>4} | {_agg(nc[metric_ed]):>11.4e} {_agg(bl[metric_ed]):>11.4e} | "
              f"{_agg(nc[metric_ar]):>11.4e} {_agg(bl[metric_ar]):>11.4e}")

    table("rmse_ed", "rmse_ar", "Raw K-step velocity RMSE")
    table("nrmse_ed", "nrmse_ar", "Normalized K-step RMSE (RMSE / test-velocity std)")

    print("\n[cmp] ARX gain over EDMDc (aggregate NRMSE):")
    for tag, r in res.items():
        ge, ga = _agg(r["nrmse_ed"]), _agg(r["nrmse_ar"])
        print(f"      {tag:>15}: EDMDc {ge:.4f} -> ARX {ga:.4f}  ({(1-ga/ge)*100:+.1f}%)")

    # Combined bar chart of aggregate NRMSE.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = ["EDMDc", "EDMDc+ARX"]
        cube = [_agg(nc["nrmse_ed"]), _agg(nc["nrmse_ar"])]
        base = [_agg(bl["nrmse_ed"]), _agg(bl["nrmse_ar"])]
        x = np.arange(len(labels)); w = 0.35
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.bar(x - w / 2, cube, w, label="normalized [-1,1] / inscribed")
        ax.bar(x + w / 2, base, w, label="baseline Newton / inscribed")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel(f"aggregate {K}-step NRMSE")
        ax.set_title("Modeling accuracy: input representation comparison (gripper)")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        for xi, (cv, bv) in enumerate(zip(cube, base)):
            ax.text(xi - w / 2, cv, f"{cv:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + w / 2, bv, f"{bv:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        out = Path(__file__).resolve().parent / "data" / "plots" / "compare_inputs_nrmse.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        print(f"\n[cmp] wrote plot {out}")
    except Exception as e:
        print(f"[cmp] plot skipped ({e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
