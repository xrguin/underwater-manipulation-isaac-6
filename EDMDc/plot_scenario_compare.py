"""Overlay no-cube vs with-cube scenarios on shared axes.

Convention: COLOR encodes the logical series (method, or horizon), LINE STYLE
encodes the scenario (solid = no cube, dashed = with cube). Produces two
combined figures:

  1) heldout_rmse_vs_horizon_scenarios.png
       mean K-step held-out RMSE vs horizon h.
       color = method (EDMDc / EDMDc_delay), style = scenario.
  2) heldout_rmse_vs_history_scenarios.png
       mean K-step held-out RMSE vs history length (delay depth).
       color = horizon K, style = scenario.

Pure NumPy + matplotlib (Agg); does not boot Isaac. File paths are globbed by
convention from EDMDc/data/{with_gripper,with_gripper_cube}/ and EDMDc/model/.

Usage:
    python -m EDMDc.plot_scenario_compare [--hmax 6] [--K 10 30 60]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import _plot_style
from .edmdc_delay import fit_delay
from .validate_delay import kstep_rmse
from .plot_delay_vs_history import heldout_kstep_rmse

_plot_style.apply()

# scenario label -> (data subdir, dataset filename glob stem, model suffix)
SCENARIOS = [
    ("no cube", "with_gripper",      "ardusub",      "",      "-"),
    ("cube",    "with_gripper_cube", "ardusub_cube", "_cube", "--"),
]
METHOD_COLOR = {"EDMDc": "C0", "EDMDc_delay": "C3"}
K_COLORS = ["C0", "C2", "C4", "C1", "C5"]


def _latest(subdir: str, stem: str, kind: str) -> Path:
    pat = f"EDMDc/data/{subdir}/{stem}_{kind}_k*.npz"
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(f"no dataset matched {pat}")
    return Path(hits[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hmax", type=int, default=6)
    ap.add_argument("--horizons", type=int, nargs="+",
                    default=[1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100])
    ap.add_argument("--K", type=int, nargs="+", default=[10, 30, 60],
                    help="Horizons (one color each) for the history-length figure.")
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=Path, default=Path("EDMDc/data/plots"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve datasets + models per scenario.
    resolved = []
    for label, subdir, stem, msfx, style in SCENARIOS:
        train = _latest(subdir, stem, "train")
        heldout = _latest(subdir, stem, "heldout")
        edmdc = Path(f"EDMDc/model/ardusub_edmdc{msfx}.npz")
        delay = Path(f"EDMDc/model/edmdc_delay{msfx}.npz")
        resolved.append((label, style, train, heldout, edmdc, delay))
        print(f"[{label}] heldout={heldout.name}  edmdc={edmdc.name}  delay={delay.name}")

    # ----- Figure 1: RMSE vs horizon (color=method, style=scenario) -----
    hs = sorted(args.horizons)
    fig1, ax1 = plt.subplots(figsize=(_plot_style.FIG_W, 3.0))
    for label, style, _train, heldout, edmdc, delay in resolved:
        d = np.load(heldout)
        X, U = d["X"], d["U"]
        traj_idx = d["traj_idx"]
        step_idx = d["step_idx"] if "step_idx" in d.files else None
        for mlabel, mpath in (("EDMDc", edmdc), ("EDMDc_delay", delay)):
            ys = []
            for K in hs:
                r = kstep_rmse(mpath, X, U, traj_idx, step_idx, K, warmup=3)
                ys.append(np.mean(r) if r is not None else np.nan)
            ax1.plot(hs, ys, style, color=METHOD_COLOR[mlabel], lw=1.6,
                     marker="o", ms=3)
            print(f"[horizon][{label}] {mlabel}: {ys[-3]:.3e} @h={hs[-3]}")
    # Legends: color -> method, style -> scenario.
    h1 = [Line2D([], [], color=METHOD_COLOR[m], ls=r[1], lw=1.6,
                 marker="o", ms=3, label=f"{m}, {r[0]}")
          for m in METHOD_COLOR for r in resolved]
    fig1.legend(handles=h1, loc="upper center", ncol=4, frameon=False,
                bbox_to_anchor=(0.5, 1.01), handlelength=2.4, columnspacing=1.3)
    ax1.set_xlabel("prediction horizon h [steps]")
    ax1.set_ylabel("mean held-out nu RMSE")
    ax1.grid(alpha=0.3)
    fig1.tight_layout(rect=(0, 0, 1, 0.92))
    f1 = args.out_dir / "heldout_rmse_vs_horizon_scenarios.png"
    fig1.savefig(f1, dpi=130)
    print(f"[plot] wrote {f1}")

    # ----- Figure 2: RMSE vs history length (color=K, style=scenario) -----
    hgrid = list(range(0, args.hmax + 1))
    warmup = args.hmax
    fig2, ax2 = plt.subplots(figsize=(_plot_style.FIG_W, 3.4))
    for label, style, train, heldout, _edmdc, _delay in resolved:
        tr = np.load(train)
        Xt, Ut, Xnt = tr["X"], tr["U"], tr["X_next"]
        traj_t = tr["traj_idx"]
        step_t = tr["step_idx"] if "step_idx" in tr.files else None
        ho = np.load(heldout)
        Xh, Uh = ho["X"], ho["U"]
        traj_h = ho["traj_idx"]
        step_h = ho["step_idx"] if "step_idx" in ho.files else None
        models = {h: fit_delay(Xt, Ut, Xnt, traj_t, step_t, ds=h, di=h,
                               lam=args.lam)[:2] for h in hgrid}
        for ki, K in enumerate(args.K):
            ys = []
            for h in hgrid:
                A_list, B_list = models[h]
                r = heldout_kstep_rmse(A_list, B_list, h, h, Xh, Uh,
                                       traj_h, step_h, K, warmup)
                ys.append(float(np.mean(r)))
            ax2.plot(hgrid, ys, style, color=K_COLORS[ki], lw=1.6,
                     marker="o", ms=4)
            print(f"[history][{label}] K={K}: {ys}")
    # Order so column-major ncol=3 fill gives row = scenario, col = horizon.
    h2 = [Line2D([], [], color=K_COLORS[ki], ls=r[1], lw=1.6,
                 marker="o", ms=4, label=f"K={K} ({K/60:.2f}s), {r[0]}")
          for ki, K in enumerate(args.K) for r in resolved]
    fig2.legend(handles=h2, loc="upper center", ncol=3, frameon=False,
                bbox_to_anchor=(0.5, 1.02), handlelength=2.4, columnspacing=1.3)
    ax2.set_xlabel("history length  h  (state-delay = input-delay; h=0 ⇒ EDMDc)")
    ax2.set_ylabel("mean held-out nu RMSE")
    ax2.set_xticks(hgrid)
    ax2.grid(alpha=0.3)
    fig2.tight_layout(rect=(0, 0, 1, 0.86))
    f2 = args.out_dir / "heldout_rmse_vs_history_scenarios.png"
    fig2.savefig(f2, dpi=130)
    print(f"[plot] wrote {f2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
