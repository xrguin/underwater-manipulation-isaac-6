"""Plot EDMDc A-matrix eigenvalues against the unit circle.

Loads one or more trained model npz files (``EDMDc/model/*.npz``), reports
the spectral radius rho(A) = max |lambda| (discrete-time stability:
|lambda| > 1 means that mode grows every step -> unstable model; one
eigenvalue at exactly 1.0 is the affine/constant-feature offset mode and
is expected), and draws all eigenvalues in the complex plane with the
unit circle. Multiple models overlay with distinct colors for direct
comparison (e.g. yaw-held tank model vs yaw-excited free model).

No Isaac — runs in seconds:

    conda activate marinegym && python -m EDMDc.plot_eigs \\
        EDMDc/model/edmdc_free20hz_20260718.npz \\
        EDMDc/model/edmdc_tank20hz_20260715.npz
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", type=Path,
                   help="Trained EDMDc model npz file(s).")
    p.add_argument("--out", type=str, default=None,
                   help="Output png (default EDMDc/data/plots/eigs_<TS>.png).")
    return p.parse_args()


def resolve_dt(model: dict, project: Path) -> float | None:
    """Model files don't store dt; recover it from the source dataset if
    that file still exists locally."""
    try:
        src = project / str(model["source_npz"])
        if src.exists():
            return float(np.load(src)["dt"])
    except Exception:
        pass
    return None


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=1.0, label="unit circle")

    for i, path in enumerate(args.models):
        m = np.load(path)
        A = m["A"]
        eigs = np.linalg.eigvals(A)
        mags = np.abs(eigs)
        rho = float(mags.max())
        n_unstable = int(np.sum(mags > 1.0 + 1e-9))
        dt = resolve_dt(m, project)

        # slowest genuinely dynamical mode (exclude the offset eig ~ 1.0)
        dyn = mags[mags < 1.0 - 1e-6]
        slow = float(dyn.max()) if dyn.size else float("nan")
        tau = (-dt / np.log(slow)) if (dt and slow < 1.0) else None

        print(f"{path.name}")
        print(f"  rho(A) = {rho:.6f}  ({n_unstable} eigenvalue(s) outside "
              f"the unit circle -> {'UNSTABLE' if n_unstable else 'stable'})")
        print(f"  slowest dynamical mode |lambda| = {slow:.4f}"
              + (f"  (tau = {tau:.2f} s at dt={dt:g})" if tau else ""))

        color = f"C{i}"
        ax.scatter(eigs.real, eigs.imag, s=36, facecolors="none",
                   edgecolors=color, linewidths=1.4,
                   label=f"{path.stem}  (rho={rho:.4f})")
        # highlight anything unstable
        bad = mags > 1.0 + 1e-9
        if bad.any():
            ax.scatter(eigs[bad].real, eigs[bad].imag, s=90, marker="x",
                       color=color, linewidths=2.0)

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.set_title("EDMDc A-matrix spectrum vs unit circle\n"
                 "(x marks = unstable modes; the point at +1 is the "
                 "affine offset mode)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = project / "EDMDc" / "data" / "plots" / f"eigs_{ts}.png"
    else:
        out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[plot-eigs] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
