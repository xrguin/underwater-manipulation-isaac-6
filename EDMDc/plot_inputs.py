"""Plot the distribution of APRBS control inputs from a collected .npz.

Loads one or more .npz files saved by ``collect_fossen.py`` / ``collect_isaac.py``
and renders 2x2 histograms of the 4 task-DOF axes (Fx, Fy, Fz, Tz). Drain-phase
samples (U = 0) are excluded by default; ``--only-complete`` further restricts
the input set to trajectories that finished the APRBS phase without truncation.

When multiple .npz files are passed, histograms for each are overlaid on the
same axes for direct distribution comparison (e.g., fossen vs isaac).

Output: ``EDMDc/data/plots/input_dist_<YYYYMMDD_HHMMSS>.png`` by default.

Usage:
    # Plot one dataset
    python -m EDMDc.plot_inputs EDMDc/data/numpy/fossen_20260513_023659.npz

    # Compare fossen vs isaac side by side
    python -m EDMDc.plot_inputs \\
        EDMDc/data/numpy/fossen_*.npz \\
        EDMDc/data/simulator/isaac_*.npz \\
        --labels fossen isaac

    # Restrict to trajectories that completed APRBS without truncation
    python -m EDMDc.plot_inputs <files> --only-complete
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plot_inputs. "
        "Install with `pip install matplotlib`."
    ) from e

from ._common import project_root


# 4-DOF task wrench: indices into the 6-vector (Fx, Fy, Fz, Tx, Ty, Tz) and labels.
INPUT_NAMES_4DOF = ["Fx (surge)", "Fy (sway)", "Fz (heave)", "Tz (yaw)"]
INPUT_AXES_4DOF = [0, 1, 2, 5]
INPUT_UNITS = ["N", "N", "N", "N·m"]


def select_aprbs_inputs(
    npz_path: Path, only_complete: bool
) -> tuple[np.ndarray, int, int]:
    """Return (U_aprbs, n_total_trajectories, n_selected_trajectories).

    Filters:
      * Always drops drain-phase rows where U == 0 (so the giant zero bar
        doesn't dominate the histogram).
      * If ``only_complete``, also keeps only trajectories whose maximum
        ``step_idx`` reached the end of the APRBS phase — i.e., those that
        were not truncated by the envelope/tipover check mid-APRBS.
    """
    d = np.load(npz_path)
    U = d["U"]
    traj_idx = d["traj_idx"]
    step_idx = d["step_idx"]
    n_total_trajs = int(d["n_trajectories"])

    aprbs_mask = np.linalg.norm(U, axis=1) > 0.0

    if only_complete:
        n_steps = int(round(float(d["episode_seconds"]) / float(d["dt"])))
        completed_trajs: set[int] = set()
        for t in np.unique(traj_idx):
            max_step = int(step_idx[traj_idx == t].max())
            if max_step >= n_steps:           # reached drain ⇒ APRBS completed
                completed_trajs.add(int(t))
        comp_mask = np.isin(traj_idx, list(completed_trajs))
        combined = aprbs_mask & comp_mask
        n_selected = len(completed_trajs)
    else:
        combined = aprbs_mask
        n_selected = int(len(np.unique(traj_idx)))

    return U[combined], n_total_trajs, n_selected


def plot_distribution(
    npz_paths: list[Path],
    labels: list[str] | None = None,
    only_complete: bool = False,
    bins: int = 40,
    out: Path | None = None,
) -> Path:
    """Render the 4-axis APRBS input histogram and write to `out` (or a
    timestamped default under EDMDc/data/plots/). Returns the output path.
    """
    npz_paths = [Path(p) for p in npz_paths]
    if labels is None:
        labels = [p.stem for p in npz_paths]
    if len(labels) != len(npz_paths):
        raise ValueError(
            f"labels count ({len(labels)}) must match number of .npz files "
            f"({len(npz_paths)})."
        )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for npz_path, label in zip(npz_paths, labels):
        U, n_total, n_sel = select_aprbs_inputs(npz_path, only_complete=only_complete)
        legend_label = (
            f"{label}: {n_sel}/{n_total} trajs, {U.shape[0]:,} APRBS samples"
        )
        for ax, name, axis, unit in zip(
            axes, INPUT_NAMES_4DOF, INPUT_AXES_4DOF, INPUT_UNITS
        ):
            ax.hist(U[:, axis], bins=bins, alpha=0.55,
                    label=legend_label, edgecolor="none")
            ax.set_title(name)
            ax.set_xlabel(f"value [{unit}]")
            ax.set_ylabel("count")

    for ax in axes:
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    suffix = ("trajectories with completed APRBS only"
              if only_complete else "all retained APRBS samples")
    fig.suptitle(f"APRBS control-input distribution ({suffix})", fontsize=12)
    fig.tight_layout()

    if out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = project_root() / "EDMDc" / "data" / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"input_dist_{ts}.png"
    else:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot_inputs] wrote {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "npz", nargs="+", type=Path,
        help="One or more .npz files produced by collect_fossen / collect_isaac.",
    )
    ap.add_argument(
        "--labels", nargs="+", default=None,
        help="Legend labels (one per .npz). Default = filename stem.",
    )
    ap.add_argument(
        "--only-complete", action="store_true",
        help="Restrict to trajectories that finished APRBS without truncation.",
    )
    ap.add_argument(
        "--bins", type=int, default=40,
        help="Histogram bin count per subplot (default 40).",
    )
    ap.add_argument(
        "--out", type=str, default=None,
        help="Output PNG path. Default: EDMDc/data/plots/input_dist_<YYYYMMDD_HHMMSS>.png.",
    )
    args = ap.parse_args()
    plot_distribution(
        npz_paths=args.npz,
        labels=args.labels,
        only_complete=args.only_complete,
        bins=args.bins,
        out=Path(args.out) if args.out else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
