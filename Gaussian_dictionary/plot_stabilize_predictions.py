"""Plot fair stabilize-data predictions for Gaussian, ARX10, and 34-D EDMDc."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from EDMDc.edmdc import decode_nu, lift as lift34, rollout as rollout34

from .arx_history import ARXHistoryModel, contiguous_segments
from .gaussian_edmdc import (
    STATE_INDICES,
    STATE_NAMES,
    GaussianEDMDcModel,
    project_root,
    select_controlled_state,
)


COLORS = {
    "Gaussian 2-RBF": "#1f77b4",  # Matplotlib C0
    "ARX(10) Gaussian": "#ff7f0e",  # Matplotlib C1
    "Gaussian 16-RBF": "#2ca02c",  # Matplotlib C2
    "EDMDc 34-D": "#9467bd",  # Matplotlib C4
}
LINESTYLES = {
    "Gaussian 2-RBF": "-",
    "ARX(10) Gaussian": "--",
    "Gaussian 16-RBF": "-.",
    "EDMDc 34-D": ":",
}
UNITS = ("m/s", "m/s", "m/s", "rad/s")


def _common_starts(
    segments: list[np.ndarray], history: int, horizon: int,
) -> np.ndarray:
    starts = []
    for segment in segments:
        last_position = segment.size - horizon
        if last_position >= history - 1:
            starts.append(segment[history - 1:last_position + 1])
    return np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)


def _gaussian_endpoint(
    model: GaussianEDMDcModel,
    state4: np.ndarray,
    control: np.ndarray,
    starts: np.ndarray,
    horizon: int,
) -> np.ndarray:
    lifted = model.lift(state4[starts])
    for j in range(horizon):
        lifted = lifted @ model.A.T + control[starts + j] @ model.B.T
    return model.decode(lifted)


def _arx_endpoint(
    model: ARXHistoryModel,
    state4: np.ndarray,
    control: np.ndarray,
    raw_features: np.ndarray,
    starts: np.ndarray,
    horizon: int,
) -> np.ndarray:
    prior = [raw_features[starts - lag] for lag in range(1, model.m)]
    controls = np.stack([control[starts + j] for j in range(horizon)], axis=1)
    return model.endpoint_rollout(state4[starts], controls, prior)


def _edmdc34_endpoint(
    A: np.ndarray,
    B: np.ndarray,
    state6: np.ndarray,
    control: np.ndarray,
    starts: np.ndarray,
    horizon: int,
) -> np.ndarray:
    lifted = lift34(state6[starts])
    for j in range(horizon):
        lifted = lifted @ A.T + control[starts + j] @ B.T
    return decode_nu(lifted)[:, list(STATE_INDICES)]


def _plot_one_step(
    predictions: dict[str, np.ndarray],
    truth: np.ndarray,
    n_samples: int,
    subtitle: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    names = list(predictions)
    fig, axes = plt.subplots(len(names), 4, figsize=(16, 3.5 * len(names)))
    for row, name in enumerate(names):
        prediction = predictions[name]
        for axis_i, (axis_name, unit) in enumerate(zip(STATE_NAMES, UNITS)):
            ax = axes[row, axis_i]
            x = truth[:, axis_i]
            y = prediction[:, axis_i]
            rmse = float(np.sqrt(np.mean((y - x) ** 2)))
            ax.scatter(x, y, s=2, alpha=0.12, color=COLORS[name],
                       edgecolors="none", rasterized=True)
            low = float(min(x.min(), y.min()))
            high = float(max(x.max(), y.max()))
            pad = 0.04 * (high - low + 1e-12)
            ax.plot([low - pad, high + pad], [low - pad, high + pad],
                    color="0.25", linestyle="--", linewidth=1)
            ax.set_xlim(low - pad, high + pad)
            ax.set_ylim(low - pad, high + pad)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.25)
            ax.set_title(f"{axis_name}: RMSE {rmse:.4f} {unit}")
            ax.set_xlabel(f"recorded {axis_name}")
            ax.set_ylabel(f"predicted {axis_name}")
            if axis_i == 0:
                ax.text(-0.32, 0.5, name, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=11,
                        color=COLORS[name], fontweight="bold")
    fig.suptitle(
        f"One-step prediction on stabilize test trajectories\n"
        f"n={n_samples:,} common causal origins · {subtitle}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _plot_kstep(
    horizons: list[int],
    totals: dict[str, list[float]],
    counts: list[int],
    dt: float,
    subtitle: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    seconds = np.asarray(horizons) * dt
    for name in ("Gaussian 2-RBF", "ARX(10) Gaussian",
                 "Gaussian 16-RBF", "EDMDc 34-D"):
        ax.plot(seconds, totals[name], marker="o", linewidth=2.2,
                linestyle=LINESTYLES[name], color=COLORS[name], label=name)
    ax.set_xlabel("prediction horizon [s]")
    ax.set_ylabel("total RMSE over [u, v, w, r]\n(mixed m/s and rad/s)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    ax.set_title(
        "Sliding-origin recursive prediction RMSE\n"
        f"{subtitle} · common origins: {min(counts):,} to {max(counts):,}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _plot_trajectory(
    time_s: np.ndarray,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    traj_id: int,
    subtitle: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for axis_i, (ax, axis_name, unit) in enumerate(
        zip(axes.flat, STATE_NAMES, UNITS)
    ):
        ax.plot(time_s, truth[:, axis_i], color="0.1", linewidth=2.0,
                label="Isaac recorded")
        for name, prediction in predictions.items():
            ax.plot(time_s, prediction[:, axis_i], color=COLORS[name],
                    linestyle=LINESTYLES[name], linewidth=1.5, label=name)
        ax.set_title(f"{axis_name} [{unit}]")
        ax.set_ylabel(axis_name)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("time from prediction origin [s]")
    axes[1, 1].set_xlabel("time from prediction origin [s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.91))
    fig.suptitle(
        f"Full recursive body-state rollout — trajectory {traj_id}\n{subtitle}",
        fontsize=12, y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=(
        root / "EDMDc/data/stablize/isaac_stabilize_20hz_test.npz"))
    parser.add_argument("--train-data", type=Path, default=(
        root / "EDMDc/data/stablize/isaac_stabilize_20hz_train.npz"))
    parser.add_argument("--gaussian-2", type=Path, default=(
        root / "Gaussian_dictionary/model/stab_free20_gauss_2rbf.npz"))
    parser.add_argument("--gaussian-16", type=Path, default=(
        root / "Gaussian_dictionary/model/stab_free20_gauss_16rbf.npz"))
    parser.add_argument("--arx", type=Path, default=(
        root / "Gaussian_dictionary/model/arx10_stab_free20_gauss_2rbf.npz"))
    parser.add_argument("--edmdc-34", type=Path, default=(
        root / "EDMDc/model/stab_free20_edmdc_34d.npz"))
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=[1, 5, 10, 25, 50, 100])
    parser.add_argument("--traj-idx", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")

    gaussian2 = GaussianEDMDcModel.load(args.gaussian_2)
    gaussian16 = GaussianEDMDcModel.load(args.gaussian_16)
    arx = ARXHistoryModel.load(args.arx, args.gaussian_2)
    with np.load(args.edmdc_34, allow_pickle=True) as model34:
        A34 = np.asarray(model34["A"], dtype=np.float64)
        B34 = np.asarray(model34["B"], dtype=np.float64)

    with np.load(args.data, allow_pickle=False) as data:
        state6 = np.asarray(data["X"], dtype=np.float64)
        state4 = select_controlled_state(state6)
        next4 = select_controlled_state(data["X_next"])
        control = np.asarray(data["U"], dtype=np.float64)
        traj_idx = np.asarray(data["traj_idx"], dtype=np.int64)
        step_idx = np.asarray(data["step_idx"], dtype=np.int64)
        dt = float(data["dt"])

    overlap = False
    if args.train_data.exists():
        with np.load(args.train_data, allow_pickle=False) as train:
            overlap = (
                state6.shape[0] <= train["X"].shape[0]
                and np.array_equal(state6, train["X"][:state6.shape[0]])
                and np.array_equal(control, train["U"][:control.shape[0]])
            )
    subtitle = (
        "designated test archive duplicates training trajectories 0–4"
        if overlap else "independent test archive"
    )
    print(f"[plot] data qualification: {subtitle}")

    segments = contiguous_segments(traj_idx, step_idx)
    raw_arx = np.hstack([gaussian2.lift(state4), control])
    predictions_by_horizon: dict[int, dict[str, np.ndarray]] = {}
    truth_by_horizon: dict[int, np.ndarray] = {}
    totals = {name: [] for name in COLORS}
    counts: list[int] = []
    metric_rows: list[dict[str, object]] = []

    for horizon in args.horizons:
        starts = _common_starts(segments, arx.m, horizon)
        if starts.size == 0:
            continue
        truth = next4[starts + horizon - 1]
        predictions = {
            "Gaussian 2-RBF": _gaussian_endpoint(
                gaussian2, state4, control, starts, horizon),
            "ARX(10) Gaussian": _arx_endpoint(
                arx, state4, control, raw_arx, starts, horizon),
            "Gaussian 16-RBF": _gaussian_endpoint(
                gaussian16, state4, control, starts, horizon),
            "EDMDc 34-D": _edmdc34_endpoint(
                A34, B34, state6, control, starts, horizon),
        }
        predictions_by_horizon[horizon] = predictions
        truth_by_horizon[horizon] = truth
        counts.append(starts.size)
        for name, prediction in predictions.items():
            error = prediction - truth
            axis_rmse = np.sqrt(np.mean(error ** 2, axis=0))
            total = float(np.sqrt(np.mean(error ** 2)))
            totals[name].append(total)
            for axis_name, value in zip(STATE_NAMES, axis_rmse):
                metric_rows.append({
                    "model": name, "horizon_steps": horizon,
                    "horizon_seconds": horizon * dt, "axis": axis_name,
                    "rmse": float(value), "n_origins": starts.size,
                })
            metric_rows.append({
                "model": name, "horizon_steps": horizon,
                "horizon_seconds": horizon * dt, "axis": "total",
                "rmse": total, "n_origins": starts.size,
            })

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        root / "Gaussian_dictionary/results"
        / f"stabilize_predictions_corrected_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_horizons = [h for h in args.horizons if h in predictions_by_horizon]
    _plot_one_step(
        predictions_by_horizon[1], truth_by_horizon[1], counts[0], subtitle,
        out_dir / "one_step_scatter.png",
    )
    _plot_kstep(
        valid_horizons, totals, counts, dt, subtitle,
        out_dir / "sliding_kstep_rmse.png",
    )

    rows = np.flatnonzero(traj_idx == args.traj_idx)
    rows = rows[np.argsort(step_idx[rows], kind="stable")]
    if rows.size < arx.m:
        raise ValueError("chosen trajectory is too short for ARX history")
    origin_pos = arx.m - 1
    origin = rows[origin_pos]
    trajectory_controls = control[rows[origin_pos:]]
    truth_trajectory = np.vstack([
        state4[origin:origin + 1], next4[rows[origin_pos:]],
    ])
    prior = [raw_arx[rows[origin_pos - lag]] for lag in range(1, arx.m)]
    model_trajectories = {
        "Gaussian 2-RBF": gaussian2.rollout(
            state4[origin], trajectory_controls),
        "ARX(10) Gaussian": arx.rollout(
            state4[origin], trajectory_controls, prior),
        "Gaussian 16-RBF": gaussian16.rollout(
            state4[origin], trajectory_controls),
        "EDMDc 34-D": select_controlled_state(rollout34(
            A34, B34, state6[origin], trajectory_controls)),
    }
    time_s = np.arange(truth_trajectory.shape[0]) * dt
    _plot_trajectory(
        time_s, truth_trajectory, model_trajectories, args.traj_idx, subtitle,
        out_dir / "trajectory_0_velocity_rollout.png",
    )

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "model", "horizon_steps", "horizon_seconds", "axis", "rmse",
            "n_origins",
        ])
        writer.writeheader()
        writer.writerows(metric_rows)
    (out_dir / "README.txt").write_text(
        "Evaluation uses common causal origins and X_next[k] as the one-step "
        "target.\nData qualification: " + subtitle + ".\n"
        "No 3D path is produced for Gaussian/ARX models because they do not "
        "predict p or q and the dataset does not store pose.\n",
        encoding="utf-8",
    )

    print("\nTotal sliding-origin RMSE over [u,v,w,r]")
    for horizon_i, horizon in enumerate(valid_horizons):
        values = ", ".join(
            f"{name}={totals[name][horizon_i]:.5f}" for name in COLORS
        )
        print(f"  {horizon * dt:4.2f}s (n={counts[horizon_i]:,}): {values}")
    print(f"[plot] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
