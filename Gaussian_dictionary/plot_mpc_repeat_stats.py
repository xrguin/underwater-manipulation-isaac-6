"""Plot mean and standard deviation across repeated Isaac MPC comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


MODEL_KEYS = ("gaussian", "arx10")
MODEL_LABELS = {
    "gaussian": "Gaussian EDMDc",
    "arx10": "ARX(10)-Gaussian",
}
MODEL_COLORS = {
    "gaussian": "#1f77b4",  # Matplotlib C0
    "arx10": "#ff7f0e",    # Matplotlib C1
}
AXIS_NAMES = ("u", "v", "w", "r")
AXIS_LABELS = (
    "u — surge [m/s]",
    "v — sway-right [m/s]",
    "w — heave-up [m/s]",
    "r — yaw-right [rad/s]",
)


def _mean_std(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sample statistics, preserving exact zero for identical trials."""

    mean = np.mean(sample, axis=0)
    std = np.std(sample, axis=0, ddof=1)
    std[np.ptp(sample, axis=0) == 0.0] = 0.0
    return mean, std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archives", type=Path, nargs="+",
        help="Repeated run_mpc_surge_compare NPZ archives.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stats-output", type=Path, default=None,
        help="Aggregate NPZ path; defaults beside the plot.",
    )
    return parser.parse_args()


def _load_trials(
    paths: list[Path],
) -> tuple[np.ndarray, dict[str, np.ndarray], float, float, float]:
    if len(paths) < 2:
        raise ValueError("at least two repeated archives are required")

    time_s: np.ndarray | None = None
    target_u: float | None = None
    hold_s: float | None = None
    post_zero_s: float | None = None
    trials: dict[str, list[np.ndarray]] = {key: [] for key in MODEL_KEYS}

    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            trial_time = np.asarray(data["gaussian_time_s"], dtype=float)
            trial_target = float(data["target_u_mps"])
            trial_hold = float(data["duration_s"])
            trial_post = float(data["post_zero_duration_s"])
            if time_s is None:
                time_s = trial_time
                target_u = trial_target
                hold_s = trial_hold
                post_zero_s = trial_post
            else:
                if not np.array_equal(time_s, trial_time):
                    raise ValueError(f"{path}: time grid differs")
                if not np.isclose(target_u, trial_target):
                    raise ValueError(f"{path}: target differs")
                if not np.isclose(hold_s, trial_hold):
                    raise ValueError(f"{path}: hold duration differs")
                if not np.isclose(post_zero_s, trial_post):
                    raise ValueError(f"{path}: zero duration differs")

            for model in MODEL_KEYS:
                model_time = np.asarray(data[f"{model}_time_s"], dtype=float)
                state = np.asarray(data[f"{model}_state4"], dtype=float)
                if not np.array_equal(trial_time, model_time):
                    raise ValueError(f"{path}: {model} time grid differs")
                if state.shape != (trial_time.size, 4):
                    raise ValueError(
                        f"{path}: {model} state shape is {state.shape}"
                    )
                if not np.all(np.isfinite(state)):
                    raise ValueError(f"{path}: {model} contains nonfinite data")
                trials[model].append(state)

    assert time_s is not None
    assert target_u is not None
    assert hold_s is not None
    assert post_zero_s is not None
    stacked = {
        model: np.stack(model_trials, axis=0)
        for model, model_trials in trials.items()
    }
    return time_s, stacked, target_u, hold_s, post_zero_s


def _plot(
    output: Path,
    time_s: np.ndarray,
    trials: dict[str, np.ndarray],
    target_u: float,
    hold_s: float,
    post_zero_s: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink = "#263238"
    surface = "#fcfcfb"
    grid_hue = "#d7dee3"
    zero_hue = "#9aa6ad"
    total_s = hold_s + post_zero_s
    reference_t = np.asarray([0.0, hold_s, hold_s, total_s])

    figure, axes_array = plt.subplots(
        2, 2, figsize=(15.2, 10.2), sharex=True, facecolor="white"
    )
    axes = list(axes_array.flat)
    for component, axis in enumerate(axes):
        axis.set_facecolor(surface)
        axis.grid(True, color=grid_hue, linewidth=0.75, alpha=0.85)
        axis.axhline(0.0, color=zero_hue, linewidth=0.8, zorder=1)
        limits = [np.asarray([0.0])]

        for model in MODEL_KEYS:
            sample = trials[model][:, :, component]
            mean, std = _mean_std(sample)
            low = mean - std
            high = mean + std
            limits.extend([low, high])
            axis.fill_between(
                time_s, low, high, color=MODEL_COLORS[model],
                alpha=0.18, linewidth=0.0, zorder=2.0,
            )
            axis.plot(
                time_s, mean, color=MODEL_COLORS[model],
                linewidth=1.8, solid_capstyle="round", zorder=2.5,
            )

        reference_y = (
            np.asarray([target_u, target_u, 0.0, 0.0])
            if component == 0 else np.zeros(4)
        )
        limits.append(reference_y)
        axis.plot(
            reference_t, reference_y, color="#263238", linewidth=1.2,
            linestyle=(0, (1.0, 2.0)), zorder=3.0,
        )
        combined = np.concatenate(limits)
        low = float(np.min(combined))
        high = float(np.max(combined))
        span = max(high - low, 0.02 if component < 3 else 0.01)
        axis.set_ylim(low - 0.10 * span, high + 0.10 * span)
        axis.set_xlim(0.0, total_s)
        axis.set_title(
            AXIS_NAMES[component], loc="left", color=ink,
            fontsize=14, fontweight="semibold",
        )
        axis.set_ylabel(AXIS_LABELS[component], color=ink, fontsize=12)
        axis.tick_params(colors=ink, labelsize=10.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#aeb8bf")

    for axis in axes[2:]:
        axis.set_xlabel(
            "Time since MPC engagement [s]", color=ink, fontsize=12
        )

    handles = [
        Line2D(
            [], [], color=MODEL_COLORS[model], linewidth=2.2,
            label=f"{MODEL_LABELS[model]} mean ± 1 std",
        )
        for model in MODEL_KEYS
    ]
    handles.append(Line2D(
        [], [], color="#263238", linewidth=1.2,
        linestyle=(0, (1.0, 2.0)), label="reference",
    ))
    figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=3, frameon=False, fontsize=10.5, columnspacing=1.3,
        handlelength=2.4,
    )
    count = trials["gaussian"].shape[0]
    figure.suptitle(
        f"Isaac 6 MPC4 velocity tracking across {count} tests",
        color=ink, fontsize=17, y=0.985,
    )
    figure.text(
        0.5, 0.949,
        (
            "Mean ± one sample standard deviation; reference "
            f"u={target_u:.2f}, v=0.00, w=0.00, r=0.00"
        ),
        ha="center", va="center", color="#59636a", fontsize=11,
    )
    figure.text(
        0.5, 0.018,
        (
            "Source: repeated Isaac 6 closed-loop tests; w is positive-up, "
            "r is yaw-right. No resampling, smoothing, or clipping."
        ),
        ha="center", va="bottom", color="#59636a", fontsize=9.5,
    )
    figure.subplots_adjust(
        left=0.085, right=0.985, bottom=0.085, top=0.865,
        hspace=0.22, wspace=0.18,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    paths = [path.expanduser().resolve() for path in args.archives]
    output = args.output.expanduser().resolve()
    stats_output = (
        args.stats_output.expanduser().resolve()
        if args.stats_output is not None
        else output.with_suffix(".npz")
    )
    time_s, trials, target_u, hold_s, post_zero_s = _load_trials(paths)
    _plot(output, time_s, trials, target_u, hold_s, post_zero_s)

    payload: dict[str, np.ndarray] = {
        "time_s": time_s,
        "target_u_mps": np.asarray(target_u),
        "duration_s": np.asarray(hold_s),
        "post_zero_duration_s": np.asarray(post_zero_s),
        "source_archives": np.asarray([str(path) for path in paths]),
        "trial_count": np.asarray(len(paths)),
        "state_names": np.asarray(AXIS_NAMES),
    }
    for model in MODEL_KEYS:
        mean, std = _mean_std(trials[model])
        payload[f"{model}_trials_state4"] = trials[model]
        payload[f"{model}_mean_state4"] = mean
        payload[f"{model}_std_state4"] = std
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stats_output, **payload)
    print(f"plot:  {output}")
    print(f"stats: {stats_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
