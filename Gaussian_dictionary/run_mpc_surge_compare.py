"""Compare Gaussian EDMDc and ARX(10)-Gaussian MPC in Isaac Sim 6.

Both controllers track the same body-frame velocity reference
``[u, v, w, r] = [target_u, 0, 0, 0]`` from the same settled tank state,
then track zero for a configurable recovery interval. Physics runs at 60 Hz
and control at the models' native 20 Hz. ArduSub-style roll/pitch STABILIZE
remains enabled, with no depth or yaw hold.

The ARX archive is the real-robot controller's causal history model, so this
runner uses the pure-math ``control_v2`` MPC implementation from
``EDMDc_bluerov`` read-only. It verifies that the ARX base dictionary is
byte-identical to the explicitly supplied Gaussian model before running.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_GAUSSIAN = Path(
    "/home/miaodong/Documents/underwater-manipulation-ardusub-sim/"
    "Gaussian_dictionary/model/gaussian_2rbf_pilot.npz"
)
DEFAULT_ARX = (
    PROJECT / "Gaussian_dictionary" / "model"
    / "arx10_gaussian_free20_pilot.npz"
)
DEFAULT_CONTROL_ROOT = Path("/home/miaodong/Documents/EDMDc_bluerov")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gaussian-model", type=Path,
                        default=DEFAULT_GAUSSIAN)
    parser.add_argument("--arx-model", type=Path, default=DEFAULT_ARX)
    parser.add_argument("--real-control-root", type=Path,
                        default=DEFAULT_CONTROL_ROOT,
                        help="Read-only repository containing the validated "
                             "control_v2 Gaussian/ARX MPC math.")
    parser.add_argument("--target-u", type=float, default=0.2,
                        help="Body surge velocity target in m/s (default 0.2).")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Active reference duration in seconds (default 5).")
    parser.add_argument(
        "--post-zero-duration", type=float, default=3.0,
        help="Zero-reference recovery after the active segment in seconds "
             "(default 3).",
    )
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--physics-hz", type=float, default=60.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=None,
                        help="Output prefix; defaults under "
                             "Gaussian_dictionary/results/.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _controlled_state(nu6: np.ndarray) -> np.ndarray:
    return np.asarray(nu6, dtype=float)[[0, 1, 2, 5]]


def _metrics(
    time_s: np.ndarray,
    state4: np.ndarray,
    command4: np.ndarray,
    pose: np.ndarray,
    roll_pitch: np.ndarray,
    target_u: float,
    hold_s: float,
) -> dict[str, float]:
    u = state4[:, 0]
    reference = np.where(time_s <= hold_s, target_u, 0.0)
    error = u - reference
    tracking = time_s <= hold_s
    tracking_tail = tracking & (time_s >= max(0.0, hold_s - 1.0))
    zero_tail = time_s >= max(hold_s, float(time_s[-1]) - 1.0)
    threshold = 0.9 * target_u
    reached = np.flatnonzero(tracking & (u >= threshold))
    rise_time = float(time_s[reached[0]]) if reached.size else float("nan")
    return {
        "u_full_reference_rmse_mps": float(np.sqrt(np.mean(error ** 2))),
        "u_tracking_rmse_mps": float(np.sqrt(np.mean(
            (u[tracking] - target_u) ** 2))),
        "u_tracking_mae_mps": float(np.mean(
            np.abs(u[tracking] - target_u))),
        "u_at_release_mps": float(u[np.flatnonzero(tracking)[-1]]),
        "u_final_mps": float(u[-1]),
        "u_tracking_last1s_mean_mps": float(np.mean(u[tracking_tail])),
        "u_tracking_last1s_std_mps": float(np.std(u[tracking_tail])),
        "u_zero_last1s_mean_mps": float(np.mean(u[zero_tail])),
        "u_zero_last1s_std_mps": float(np.std(u[zero_tail])),
        "u_peak_mps": float(np.max(u[tracking])),
        "u_overshoot_mps": float(max(0.0, np.max(u) - target_u)),
        "rise_time_90_s": rise_time,
        "surge_cmd_rms": float(np.sqrt(np.mean(command4[:, 0] ** 2))),
        "surge_cmd_peak": float(np.max(np.abs(command4[:, 0]))),
        "max_abs_v_mps": float(np.max(np.abs(state4[:, 1]))),
        "max_abs_w_mps": float(np.max(np.abs(state4[:, 2]))),
        "max_abs_r_radps": float(np.max(np.abs(state4[:, 3]))),
        "max_abs_roll_deg": float(np.max(np.abs(
            np.rad2deg(roll_pitch[:, 0])))),
        "max_abs_pitch_deg": float(np.max(np.abs(
            np.rad2deg(roll_pitch[:, 1])))),
        "forward_distance_m": float(pose[-1, 0] - pose[0, 0]),
        "finite": float(
            np.all(np.isfinite(state4))
            and np.all(np.isfinite(command4))
            and np.all(np.isfinite(pose))
        ),
    }


def _save_plot(
    png_path: Path,
    runs: dict[str, dict[str, np.ndarray]],
    target_u: float,
    hold_s: float,
    total_s: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"gaussian": "tab:blue", "arx10": "tab:orange"}
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for name, run in runs.items():
        t = run["time_s"]
        state = run["state4"]
        command = run["command4"]
        axes[0].plot(t, state[:, 0], color=colors[name], label=name)
        axes[1].plot(t, command[:, 0], color=colors[name], label=name)
        axes[2].plot(t, state[:, 1], color=colors[name],
                     label=f"{name} v")
        axes[2].plot(t, state[:, 2], color=colors[name], linestyle="--",
                     label=f"{name} w")
        axes[2].plot(t, state[:, 3], color=colors[name], linestyle=":",
                     label=f"{name} r")
    axes[0].plot(
        [0.0, hold_s, hold_s, total_s],
        [target_u, target_u, 0.0, 0.0],
        color="black", linestyle="--", linewidth=1.0,
        label="u reference",
    )
    axes[0].axvline(hold_s, color="0.4", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("surge u [m/s]")
    axes[1].set_ylabel("surge command")
    axes[2].set_ylabel("v,w [m/s]; r [rad/s]")
    axes[2].set_xlabel("active-control time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", ncol=2)
    fig.suptitle(
        "Isaac 6 MPC surge tracking and zero-reference recovery "
        "with ArduSub STABILIZE"
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def _save_velocity_tracking_plot(
    png_path: Path,
    runs: dict[str, dict[str, np.ndarray]],
    target_u: float,
    hold_s: float,
    total_s: float,
) -> None:
    """Match the real-robot MPC4 u/v/w/r tracking-figure presentation."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    axis_names = ("u", "v", "w", "r")
    axis_labels = (
        "u — surge [m/s]",
        "v — sway-right [m/s]",
        "w — heave-up [m/s]",
        "r — yaw-right [rad/s]",
    )
    colors = {"gaussian": "#4d98cb", "arx10": "#08306b"}
    labels = {
        "gaussian": "Gaussian EDMDc",
        "arx10": "ARX(10)-Gaussian",
    }
    ink = "#263238"
    surface = "#fcfcfb"
    grid_hue = "#d7dee3"
    zero_hue = "#9aa6ad"
    reference_hue = "#b66a00"

    figure, axes_array = plt.subplots(
        2, 2, figsize=(15.2, 10.2), sharex=True, facecolor="white"
    )
    axes = list(axes_array.flat)
    reference_t = np.asarray([0.0, hold_s, hold_s, total_s])

    for component, axis in enumerate(axes):
        axis.set_facecolor(surface)
        axis.grid(True, color=grid_hue, linewidth=0.75, alpha=0.85)
        axis.axhline(0.0, color=zero_hue, linewidth=0.8, zorder=1)

        values = [np.asarray([0.0])]
        for name, run in runs.items():
            measured = run["state4"][:, component]
            values.append(measured)
            axis.plot(
                run["time_s"], measured,
                color=colors[name], linewidth=1.6, alpha=0.94,
                solid_capstyle="round", zorder=2.2,
            )

        reference_y = (
            np.asarray([target_u, target_u, 0.0, 0.0])
            if component == 0 else np.zeros(4)
        )
        values.append(reference_y)
        axis.plot(
            reference_t, reference_y,
            color=reference_hue, linewidth=1.2,
            linestyle=(0, (1.0, 2.0)), zorder=2.8,
        )
        combined = np.concatenate(values)
        low = float(np.min(combined))
        high = float(np.max(combined))
        span = max(high - low, 0.02 if component < 3 else 0.01)
        axis.set_ylim(low - 0.10 * span, high + 0.10 * span)
        axis.set_xlim(0.0, total_s)
        axis.set_title(
            axis_names[component], loc="left", color=ink,
            fontsize=14, fontweight="semibold",
        )
        axis.set_ylabel(axis_labels[component], color=ink, fontsize=12)
        axis.tick_params(colors=ink, labelsize=10.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#aeb8bf")

    for axis in axes[2:]:
        axis.set_xlabel(
            "Time since MPC engagement [s]", color=ink, fontsize=12
        )

    handles = [
        Line2D([], [], color=colors[name], linewidth=2.0, label=labels[name])
        for name in ("gaussian", "arx10")
    ]
    handles.append(Line2D(
        [], [], color=reference_hue, linewidth=1.2,
        linestyle=(0, (1.0, 2.0)), label="reference actually applied",
    ))
    figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=3, frameon=False, fontsize=10.5, columnspacing=1.3,
        handlelength=2.4,
    )
    figure.suptitle(
        "Isaac 6 MPC4 velocity tracking", color=ink, fontsize=17, y=0.985
    )
    figure.text(
        0.5, 0.949,
        (
            "Gaussian EDMDc vs ARX(10)-Gaussian; logged peak reference "
            f"u={target_u:.2f}, v=0.00, w=0.00, r=0.00"
        ),
        ha="center", va="center", color="#59636a", fontsize=11,
    )
    figure.text(
        0.5, 0.018,
        (
            "Source: Isaac 6 closed-loop state feedback; w is positive-up, "
            "r is yaw-right. No resampling, smoothing, or clipping."
        ),
        ha="center", va="bottom", color="#59636a", fontsize=9.5,
    )
    figure.subplots_adjust(
        left=0.085, right=0.985, bottom=0.085, top=0.865,
        hspace=0.22, wspace=0.18,
    )
    figure.savefig(png_path, dpi=160, facecolor="white")
    plt.close(figure)


def _save_results(
    prefix: Path,
    *,
    args: argparse.Namespace,
    gaussian_path: Path,
    arx_path: Path,
    gaussian_hash: str,
    control_dt: float,
    physics_dt: float,
    runs: dict[str, dict[str, np.ndarray]],
    initial_states: dict[str, np.ndarray],
) -> tuple[Path, Path, Path, Path]:
    """Write results before SimulationApp.close(), which terminates Kit."""

    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = prefix.with_suffix(".npz")
    csv_path = prefix.with_suffix(".csv")
    png_path = prefix.with_suffix(".png")
    velocity_png_path = prefix.with_name(
        f"{prefix.name}_velocity_tracking"
    ).with_suffix(".png")

    payload: dict[str, np.ndarray] = {
        "target_u_mps": np.asarray(float(args.target_u)),
        "duration_s": np.asarray(float(args.duration)),
        "post_zero_duration_s": np.asarray(float(args.post_zero_duration)),
        "control_dt_s": np.asarray(control_dt),
        "physics_dt_s": np.asarray(physics_dt),
        "gaussian_model_path": np.asarray(str(gaussian_path)),
        "arx_model_path": np.asarray(str(arx_path)),
        "shared_base_sha256": np.asarray(gaussian_hash),
        "state_names": np.asarray(["u", "v", "w", "r"]),
        "command_names": np.asarray(["surge", "sway", "heave", "yaw"]),
    }
    for name, run in runs.items():
        for key, value in run.items():
            if key != "metrics":
                payload[f"{name}_{key}"] = np.asarray(value)
        payload[f"{name}_initial_state"] = initial_states[name]
        for key, value in run["metrics"].items():
            payload[f"{name}_metric_{key}"] = np.asarray(value)
    np.savez_compressed(npz_path, **payload)

    metric_names = list(runs["gaussian"]["metrics"])
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", *metric_names])
        writer.writeheader()
        for name, run in runs.items():
            writer.writerow({"model": name, **run["metrics"]})
    _save_plot(
        png_path, runs, float(args.target_u), float(args.duration),
        float(args.duration + args.post_zero_duration),
    )
    _save_velocity_tracking_plot(
        velocity_png_path, runs, float(args.target_u), float(args.duration),
        float(args.duration + args.post_zero_duration),
    )
    return npz_path, csv_path, png_path, velocity_png_path


def main() -> int:
    args = parse_args()
    gaussian_path = args.gaussian_model.resolve()
    arx_path = args.arx_model.resolve()
    control_root = args.real_control_root.resolve()
    for path in (gaussian_path, arx_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (control_root / "control_v2").is_dir():
        raise FileNotFoundError(control_root / "control_v2")

    physics_per_control = int(round(args.physics_hz / args.control_hz))
    if not np.isclose(
            physics_per_control * args.control_hz, args.physics_hz):
        raise ValueError("physics_hz/control_hz must be an integer")
    control_dt = 1.0 / float(args.control_hz)
    physics_dt = 1.0 / float(args.physics_hz)
    if args.duration <= 0.0 or args.post_zero_duration < 0.0:
        raise ValueError("duration must be positive and post-zero duration "
                         "must be nonnegative")
    total_duration = float(args.duration + args.post_zero_duration)
    n_control = int(round(total_duration / control_dt))

    sys.path.insert(0, str(control_root))
    from control_v2.vel_mpc_controller import (  # noqa: E402
        GaussianEDMDcModel,
        VelocityProfile,
    )
    from control_v2.vel_mpc4_controller import QuadVelocityMPC  # noqa: E402
    from control_v2.vel_mpc_mz_controller import (  # noqa: E402
        ArxLiftedModel,
        MzQuadMPC,
    )

    profile = VelocityProfile(
        u_ref=float(args.target_u),
        v_ref=0.0, w_ref=0.0, r_ref=0.0,
        hold_s=float(args.duration),
        settle_s=float(args.post_zero_duration),
    )
    gaussian_controller = QuadVelocityMPC(
        GaussianEDMDcModel(str(gaussian_path)), profile=profile)
    arx_model = ArxLiftedModel(str(arx_path))
    arx_controller = MzQuadMPC(arx_model, profile=profile)

    gaussian_hash = _sha256(gaussian_path)
    arx_base_hash = _sha256(Path(arx_model.base.path).resolve())
    if gaussian_hash != arx_base_hash:
        raise RuntimeError(
            "ARX base dictionary does not match --gaussian-model: "
            f"{arx_model.base.path}"
        )
    if not np.isclose(arx_model.dt, control_dt):
        raise ValueError(
            f"ARX dt={arx_model.dt:g} does not match control dt={control_dt:g}"
        )

    print(f"[mpc-compare] Gaussian: {gaussian_path}")
    print(f"[mpc-compare] ARX(10):  {arx_path}")
    print(f"[mpc-compare] shared base SHA256: {gaussian_hash}")
    print(f"[mpc-compare] target u={args.target_u:g} m/s for "
          f"{args.duration:g} s, then zero for "
          f"{args.post_zero_duration:g} s; "
          f"control={args.control_hz:g} Hz, "
          f"physics={args.physics_hz:g} Hz")

    if args.out_prefix is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = (
            PROJECT / "Gaussian_dictionary" / "results"
            / f"mpc_surge_compare_{stamp}"
        )
    else:
        prefix = args.out_prefix.resolve()

    from EDMDc.isaac_scene import GripperScene  # noqa: E402

    env_usd = PROJECT / "assets" / "environment_tank.usda"
    scene = GripperScene(
        dt=physics_dt,
        headless=not args.gui,
        env_usd=str(env_usd),
        retrim_level=True,
        ardusub_stabilize=True,
    )
    runs: dict[str, dict[str, np.ndarray]] = {}
    initial_states: dict[str, np.ndarray] = {}
    try:
        for name, controller in (
            ("gaussian", gaussian_controller),
            ("arx10", arx_controller),
        ):
            print(f"[mpc-compare] settling {name} trial ...")
            scene.reset_to_spawn()
            controller.reset()
            settled = scene.settle(max_s=8.0, min_s=4.0, tol=0.02)
            initial_states[name] = np.concatenate([
                settled.pos_w, settled.nu,
                [settled.roll, settled.pitch, settled.yaw],
            ])

            # The ARX controller requires m-1 real causal blocks. Give both
            # trials the same nine zero-command control periods before the
            # five-second reference begins.
            warmup_cycles = int(arx_model.m - 1)
            for k in range(warmup_cycles):
                state = scene.read_state()
                if name == "arx10":
                    decision = controller.step(
                        _controlled_state(state.nu), k * control_dt)
                    warm_cmd = np.asarray(decision.effective, dtype=float)
                    if np.max(np.abs(warm_cmd)) > 1e-12:
                        raise RuntimeError(
                            "ARX causal warm-up produced nonzero command")
                else:
                    warm_cmd = np.zeros(4)
                for _ in range(physics_per_control):
                    scene.apply_wrench(warm_cmd)

            times = []
            states = []
            commands = []
            poses = []
            roll_pitch = []
            solver_cost = []
            saturation = []
            for k in range(n_control):
                before = scene.read_state()
                x4 = _controlled_state(before.nu)
                if name == "gaussian":
                    decision = controller.step(x4, k * control_dt)
                else:
                    decision = controller.step(
                        x4, (warmup_cycles + k) * control_dt)
                command = np.asarray(decision.effective, dtype=float)
                if command.shape != (4,) or not np.all(np.isfinite(command)):
                    raise RuntimeError(
                        f"{name} returned invalid command {command}")
                for _ in range(physics_per_control):
                    scene.apply_wrench(command)
                after = scene.read_state()

                times.append((k + 1) * control_dt)
                states.append(_controlled_state(after.nu))
                commands.append(command)
                poses.append(after.pos_w.copy())
                roll_pitch.append([after.roll, after.pitch])
                solver_cost.append(float(decision.cost))
                saturation.append(np.asarray(
                    decision.saturated, dtype=bool))

            run = {
                "time_s": np.asarray(times),
                "state4": np.asarray(states),
                "command4": np.asarray(commands),
                "pose_w": np.asarray(poses),
                "roll_pitch": np.asarray(roll_pitch),
                "solver_cost": np.asarray(solver_cost),
                "saturated": np.asarray(saturation),
            }
            run["metrics"] = _metrics(
                run["time_s"], run["state4"], run["command4"],
                run["pose_w"], run["roll_pitch"], float(args.target_u),
                float(args.duration))
            runs[name] = run
            m = run["metrics"]
            print(
                f"[mpc-compare] {name:8s} "
                f"tracking_RMSE={m['u_tracking_rmse_mps']:.4f} m/s, "
                f"tracking_last1s={m['u_tracking_last1s_mean_mps']:.4f}±"
                f"{m['u_tracking_last1s_std_mps']:.4f}, "
                f"zero_final={m['u_final_mps']:.4f}, "
                f"peak={m['u_peak_mps']:.4f}, "
                f"cmd_peak={m['surge_cmd_peak']:.3f}, "
                f"distance={m['forward_distance_m']:.3f} m"
            )

        npz_path, csv_path, png_path, velocity_png_path = _save_results(
            prefix,
            args=args,
            gaussian_path=gaussian_path,
            arx_path=arx_path,
            gaussian_hash=gaussian_hash,
            control_dt=control_dt,
            physics_dt=physics_dt,
            runs=runs,
            initial_states=initial_states,
        )
        print(f"[mpc-compare] results: {npz_path}")
        print(f"[mpc-compare] summary: {csv_path}")
        print(f"[mpc-compare] plot:    {png_path}")
        print(f"[mpc-compare] velocity tracking: {velocity_png_path}")
    finally:
        scene.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
