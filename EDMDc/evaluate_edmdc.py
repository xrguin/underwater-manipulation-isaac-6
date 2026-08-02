"""Evaluate a trained EDMDc model against a held-out dataset.

Reports + plots (all saved under EDMDc/data/plots/; windows kept open):

  1. One-step prediction quality — six predicted-vs-true scatter subplots
     (one per body-velocity axis) with y=x line + per-axis RMSE in the title.
     File: onestep_<TS>.png.

  2. Multi-step rollout RMSE vs horizon — log-y line plot of total and
     per-axis RMSE at user-specified horizons.  Every valid sliding origin is
     evaluated, rather than only the first state of each trajectory.
     File: rollout_sliding_<TS>.png.

  3. Open-loop velocity tracking on a single picked trajectory — six
     time-series subplots overlaying recorded ν against the EDMDc rollout
     under the same control sequence.
     File: velocity_compare_corrected_<TS>.png.

  4. Derived open-loop 3D path comparison — both body-velocity sequences are
     integrated with nonsingular quaternion kinematics.  When possible, the
     deterministic collector initial pose is recovered from dataset metadata.
     The plot is explicitly labelled as velocity-derived because the current
     dataset schema does not store pose samples.
     Files: trajectory_3d_quaternion_<H>s_<TS>.png for the same maximum
     horizon as the rollout chart, plus trajectory_3d_quaternion_full_<TS>.png
     when the selected trajectory is longer.

  5. Derived pose separation — position and attitude separation accumulated by
     the two quaternion integrations.
     File: derived_pose_error_<TS>.png.

All plot windows stay open via plt.show() until you close them.

Usage:
    python -m EDMDc.evaluate_edmdc \\
        EDMDc/model/edmdc_<TS>.npz \\
        EDMDc/data/numpy/fossen_<test-TS>.npz \\
        [--traj-idx 0]
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
        "matplotlib is required for EDMDc.evaluate_edmdc. "
        "Install with `pip install matplotlib`."
    ) from e

from ._common import default_yaml, project_root
from .edmdc import NU_NAMES, decode_nu, lift, rollout
from .fossen_integrator import FossenIntegratorParams, RHO_WATER, step_fossen


# Axis name -> 6-vector index for synthetic-input modes.
AXIS_IDX = {"Fx": 0, "Fy": 1, "Fz": 2, "Tx": 3, "Ty": 4, "Tz": 5}


_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def _plot_one_step(nu_true: np.ndarray, nu_pred: np.ndarray,
                   model_stem: str, test_stem: str,
                   out_path: Path) -> None:
    """Six pred-vs-true scatter subplots (one per axis) + y=x line."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes_flat = axes.flatten()
    for i, (ax, name) in enumerate(zip(axes_flat, NU_NAMES)):
        true_i = nu_true[:, i]
        pred_i = nu_pred[:, i]
        rmse = float(np.sqrt(np.mean((true_i - pred_i) ** 2)))

        ax.scatter(true_i, pred_i, s=2, alpha=0.15, color="C0",
                   rasterized=True, edgecolors="none")
        lim_lo = float(min(true_i.min(), pred_i.min()))
        lim_hi = float(max(true_i.max(), pred_i.max()))
        pad = 0.05 * (lim_hi - lim_lo + 1e-9)
        lim_lo -= pad; lim_hi += pad
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                "k--", alpha=0.6, linewidth=1, label="y = x")
        ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel(f"true  {name}")
        ax.set_ylabel(f"pred  {name}")
        ax.set_title(f"{name}:  RMSE = {rmse:.4f}")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"EDMDc one-step prediction (pred vs true)\n"
        f"model {model_stem}  /  test {test_stem}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[eval] wrote one-step plot to {out_path}")


def _build_synthetic_tau(
    mode: str, axis: str, amp: float, on_seconds: float,
    total_seconds: float, dt: float,
) -> np.ndarray:
    """Synthetic single-axis control sequence for the velocity-tracking probe.

    mode = "step": tau[axis] = amp for 0 <= t < on_seconds, then 0.
    """
    if axis not in AXIS_IDX:
        raise ValueError(f"--axis must be one of {list(AXIS_IDX)}; got {axis!r}.")
    n_total = int(round(total_seconds / dt))
    n_on    = int(round(on_seconds / dt))
    tau = np.zeros((n_total, 6), dtype=np.float64)
    if mode == "step":
        tau[:n_on, AXIS_IDX[axis]] = amp
    else:
        raise ValueError(f"Unsupported synthetic input mode: {mode!r}.")
    return tau


def _fossen_rollout(
    fparams: FossenIntegratorParams,
    eta0: np.ndarray, nu0: np.ndarray,
    tau_seq: np.ndarray, dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll Fossen forward through `tau_seq` from (eta0, nu0).

    Returns:
        nu_seq:  (N + 1, 6) — body velocity per step (nu_seq[0] = nu0).
        eta_seq: (N + 1, 6) — pose per step (eta_seq[0] = eta0).
    """
    N = tau_seq.shape[0]
    nu_seq  = np.zeros((N + 1, 6), dtype=np.float64)
    eta_seq = np.zeros((N + 1, 6), dtype=np.float64)
    nu_seq[0]  = nu0
    eta_seq[0] = eta0
    for k in range(N):
        eta_seq[k + 1], nu_seq[k + 1] = step_fossen(
            eta_seq[k], nu_seq[k], tau_seq[k], fparams, dt,
        )
    return nu_seq, eta_seq


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for wxyz quaternions."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def _quat_from_euler_zyx(euler: np.ndarray) -> np.ndarray:
    """ZYX roll/pitch/yaw to a body-to-world wxyz quaternion."""
    roll, pitch, yaw = np.asarray(euler, dtype=np.float64)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ], dtype=np.float64)


def _quat_to_euler_zyx(q: np.ndarray) -> np.ndarray:
    """Body-to-world wxyz quaternion to principal ZYX Euler angles."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    roll = np.arctan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
    sin_pitch = np.clip(2.0 * (w*y - z*x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    yaw = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Body-to-world rotation matrix from a normalized wxyz quaternion."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z), 2.0*(x*z + w*y)],
        [2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)],
        [2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y)],
    ], dtype=np.float64)


def _rotation_increment(omega_body: np.ndarray, seconds: float) -> np.ndarray:
    """Quaternion exponential for a constant body angular rate."""
    rotation_vector = np.asarray(omega_body, dtype=np.float64) * float(seconds)
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return np.array([1.0, *(0.5 * rotation_vector)], dtype=np.float64)
    axis = rotation_vector / angle
    return np.concatenate(([np.cos(angle / 2.0)], axis * np.sin(angle / 2.0)))


def _integrate_kinematics_quaternion(
    nu_seq: np.ndarray,
    dt: float,
    eta0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate body twist with nonsingular quaternion kinematics.

    A constant-body-rate quaternion exponential advances orientation over each
    sample.  Translation uses the interval midpoint attitude.  This avoids the
    tan(theta)/cos(theta) singularity in the former Euler-rate integrator.

    Args:
        nu_seq: (N, 6) body velocity per step.
        dt:     timestep [s].
        eta0:   (6,) starting pose; default zeros.

    Returns:
        eta_seq:  (N + 1, 6) [x, y, z, phi, theta, psi] in the dataset world
                  frame. Euler angles are principal-value views of quaternion
                  orientation and are not integrated directly.
        quat_seq: (N + 1, 4) normalized body-to-world wxyz quaternions.
    """
    nu_seq = np.asarray(nu_seq, dtype=np.float64)
    if nu_seq.ndim != 2 or nu_seq.shape[1] != 6:
        raise ValueError(f"nu_seq must have shape (N, 6), got {nu_seq.shape}")
    N = nu_seq.shape[0]
    eta = np.zeros((N + 1, 6), dtype=np.float64)
    quat = np.zeros((N + 1, 4), dtype=np.float64)
    if eta0 is not None:
        eta[0] = np.asarray(eta0, dtype=np.float64)
    quat[0] = _quat_from_euler_zyx(eta[0, 3:6])
    for k in range(N):
        omega = nu_seq[k, 3:6]
        q_mid = _quat_multiply(quat[k], _rotation_increment(omega, 0.5 * dt))
        q_mid /= np.linalg.norm(q_mid)
        eta[k + 1, 0:3] = (
            eta[k, 0:3] + dt * (_quat_to_rotmat(q_mid) @ nu_seq[k, 0:3])
        )
        quat[k + 1] = _quat_multiply(
            quat[k], _rotation_increment(omega, dt),
        )
        quat[k + 1] /= np.linalg.norm(quat[k + 1])
        eta[k + 1, 3:6] = _quat_to_euler_zyx(quat[k + 1])
    return eta, quat


def _dataset_initial_eta(data, chosen_idx: int) -> tuple[np.ndarray, str]:
    """Return the best available initial pose and a provenance description."""
    traj_idx = np.asarray(data["traj_idx"])
    selected = np.flatnonzero(traj_idx == chosen_idx)
    if selected.size == 0:
        raise ValueError(f"trajectory {chosen_idx} is not present")

    # Some older/pilot datasets contain pose even though the current collector
    # deliberately retains the six-state X/U/X_next schema.
    for key in ("Eta", "eta", "pose"):
        if key in data.files:
            pose = np.asarray(data[key], dtype=np.float64)
            if pose.ndim == 2 and pose.shape[0] == traj_idx.shape[0] and pose.shape[1] >= 6:
                return pose[selected[0], :6].copy(), f"recorded {key}[0]"

    if "spawn_flat_pool" in data.files and bool(data["spawn_flat_pool"]):
        x_world = float(data["spawn_x_world"]) if "spawn_x_world" in data.files else 0.0
        return np.array([x_world, 0.0, -1.0, 0.0, 0.0, 0.0]), "collector fixed spawn"

    required = {"seed", "n_attempts", "n_committed"}
    if required.issubset(data.files):
        n_attempts = int(data["n_attempts"])
        n_committed = int(data["n_committed"])
        # With no discarded attempts, committed trajectory label == RNG index.
        # The collector has no CLI overrides for these IC distribution values.
        if n_attempts == n_committed:
            from .collect_fossen import (
                CollectionConfig, sample_initial_eta_v, trajectory_rng,
            )
            cfg = CollectionConfig(
                n_trajectories=n_committed,
                episode_seconds=float(data["episode_seconds"]),
                dt=float(data["dt"]),
                seed=int(data["seed"]),
            )
            eta0, _ = sample_initial_eta_v(
                trajectory_rng(cfg.seed, chosen_idx), cfg,
            )
            return eta0, "deterministically reconstructed collector IC"

    return np.zeros(6, dtype=np.float64), "unknown (zero fallback)"


def _plot_velocity_compare(t: np.ndarray, nu_truth: np.ndarray, nu_pred: np.ndarray,
                           traj_idx: int, model_stem: str, test_stem: str,
                           out_path: Path, reference_label: str) -> None:
    """Six body-velocity/rate traces for one open-loop rollout."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    axes_flat = axes.flatten()
    for i, (ax, name) in enumerate(zip(axes_flat, NU_NAMES)):
        ax.plot(t, nu_truth[:, i], linewidth=2, color="C0", label=reference_label)
        ax.plot(t, nu_pred[:, i], linewidth=1.5, color="C1",
                linestyle="--",                       label="EDMDc rollout")
        ax.set_title(f"{name}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Open-loop body velocity/rate rollout — trajectory {traj_idx}\n"
        f"model {model_stem}  /  test {test_stem}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[eval] wrote velocity-compare plot to {out_path}")


def _plot_trajectory_3d(eta_truth: np.ndarray, eta_pred: np.ndarray,
                        traj_idx: int, model_stem: str, test_stem: str,
                        out_path: Path, initial_pose_source: str,
                        reference_label: str, duration_s: float) -> None:
    """Plot quaternion-integrated paths without claiming recorded position."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(eta_truth[:, 0], eta_truth[:, 1], eta_truth[:, 2],
            linewidth=2, color="C0", label=f"{reference_label}, integrated")
    ax.plot(eta_pred[:,  0], eta_pred[:,  1], eta_pred[:,  2],
            linewidth=1.5, color="C1", linestyle="--", label="EDMDc, integrated")
    ax.scatter(*eta_truth[0,  :3], color="green",  s=80, marker="o", label="start")
    ax.scatter(*eta_truth[-1, :3], color="C0", s=80, marker="s", label="reference end")
    ax.scatter(*eta_pred[-1, :3], color="C1", s=80, marker="^", label="EDMDc end")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_zlabel("world z [m, up]")
    ax.set_title(
        f"Velocity-derived path over {duration_s:g} s "
        f"(quaternion kinematics; pose not recorded)\n"
        f"trajectory {traj_idx} · IC: {initial_pose_source} · model {model_stem}",
        fontsize=11,
    )
    xyz = np.vstack([eta_truth[:, :3], eta_pred[:, :3]])
    span = np.maximum(np.ptp(xyz, axis=0), 1e-3)
    ax.set_box_aspect(span)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[eval] wrote 3D trajectory plot to {out_path}")


def _plot_rollout(hs: list[int], rmse_total: list[float],
                  rmse_axis: np.ndarray, dt: float,
                  model_stem: str, test_stem: str,
                  out_path: Path, sample_counts: list[int]) -> None:
    """Multi-step rollout RMSE vs horizon, log-y, total + per-axis."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.semilogy(hs, rmse_total, "-o", linewidth=2, label="total", color="black")
    for i, n in enumerate(NU_NAMES):
        ax.semilogy(hs, rmse_axis[:, i], "--", label=n, alpha=0.75)

    # Secondary x-axis for seconds.
    ax2 = ax.secondary_xaxis(
        "top", functions=(lambda x: x * dt, lambda x: x / dt),
    )
    ax2.set_xlabel("rollout horizon (seconds)")

    ax.set_xlabel("rollout horizon (steps)")
    ax.set_ylabel("RMSE")
    ax.set_title(
        f"EDMDc multi-step rollout RMSE (all valid sliding origins)\n"
        f"model {model_stem}  /  test {test_stem}",
        fontsize=11,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9, ncol=2)
    if sample_counts:
        fig.text(
            0.5, 0.01,
            f"Valid origins per horizon: {min(sample_counts):,} to "
            f"{max(sample_counts):,}",
            ha="center", fontsize=9,
        )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=120)
    print(f"[eval] wrote rollout plot to {out_path}")


def _plot_derived_pose_error(
    t: np.ndarray,
    eta_reference: np.ndarray,
    eta_pred: np.ndarray,
    quat_reference: np.ndarray,
    quat_pred: np.ndarray,
    model_stem: str,
    out_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot separation of two pose reconstructions derived from body twists."""
    position_error = np.linalg.norm(eta_pred[:, :3] - eta_reference[:, :3], axis=1)
    dots = np.abs(np.sum(quat_reference * quat_pred, axis=1))
    attitude_error_deg = np.rad2deg(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True)
    axes[0].plot(t, position_error, color="C1", linewidth=2)
    axes[0].set_ylabel("position separation [m]")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(
        f"Derived pose separation from quaternion integration\nmodel {model_stem}",
        fontsize=11,
    )
    axes[1].plot(t, attitude_error_deg, color="C2", linewidth=2)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("attitude separation [deg]")
    axes[1].grid(alpha=0.3)
    fig.text(
        0.5, 0.01,
        "Diagnostic only: the dataset stores body velocity/rate, not pose.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=120)
    print(f"[eval] wrote derived-pose-error plot to {out_path}")
    return position_error, attitude_error_deg


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("model", type=Path, help="Trained EDMDc model .npz.")
    ap.add_argument("data",  type=Path, help="Test dataset .npz from collect_fossen.")
    ap.add_argument(
        "--horizons", nargs="+", type=int,
        default=[1, 5, 10, 30, 60, 120, 300],
        help="Step horizons to evaluate (default: 1 5 10 30 60 120 300).",
    )
    ap.add_argument(
        "--max-trajs", type=int, default=None,
        help="Cap number of test trajectories used for rollout.",
    )
    ap.add_argument(
        "--traj-idx", type=int, default=0,
        help="Which trajectory in the test set to use for the velocity-compare "
             "and 3D path plots when --input-mode=dataset (default 0).",
    )
    # ----- synthetic-input probe (replaces the dataset trajectory for the
    #       velocity-compare + 3D plots when set to anything but 'dataset') -----
    ap.add_argument(
        "--input-mode", choices=("dataset", "step"), default="dataset",
        help="Control input for the velocity-compare + 3D plots. 'dataset' "
             "uses the APRBS tau saved with the test trajectory (default). "
             "'step' applies a synthetic single-axis step from rest.",
    )
    ap.add_argument(
        "--axis", default="Fx", choices=tuple(AXIS_IDX),
        help="Axis to excite in step mode (default Fx).",
    )
    ap.add_argument(
        "--step-amp", type=float, default=50.0,
        help="Step magnitude in step mode (N or N*m). Default 50.",
    )
    ap.add_argument(
        "--step-on", type=float, default=2.0,
        help="Seconds the step is held ON before returning to zero. Default 2.",
    )
    ap.add_argument(
        "--step-total", type=float, default=5.0,
        help="Total episode length in step mode (default 5 s).",
    )
    ap.add_argument(
        "--mass-kg", type=float, default=13.5,
        help="Vehicle mass for the Fossen rollout in step mode (default 13.5).",
    )
    ap.add_argument(
        "--yaml", type=str, default=None,
        help="Vehicle YAML for the Fossen rollout in step mode. "
             "Default: EDMDc._common.default_yaml().",
    )
    ap.add_argument(
        "--no-show", action="store_true",
        help="Save plots but don't open the matplotlib windows.",
    )
    args = ap.parse_args()

    # ---- Load model ----------------------------------------------------------
    m = np.load(args.model, allow_pickle=True)
    A, B = m["A"], m["B"]
    eig_max = float(m["eig_max_abs"])
    print(f"[eval] model: A {A.shape}, B {B.shape}, eig_max(A) = {eig_max:.4f}")

    # ---- Load test data ------------------------------------------------------
    d = np.load(args.data)
    nu      = d["X"]
    nu_next = d["X_next"]
    tau     = d["U"]
    traj_idx = d["traj_idx"]
    step_idx = d["step_idx"]
    dt = float(d["dt"])
    n_test_trajs = int(d["n_trajectories"])
    print(f"[eval] test data: {nu.shape[0]:,} snapshots from {n_test_trajs} trajs "
          f"({args.data.name})")

    # ---- One-step prediction -------------------------------------------------
    Z      = lift(nu).T
    Z_pred = A @ Z + B @ tau.T
    nu_pred = decode_nu(Z_pred.T)
    err = nu_next - nu_pred
    rmse_1step = np.sqrt(np.mean(err ** 2, axis=0))
    print("[eval] 1-step nu RMSE:")
    for n, r in zip(NU_NAMES, rmse_1step):
        print(f"          {n}: {r:.4f}")
    print(f"          total: {float(np.sqrt(np.mean(err**2))):.4f}")

    # Mirror the data layout: if the test .npz lives under a named subdir of
    # data/ (e.g. data/with_gripper/...), write plots into the same-named
    # subdir of data/plots/ so each variant keeps its evaluation artifacts.
    plots_dir = project_root() / "EDMDc" / "data" / "plots"
    try:
        rel = args.data.resolve().relative_to((project_root() / "EDMDc" / "data").resolve())
        if len(rel.parts) > 1:
            plots_dir = plots_dir / rel.parts[0]
    except ValueError:
        pass
    plots_dir.mkdir(parents=True, exist_ok=True)
    onestep_path = plots_dir / f"onestep_corrected_{_RUN_TS}.png"
    _plot_one_step(nu_next, nu_pred,
                   model_stem=args.model.stem, test_stem=args.data.stem,
                   out_path=onestep_path)

    # ---- Multi-step rollout --------------------------------------------------
    unique_trajs = list(np.unique(traj_idx))
    if args.max_trajs is not None:
        unique_trajs = unique_trajs[: args.max_trajs]
    print(f"[eval] rolling out {len(unique_trajs)} trajectories")

    horizon_errs: dict[int, list[np.ndarray]] = {h: [] for h in args.horizons}
    for t in unique_trajs:
        sel = traj_idx == t
        order = np.argsort(step_idx[sel])
        nu_traj = nu[sel][order]
        nu_next_traj = nu_next[sel][order]
        tau_traj = tau[sel][order]
        steps_traj = step_idx[sel][order]

        # Evaluate within each contiguous segment, so a truncated/gapped
        # dataset never smuggles a reset across an open-loop rollout window.
        boundaries = np.flatnonzero(np.diff(steps_traj) != 1) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [len(steps_traj)]))
        for seg_start, seg_stop in zip(starts, stops):
            nu_seg = nu_traj[seg_start:seg_stop]
            nu_next_seg = nu_next_traj[seg_start:seg_stop]
            tau_seg = tau_traj[seg_start:seg_stop]
            n_steps = nu_seg.shape[0]
            for h in args.horizons:
                if not 1 <= h <= n_steps:
                    continue
                n_origins = n_steps - h + 1
                z_hat = lift(nu_seg[:n_origins]).T
                for j in range(h):
                    z_hat = A @ z_hat + B @ tau_seg[j:j + n_origins].T
                pred_h = decode_nu(z_hat.T)
                truth_h = nu_next_seg[h - 1:h - 1 + n_origins]
                horizon_errs[h].append(pred_h - truth_h)

    print("\n[eval] multi-step rollout RMSE vs horizon:")
    hs_with_data: list[int] = []
    rmse_total:   list[float] = []
    rmse_axis_h:  list[np.ndarray] = []
    for h in args.horizons:
        if not horizon_errs[h]:
            continue
        E = np.vstack(horizon_errs[h])
        rmse_axis = np.sqrt(np.mean(E ** 2, axis=0))
        rmse_tot  = float(np.sqrt(np.mean(E ** 2)))
        n = len(E)
        per_axis = ", ".join(f"{n_}={r_:.3f}" for n_, r_ in zip(NU_NAMES, rmse_axis))
        print(f"  H = {h:4d} ({h * dt:5.2f}s)  RMSE total = {rmse_tot:.4f}  "
              f"({per_axis}, n_origins = {n})")
        hs_with_data.append(h)
        rmse_total.append(rmse_tot)
        rmse_axis_h.append(rmse_axis)

    rollout_path = plots_dir / f"rollout_sliding_{_RUN_TS}.png"
    _plot_rollout(
        hs_with_data, rmse_total, np.asarray(rmse_axis_h),
        dt=dt, model_stem=args.model.stem, test_stem=args.data.stem,
        out_path=rollout_path,
        sample_counts=[sum(len(e) for e in horizon_errs[h]) for h in hs_with_data],
    )

    # ---- Open-loop velocity & 3D trajectory comparison ----------------------
    if args.input_mode == "dataset":
        if args.traj_idx not in np.unique(traj_idx):
            print(f"[eval] WARNING: traj-idx {args.traj_idx} not in dataset; "
                  f"falling back to first available ({int(unique_trajs[0])}).")
            chosen_idx = int(unique_trajs[0])
        else:
            chosen_idx = int(args.traj_idx)
        sel = traj_idx == chosen_idx
        nu_traj_truth = nu[sel]
        nu_next_truth = nu_next[sel]
        tau_traj      = tau[sel]
        N = nu_traj_truth.shape[0]
        print(f"\n[eval] open-loop compare on dataset trajectory {chosen_idx} "
              f"({N} steps = {N * dt:.2f}s)")

        # Fossen ground truth (already in data): prepend first nu, then nu_next.
        nu_truth_full = np.vstack([nu_traj_truth[0:1], nu_next_truth])
        # EDMDc rollout uses the same starting nu and the same tau sequence.
        nu_pred_full = rollout(A, B, nu_traj_truth[0], tau_traj)
        label_tag = f"dataset traj {chosen_idx}"
        reference_label = "Isaac recorded state"
        eta0, eta0_source = _dataset_initial_eta(d, chosen_idx)
        eta_truth_direct = None
        print(f"[eval] initial pose for derived path: {eta0.round(6).tolist()} "
              f"({eta0_source})")

    else:  # synthetic single-axis input (e.g., step)
        # Build the synthetic tau sequence.
        tau_traj = _build_synthetic_tau(
            mode=args.input_mode, axis=args.axis,
            amp=args.step_amp, on_seconds=args.step_on,
            total_seconds=args.step_total, dt=dt,
        )
        N = tau_traj.shape[0]
        # Load Fossen params and run from rest.
        yaml_path = args.yaml or str(default_yaml())
        fparams = FossenIntegratorParams.from_yaml(yaml_path, mass_kg=args.mass_kg)
        if bool(d.get("neutral_buoyancy", False)):
            fparams.volume = float(args.mass_kg) / RHO_WATER
            print(f"[eval] neutral buoyancy ON (matching training): "
                  f"volume override -> {fparams.volume:.6f} m^3")
        eta0 = np.zeros(6, dtype=np.float64)
        nu0  = np.zeros(6, dtype=np.float64)
        print(f"\n[eval] synthetic '{args.input_mode}' input on {args.axis} = "
              f"{args.step_amp:g} for {args.step_on:g}s, then 0 to "
              f"{args.step_total:g}s ({N} steps). Starting from rest.")

        nu_truth_full, eta_truth_direct = _fossen_rollout(
            fparams, eta0, nu0, tau_traj, dt,
        )
        nu_pred_full = rollout(A, B, nu0, tau_traj)
        chosen_idx = -1            # not a dataset trajectory
        label_tag = (f"step {args.axis}={args.step_amp:g} "
                     f"({args.step_on:g}s ON / {args.step_total:g}s total)")
        reference_label = "Fossen reference"
        eta0_source = "specified zero pose"

    t_axis = np.arange(N + 1) * dt
    velcmp_path = plots_dir / f"velocity_compare_corrected_{_RUN_TS}.png"
    _plot_velocity_compare(
        t_axis, nu_truth_full, nu_pred_full,
        traj_idx=chosen_idx,
        model_stem=args.model.stem,
        test_stem=label_tag if args.input_mode != "dataset" else args.data.stem,
        out_path=velcmp_path,
        reference_label=reference_label,
    )

    eta_truth_derived, quat_truth = _integrate_kinematics_quaternion(
        nu_truth_full[:-1], dt=dt, eta0=eta0,
    )
    eta_pred, quat_pred = _integrate_kinematics_quaternion(
        nu_pred_full[:-1], dt=dt, eta0=eta0,
    )
    # Synthetic Fossen probes have directly integrated pose; Isaac datasets do
    # not.  Keep quaternion-derived orientation for the separation diagnostic
    # so both curves use identical numerical kinematics.
    eta_truth_plot = (
        eta_truth_direct if eta_truth_direct is not None else eta_truth_derived
    )
    max_requested_horizon = max(
        (h for h in args.horizons if h > 0), default=N,
    )
    path_steps = min(N, max_requested_horizon)
    path_duration = path_steps * dt
    duration_tag = f"{path_duration:g}s".replace(".", "p")
    traj3d_path = (
        plots_dir / f"trajectory_3d_quaternion_{duration_tag}_{_RUN_TS}.png"
    )
    _plot_trajectory_3d(
        eta_truth_plot[:path_steps + 1], eta_pred[:path_steps + 1],
        traj_idx=chosen_idx,
        model_stem=args.model.stem,
        test_stem=label_tag if args.input_mode != "dataset" else args.data.stem,
        out_path=traj3d_path,
        initial_pose_source=eta0_source,
        reference_label=reference_label,
        duration_s=path_duration,
    )
    if path_steps < N:
        traj3d_full_path = (
            plots_dir / f"trajectory_3d_quaternion_full_{_RUN_TS}.png"
        )
        _plot_trajectory_3d(
            eta_truth_plot, eta_pred,
            traj_idx=chosen_idx,
            model_stem=args.model.stem,
            test_stem=(label_tag if args.input_mode != "dataset"
                       else args.data.stem),
            out_path=traj3d_full_path,
            initial_pose_source=eta0_source,
            reference_label=reference_label,
            duration_s=N * dt,
        )

    pose_error_path = plots_dir / f"derived_pose_error_{_RUN_TS}.png"
    position_error, attitude_error_deg = _plot_derived_pose_error(
        t_axis, eta_truth_derived, eta_pred, quat_truth, quat_pred,
        model_stem=args.model.stem, out_path=pose_error_path,
    )
    velocity_error = nu_pred_full - nu_truth_full
    print("\n[eval] full-trajectory diagnostics:")
    print(f"  body linear-velocity RMSE: "
          f"{float(np.sqrt(np.mean(velocity_error[:, :3] ** 2))):.4f} m/s")
    print(f"  body angular-rate RMSE: "
          f"{float(np.sqrt(np.mean(velocity_error[:, 3:] ** 2))):.4f} rad/s")
    print(f"  final derived position separation: {position_error[-1]:.4f} m")
    print(f"  maximum derived position separation: {position_error.max():.4f} m")
    print(f"  final derived attitude separation: {attitude_error_deg[-1]:.2f} deg")
    print("  NOTE: derived pose is a diagnostic integration; pose was not saved "
          "in this dataset.")

    # ---- Hold windows open until user closes them ---------------------------
    if not args.no_show:
        print("\n[eval] close the plot windows to exit.")
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
