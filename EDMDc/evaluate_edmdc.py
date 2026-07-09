"""Evaluate a trained EDMDc model against a held-out fossen .npz.

Reports + plots (all saved under EDMDc/data/plots/; windows kept open):

  1. One-step prediction quality — six predicted-vs-true scatter subplots
     (one per body-velocity axis) with y=x line + per-axis RMSE in the title.
     File: onestep_<TS>.png.

  2. Multi-step rollout RMSE vs horizon — log-y line plot of total and
     per-axis RMSE at user-specified horizons.
     File: rollout_<TS>.png.

  3. Open-loop velocity tracking on a single picked trajectory — six
     time-series subplots overlaying Fossen ground-truth ν against the
     EDMDc rollout under the same control sequence.
     File: velocity_compare_<TS>.png.

  4. Open-loop 3D path comparison — both ν sequences integrated through
     the kinematics η̇ = J(η)ν from η₀ = 0 and plotted in 3D.
     File: trajectory_3d_<TS>.png.

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


def _integrate_kinematics(nu_seq: np.ndarray, dt: float,
                          eta0: np.ndarray | None = None) -> np.ndarray:
    """Forward-Euler integrate body velocity through the kinematic transform
    η̇ = J(η) ν (ZYX intrinsic Euler convention).

    Args:
        nu_seq: (N, 6) body velocity per step.
        dt:     timestep [s].
        eta0:   (6,) starting pose; default zeros.

    Returns:
        eta_seq: (N + 1, 6) [x, y, z, phi, theta, psi] in world (NED) frame.
    """
    N = nu_seq.shape[0]
    eta = np.zeros((N + 1, 6), dtype=np.float64)
    if eta0 is not None:
        eta[0] = eta0
    for k in range(N):
        phi, theta, psi = eta[k, 3], eta[k, 4], eta[k, 5]
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth,  sth  = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi),   np.sin(psi)
        # Body -> world rotation (ZYX intrinsic Euler).
        R = np.array([
            [cth * cpsi, sphi*sth*cpsi - cphi*spsi, cphi*sth*cpsi + sphi*spsi],
            [cth * spsi, sphi*sth*spsi + cphi*cpsi, cphi*sth*spsi - sphi*cpsi],
            [-sth,       sphi * cth,                cphi * cth],
        ])
        # Body angular rate -> Euler-angle rate transform.
        tth = np.tan(theta) if abs(cth) > 1e-6 else 0.0
        sec_th = 1.0 / cth if abs(cth) > 1e-6 else 0.0
        T = np.array([
            [1.0, sphi * tth,    cphi * tth],
            [0.0, cphi,          -sphi],
            [0.0, sphi * sec_th, cphi * sec_th],
        ])
        eta[k + 1, 0:3] = eta[k, 0:3] + dt * (R @ nu_seq[k, 0:3])
        eta[k + 1, 3:6] = eta[k, 3:6] + dt * (T @ nu_seq[k, 3:6])
    return eta


def _plot_velocity_compare(t: np.ndarray, nu_truth: np.ndarray, nu_pred: np.ndarray,
                           traj_idx: int, model_stem: str, test_stem: str,
                           out_path: Path) -> None:
    """6 time-series subplots overlaying Fossen ν vs EDMDc ν on one trajectory."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    axes_flat = axes.flatten()
    for i, (ax, name) in enumerate(zip(axes_flat, NU_NAMES)):
        ax.plot(t, nu_truth[:, i], linewidth=2,    color="C0", label="Fossen (truth)")
        ax.plot(t, nu_pred[:,  i], linewidth=1.5,  color="C3",
                linestyle="--",                       label="EDMDc rollout")
        ax.set_title(f"{name}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Open-loop velocity tracking — trajectory {traj_idx}\n"
        f"model {model_stem}  /  test {test_stem}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[eval] wrote velocity-compare plot to {out_path}")


def _plot_trajectory_3d(eta_truth: np.ndarray, eta_pred: np.ndarray,
                        traj_idx: int, model_stem: str, test_stem: str,
                        out_path: Path) -> None:
    """3D path plot: η integrated from both ν sequences starting at η₀ = 0."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(eta_truth[:, 0], eta_truth[:, 1], eta_truth[:, 2],
            linewidth=2, color="C0", label="Fossen (truth)")
    ax.plot(eta_pred[:,  0], eta_pred[:,  1], eta_pred[:,  2],
            linewidth=1.5, color="C3", linestyle="--", label="EDMDc rollout")
    ax.scatter(*eta_truth[0,  :3], color="green",  s=80, marker="o", label="start")
    ax.scatter(*eta_truth[-1, :3], color="C0",     s=80, marker="s", label="Fossen end")
    ax.scatter(*eta_pred[-1,  :3], color="C3",     s=80, marker="^", label="EDMDc end")
    ax.set_xlabel("x  [m, North]")
    ax.set_ylabel("y  [m, East]")
    ax.set_zlabel("z  [m, Down]")
    ax.invert_zaxis()                          # depth increases downward
    ax.set_title(
        f"3D path (η₀ = 0, kinematic integration of ν)\n"
        f"trajectory {traj_idx}  ·  model {model_stem}",
        fontsize=11,
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[eval] wrote 3D trajectory plot to {out_path}")


def _plot_rollout(hs: list[int], rmse_total: list[float],
                  rmse_axis: np.ndarray, dt: float,
                  model_stem: str, test_stem: str,
                  out_path: Path) -> None:
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
        f"EDMDc multi-step rollout RMSE\n"
        f"model {model_stem}  /  test {test_stem}",
        fontsize=11,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[eval] wrote rollout plot to {out_path}")


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
    onestep_path = plots_dir / f"onestep_{_RUN_TS}.png"
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
        nu_traj      = nu[sel]
        nu_next_traj = nu_next[sel]
        tau_traj     = tau[sel]
        n_steps = nu_traj.shape[0]
        if n_steps < 2:
            continue
        nu_hat = rollout(A, B, nu_traj[0], tau_traj)
        for h in args.horizons:
            if 1 <= h <= n_steps:
                horizon_errs[h].append(nu_hat[h] - nu_next_traj[h - 1])

    print("\n[eval] multi-step rollout RMSE vs horizon:")
    hs_with_data: list[int] = []
    rmse_total:   list[float] = []
    rmse_axis_h:  list[np.ndarray] = []
    for h in args.horizons:
        if not horizon_errs[h]:
            continue
        E = np.asarray(horizon_errs[h])
        rmse_axis = np.sqrt(np.mean(E ** 2, axis=0))
        rmse_tot  = float(np.sqrt(np.mean(E ** 2)))
        n = len(E)
        per_axis = ", ".join(f"{n_}={r_:.3f}" for n_, r_ in zip(NU_NAMES, rmse_axis))
        print(f"  H = {h:4d} ({h * dt:5.2f}s)  RMSE total = {rmse_tot:.4f}  "
              f"({per_axis}, n_trajs = {n})")
        hs_with_data.append(h)
        rmse_total.append(rmse_tot)
        rmse_axis_h.append(rmse_axis)

    rollout_path = plots_dir / f"rollout_{_RUN_TS}.png"
    _plot_rollout(
        hs_with_data, rmse_total, np.asarray(rmse_axis_h),
        dt=dt, model_stem=args.model.stem, test_stem=args.data.stem,
        out_path=rollout_path,
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

        nu_truth_full, _ = _fossen_rollout(fparams, eta0, nu0, tau_traj, dt)
        nu_pred_full = rollout(A, B, nu0, tau_traj)
        chosen_idx = -1            # not a dataset trajectory
        label_tag = (f"step {args.axis}={args.step_amp:g} "
                     f"({args.step_on:g}s ON / {args.step_total:g}s total)")

    t_axis = np.arange(N + 1) * dt
    velcmp_path = plots_dir / f"velocity_compare_{_RUN_TS}.png"
    _plot_velocity_compare(
        t_axis, nu_truth_full, nu_pred_full,
        traj_idx=chosen_idx,
        model_stem=args.model.stem,
        test_stem=label_tag if args.input_mode != "dataset" else args.data.stem,
        out_path=velcmp_path,
    )

    eta_truth = _integrate_kinematics(nu_truth_full[:-1], dt=dt)
    eta_pred  = _integrate_kinematics(nu_pred_full[:-1],  dt=dt)
    traj3d_path = plots_dir / f"trajectory_3d_{_RUN_TS}.png"
    _plot_trajectory_3d(
        eta_truth, eta_pred,
        traj_idx=chosen_idx,
        model_stem=args.model.stem,
        test_stem=label_tag if args.input_mode != "dataset" else args.data.stem,
        out_path=traj3d_path,
    )

    # ---- Hold windows open until user closes them ---------------------------
    if not args.no_show:
        print("\n[eval] close the plot windows to exit.")
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
