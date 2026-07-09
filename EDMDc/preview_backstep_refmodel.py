"""Preview the ghost trajectory that a 3rd-order Fossen reference model would
produce through the default waypoint sequence of ``cascade_nav_38_lqr.py``.

No Isaac, no LQR, no plant. We're plotting the **outer-loop math only**:
given the waypoint sequence, what does the reference model output as
(eta_d, eta_dot_d, eta_ddot_d) over time? The eta_dot_d trace is the
velocity profile that backstepping would hand to the inner LQR (after
rotation R^T into the body frame; here we plot in world frame since the
plot is meant for tuning, not execution). The yaw panels show the
scalar 3rd-order yaw ghost that the main script runs in parallel, and
whose `psi_dot_d` becomes the r (yaw-rate) reference driving Tz.

If you change the bandwidth OMEGA_N or OMEGA_YAW, the curves change. If
you change LAMBDA (the backstepping correction gain), nothing in this
preview changes -- because LAMBDA only kicks in when there is a position
error between the real vehicle and the ghost, and in this preview there
is no real vehicle.

Run::

    python EDMDc/preview_backstep_refmodel.py
    python EDMDc/preview_backstep_refmodel.py --omega 0.30 0.30 0.20
    python EDMDc/preview_backstep_refmodel.py --omega-yaw 0.40
    python EDMDc/preview_backstep_refmodel.py --leg-timeout-s 25

The output PNG lives next to the script.

Parameters worth tuning
-----------------------
- OMEGA_N (per-axis position bandwidth, rad/s)
    Single biggest knob for the position ghost. Peak ghost velocity for
    a step of size L (with zeta = 1) is

        eta_dot_peak ~= 0.271 * L * omega_n

    so pick omega_n so that the peak stays inside v_max with headroom
    for the LAMBDA*e_p correction term. Default: 0.30 / 0.30 / 0.20 rad/s.

- OMEGA_YAW (yaw bandwidth, rad/s)
    Same role as OMEGA_N but scalar, for the yaw ghost. Peak r reference
    is approximately 0.271 * |delta_psi| * omega_yaw. Default 0.30.

- ZETA / ZETA_YAW (damping ratio)
    Controls overshoot. 1.0 = critical damping, no overshoot. Never < 1.0.

- LAMBDA (backstepping correction gain, per axis)
    NOT used in this preview (no real vehicle).

- HOLD_S, ARRIVAL_TOL
    Dwell time at each waypoint and arrival tolerance. Defaults 2.0 s,
    0.15 m, matching the main script.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size":       13,
    "axes.titlesize":  15,
    "axes.labelsize":  14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
import matplotlib.pyplot as plt
import numpy as np


# ----- Defaults aligned with cascade_nav_38_lqr.py -----
START = np.array([0.0, 0.0, -0.5])    # approx. settled position
WAYPOINTS_DEFAULT = np.array([
    [4.0,  3.0, -1.0],
    [4.0, -3.0, -1.5],
    [0.0,  0.0, -0.5],
])
V_MAX = np.array([0.5, 0.4, 0.15])    # u_max, v_max, w_max from main script
R_MAX = 0.6                            # r_max from main script (rad/s)


def step_ref_model(x_d: np.ndarray,
                   target: np.ndarray,
                   omega_n: np.ndarray,
                   zeta: float,
                   dt: float) -> np.ndarray:
    """Advance one tick of a per-axis-decoupled 3rd-order reference model.

    State layout: x_d = [eta_d (3,), eta_dot_d (3,), eta_ddot_d (3,)]
    """
    eta_d  = x_d[0:3]
    eta_d1 = x_d[3:6]
    eta_d2 = x_d[6:9]

    # Per-axis triple-pole filter (zeta = 1 -> triple pole at -omega).
    # General form (Fossen 12.1.1):
    #   eta_d3 = -omega^3 (eta_d - target)
    #            - (2 zeta + 1) omega^2 eta_d1
    #            - (2 zeta + 1) omega   eta_d2
    eta_d3 = (
        -omega_n**3 * (eta_d - target)
        - (2.0 * zeta + 1.0) * omega_n**2 * eta_d1
        - (2.0 * zeta + 1.0) * omega_n    * eta_d2
    )
    return np.concatenate([
        eta_d  + eta_d1 * dt,
        eta_d1 + eta_d2 * dt,
        eta_d2 + eta_d3 * dt,
    ])


def step_yaw_ref(x_yaw: np.ndarray,
                 target_psi: float,
                 omega_n: float,
                 zeta: float,
                 dt: float) -> np.ndarray:
    """Scalar version of step_ref_model for yaw, with error wrapped to
    (-pi, pi] so the ghost takes the short way around. Mirrors the
    helper of the same name in ``backstep_edmdc_lqr.py``.

    State layout: x_yaw = [psi_d, psi_dot_d, psi_ddot_d].
    """
    psi_d, psi_d1, psi_d2 = x_yaw
    err = float((psi_d - target_psi + np.pi) % (2 * np.pi) - np.pi)
    psi_d3 = (
        -omega_n**3 * err
        - (2.0 * zeta + 1.0) * omega_n**2 * psi_d1
        - (2.0 * zeta + 1.0) * omega_n    * psi_d2
    )
    return np.array([psi_d + psi_d1 * dt,
                     psi_d1 + psi_d2 * dt,
                     psi_d2 + psi_d3 * dt])


def simulate(start: np.ndarray,
             waypoints: np.ndarray,
             omega_n: np.ndarray,
             zeta: float,
             dt: float,
             hold_s: float,
             arrival_tol: float,
             leg_timeout_s: float,
             omega_yaw: float = 0.30,
             zeta_yaw: float = 1.0,
             psi0: float = 0.0) -> dict:
    """Walk the position+yaw ghosts through the waypoint sequence and log
    everything. Yaw target for each leg = atan2(dy, dx) computed once when
    the leg starts (matches main script convention)."""
    n_wp = waypoints.shape[0]
    hold_ticks = max(1, int(round(hold_s / dt)))
    leg_timeout_ticks = max(1, int(round(leg_timeout_s / dt)))

    # Generous buffer: hold + timeout per leg.
    max_ticks = (hold_ticks + leg_timeout_ticks) * n_wp + 200

    log_t   = np.empty(max_ticks)
    log_eta = np.empty((max_ticks, 3))
    log_etd = np.empty((max_ticks, 3))
    log_eta2= np.empty((max_ticks, 3))
    log_target_idx = np.empty(max_ticks, dtype=np.int32)
    log_psi_d      = np.empty(max_ticks)
    log_psi_dot_d  = np.empty(max_ticks)
    log_psi_ddot_d = np.empty(max_ticks)
    log_psi_target = np.empty(max_ticks)
    arrival_steps = [-1] * n_wp
    switch_steps  = [0] + [-1] * (n_wp - 1)

    x_d   = np.concatenate([start.copy(), np.zeros(3), np.zeros(3)])
    x_yaw = np.array([psi0, 0.0, 0.0])
    target_idx = 0
    target = waypoints[0].copy()
    # Leg 1 yaw target: from current ghost position toward wp1 (matches
    # main script's `np.arctan2(target[1]-pos0[1], target[0]-pos0[0])`).
    psi_target = float(np.arctan2(target[1] - x_d[1], target[0] - x_d[0]))
    k = 0
    while k < max_ticks:
        log_t[k] = k * dt
        log_eta[k]  = x_d[0:3]
        log_etd[k]  = x_d[3:6]
        log_eta2[k] = x_d[6:9]
        log_target_idx[k] = target_idx
        log_psi_d[k]      = x_yaw[0]
        log_psi_dot_d[k]  = x_yaw[1]
        log_psi_ddot_d[k] = x_yaw[2]
        log_psi_target[k] = psi_target

        err = float(np.linalg.norm(x_d[0:3] - target))
        if arrival_steps[target_idx] < 0 and err < arrival_tol:
            arrival_steps[target_idx] = k

        # Switch after dwell elapsed, OR if leg timed out.
        if target_idx < n_wp - 1:
            arrived = arrival_steps[target_idx]
            leg_start = switch_steps[target_idx]
            if (arrived >= 0 and (k - arrived) >= hold_ticks) or \
               (k - leg_start) >= (hold_ticks + leg_timeout_ticks):
                target_idx += 1
                target = waypoints[target_idx].copy()
                switch_steps[target_idx] = k + 1
                # Retarget yaw for the new leg, from current ghost position
                # to the new waypoint (xy plane).
                psi_target = float(np.arctan2(
                    target[1] - x_d[1], target[0] - x_d[0]))
        else:
            if (arrival_steps[target_idx] >= 0
                    and (k - arrival_steps[target_idx]) >= hold_ticks):
                k += 1
                break

        x_d   = step_ref_model(x_d, target, omega_n, zeta, dt)
        x_yaw = step_yaw_ref(x_yaw, psi_target, omega_yaw, zeta_yaw, dt)
        k += 1

    return dict(
        t=log_t[:k],
        eta=log_eta[:k],
        eta_dot=log_etd[:k],
        eta_ddot=log_eta2[:k],
        target_idx=log_target_idx[:k],
        arrival_steps=np.asarray(arrival_steps, dtype=np.int64),
        switch_steps=np.asarray(switch_steps, dtype=np.int64),
        psi_d=log_psi_d[:k],
        psi_dot_d=log_psi_dot_d[:k],
        psi_ddot_d=log_psi_ddot_d[:k],
        psi_target=log_psi_target[:k],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omega", type=float, nargs=3, default=[0.30, 0.30, 0.20],
                    metavar=("OMEGA_X", "OMEGA_Y", "OMEGA_Z"),
                    help="Per-axis position bandwidth in rad/s. Default "
                         "0.30 0.30 0.20.")
    ap.add_argument("--zeta", type=float, default=1.0,
                    help="Position damping ratio. >= 1.0 recommended.")
    ap.add_argument("--omega-yaw", type=float, default=0.30,
                    help="Yaw bandwidth in rad/s. Default 0.30.")
    ap.add_argument("--zeta-yaw", type=float, default=1.0,
                    help="Yaw damping ratio. >= 1.0 recommended.")
    ap.add_argument("--psi0", type=float, default=0.0,
                    help="Initial yaw of the ghost (rad). Default 0.")
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument("--arrival-tol", type=float, default=0.15)
    ap.add_argument("--leg-timeout-s", type=float, default=30.0,
                    help="Cap each leg at this many seconds (safety, in case "
                         "the ghost never reaches the tolerance).")
    ap.add_argument("--start", type=float, nargs=3, default=START.tolist())
    ap.add_argument("--waypoints", type=float, nargs="+",
                    default=WAYPOINTS_DEFAULT.flatten().tolist(),
                    help="Flat list of waypoint xyz floats, multiple of 3.")
    args = ap.parse_args()

    if len(args.waypoints) % 3 != 0 or len(args.waypoints) == 0:
        raise SystemExit("--waypoints must be a non-empty multiple of 3.")
    waypoints = np.asarray(args.waypoints, dtype=np.float64).reshape(-1, 3)
    start     = np.asarray(args.start, dtype=np.float64)
    omega_n   = np.asarray(args.omega, dtype=np.float64)

    # Predicted peak velocity per axis (heuristic, zeta=1, longest leg as L)
    longest_leg = max(
        float(np.linalg.norm(waypoints[0] - start)),
        *[float(np.linalg.norm(waypoints[i+1] - waypoints[i]))
          for i in range(len(waypoints) - 1)],
    )
    pred_peak = 0.271 * longest_leg * omega_n
    # Predicted peak r reference (using longest yaw delta = pi as worst-case)
    pred_r_peak = 0.271 * np.pi * args.omega_yaw
    print("[preview] start          =", start.round(3).tolist())
    print("[preview] waypoints      =")
    for i, w in enumerate(waypoints):
        print(f"             wp{i+1}: {w.tolist()}")
    print(f"[preview] omega_n        = {omega_n.tolist()} rad/s   (position)")
    print(f"[preview] omega_yaw      = {args.omega_yaw} rad/s  (yaw)")
    print(f"[preview] zeta / zeta_yaw= {args.zeta} / {args.zeta_yaw}")
    print(f"[preview] longest leg    = {longest_leg:.2f} m")
    print(f"[preview] predicted peak |eta_dot_d| per axis = "
          f"{pred_peak.round(3).tolist()} m/s")
    print(f"[preview] predicted peak |psi_dot_d| (worst-case 180 deg turn) "
          f"= {pred_r_peak:.3f} rad/s")
    print(f"[preview] v_max envelope = {V_MAX.tolist()} m/s")
    print(f"[preview] r_max envelope = {R_MAX} rad/s")
    if np.any(pred_peak > V_MAX):
        print("[preview] WARNING: predicted peak exceeds v_max on some axis "
              "-- shrink omega_n or expect inner-loop clamping.")
    if pred_r_peak > R_MAX:
        print("[preview] WARNING: predicted peak r reference exceeds r_max "
              "-- shrink omega_yaw or expect inner-loop clamping.")

    sim = simulate(start=start,
                   waypoints=waypoints,
                   omega_n=omega_n,
                   zeta=args.zeta,
                   dt=args.dt,
                   hold_s=args.hold_s,
                   arrival_tol=args.arrival_tol,
                   leg_timeout_s=args.leg_timeout_s,
                   omega_yaw=args.omega_yaw,
                   zeta_yaw=args.zeta_yaw,
                   psi0=args.psi0)
    t  = sim["t"]
    eta  = sim["eta"]
    etd  = sim["eta_dot"]
    etd2 = sim["eta_ddot"]
    target_idx = sim["target_idx"]
    arrival = sim["arrival_steps"]
    switches = sim["switch_steps"]
    psi_d      = sim["psi_d"]
    psi_dot_d  = sim["psi_dot_d"]
    psi_target = sim["psi_target"]

    actual_peak = np.max(np.abs(etd), axis=0)
    actual_r_peak = float(np.max(np.abs(psi_dot_d)))
    print(f"[preview] simulated peak |eta_dot_d| per axis = "
          f"{actual_peak.round(3).tolist()} m/s")
    print(f"[preview] simulated peak |psi_dot_d| = {actual_r_peak:.3f} rad/s")
    for i, s_arr in enumerate(arrival):
        tag = f"t={t[s_arr]:.2f}s" if s_arr >= 0 else "NOT REACHED"
        print(f"[preview] wp{i+1} ghost arrival: {tag}")

    # ---- Plot ----
    fig, axes = plt.subplots(6, 1, figsize=(13, 18), sharex=True)
    axis_names = ["x", "y", "z"]
    colors = ["C0", "C1", "C2"]

    # (0) position: ghost eta_d + waypoint stair
    ax = axes[0]
    for i in range(3):
        ax.plot(t, eta[:, i], color=colors[i], lw=1.5,
                label=f"eta_d {axis_names[i]}")
        wp_step = np.array([waypoints[idx, i] for idx in target_idx])
        ax.plot(t, wp_step, color=colors[i], lw=0.7, ls="--", alpha=0.6)
    for s in switches[1:]:
        if s > 0:
            ax.axvline(t[s], color="k", ls=":", lw=0.7, alpha=0.5)
    for i, s_arr in enumerate(arrival):
        if s_arr >= 0:
            ax.plot(t[s_arr], eta[s_arr, 0], "k.", markersize=6)
            ax.annotate(f"wp{i+1}", (t[s_arr], waypoints[i, 0]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=11, color="k")
    ax.set_ylabel("position [m]")
    ax.set_title(f"3rd-order ref model: ghost trajectory  "
                 f"(omega = {omega_n.tolist()}, zeta = {args.zeta})")
    ax.grid(alpha=0.3); ax.legend(loc="best", ncol=3)

    # (1) velocity profile -- THE KEY PLOT for Fx, Fy, Fz references
    ax = axes[1]
    for i in range(3):
        ax.plot(t, etd[:, i], color=colors[i], lw=1.6,
                label=f"eta_dot_d {axis_names[i]}")
        ax.axhline(V_MAX[i],  color=colors[i], ls=":", lw=0.6, alpha=0.5)
        ax.axhline(-V_MAX[i], color=colors[i], ls=":", lw=0.6, alpha=0.5)
    for s in switches[1:]:
        if s > 0:
            ax.axvline(t[s], color="k", ls=":", lw=0.7, alpha=0.5)
    ax.set_ylabel("velocity [m/s]")
    ax.set_title("ghost velocity profile -> u_ref, v_ref, w_ref after R_z^T")
    ax.grid(alpha=0.3); ax.legend(loc="best", ncol=3)

    # (2) acceleration profile
    ax = axes[2]
    for i in range(3):
        ax.plot(t, etd2[:, i], color=colors[i], lw=1.3,
                label=f"eta_ddot_d {axis_names[i]}")
    for s in switches[1:]:
        if s > 0:
            ax.axvline(t[s], color="k", ls=":", lw=0.7, alpha=0.5)
    ax.set_ylabel("acceleration [m/s^2]")
    ax.set_title("ghost acceleration profile (= eta_ddot_d, free feedforward)")
    ax.grid(alpha=0.3); ax.legend(loc="best", ncol=3)

    # (3) translational speed magnitude
    ax = axes[3]
    speed = np.linalg.norm(etd, axis=1)
    ax.plot(t, speed, "C3", lw=1.6, label="||eta_dot_d||")
    for s in switches[1:]:
        if s > 0:
            ax.axvline(t[s], color="k", ls=":", lw=0.7, alpha=0.5)
    ax.set_ylabel("speed [m/s]")
    ax.set_title("translational speed magnitude")
    ax.grid(alpha=0.3); ax.legend(loc="best")

    # (4) yaw heading: psi_d vs target stair
    ax = axes[4]
    ax.plot(t, np.rad2deg(psi_d), color="C4", lw=1.6,
            label="psi_d (yaw ghost)")
    ax.plot(t, np.rad2deg(psi_target), color="k", lw=0.8, ls="--",
            alpha=0.7, label="psi_target per leg")
    for s in switches[1:]:
        if s > 0:
            ax.axvline(t[s], color="k", ls=":", lw=0.7, alpha=0.5)
    ax.set_ylabel("yaw [deg]")
    ax.set_title(f"yaw ghost heading  (omega_yaw = {args.omega_yaw}, "
                 f"zeta_yaw = {args.zeta_yaw})")
    ax.grid(alpha=0.3); ax.legend(loc="best")

    # (5) yaw rate: psi_dot_d -- this IS the r reference fed to the LQR
    ax = axes[5]
    ax.plot(t, psi_dot_d, color="C5", lw=1.6, label="psi_dot_d = r_ref")
    ax.axhline(R_MAX,  color="C5", ls=":", lw=0.6, alpha=0.5)
    ax.axhline(-R_MAX, color="C5", ls=":", lw=0.6, alpha=0.5)
    for s in switches[1:]:
        if s > 0:
            ax.axvline(t[s], color="k", ls=":", lw=0.7, alpha=0.5)
    ax.set_xlabel("t [s]"); ax.set_ylabel("yaw rate [rad/s]")
    ax.set_title("yaw rate ghost -> r reference (drives Tz via LQR)")
    ax.grid(alpha=0.3); ax.legend(loc="best")

    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent / "data" / "plots" / "preview_refmodel"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = out_dir / f"preview_backstep_refmodel_{ts}.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"[preview] plot -> {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
