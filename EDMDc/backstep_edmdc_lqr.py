"""Cascade navigation: outer Lyapunov backstepping (3rd-order ghost) + inner
38-D EDMDc-LQR.

This replaces the PD outer loop of ``cascade_nav_38_lqr.py`` with a
Lyapunov-derived backstepping law against a 3rd-order Fossen reference model
(Fossen 12.1.1). The inner EDMDc-LQR is unchanged.

Outer loop (each tick):

    1. Advance the 3rd-order POSITION ghost (eta_d, eta_dot_d, eta_ddot_d)
       toward the active waypoint via ``step_ref_model``.
    2. Advance the 3rd-order YAW ghost (psi_d, psi_dot_d, psi_ddot_d) toward
       psi_target = atan2(target_y - y, target_x - x) (set ONCE per leg).
    3. Compute the backstepping velocity reference:

          v_world_ref = eta_dot_d + Lambda * (eta_d - eta_p)
          v_body_ref  = R_z(psi)^T @ v_world_ref
          r_ref       = psi_dot_d + k_psi * wrap_pi(psi_d - psi)

       The first term in each line is the FEEDFORWARD from the ghost (carries
       the smoothness). The second is the Lyapunov correction that closes the
       position/yaw loop -- without it, the vehicle would drift parallel to
       the ghost forever (the LQR is position-blind).
    4. Clamp each body-frame component to its envelope and hand to the LQR.
    5. Switch the active waypoint after VEHICLE arrival + dwell. The ghost is
       NOT reset on switch -- the filter smoothly retargets the new waypoint,
       carrying its momentum across.

Lyapunov certificates:

    V_pos = 1/2 |e_p|^2          ==>  dV_pos/dt = -e_p^T Lambda e_p   <= 0
    V_yaw = 1/2 e_psi^2          ==>  dV_yaw/dt = -k_psi e_psi^2      <= 0

The inner EDMDc-LQR has its own quadratic Lyapunov function from DARE
(V_in = z_tilde^T P z_tilde). The composite V = V_pos + V_yaw + alpha V_in
gives a cascade ISS proof in the inner-loop tracking residual -- see slides
``Lyapunov Backstepping ...''. Strict (no-LQR) Step-2 backstepping would add
an R^T(psi) e_p term directly to the wrench; we deliberately forfeit that
cross-cancellation by keeping the LQR as a black-box inner loop.

Run:

    conda activate marinegym
    python -m EDMDc.backstep_edmdc_lqr                                # GUI, sweep-best (R_scale=0.01)
    python -m EDMDc.backstep_edmdc_lqr --r-scale 1.0                  # revert to library defaults
    python -m EDMDc.backstep_edmdc_lqr --omega 0.32 0.17 0.39         # tune position ghost bandwidth
    python -m EDMDc.backstep_edmdc_lqr --headless --no-video          # fast headless

Defaults (no flags): R-scale=0.01 -> R_diag=(1e-6, 1e-6, 1e-6, 1e-5). This is
the composite-best from the LQR R-scale sweep (`analyze_lqr_R_sweep.py`).
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from .isaac_scene import GripperScene, ChaseCam, resolve_ffmpeg
from .live_plot_client import LivePlot2D, LivePlot3D, LivePlotThrusters
from .preview_backstep_refmodel import step_ref_model


# ----- CLI -----

def _latest_model() -> Path | None:
    md = Path(__file__).resolve().parent / "model"
    if not md.is_dir():
        return None
    # Glob 34-D models (exclude the 38-D variants).
    cands = sorted(p for p in md.glob("edmdc_gripper_*.npz") if "_38_" not in p.name)
    return cands[-1] if cands else None


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=_latest_model())
    ap.add_argument("--waypoints", type=float, nargs="+",
                    default=[4.0,  3.0, -1.0,
                             4.0, -3.0, -1.5,
                             0.0,  0.0, -0.5],
                    metavar="X Y Z",
                    help="Sequence of world-frame waypoints (must be a multiple "
                         "of 3 floats). Default: 3-waypoint triangle.")
    ap.add_argument("--hold-s", type=float, default=2.0,
                    help="Dwell time at each waypoint after VEHICLE arrival "
                         "before retargeting the ghost. Default 2 s.")
    ap.add_argument("--duration", type=float, default=120.0,
                    help="Total run time in seconds (after settle). Default 120.")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-live-plot", action="store_true",
                    help="Skip the out-of-process matplotlib live plots "
                         "(velocity + 3D path + thrusters).")
    ap.add_argument("--no-live-thrusters", action="store_true",
                    help="Skip ONLY the per-thruster live plot. Useful when "
                         "you want velocity/3D plots but not the extra "
                         "subprocess for thrusters.")
    ap.add_argument("--thruster-max-N", type=float, default=0.0,
                    help="If > 0, the thruster plot draws dotted +/- lines "
                         "at this magnitude as saturation references. "
                         "Default 0 (auto-scale only). Try ~50 for BlueROV "
                         "T200s if you want a reference.")
    ap.add_argument("--floor-grid", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--mass-kg", type=float, default=13.5)
    ap.add_argument("--neutral-buoyancy", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--spawn", type=float, nargs=3, default=[0.0, 0.0, -1.6],
                    metavar=("X", "Y", "Z"),
                    help="World-frame spawn position passed to GripperScene "
                         "(x, y, z). Default (0, 0, -1) = 1 m below the surface, "
                         "matching DEFAULT_SPAWN in isaac_scene.py.")

    # --- 3rd-order POSITION reference model ---
    ap.add_argument("--omega", type=float, nargs=3, default=[0.32, 0.17, 0.39],
                    metavar=("OMEGA_X", "OMEGA_Y", "OMEGA_Z"),
                    help="Per-axis bandwidth of the position ghost (rad/s). "
                         "Peak |eta_dot_d| ~ 0.271 * L * omega_n -- pick so "
                         "it leaves ~30%% headroom under v_max. Defaults sized "
                         "for the canonical triangle waypoints under "
                         "(u,v,w)_max = (0.5, 0.4, 0.15) m/s.")
    ap.add_argument("--zeta", type=float, default=1.0,
                    help="Damping ratio of the position ghost. zeta=1 puts all "
                         "three poles at -omega_n (monotone, no overshoot). "
                         "Do not go below 1.0.")

    # --- 3rd-order YAW reference model ---
    ap.add_argument("--omega-yaw", type=float, default=1.0,
                    help="Bandwidth of the yaw ghost (rad/s). Default 1.0.")
    ap.add_argument("--zeta-yaw", type=float, default=1.0)

    # --- Backstepping gains ---
    ap.add_argument("--Lambda", type=float, nargs=3, default=[0.2, 0.2, 0.2],
                    metavar=("LAM_X", "LAM_Y", "LAM_Z"),
                    help="Diagonal entries of the backstep correction gain "
                         "Lambda (1/s). Each axis converges with time "
                         "constant 1/Lambda_i. Default 0.2 matches the Kp_pos "
                         "in cascade_nav_38_lqr.py so the two outer loops are "
                         "directly comparable.")
    ap.add_argument("--k-psi", type=float, default=1.5,
                    help="Backstep yaw correction gain (1/s).")

    # --- Body-frame velocity envelope passed to LQR ---
    ap.add_argument("--u-max", type=float, default=0.5)
    ap.add_argument("--v-max", type=float, default=0.4)
    ap.add_argument("--w-max", type=float, default=0.15)
    ap.add_argument("--r-max", type=float, default=0.5)

    ap.add_argument("--arrival-tol", type=float, default=0.15,
                    help="Vehicle arrival tolerance (m) on |wp - eta_p|.")

    # --- Settle ---
    ap.add_argument("--settle-s", type=float, default=5.0)
    ap.add_argument("--settle-min-s", type=float, default=4.0)
    ap.add_argument("--settle-tol", type=float, default=0.02)

    # --- LQR cost-weight overrides ---
    # Default --r-scale = 0.01 -> R_diag = (1e-6, 1e-6, 1e-6, 1e-5). This is the
    # sweep-best from `analyze_lqr_R_sweep.py` on the 3-wp triangle (composite
    # z-score winner). Pass --r-scale 1.0 to revert to the library defaults
    # in `lqr.py:LQRConfig` (R_diag = 1e-4, 1e-4, 1e-4, 1e-3).
    ap.add_argument("--r-scale", type=float, default=0.01,
                    help="Multiplier applied to LQRConfig.R_diag (4 active axes). "
                         "Default 0.01 = sweep-best (R_diag = 1e-6, 1e-6, 1e-6, 1e-5). "
                         "Pass --r-scale 1.0 to revert to library defaults.")
    ap.add_argument("--r-diag", type=float, nargs=4, default=None,
                    metavar=("R_FX", "R_FY", "R_FZ", "R_TZ"),
                    help="Explicit 4-axis R values (active axes only) overriding "
                         "both the default and --r-scale.")
    ap.add_argument("--tag", type=str, default="",
                    help="Optional suffix appended to output npz/png filenames "
                         "(useful for sweep drivers).")
    return ap


# ----- 3rd-order yaw ref model (wrap-aware) -----
#
# Same Fossen triple-pole filter as ``step_ref_model`` but scalar, with the
# tracking error wrapped to (-pi, pi] so the ghost takes the short way around.
def step_yaw_ref(x_yaw: np.ndarray,
                 target_psi: float,
                 omega_n: float,
                 zeta: float,
                 dt: float) -> np.ndarray:
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


def wrap_to_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


# ----- Parse CLI BEFORE Isaac boot -----

args = _build_argparser().parse_args()
if args.model is None:
    raise SystemExit("No 34-D EDMDc/model/edmdc_gripper_*.npz found.")
if len(args.waypoints) == 0 or len(args.waypoints) % 3 != 0:
    raise SystemExit(
        f"--waypoints must be a non-empty multiple of 3 floats; "
        f"got {len(args.waypoints)} values.")
WAYPOINTS = np.asarray(args.waypoints, dtype=np.float64).reshape(-1, 3)
N_WAYPOINTS = int(WAYPOINTS.shape[0])
OMEGA = np.asarray(args.omega, dtype=np.float64)
LAMBDA = np.asarray(args.Lambda, dtype=np.float64)
print(f"[backstep-lqr] model = {args.model}")
print(f"[backstep-lqr] {N_WAYPOINTS} waypoints:")
for _i, _wp in enumerate(WAYPOINTS):
    print(f"  wp{_i}: {_wp.tolist()}")

FFMPEG_PATH = None if args.no_video else resolve_ffmpeg()


# ====== Main ======

def main() -> int:
    scene = GripperScene(
        dt=args.dt,
        mass_kg=args.mass_kg,
        headless=bool(args.headless),
        neutral_buoyancy=bool(args.neutral_buoyancy),
        spawn_pos=np.asarray(args.spawn, dtype=np.float64),
    )
    if args.floor_grid:
        scene.add_floor_grid()
    chase = ChaseCam(scene, ffmpeg_path=FFMPEG_PATH)

    from .lqr import EDMDcLQR, LQRConfig
    default_cfg = LQRConfig()
    if args.r_diag is not None:
        scaled_R = tuple(float(r) for r in args.r_diag)
        print(f"[backstep-lqr] R_diag override: {scaled_R}")
    else:
        scaled_R = tuple(r * float(args.r_scale) for r in default_cfg.R_diag)
        if abs(args.r_scale - 1.0) > 1e-9:
            print(f"[backstep-lqr] R-scale = {args.r_scale}: R_diag = {scaled_R}")
    cfg = LQRConfig(
        active_axes=default_cfg.active_axes,
        Q_nu_diag=default_cfg.Q_nu_diag,
        Q_aux_reg=default_cfg.Q_aux_reg,
        R_diag=scaled_R,
        u_min=default_cfg.u_min,
        u_max=default_cfg.u_max,
    )
    lqr = EDMDcLQR.from_npz(args.model, config=cfg)
    print(f"[backstep-lqr] LQR K shape: {lqr.K.shape}, closed-loop max|lambda| "
          f"= {lqr.cl_max_abs:.4f}")

    # ---------- Settle ----------
    print(f"[backstep-lqr] settling up to {args.settle_s:g} s...")
    state0 = scene.settle(max_s=args.settle_s,
                          min_s=args.settle_min_s,
                          tol=args.settle_tol,
                          on_step=lambda _s: chase.follow_and_record())
    pos0 = state0.pos_w
    yaw0 = float(state0.yaw)
    print(f"[backstep-lqr] settled p={pos0.round(3).tolist()}, "
          f"yaw={np.rad2deg(yaw0):+.2f}, "
          f"roll={np.rad2deg(state0.roll):+.2f}, "
          f"pitch={np.rad2deg(state0.pitch):+.2f} deg")

    # ---------- Outer-loop setup ----------
    # Ghost states are initialised AT THE SETTLED VEHICLE STATE so e_p(0)=0
    # and e_psi(0)=0 -- no initial transient kick into the LQR.
    eta_ghost = np.concatenate([pos0.copy(), np.zeros(3), np.zeros(3)])  # 9D
    yaw_ghost = np.array([yaw0, 0.0, 0.0])                               # 3D

    current_wp_idx = 0
    target = WAYPOINTS[current_wp_idx].copy()
    # psi_target computed ONCE per leg (matches cascade_nav_38_lqr.py
    # convention: no atan2 jitter near the waypoint). The yaw ghost will
    # filter steps in psi_target smoothly when we retarget.
    psi_target = float(np.arctan2(target[1] - pos0[1], target[0] - pos0[0]))
    print(f"[backstep-lqr] starting wp {current_wp_idx+1}/{N_WAYPOINTS} = "
          f"{target.tolist()}, psi_target = {np.rad2deg(psi_target):+.2f} deg")
    print(f"[backstep-lqr] omega_pos = {OMEGA.tolist()} rad/s, "
          f"Lambda = {LAMBDA.tolist()} 1/s, "
          f"omega_yaw = {args.omega_yaw}, k_psi = {args.k_psi}; "
          f"clamps: u_max={args.u_max}, v_max={args.v_max}, w_max={args.w_max}, "
          f"r_max={args.r_max}")

    # ---------- Video + live plots ----------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    PROJECT = Path(__file__).resolve().parents[1]
    if not args.no_video:
        out_video = PROJECT / "EDMDc" / "recording" / f"backstep_edmdc_lqr_34_{ts}.mp4"
        chase.start_recording(out_video)

    plots_dir = PROJECT / "EDMDc" / "data" / "plots" / "with_gripper"
    plots_dir.mkdir(parents=True, exist_ok=True)
    live_vel_png_path = plots_dir / f"backstep_lqr_{ts}_live_vel.png"
    live_3d_png_path  = plots_dir / f"backstep_lqr_{ts}_live_3d.png"
    live_thr_png_path = plots_dir / f"backstep_lqr_{ts}_live_thr.png"

    live = None
    live3d = None
    live_thr = None
    if not args.no_live_plot:
        wp_brief = " -> ".join(f"({w[0]:.1f},{w[1]:.1f},{w[2]:.1f})"
                                for w in WAYPOINTS)
        live = LivePlot2D(
            dt=args.dt,
            title=(f"Backstep+LQR: {N_WAYPOINTS} wp [{wp_brief}] -- "
                   f"nu vs backstep ref (live)"),
            window_seconds=args.duration,
            label_meas="Isaac (nu)", label_ref="backstep ref",
            save_on_close=str(live_vel_png_path),
        )
        live3d = LivePlot3D(
            title=(f"Backstep nav 3D path -- {N_WAYPOINTS} waypoints"),
            save_on_close=str(live_3d_png_path),
        )
        live3d.push_meta(start=pos0,
                         waypoints=WAYPOINTS,
                         current_idx=current_wp_idx)
        if not args.no_live_thrusters:
            live_thr = LivePlotThrusters(
                dt=args.dt,
                title=(f"Thruster commands -- {scene.num_rotors} rotors "
                       f"(auto-scaled per panel)"),
                window_seconds=args.duration,
                n_thrusters=int(scene.num_rotors),
                max_thrust=float(args.thruster_max_N),
                save_on_close=str(live_thr_png_path),
            )

    # ---------- Main control loop ----------
    n_steps    = int(round(args.duration / args.dt))
    hold_steps = max(1, int(round(args.hold_s / args.dt)))
    log_t          = np.empty(n_steps)
    log_pos        = np.empty((n_steps, 3))
    log_eta        = np.empty((n_steps, 3))   # roll, pitch, yaw
    log_nu         = np.empty((n_steps, 6))
    log_nu_ref     = np.empty((n_steps, 6))
    log_eta_d      = np.empty((n_steps, 3))   # ghost position
    log_eta_dot_d  = np.empty((n_steps, 3))   # ghost velocity (= feedforward)
    log_eta_ddot_d = np.empty((n_steps, 3))   # ghost acceleration (logged for diag)
    log_psi_d      = np.empty(n_steps)
    log_psi_dot_d  = np.empty(n_steps)
    log_e_pos      = np.empty((n_steps, 3))   # backstep error eta_d - eta_p
    log_e_wp       = np.empty((n_steps, 3))   # true error wp - eta_p (for diag)
    log_e_yaw      = np.empty(n_steps)
    log_u          = np.empty((n_steps, 4))
    log_thrusts    = np.empty((n_steps, scene.num_rotors))
    log_target_idx = np.empty(n_steps, dtype=np.int32)
    arrival_steps  = [-1] * N_WAYPOINTS

    u_held = np.zeros(4, dtype=np.float64)
    for k in range(n_steps):
        t = k * args.dt
        s = scene.read_state()

        # --- Outer loop: Lyapunov backstepping ---
        eta_d     = eta_ghost[0:3]
        eta_dot_d = eta_ghost[3:6]
        psi_d     = float(yaw_ghost[0])
        psi_dot_d = float(yaw_ghost[1])

        # Position backstepping. Note: error is ghost - vehicle (NOT
        # waypoint - vehicle). The waypoint enters only via the ghost's
        # setpoint; the LQR sees the smoothly-filtered eta_d.
        e_pos = eta_d - s.pos_w
        v_world_ref = eta_dot_d + LAMBDA * e_pos
        v_body_ref = s.R_body.T @ v_world_ref

        # Yaw backstepping (wrap-aware).
        e_yaw = wrap_to_pi(psi_d - s.yaw)
        r_ref = psi_dot_d + args.k_psi * e_yaw

        # Per-axis body-frame clamp. NOTE: in strict Lyapunov form this should
        # be a magnitude-preserving clip in world frame. Per-axis is good
        # enough when the envelope is roughly isotropic; for highly
        # anisotropic envelopes (e.g. our w_max << u_max) the rotated command
        # can be reshaped by the clamp. Lyapunov decay survives a positive
        # scaling, which is what happens most of the time.
        nu_ref = np.zeros(6, dtype=np.float64)
        nu_ref[0] = float(np.clip(v_body_ref[0], -args.u_max, args.u_max))
        nu_ref[1] = float(np.clip(v_body_ref[1], -args.v_max, args.v_max))
        nu_ref[2] = float(np.clip(v_body_ref[2], -args.w_max, args.w_max))
        nu_ref[5] = float(np.clip(r_ref,         -args.r_max, args.r_max))

        # --- Inner EDMDc-LQR ---
        u_held = lqr.step(s.nu, nu_ref)

        # --- Plant ---
        thrusts = scene.apply_wrench(u_held)
        chase.follow_and_record()

        # --- Log (current ghost state, current command) ---
        log_t[k] = t
        log_pos[k] = s.pos_w
        log_eta[k] = (s.roll, s.pitch, s.yaw)
        log_nu[k]  = s.nu
        log_nu_ref[k] = nu_ref
        log_eta_d[k]      = eta_d
        log_eta_dot_d[k]  = eta_dot_d
        log_eta_ddot_d[k] = eta_ghost[6:9]
        log_psi_d[k]      = psi_d
        log_psi_dot_d[k]  = psi_dot_d
        log_e_pos[k] = e_pos
        log_e_wp[k]  = target - s.pos_w
        log_e_yaw[k] = e_yaw
        log_u[k]  = u_held
        log_thrusts[k] = thrusts
        log_target_idx[k] = current_wp_idx

        # --- Vehicle arrival detect (vs. the WAYPOINT, not the ghost). ---
        e_to_wp_norm = float(np.linalg.norm(log_e_wp[k]))
        if (arrival_steps[current_wp_idx] < 0
                and e_to_wp_norm < args.arrival_tol):
            arrival_steps[current_wp_idx] = k
            print(f"[backstep-lqr] arrived at wp {current_wp_idx+1}/"
                  f"{N_WAYPOINTS} = {target.tolist()} at t = {t:.2f} s "
                  f"(|wp-eta_p| = {e_to_wp_norm:.3f} m)")

        if live is not None:
            live.push(t, s.nu, nu_ref)
        if live3d is not None:
            live3d.push(t, s.pos_w)
        if live_thr is not None:
            live_thr.push(t, thrusts)

        # --- Advance the ghost (after logging so logs reflect the state
        #     used by the control). ---
        eta_ghost = step_ref_model(eta_ghost, target, OMEGA, args.zeta, args.dt)
        yaw_ghost = step_yaw_ref(yaw_ghost, psi_target,
                                 args.omega_yaw, args.zeta_yaw, args.dt)

        # --- Switch active waypoint after vehicle arrival + dwell. ---
        # The ghost is NOT reset on switch -- it carries its current
        # (position, velocity, acceleration) and starts re-filtering the new
        # setpoint. The triple-pole filter smooths the step in r^n into a
        # continuous re-acceleration profile.
        if (arrival_steps[current_wp_idx] >= 0
                and current_wp_idx < N_WAYPOINTS - 1
                and (k - arrival_steps[current_wp_idx]) >= hold_steps):
            current_wp_idx += 1
            target = WAYPOINTS[current_wp_idx].copy()
            psi_target = float(np.arctan2(target[1] - s.pos_w[1],
                                          target[0] - s.pos_w[0]))
            print(f"[backstep-lqr] switching to wp {current_wp_idx+1}/"
                  f"{N_WAYPOINTS} = {target.tolist()}, "
                  f"psi_target = {np.rad2deg(psi_target):+.2f} deg "
                  f"(ghost carries forward, no reset)")
            if live3d is not None:
                live3d.push_meta(current_idx=current_wp_idx)

    # ---------- Save ----------
    out_npz_dir = PROJECT / "EDMDc" / "data" / "with_gripper" / "backstep_lqr"
    out_npz_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("_" + args.tag) if args.tag else ""
    npz_path = out_npz_dir / f"{ts}{suffix}.npz"
    np.savez(npz_path,
             t=log_t, pos=log_pos, eta=log_eta, nu=log_nu,
             nu_ref=log_nu_ref,
             eta_d=log_eta_d, eta_dot_d=log_eta_dot_d,
             eta_ddot_d=log_eta_ddot_d,
             psi_d=log_psi_d, psi_dot_d=log_psi_dot_d,
             e_pos=log_e_pos, e_wp=log_e_wp, e_yaw=log_e_yaw,
             u=log_u, thrusts=log_thrusts,
             waypoints=WAYPOINTS,
             target_idx=log_target_idx,
             arrival_steps=np.asarray(arrival_steps, dtype=np.int64),
             hold_steps=hold_steps,
             omega=OMEGA, zeta=args.zeta, Lambda=LAMBDA,
             omega_yaw=args.omega_yaw, zeta_yaw=args.zeta_yaw,
             k_psi=args.k_psi,
             dt=args.dt,
             input_units="ardusub_normalized_[-1,1]",
             input_names=np.array(["surge", "sway", "heave", "yaw"]),
             r_scale=float(args.r_scale),
             r_diag=np.asarray(scaled_R, dtype=np.float64),
             tag=args.tag)
    print(f"[backstep-lqr] saved -> {npz_path}")
    print(f"[backstep-lqr] arrivals: " + ", ".join(
        f"wp{i+1}={'t='+f'{log_t[s_]:.2f}s' if s_ >= 0 else 'NOT REACHED'}"
        for i, s_ in enumerate(arrival_steps)))

    if not args.no_video:
        chase.stop_recording()

    # ---------- Diagnostic plots ----------
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))

    # (0,0) -- top-down x-y: vehicle vs ghost, waypoints + arrivals
    ax = axes[0, 0]
    ax.plot(log_pos[:, 0], log_pos[:, 1], "C0", lw=1.4, label="vehicle path",
            zorder=2)
    ax.plot(log_eta_d[:, 0], log_eta_d[:, 1], "C3", lw=0.9, alpha=0.75,
            label="ghost eta_d", zorder=1)
    ax.plot(pos0[0], pos0[1], "go", markersize=8, label="start", zorder=3)
    for i, wp in enumerate(WAYPOINTS):
        ax.plot(wp[0], wp[1], "r*", markersize=12, zorder=3)
        ax.annotate(f"wp{i+1}", (wp[0], wp[1]),
                    textcoords="offset points", xytext=(7, 6),
                    fontsize=9, color="r")
        s_arr = arrival_steps[i]
        if s_arr >= 0:
            ax.plot(log_pos[s_arr, 0], log_pos[s_arr, 1], "kx",
                    markersize=10, zorder=4)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("top-down: vehicle vs ghost")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9); ax.set_aspect("equal")

    # (0,1) -- depth: vehicle vs ghost vs active waypoint
    ax = axes[0, 1]
    ax.plot(log_t, log_pos[:, 2], "C0", lw=1.4, label="vehicle z")
    ax.plot(log_t, log_eta_d[:, 2], "C3", lw=0.9, ls="--", label="ghost z_d")
    z_ref_step = WAYPOINTS[log_target_idx, 2]
    ax.plot(log_t, z_ref_step, "r:", lw=0.7, label="active wp z")
    for s_arr in arrival_steps:
        if s_arr >= 0:
            ax.axvline(log_t[s_arr], color="k", ls=":", lw=0.7, alpha=0.6)
    ax.set_xlabel("t [s]"); ax.set_ylabel("z [m]")
    ax.set_title("depth")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)

    # (1,0) -- body-frame nu vs backstep ref
    ax = axes[1, 0]
    names = ["u", "v", "w", "p", "q", "r"]
    for i in (0, 1, 2, 5):
        ax.plot(log_t, log_nu[:, i], color=f"C{i}", lw=1.3, alpha=0.9,
                label=f"{names[i]}")
        ax.plot(log_t, log_nu_ref[:, i], color=f"C{i}", lw=0.9, ls="--",
                label=f"{names[i]} ref")
    ax.set_xlabel("t [s]"); ax.set_ylabel("velocity")
    ax.set_title("body nu vs backstep ref = R^T(eta_dot_d + Lambda e_p)")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=7, ncol=4)

    # (1,1) -- attitude vs ghost yaw
    ax = axes[1, 1]
    ax.plot(log_t, np.rad2deg(log_eta[:, 0]), label="roll", lw=1.3)
    ax.plot(log_t, np.rad2deg(log_eta[:, 1]), label="pitch", lw=1.3)
    ax.plot(log_t, np.rad2deg(log_eta[:, 2]), label="yaw", lw=1.3)
    ax.plot(log_t, np.rad2deg(log_psi_d), "C2", ls=":", lw=0.9,
            label="psi_d (ghost)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("deg")
    ax.set_title("attitude + yaw ghost")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)

    # (2,0) -- error norms: |eta_d - eta_p| AND |wp - eta_p|
    ax = axes[2, 0]
    e_ghost_norm = np.linalg.norm(log_e_pos, axis=1)
    e_wp_norm    = np.linalg.norm(log_e_wp,  axis=1)
    ax.plot(log_t, e_ghost_norm, "C0", lw=1.4,
            label="|eta_d - eta_p|  (backstep error)")
    ax.plot(log_t, e_wp_norm, "C3", lw=1.0, alpha=0.8,
            label="|wp - eta_p|   (true error)")
    ax.axhline(args.arrival_tol, ls="--", color="k", lw=0.6,
               label=f"arrival tol = {args.arrival_tol}")
    for s_arr in arrival_steps:
        if s_arr >= 0:
            ax.axvline(log_t[s_arr], color="k", ls=":", lw=0.7, alpha=0.6)
    ax.set_xlabel("t [s]"); ax.set_ylabel("|e| [m]")
    ax.set_title("position errors (Lyapunov: |eta_d - eta_p| -> 0 exponentially)")
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8)

    # (2,1) -- normalized ArduSub command
    ax = axes[2, 1]
    wnames = ["surge", "sway", "heave", "yaw"]
    for i in range(4):
        ax.plot(log_t, log_u[:, i], label=wnames[i], lw=1.0, alpha=0.85)
    for s_arr in arrival_steps:
        if s_arr >= 0:
            ax.axvline(log_t[s_arr], color="k", ls=":", lw=0.7, alpha=0.6)
    ax.set_xlabel("t [s]"); ax.set_ylabel("normalized command")
    ax.set_title("LQR ArduSub command")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8, ncol=2)

    n_reached = sum(1 for s_ in arrival_steps if s_ >= 0)
    fig.suptitle(
        f"Backstepping (3rd-order ghost + Lambda e_p) + EDMDc-LQR-38  -  "
        f"{N_WAYPOINTS} waypoints (reached {n_reached}/{N_WAYPOINTS})  -  "
        f"TS {ts}",
        fontsize=12)
    fig.tight_layout()
    png_path = plots_dir / f"backstep_lqr_{ts}{suffix}.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"[backstep-lqr] plot -> {png_path}")

    if live is not None:
        live.close()
        print(f"[backstep-lqr] live velocity plot saved -> {live_vel_png_path}")
    if live3d is not None:
        live3d.close()
        print(f"[backstep-lqr] live 3D plot saved -> {live_3d_png_path}")
    if live_thr is not None:
        live_thr.close()
        print(f"[backstep-lqr] live thruster plot saved -> {live_thr_png_path}")
    chase.close()
    scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
