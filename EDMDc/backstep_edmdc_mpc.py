"""Cascade navigation: outer Lyapunov backstepping (3rd-order ghost) + inner
34-D EDMDc-MPC.

Mirrors ``backstep_edmdc_lqr.py`` exactly, but the inner-loop tracker is the
constrained MPC from ``mpc.py`` instead of the unconstrained LQR. Same outer
backstepping math, same 3-waypoint default, same logging schema — direct
apples-to-apples comparison with the LQR baseline.

Defaults baked in (no flags needed):
  * normalized ArduSub command bounds [-1,1]^4
  * --mpc-N           5
  * R_diag fallback   (1e-3, 1e-3, 1e-3, 1e-3)

CLI knobs vs the LQR script:
  --inscribed-caps    compatibility flag; normalized command bounds are used.
  --mpc-N             prediction horizon in steps.
  --mpc-decimation    solve every N sim steps and hold u between. Default 2.
  --r-scale           scale the *library default* R_diag by this factor
                      (only used when no --r-diag is given AND scale != 1.0).
  --r-diag            explicit 4-axis R override.

Run:
    conda activate marinegym
    python -m EDMDc.backstep_edmdc_mpc                                # GUI, sweep-best
    python -m EDMDc.backstep_edmdc_mpc --headless --no-video           # batch
    python -m EDMDc.backstep_edmdc_mpc --r-diag 1e-3 1e-3 1e-3 1e-3
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
    ap.add_argument("--no-diagnostic-plot", action="store_true",
                    help="Skip the final per-run diagnostic PNG. The .npz log "
                         "is still saved. Useful for large Monte Carlo runs.")
    ap.add_argument("--no-live-thrusters", action="store_true",
                    help="Skip ONLY the per-thruster live plot. Useful when "
                         "you want velocity/3D plots but not the extra "
                         "subprocess for thrusters.")
    ap.add_argument("--thruster-max-N", type=float, default=0.0,
                    help="If > 0, the thruster plot draws dotted +/- lines "
                         "at this magnitude as saturation references.")
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
    ap.add_argument("--payload-cube", action="store_true",
                    help="Run the payload-cube scenario by welding the 1 kg "
                         "negatively-buoyant cube to the gripper.")

    # --- 3rd-order POSITION reference model ---
    ap.add_argument("--omega", type=float, nargs=3, default=[0.32, 0.17, 0.39],
                    metavar=("OMEGA_X", "OMEGA_Y", "OMEGA_Z"),
                    help="Per-axis bandwidth of the position ghost (rad/s).")
    ap.add_argument("--zeta", type=float, default=1.0,
                    help="Damping ratio of the position ghost (>=1).")

    # --- 3rd-order YAW reference model ---
    ap.add_argument("--omega-yaw", type=float, default=1.0,
                    help="Bandwidth of the yaw ghost (rad/s). Default 1.0.")
    ap.add_argument("--zeta-yaw", type=float, default=1.0)

    # --- Backstepping gains ---
    ap.add_argument("--Lambda", type=float, nargs=3, default=[0.2, 0.2, 0.2],
                    metavar=("LAM_X", "LAM_Y", "LAM_Z"),
                    help="Diagonal entries of the backstep correction gain "
                         "Lambda (1/s).")
    ap.add_argument("--k-psi", type=float, default=1.5,
                    help="Backstep yaw correction gain (1/s).")

    # --- Body-frame velocity envelope passed to MPC ---
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

    # --- MPC-specific ---
    ap.add_argument("--inscribed-caps", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Compatibility flag. The ArduSub build always uses "
                         "normalized command bounds [-1,1]^4.")
    ap.add_argument("--mpc-N", type=int, default=5,
                    help="MPC prediction horizon in steps. Default 5 "
                         "(sweep-best from analyze_mpc_R_N_sweep.py; cheap QP).")
    ap.add_argument("--mpc-decimation", type=int, default=2,
                    help="Solve MPC every N sim steps; hold u between.")
    ap.add_argument("--mpc-backend", default="condensed_osqp",
                    choices=("auto", "condensed_osqp"),
                    help="MPC solver backend (condensed direct OSQP; auto is "
                         "an alias kept for old command lines).")
    ap.add_argument("--mpc-osqp-eps", type=float, default=1e-5,
                    help="Absolute and relative tolerance for the condensed "
                         "OSQP backend. Default 1e-5.")
    ap.add_argument("--mpc-osqp-max-iter", type=int, default=10000,
                    help="Max OSQP iterations for the condensed backend.")
    ap.add_argument("--mpc-osqp-polish", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Enable OSQP polishing for the condensed backend.")
    ap.add_argument("--r-scale", type=float, default=1.0,
                    help="Multiplier applied to MPCConfig.R_diag. Only used "
                         "when --r-scale != 1.0 AND --r-diag is not given; "
                         "scales the library defaults.")
    ap.add_argument("--r-diag", type=float, nargs=4, default=None,
                    metavar=("R_SURGE", "R_SWAY", "R_HEAVE", "R_YAW"),
                    help="Explicit 4-axis command R values.")
    ap.add_argument("--tag", type=str, default="",
                    help="Optional suffix appended to output filenames (before .npz).")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Root directory for outputs. Defaults to EDMDc/data; "
                         "npz goes to <root>/<scenario>/backstep_mpc and plots "
                         "to <root>/plots/<scenario>.")
    return ap


# ----- 3rd-order yaw ref model (wrap-aware) -----

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
MPC_N = int(args.mpc_N)
MPC_DECIMATION = int(args.mpc_decimation)
print(f"[backstep-mpc] model = {args.model}")
print(f"[backstep-mpc] {N_WAYPOINTS} waypoints:")
for _i, _wp in enumerate(WAYPOINTS):
    print(f"  wp{_i}: {_wp.tolist()}")
print(f"[backstep-mpc] MPC horizon N={MPC_N}, decimation={MPC_DECIMATION}, "
      f"inscribed_caps_flag={bool(args.inscribed_caps)}")

FFMPEG_PATH = None if args.no_video else resolve_ffmpeg()


def _make_mpc_config():
    from .mpc import MPCConfig
    default_cfg = MPCConfig()
    if args.r_diag is not None:
        scaled_R = tuple(float(r) for r in args.r_diag)
        print(f"[backstep-mpc] R_diag override: {scaled_R}")
    elif abs(args.r_scale - 1.0) > 1e-9:
        scaled_R = tuple(r * float(args.r_scale) for r in default_cfg.R_diag)
        print(f"[backstep-mpc] R-scale = {args.r_scale} on library defaults: "
              f"R_diag = {scaled_R}")
    else:
        scaled_R = default_cfg.R_diag
        print(f"[backstep-mpc] using default R_diag: {scaled_R}")
    kwargs = dict(
        N=MPC_N,
        dt=args.dt,
        R_diag=scaled_R,
        solver_backend=args.mpc_backend,
        osqp_eps_abs=args.mpc_osqp_eps,
        osqp_eps_rel=args.mpc_osqp_eps,
        osqp_max_iter=args.mpc_osqp_max_iter,
        osqp_polish=bool(args.mpc_osqp_polish),
    )
    return MPCConfig(**kwargs)


# ====== Main ======

def main() -> int:
    scene = GripperScene(
        dt=args.dt,
        mass_kg=args.mass_kg,
        headless=bool(args.headless),
        neutral_buoyancy=bool(args.neutral_buoyancy),
        payload_cube=bool(args.payload_cube),
        spawn_pos=np.asarray(args.spawn, dtype=np.float64),
    )
    if args.floor_grid:
        scene.add_floor_grid()
    chase = ChaseCam(scene, ffmpeg_path=FFMPEG_PATH)

    from .mpc import EDMDcMPC
    mpc = EDMDcMPC.from_npz(args.model, config=_make_mpc_config())
    model_kind = getattr(mpc, "model_kind", "edmdc")
    print(f"[backstep-mpc] MPC built: kind={model_kind}, state_dim={mpc.d}, input_dim={mpc.m}, "
          f"backend={getattr(mpc, 'solver_backend', 'unknown')}, "
          f"u_min={mpc.cfg.u_min}, u_max={mpc.cfg.u_max}")

    # ---------- Settle ----------
    print(f"[backstep-mpc] settling up to {args.settle_s:g} s...")
    state0 = scene.settle(max_s=args.settle_s,
                          min_s=args.settle_min_s,
                          tol=args.settle_tol,
                          on_step=lambda _s: chase.follow_and_record())
    pos0 = state0.pos_w
    yaw0 = float(state0.yaw)
    print(f"[backstep-mpc] settled p={pos0.round(3).tolist()}, "
          f"yaw={np.rad2deg(yaw0):+.2f}, "
          f"roll={np.rad2deg(state0.roll):+.2f}, "
          f"pitch={np.rad2deg(state0.pitch):+.2f} deg")

    # ---------- Outer-loop setup ----------
    eta_ghost = np.concatenate([pos0.copy(), np.zeros(3), np.zeros(3)])
    yaw_ghost = np.array([yaw0, 0.0, 0.0])

    current_wp_idx = 0
    target = WAYPOINTS[current_wp_idx].copy()
    psi_target = float(np.arctan2(target[1] - pos0[1], target[0] - pos0[0]))
    print(f"[backstep-mpc] starting wp {current_wp_idx+1}/{N_WAYPOINTS} = "
          f"{target.tolist()}, psi_target = {np.rad2deg(psi_target):+.2f} deg")
    print(f"[backstep-mpc] omega_pos = {OMEGA.tolist()} rad/s, "
          f"Lambda = {LAMBDA.tolist()} 1/s, "
          f"omega_yaw = {args.omega_yaw}, k_psi = {args.k_psi}")

    # ---------- Video + live plots ----------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    PROJECT = Path(__file__).resolve().parents[1]
    if not args.no_video:
        out_video = PROJECT / "EDMDc" / "recording" / f"backstep_edmdc_mpc_34_{ts}.mp4"
        chase.start_recording(out_video)

    scenario_subdir = "with_gripper_cube" if args.payload_cube else "with_gripper"
    data_root = args.out_root if args.out_root is not None \
        else PROJECT / "EDMDc" / "data"
    plots_dir = data_root / "plots" / scenario_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)
    live_vel_png_path = plots_dir / f"backstep_mpc_{ts}_live_vel.png"
    live_3d_png_path  = plots_dir / f"backstep_mpc_{ts}_live_3d.png"
    live_thr_png_path = plots_dir / f"backstep_mpc_{ts}_live_thr.png"

    live = None
    live3d = None
    live_thr = None
    if not args.no_live_plot:
        wp_brief = " -> ".join(f"({w[0]:.1f},{w[1]:.1f},{w[2]:.1f})"
                                for w in WAYPOINTS)
        live = LivePlot2D(
            dt=args.dt,
            title=(f"Backstep+MPC: {N_WAYPOINTS} wp [{wp_brief}] -- "
                   f"nu vs backstep ref (live)"),
            window_seconds=args.duration,
            label_meas="Isaac (nu)", label_ref="backstep ref",
            save_on_close=str(live_vel_png_path),
        )
        live3d = LivePlot3D(
            title=(f"Backstep+MPC nav 3D path -- {N_WAYPOINTS} waypoints"),
            save_on_close=str(live_3d_png_path),
        )
        live3d.push_meta(start=pos0,
                         waypoints=WAYPOINTS,
                         current_idx=current_wp_idx)
        if not args.no_live_thrusters:
            live_thr = LivePlotThrusters(
                dt=args.dt,
                title=(f"Thruster commands (MPC) -- {scene.num_rotors} rotors"),
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
    log_eta        = np.empty((n_steps, 3))
    log_nu         = np.empty((n_steps, 6))
    log_nu_ref     = np.empty((n_steps, 6))
    log_eta_d      = np.empty((n_steps, 3))
    log_eta_dot_d  = np.empty((n_steps, 3))
    log_eta_ddot_d = np.empty((n_steps, 3))
    log_psi_d      = np.empty(n_steps)
    log_psi_dot_d  = np.empty(n_steps)
    log_e_pos      = np.empty((n_steps, 3))
    log_e_wp       = np.empty((n_steps, 3))
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

        e_pos = eta_d - s.pos_w
        v_world_ref = eta_dot_d + LAMBDA * e_pos
        v_body_ref = s.R_body.T @ v_world_ref

        e_yaw = wrap_to_pi(psi_d - s.yaw)
        r_ref = psi_dot_d + args.k_psi * e_yaw

        nu_ref = np.zeros(6, dtype=np.float64)
        nu_ref[0] = float(np.clip(v_body_ref[0], -args.u_max, args.u_max))
        nu_ref[1] = float(np.clip(v_body_ref[1], -args.v_max, args.v_max))
        nu_ref[2] = float(np.clip(v_body_ref[2], -args.w_max, args.w_max))
        nu_ref[5] = float(np.clip(r_ref,         -args.r_max, args.r_max))

        # --- Inner EDMDc-MPC (with decimation: solve every MPC_DECIMATION ticks)
        if k % MPC_DECIMATION == 0:
            u_held = mpc.step(s.nu, nu_ref)
        elif hasattr(mpc, "observe"):
            mpc.observe(s.nu)

        # --- Plant ---
        thrusts = scene.apply_wrench(u_held)
        chase.follow_and_record()

        # --- Log ---
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

        e_to_wp_norm = float(np.linalg.norm(log_e_wp[k]))
        if (arrival_steps[current_wp_idx] < 0
                and e_to_wp_norm < args.arrival_tol):
            arrival_steps[current_wp_idx] = k
            print(f"[backstep-mpc] arrived at wp {current_wp_idx+1}/"
                  f"{N_WAYPOINTS} = {target.tolist()} at t = {t:.2f} s "
                  f"(|wp-eta_p| = {e_to_wp_norm:.3f} m)")

        if live is not None:
            live.push(t, s.nu, nu_ref)
        if live3d is not None:
            live3d.push(t, s.pos_w)
        if live_thr is not None:
            live_thr.push(t, thrusts)

        eta_ghost = step_ref_model(eta_ghost, target, OMEGA, args.zeta, args.dt)
        yaw_ghost = step_yaw_ref(yaw_ghost, psi_target,
                                 args.omega_yaw, args.zeta_yaw, args.dt)

        if (arrival_steps[current_wp_idx] >= 0
                and current_wp_idx < N_WAYPOINTS - 1
                and (k - arrival_steps[current_wp_idx]) >= hold_steps):
            current_wp_idx += 1
            target = WAYPOINTS[current_wp_idx].copy()
            psi_target = float(np.arctan2(target[1] - s.pos_w[1],
                                          target[0] - s.pos_w[0]))
            print(f"[backstep-mpc] switching to wp {current_wp_idx+1}/"
                  f"{N_WAYPOINTS} = {target.tolist()}, "
                  f"psi_target = {np.rad2deg(psi_target):+.2f} deg")
            if live3d is not None:
                live3d.push_meta(current_idx=current_wp_idx)

    # ---------- Save (BEFORE scene.close to survive process teardown) ----------
    solve_count = int(getattr(mpc, "solve_count", 0))
    solve_time_total = float(getattr(mpc, "solve_time_total_s", 0.0))
    solve_time_mean = solve_time_total / solve_count if solve_count else np.nan
    solve_fail_count = int(getattr(mpc, "solve_fail_count", 0))
    print(f"[backstep-mpc] MPC solve stats: backend={getattr(mpc, 'solver_backend', 'unknown')}, "
          f"count={solve_count}, failures={solve_fail_count}, "
          f"total={solve_time_total:.3f}s, mean={solve_time_mean * 1e3:.3f} ms")

    out_npz_dir = data_root / scenario_subdir / "backstep_mpc"
    out_npz_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("_" + args.tag) if args.tag else ""
    npz_path = out_npz_dir / f"{ts}{suffix}.npz"
    # Recover the active R values that the MPC ended up using (post-scale/override).
    used_R = mpc.cfg.R_diag if hasattr(mpc.cfg, "R_diag") else None
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
             mpc_N=MPC_N, mpc_decimation=MPC_DECIMATION,
             mpc_solver_backend=str(getattr(mpc, "solver_backend", "unknown")),
             mpc_last_solver_status=str(getattr(mpc, "last_solver_status", "")),
             mpc_solve_count=solve_count,
             mpc_solve_fail_count=solve_fail_count,
             mpc_solve_time_total_s=solve_time_total,
             mpc_solve_time_mean_s=solve_time_mean,
             input_units="ardusub_normalized_[-1,1]",
             input_names=np.array(["surge", "sway", "heave", "yaw"]),
             model_path=str(args.model),
             model_kind=str(model_kind),
             payload_cube=bool(args.payload_cube),
             scenario=scenario_subdir,
             inscribed_caps_flag=bool(args.inscribed_caps),
             r_scale=float(args.r_scale),
             r_diag=np.asarray(used_R if used_R is not None else (), dtype=np.float64),
             tag=args.tag)
    print(f"[backstep-mpc] saved -> {npz_path}")
    print(f"[backstep-mpc] arrivals: " + ", ".join(
        f"wp{i+1}={'t='+f'{log_t[s_]:.2f}s' if s_ >= 0 else 'NOT REACHED'}"
        for i, s_ in enumerate(arrival_steps)))

    if not args.no_video:
        chase.stop_recording()

    if args.no_diagnostic_plot:
        if live is not None:
            live.close()
            print(f"[backstep-mpc] live velocity plot saved -> {live_vel_png_path}")
        if live3d is not None:
            live3d.close()
            print(f"[backstep-mpc] live 3D plot saved -> {live_3d_png_path}")
        if live_thr is not None:
            live_thr.close()
            print(f"[backstep-mpc] live thruster plot saved -> {live_thr_png_path}")
        chase.close()
        scene.close()
        return 0

    # ---------- Diagnostic plot ----------
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))

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

    ax = axes[1, 0]
    names = ["u", "v", "w", "p", "q", "r"]
    for i in (0, 1, 2, 5):
        ax.plot(log_t, log_nu[:, i], color=f"C{i}", lw=1.3, alpha=0.9,
                label=f"{names[i]}")
        ax.plot(log_t, log_nu_ref[:, i], color=f"C{i}", lw=0.9, ls="--",
                label=f"{names[i]} ref")
    ax.set_xlabel("t [s]"); ax.set_ylabel("velocity")
    ax.set_title("body nu vs backstep ref")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=7, ncol=4)

    ax = axes[1, 1]
    ax.plot(log_t, np.rad2deg(log_eta[:, 0]), label="roll", lw=1.3)
    ax.plot(log_t, np.rad2deg(log_eta[:, 1]), label="pitch", lw=1.3)
    ax.plot(log_t, np.rad2deg(log_eta[:, 2]), label="yaw", lw=1.3)
    ax.plot(log_t, np.rad2deg(log_psi_d), "C2", ls=":", lw=0.9,
            label="psi_d (ghost)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("deg")
    ax.set_title("attitude + yaw ghost")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)

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
    ax.set_title("position errors")
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8)

    ax = axes[2, 1]
    wnames = ["surge", "sway", "heave", "yaw"]
    for i in range(4):
        ax.plot(log_t, log_u[:, i], label=wnames[i], lw=1.0, alpha=0.85)
    for s_arr in arrival_steps:
        if s_arr >= 0:
            ax.axvline(log_t[s_arr], color="k", ls=":", lw=0.7, alpha=0.6)
    ax.set_xlabel("t [s]"); ax.set_ylabel("normalized command")
    ax.set_title("MPC ArduSub command")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8, ncol=2)

    n_reached = sum(1 for s_ in arrival_steps if s_ >= 0)
    fig.suptitle(
        f"Backstepping + EDMDc-MPC-34  -  "
        f"{N_WAYPOINTS} waypoints (reached {n_reached}/{N_WAYPOINTS})  -  "
        f"ArduSub cmd [-1,1]^4  -  TS {ts}",
        fontsize=12)
    fig.tight_layout()
    png_path = plots_dir / f"backstep_mpc_{ts}{suffix}.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"[backstep-mpc] plot -> {png_path}")

    if live is not None:
        live.close()
        print(f"[backstep-mpc] live velocity plot saved -> {live_vel_png_path}")
    if live3d is not None:
        live3d.close()
        print(f"[backstep-mpc] live 3D plot saved -> {live_3d_png_path}")
    if live_thr is not None:
        live_thr.close()
        print(f"[backstep-mpc] live thruster plot saved -> {live_thr_png_path}")
    chase.close()
    scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
