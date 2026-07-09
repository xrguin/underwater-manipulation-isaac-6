"""Collect APRBS-driven snapshots from the NumPy Fossen integrator.

Design choices:

  1. State stored is **body velocity only**: nu = [u, v, w, p, q, r].
     Pose eta is held in scope for envelope checks but not saved.
  2. Per-trajectory deterministic RNG so the IC + APRBS sequence is reproducible
     from `(master_seed, trajectory_index)`. Same seed → identical inputs in
     `collect_isaac.py` (prerequisite for Fossen-vs-PhysX A/B comparison).
  3. Drain phase: each trajectory continues with tau = 0 after the APRBS
     sequence until ‖v‖ < DRAIN_VEL_TOL (or DRAIN_MAX_STEPS as a safety cap).
     Drain snapshots are recorded with U = 0 so the EDMDc model sees clean
     decay-to-rest behavior.

Defaults: 4-DOF task wrench excitation (Fx, Fy, Fz, Tz; Tx, Ty zeroed) and
neutral buoyancy ON. Use `--fx-only` for surge-only validation, or
`--no-neutral-buoyancy` for the YAML's actual mass/volume mismatch.

Snapshot schema:
    X      : (N, 6) body velocity nu at t  [u, v, w, p, q, r]
    U      : (N, 6) body wrench applied between t and t+1
    X_next : (N, 6) body velocity nu at t+1
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Run timestamp — stamped into the default --out filename so successive runs
# don't overwrite. Override with `--out` to pin a custom path.
_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

from ._common import default_yaml, project_root
from .ardusub_allocator import StaticArduSubAllocator
from .fossen_integrator import (
    FossenIntegratorParams,
    RHO_WATER,
    step_fossen,
)
from .thrusters import ThrusterConfig, T200Group, sum_to_wrench


# ---- APRBS excitation parameters --------------------------------------------
# Per-axis amplitudes (Fx, Fy, Fz, Tx, Ty, Tz). Aligned with WRENCH_CMD_MAX_*
# below so APRBS_LEVELS {-1, -0.5, 0, 0.5, 1} map to {-cap, -cap/2, 0,
# +cap/2, +cap}. The downstream _clip_wrench_command then becomes a no-op
# safety guard. Heave (+Fz up, −Fz down) is asymmetric.
APRBS_AMPS_POS = np.array([65.0, 37.5, 102.0, 7.5, 7.5, 17.0])
APRBS_AMPS_NEG = np.array([65.0, 37.5, 80.0,  7.5, 7.5, 17.0])
APRBS_LEVELS   = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
APRBS_HOLD_MIN_STEPS = 6    # 0.1 s @ dt = 1/60
APRBS_HOLD_MAX_STEPS = 30   # 0.5 s @ dt = 1/60

# Max-volume inscribed box of the 4-DOF achievable wrench polytope
# (per-rotor cap 25 N). Computed by EDMDc.find_inscribed_aprbs_box: every
# corner of this box has rotor demand exactly 25 N → no saturation
# anywhere inside, the recorded U is the realized wrench. Used by the
# continuous-uniform APRBS mode (see aprbs_sequence_continuous and the
# --continuous-aprbs CLI flag in collect_isaac_38).
INSCRIBED_BOX_CAPS_POS = np.array([22.57, 23.47, 88.17, 0.0, 0.0, 6.59])
INSCRIBED_BOX_CAPS_NEG = np.array([22.57, 23.47, 69.15, 0.0, 0.0, 6.59])

# ---- Initial-condition + pool envelope --------------------------------------
IC_RADIUS_M       = 2.0      # IC spawn sphere radius (m)
POOL_Z_HALF       = 1.8      # ±half-pool depth (m); breach if |z| > this
TRUNCATE_RADIUS_M = 6.0      # horizontal envelope radius (m)
PITCH_LIMIT       = 1.5      # rad ≈ 86°; roll & pitch tipover threshold

# Body-wrench clip caps applied to APRBS samples before integration. These
# mirror collect_isaac.py exactly so both collectors see the same input
# wrench distribution given the same seed (half the APRBS_AMPS used by the
# canonical collector; chosen to match the Isaac thruster pipeline's
# allocator + per-motor 25 N saturation).
# Order: Fx, Fy, Fz [N], Tx, Ty, Tz [N·m]. Heave (+Fz up, −Fz down) is asymmetric.
WRENCH_CMD_MAX_POS = np.array(
    [65.0, 37.5, 102.0, 7.5, 7.5, 17.0], dtype=np.float64
)
WRENCH_CMD_MAX_NEG = np.array(
    [65.0, 37.5, 80.0, 7.5, 7.5, 17.0], dtype=np.float64
)

COMMAND_DIM = 4
COMMAND_NAMES = ("surge", "sway", "heave", "yaw")
COMMAND_MAX_POS = np.ones(COMMAND_DIM, dtype=np.float64)
COMMAND_MAX_NEG = np.ones(COMMAND_DIM, dtype=np.float64)


def _clip_wrench_command(u_raw: np.ndarray) -> np.ndarray:
    """Clip APRBS body wrench to ±WRENCH_CMD_MAX_{POS,NEG} per axis."""
    u = np.asarray(u_raw, dtype=np.float64).copy()
    np.minimum(u, WRENCH_CMD_MAX_POS, out=u)
    np.maximum(u, -WRENCH_CMD_MAX_NEG, out=u)
    return u


def _clip_command(u_raw: np.ndarray) -> np.ndarray:
    u = np.asarray(u_raw, dtype=np.float64).copy().reshape(COMMAND_DIM)
    np.minimum(u, COMMAND_MAX_POS, out=u)
    np.maximum(u, -COMMAND_MAX_NEG, out=u)
    return u


def command_axis_mask(*, fx_only: bool = False) -> np.ndarray:
    mask = np.ones(COMMAND_DIM, dtype=bool)
    if fx_only:
        mask[:] = False
        mask[0] = True
    return mask


def aprbs_sequence_command4(
    rng: np.random.Generator,
    n_steps: int,
    axis_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Pre-generate normalized ArduSub commands in [-1,1]."""
    if axis_mask is None:
        axis_mask = np.ones(COMMAND_DIM, dtype=bool)
    axis_mask = np.asarray(axis_mask, dtype=bool).reshape(COMMAND_DIM)
    out = np.zeros((n_steps, COMMAND_DIM), dtype=float)
    levels = np.zeros(COMMAND_DIM, dtype=float)
    holds = np.zeros(COMMAND_DIM, dtype=int)
    for k in range(n_steps):
        for axis in range(COMMAND_DIM):
            if not axis_mask[axis]:
                continue
            if holds[axis] <= 0:
                levels[axis] = float(rng.uniform(-1.0, 1.0))
                holds[axis] = int(rng.integers(
                    APRBS_HOLD_MIN_STEPS, APRBS_HOLD_MAX_STEPS + 1,
                ))
        out[k, :] = levels
        out[k, ~axis_mask] = 0.0
        holds -= 1
    return out


# Drain phase: after APRBS ends, apply tau = 0 and keep stepping until the
# body velocity decays to rest. Snapshots in this phase carry U = 0 and let
# the EDMDc model see clean decay-to-rest behavior (damping + buoyancy).
DRAIN_VEL_TOL = 1e-2      # ‖v‖ threshold (mixed m/s + rad/s units)
DRAIN_MAX_STEPS = 600     # safety cap (~10 s at dt = 1/60)


@dataclass
class CollectionConfig:
    n_trajectories: int = 100
    episode_seconds: float = 5.0
    dt: float = 1.0 / 60.0
    init_lin_vel_sigma: float = 0.2
    init_ang_vel_sigma: float = 0.1
    init_roll_pitch_max: float = 0.15
    init_yaw_rel_max: float = np.pi
    init_radius: float = IC_RADIUS_M
    seed: int = 0
    mass_kg: float = 13.5
    yaml_path: str = ""  # filled in main() if blank
    task_dof: int = 4    # 4 = (Fx, Fy, Fz, Tz) only; 6 = full 6-DOF wrench
    fx_only: bool = False  # True ⇒ APRBS on Fx only (ignores task_dof mask).


def trajectory_rng(master_seed: int, traj_idx: int) -> np.random.Generator:
    """Deterministic per-trajectory RNG. SeedSequence-based so both
    collectors get bit-identical sequences from the same (seed, idx) pair."""
    ss = np.random.SeedSequence([int(master_seed), int(traj_idx)])
    return np.random.default_rng(ss)


def sample_initial_eta_v(
    rng: np.random.Generator, cfg: CollectionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (eta_world, v_body) for a fresh trajectory.

    Returns world-frame pose + body-frame velocity (12 numbers total).
    These are passed to ``state_from_eta_v`` then ``euler_to_trig`` to
    produce the 15-dim absolute world-coords snapshot.
    """
    eta = np.zeros(6, dtype=float)
    direction = rng.normal(0.0, 1.0, size=3)
    nrm = float(np.linalg.norm(direction))
    if nrm < 1e-9:
        direction = np.array([1.0, 0.0, 0.0])
        nrm = 1.0
    direction = direction / nrm
    radius = float(cfg.init_radius * rng.uniform(0.0, 1.0) ** (1.0 / 3.0))
    eta[0:3] = radius * direction
    eta[2] = float(np.clip(eta[2], -POOL_Z_HALF, POOL_Z_HALF))
    eta[3] = rng.uniform(-cfg.init_roll_pitch_max, cfg.init_roll_pitch_max)
    eta[4] = rng.uniform(-cfg.init_roll_pitch_max, cfg.init_roll_pitch_max)
    eta[5] = rng.uniform(-cfg.init_yaw_rel_max, cfg.init_yaw_rel_max)

    v = np.empty(6, dtype=float)
    v[0:3] = rng.normal(0.0, cfg.init_lin_vel_sigma, size=3)
    v[3:6] = rng.normal(0.0, cfg.init_ang_vel_sigma, size=3)
    return eta, v


def aprbs_sequence(
    rng: np.random.Generator, n_steps: int,
    axis_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Pre-generate the full APRBS wrench sequence for one trajectory.

    Returns (n_steps, 6). Per-axis independent amplitude-modulated PRBS.
    Hold times drawn from [APRBS_HOLD_MIN_STEPS, APRBS_HOLD_MAX_STEPS].
    Forced level transitions (no consecutive duplicates).

    Pre-generating here (rather than streaming with stateful counters) means
    the Isaac collector can re-use this exact sequence by calling the same
    function with the same trajectory_rng — no risk of RNG drift.

    Args:
        axis_mask: optional (6,) boolean. Axes set to False are zeroed in
            the output and consume zero RNG draws so masked vs unmasked runs
            produce identical sequences on the active axes for the same
            seed. Default None = all True (full 6-DOF excitation).
    """
    if axis_mask is None:
        axis_mask = np.ones(6, dtype=bool)
    else:
        axis_mask = np.asarray(axis_mask, dtype=bool).reshape(6)
    out = np.zeros((n_steps, 6), dtype=float)
    levels = np.zeros(6, dtype=float)
    holds = np.zeros(6, dtype=int)
    for k in range(n_steps):
        for axis in range(6):
            if not axis_mask[axis]:
                continue                        # skip RNG draws on masked axes
            if holds[axis] <= 0:
                cands = APRBS_LEVELS[APRBS_LEVELS != levels[axis]]
                levels[axis] = float(rng.choice(cands))
                holds[axis] = int(rng.integers(
                    APRBS_HOLD_MIN_STEPS, APRBS_HOLD_MAX_STEPS + 1,
                ))
        amps = np.where(levels >= 0.0, APRBS_AMPS_POS, APRBS_AMPS_NEG)
        out[k, :] = levels * amps
        out[k, ~axis_mask] = 0.0
        holds -= 1
    return out


def aprbs_sequence_continuous(
    rng: np.random.Generator, n_steps: int,
    caps_pos: np.ndarray | None = None,
    caps_neg: np.ndarray | None = None,
    axis_mask: np.ndarray | None = None,
    hold_min_steps: int | np.ndarray | None = None,
    hold_max_steps: int | np.ndarray | None = None,
) -> np.ndarray:
    """Continuous-uniform APRBS variant. Same hold-time structure as
    `aprbs_sequence` but each hold's amplitude is drawn from
    Uniform(-caps_neg[axis], +caps_pos[axis]) instead of the 5-level
    discrete grid.

    Default caps come from `INSCRIBED_BOX_CAPS_*`, which are the
    max-volume box inscribed in the 4-DOF achievable wrench polytope
    (per-rotor cap 25 N). Every sample is realizable → recorded U
    matches realized wrench, no allocator-saturation bias.

    Hold times can be scalar (applied to every axis) or a 6-element array
    for per-axis tuning. Useful for axes with weak damping (yaw has zero
    linear damping; longer holds let r reach the steady-state regime
    during training).

    Returns (n_steps, 6). Per-axis independent. To preserve RNG
    determinism, axes masked off consume no RNG draws.
    """
    if caps_pos is None:
        caps_pos = INSCRIBED_BOX_CAPS_POS
    if caps_neg is None:
        caps_neg = INSCRIBED_BOX_CAPS_NEG
    caps_pos = np.asarray(caps_pos, dtype=float).reshape(6)
    caps_neg = np.asarray(caps_neg, dtype=float).reshape(6)
    if axis_mask is None:
        axis_mask = np.ones(6, dtype=bool)
    else:
        axis_mask = np.asarray(axis_mask, dtype=bool).reshape(6)

    # Per-axis hold range. Scalar broadcasts to all axes.
    if hold_min_steps is None:
        hmin = np.full(6, APRBS_HOLD_MIN_STEPS, dtype=int)
    elif np.isscalar(hold_min_steps):
        hmin = np.full(6, int(hold_min_steps), dtype=int)
    else:
        hmin = np.asarray(hold_min_steps, dtype=int).reshape(6)
    if hold_max_steps is None:
        hmax = np.full(6, APRBS_HOLD_MAX_STEPS, dtype=int)
    elif np.isscalar(hold_max_steps):
        hmax = np.full(6, int(hold_max_steps), dtype=int)
    else:
        hmax = np.asarray(hold_max_steps, dtype=int).reshape(6)

    out = np.zeros((n_steps, 6), dtype=float)
    levels = np.zeros(6, dtype=float)
    holds = np.zeros(6, dtype=int)
    for k in range(n_steps):
        for axis in range(6):
            if not axis_mask[axis]:
                continue
            if holds[axis] <= 0:
                # Continuous uniform draw in [-caps_neg, +caps_pos] for this axis.
                levels[axis] = float(rng.uniform(-caps_neg[axis], caps_pos[axis]))
                holds[axis] = int(rng.integers(hmin[axis], hmax[axis] + 1))
        out[k, :] = levels
        out[k, ~axis_mask] = 0.0
        holds -= 1
    return out


# Standard 4-DOF mask: drop Tx (roll torque) and Ty (pitch torque).
# Matches the deploy-time task space (MPC plans Fx, Fy, Fz, Tz; the
# AttitudeStabilizer handles roll/pitch).
TASK_DOF_4_MASK = np.array([True, True, True, False, False, True], dtype=bool)


def axis_mask_for(task_dof: int, *, fx_only: bool = False) -> np.ndarray:
    if fx_only:
        m = np.zeros(6, dtype=bool)
        m[0] = True
        return m
    if task_dof == 4:
        return TASK_DOF_4_MASK.copy()
    if task_dof == 6:
        return np.ones(6, dtype=bool)
    raise ValueError(f"Unsupported task_dof={task_dof}; expected 4 or 6.")


def _in_envelope(x_euler: np.ndarray) -> bool:
    """Truncate when ROV leaves the validated envelope (world coords)
    or tips over in roll/pitch beyond PITCH_LIMIT."""
    px, py, pz = x_euler[:3]
    horizontal_r = float(np.hypot(px, py))
    return (
        horizontal_r <= TRUNCATE_RADIUS_M
        and abs(pz) <= POOL_Z_HALF
        and abs(x_euler[3]) < PITCH_LIMIT  # roll  phi  (tipover guard)
        and abs(x_euler[4]) < PITCH_LIMIT  # pitch theta (tipover + gimbal-safe)
    )


def collect(
    p: FossenIntegratorParams,
    cfg: CollectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run all trajectories. Returns 6-D body-velocity snapshot triples (X, U, X_next).

    Each trajectory has two phases:
      1. APRBS: ``episode_seconds`` of amplitude-modulated PRBS excitation.
      2. Drain: tau = 0 applied until ``‖v‖ < DRAIN_VEL_TOL`` or
         ``DRAIN_MAX_STEPS`` steps elapse. Drain snapshots are stored with
         ``U = 0`` and capture natural decay (damping + buoyancy).

    Stored state is body velocity nu = [u, v, w, p, q, r] only. The pose
    eta is kept in scope for envelope checks but not saved.
    """
    n_steps = int(round(cfg.episode_seconds / cfg.dt))
    X_list:    list[np.ndarray] = []
    U_list:    list[np.ndarray] = []
    Xn_list:   list[np.ndarray] = []
    traj_list: list[int] = []          # per-snapshot trajectory index
    step_list: list[int] = []          # per-snapshot step within trajectory
    n_truncated = 0
    n_drain_total = 0
    zero_command = np.zeros(COMMAND_DIM, dtype=float)

    with open(cfg.yaml_path) as f:
        thruster_cfg = ThrusterConfig.from_yaml_dict(yaml.safe_load(f))
    t200 = T200Group(thruster_cfg)
    allocator = StaticArduSubAllocator()

    mask = command_axis_mask(fx_only=cfg.fx_only)
    for traj in range(cfg.n_trajectories):
        rng = trajectory_rng(cfg.seed, traj)
        eta, v = sample_initial_eta_v(rng, cfg)
        U_seq = aprbs_sequence_command4(rng, n_steps, axis_mask=mask)
        t200.reset()

        truncated = False
        # ---- APRBS phase ----
        for k in range(n_steps):
            u = _clip_command(U_seq[k])
            commands = allocator.allocate(u)
            thrusts, _ = t200.step(commands, cfg.dt)
            user_wb = sum_to_wrench(thruster_cfg, thrusts)
            eta_next, v_next = step_fossen(eta, v, user_wb, p, cfg.dt)
            # Envelope + gimbal check on raw eta (eta[:3] = position,
            # eta[4] = pitch).
            if (not _in_envelope(eta_next)
                    or not np.all(np.isfinite(v_next))):
                n_truncated += 1
                truncated = True
                break
            X_list.append(v.copy())
            U_list.append(u.copy())
            Xn_list.append(v_next.copy())
            traj_list.append(traj)
            step_list.append(k)
            eta, v = eta_next, v_next

        # ---- Drain phase: tau = 0 until ‖v‖ < DRAIN_VEL_TOL ----
        if not truncated:
            for d in range(DRAIN_MAX_STEPS):
                commands = allocator.allocate(zero_command)
                thrusts, _ = t200.step(commands, cfg.dt)
                user_wb = sum_to_wrench(thruster_cfg, thrusts)
                eta_next, v_next = step_fossen(eta, v, user_wb, p, cfg.dt)
                if (not _in_envelope(eta_next)
                        or not np.all(np.isfinite(v_next))):
                    n_truncated += 1
                    break
                X_list.append(v.copy())
                U_list.append(zero_command.copy())
                Xn_list.append(v_next.copy())
                traj_list.append(traj)
                step_list.append(n_steps + d)
                eta, v = eta_next, v_next
                n_drain_total += 1
                if float(np.linalg.norm(v_next)) < DRAIN_VEL_TOL:
                    break

    X  = np.asarray(X_list,  dtype=float)
    U  = np.asarray(U_list,  dtype=float)
    Xn = np.asarray(Xn_list, dtype=float)
    traj_idx = np.asarray(traj_list, dtype=np.int32)
    step_idx = np.asarray(step_list, dtype=np.int32)
    n_kept = cfg.n_trajectories - n_truncated
    print(
        f"[fossen] kept {n_kept}/{cfg.n_trajectories} trajectories "
        f"({n_truncated} truncated by envelope/tipover); "
        f"{len(X):,} total snapshots "
        f"(seed={cfg.seed}, episode={cfg.episode_seconds:.1f}s + drain, "
        f"drain_total={n_drain_total})"
    )
    return X, U, Xn, traj_idx, step_idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, dest="n_trajectories")
    ap.add_argument("--episode-s", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mass-kg", type=float, default=13.5)
    ap.add_argument("--yaml", type=str, default=None)
    ap.add_argument(
        "--fx-only",
        action="store_true",
        help="APRBS on surge (Fx) only; overrides the default 4-DOF mask.",
    )
    ap.add_argument(
        "--out", type=str,
        default=str(project_root() / "EDMDc" / "data" /"numpy"/ f"fossen_{_RUN_TS}.npz"),
        help="Output .npz path. Default: EDMDc/data/fossen_<YYYYMMDD_HHMMSS>.npz.",
    )
    ap.add_argument(
        "--neutral-buoyancy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Override Fossen volume so ρgV exactly cancels m·g. Default ON "
             "so drain trajectories reach rest cleanly. Pass --no-neutral-buoyancy "
             "to keep the YAML's actual mass/volume mismatch (terminal sink).",
    )
    args = ap.parse_args()

    yaml_path = args.yaml or default_yaml()
    cfg = CollectionConfig(
        n_trajectories=args.n_trajectories,
        episode_seconds=args.episode_s,
        dt=args.dt,
        seed=args.seed,
        mass_kg=args.mass_kg,
        yaml_path=str(yaml_path),
        task_dof=4,  # hardcoded: Fx, Fy, Fz, Tz only (Tx, Ty zeroed)
        fx_only=bool(args.fx_only),
    )
    p = FossenIntegratorParams.from_yaml(str(yaml_path), mass_kg=cfg.mass_kg)
    if args.neutral_buoyancy:
        p.volume = float(cfg.mass_kg) / RHO_WATER
        print(f"[neutral-buoyancy] volume override → {p.volume:.6f} m³ "
              f"(buoyancy {RHO_WATER * 9.81 * p.volume:.2f} N == "
              f"m·g {cfg.mass_kg * 9.81:.2f} N)")
    X, U, Xn, traj_idx, step_idx = collect(p, cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, U=U, X_next=Xn,
                        traj_idx=traj_idx, step_idx=step_idx,
                        seed=cfg.seed, dt=cfg.dt,
                        episode_seconds=cfg.episode_seconds,
                        n_trajectories=cfg.n_trajectories,
                        task_dof=cfg.task_dof,
                        fx_only=cfg.fx_only,
                        mass_kg=float(cfg.mass_kg),
                        neutral_buoyancy=bool(args.neutral_buoyancy),
                        input_units="ardusub_normalized_[-1,1]",
                        input_names=np.array(COMMAND_NAMES),
                        command_max_pos=COMMAND_MAX_POS,
                        command_max_neg=COMMAND_MAX_NEG,
                        drain_vel_tol=float(DRAIN_VEL_TOL),
                        drain_max_steps=int(DRAIN_MAX_STEPS))
    sz_mb = out_path.stat().st_size / 1e6
    print(f"[fossen] wrote {out_path}  ({sz_mb:.2f} MB)")

    # Auto-plot APRBS input distribution alongside the .npz.
    try:
        from .plot_inputs import plot_distribution
        plot_distribution([out_path], labels=[out_path.stem])
    except Exception as e:
        print(f"[fossen] WARNING: input-distribution plot failed ({e}); "
              f"run `python -m EDMDc.plot_inputs {out_path}` manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
