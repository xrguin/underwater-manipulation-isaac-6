"""Pure-NumPy Fossen 6-DOF integrator.

Mirrors the wrench computation in `fossen.py` but advances the state itself
(η, ν) rather than handing the wrench off to Isaac Sim's PhysX. Used by
`data.py` to generate training trajectories without running the simulator.

State vector convention (12-dim, matches dictionary.py):
    x = [pos(3), euler_zyx(3), v_lin(3), v_ang(3)]

Equations of motion (body-frame velocity, world-frame pose):

    M·v̇ + C(v)·v + D(v)·v + g(η) = τ
    η̇ = J(η)·v

  where
    M = M_RB + M_A     (constant 6×6 diagonal: mass+added_mass per axis)
    C(v)·v             — full 6×6 C_RB + C_A (Fossen 2011 Theorem 3.2):
                         off-diagonal blocks  = -m·S(v_lin) − S(M_A_lin·v_lin)
                         lower-right block    = -S(I_b·v_ang) − S(M_A_ang·v_ang)
                         Captures both rigid-body gyroscopic terms (the part
                         PhysX supplies in the Isaac-Sim stack) and the
                         hydrodynamic Coriolis/Munk effects.
    D(v)·v             = D_L·v + D_Q·|v|·v
    g(η)               = body-frame net gravity-buoyancy:
                         (W − B) on the linear axes (gravity rotated to body)
                         + cross(r_cob_body, B_body) on the angular axes
                         (buoyancy applied at COB offset above COG)
    J(η) =  [ R_zyx(η)        0_3 ]
            [ 0_3       T_eul(η)  ]

Sign convention: in `fossen.py:198–214` the wrench passed to PhysX is
  τ_hydro_body = −M_A·v̇ − C_A(v)·v − D(v)·v
  τ_buoy_world =  ρgV·ẑ_world  (applied at COB; rotated to world before sum)
PhysX adds gravity and integrates `(M_RB + M_A)·v̇ = τ_hydro + τ_user`.
We do the same accounting here, but in body frame, and integrate ν directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


GRAVITY = 9.81
RHO_WATER = 997.0


@dataclass
class FossenIntegratorParams:
    mass: float                       # rigid-body mass (kg) — required for gravity
    volume: float                     # m³, sets buoyancy = ρgV
    cob_offset: float                 # COB above COG along body +Z (m)
    added_mass: np.ndarray            # (6,) diagonal of M_A
    linear_damping: np.ndarray        # (6,) diagonal of D_L
    quadratic_damping: np.ndarray     # (6,) diagonal of D_Q

    @classmethod
    def from_yaml(cls, yaml_path: str, mass_kg: float) -> "FossenIntegratorParams":
        """Load the Heavy YAML the same way fossen.FossenParams does, but also
        accept the mass externally (the YAML doesn't store mass — it lives in
        the USD's MassAPI, hence the explicit arg here)."""
        import yaml
        with open(yaml_path, "r") as f:
            p = yaml.safe_load(f)
        h = p["hydro_coef"]
        return cls(
            mass=float(mass_kg),
            volume=float(p["volume"]),
            cob_offset=float(p["coBM"]),
            added_mass=np.asarray(h["added_mass"], dtype=float),
            linear_damping=np.asarray(h["linear_damping"], dtype=float),
            quadratic_damping=np.asarray(h["quadratic_damping"], dtype=float),
        )

    def M_total_diag(self) -> np.ndarray:
        """Diagonal of M = M_RB + M_A. For BlueROV the rigid-body inertia tensor
        is approximately diagonal in body frame (uniform-box approximation in the
        Heavy USD); for the integrator we treat M_RB on rotational axes as the
        same uniform-box inertia computed from the chassis bbox.
        """
        # Rigid-body translational mass = mass on each linear axis.
        M_RB_lin = np.full(3, self.mass)
        # Rigid-body rotational inertia: uniform-box at the Heavy chassis bbox.
        # Matches build_heavy_visual_from_blender.py / from_cad.py.
        # For BlueROV (smaller chassis) the values would differ; for first-pass
        # data collection use the Heavy uniform-box constants.
        a, b, c = 0.4572, 0.5748, 0.2537   # Heavy chassis extents (m)
        I_xx = (1.0 / 12.0) * self.mass * (b * b + c * c)
        I_yy = (1.0 / 12.0) * self.mass * (a * a + c * c)
        I_zz = (1.0 / 12.0) * self.mass * (a * a + b * b)
        M_RB_ang = np.array([I_xx, I_yy, I_zz])
        return np.concatenate([M_RB_lin, M_RB_ang]) + self.added_mass


def R_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body→world rotation, intrinsic Z-Y-X order. Matches dictionary.py."""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr           ],
    ])


def T_euler_zyx(roll: float, pitch: float) -> np.ndarray:
    """Body angular velocity → Euler-rate transformation for ZYX convention.

    [ϕ̇, θ̇, ψ̇]ᵀ = T(η) · [p, q, r]ᵀ  where η = (roll ϕ, pitch θ, yaw ψ).

    Singular at pitch = ±π/2 — we don't approach those during random-excitation
    data collection (initial roll/pitch sampled small and damping pulls them
    back), so the singularity isn't a practical concern.
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    if abs(cp) < 1e-6:
        cp = 1e-6
    return np.array([
        [1.0, sr * sp / cp, cr * sp / cp],
        [0.0, cr,           -sr         ],
        [0.0, sr / cp,      cr / cp     ],
    ])


def fossen_vdot(
    eta: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
    p: FossenIntegratorParams,
) -> np.ndarray:
    """Body-frame acceleration v̇ from Fossen's 6-DOF EOM.

    Args:
        eta: (6,) world-frame pose [x, y, z, roll, pitch, yaw].
        v:   (6,) body-frame velocity [u, v, w, p, q, r].
        tau: (6,) body-frame applied wrench [Fx, Fy, Fz, Tx, Ty, Tz].
        p:   FossenIntegratorParams.

    Returns:
        v̇: (6,) body-frame acceleration.
    """
    M_diag = p.M_total_diag()                # (6,)
    M_inv_diag = 1.0 / M_diag                 # diagonal inverse
    I_b = M_diag[3:] - p.added_mass[3:]       # pure rigid-body angular inertia

    # Full 6×6 C(ν) = C_RB(ν) + C_A(ν), Fossen 2011 Theorem 3.2 parametrization.
    # Off-diagonal blocks: -m·S(ν_lin) and -S(M_A_lin·ν_lin). Lower-right:
    # -S(I_b·ν_ang) and -S(M_A_ang·ν_ang). PhysX provides C_RB in `fossen.py`,
    # but this pure-NumPy integrator computes both halves explicitly.
    C_RB = np.array([
        [0.0, 0.0, 0.0, 0.0, p.mass * v[2], -p.mass * v[1]],
        [0.0, 0.0, 0.0, -p.mass * v[2], 0.0, p.mass * v[0]],
        [0.0, 0.0, 0.0, p.mass * v[1], -p.mass * v[0], 0.0],
        [0.0, p.mass * v[2], -p.mass * v[1], 0.0, I_b[2] * v[5], -I_b[1] * v[4]],
        [-p.mass * v[2], 0.0, p.mass * v[0], -I_b[2] * v[5], 0.0, I_b[0] * v[3]],
        [p.mass * v[1], -p.mass * v[0], 0.0, I_b[1] * v[4], -I_b[0] * v[3], 0.0],
    ])
    a = p.added_mass
    C_A = np.array([
        [0.0, 0.0, 0.0, 0.0, a[2] * v[2], -a[1] * v[1]],
        [0.0, 0.0, 0.0, -a[2] * v[2], 0.0, a[0] * v[0]],
        [0.0, 0.0, 0.0, a[1] * v[1], -a[0] * v[0], 0.0],
        [0.0, a[2] * v[2], -a[1] * v[1], 0.0, a[5] * v[5], -a[4] * v[4]],
        [-a[2] * v[2], 0.0, a[0] * v[0], -a[5] * v[5], 0.0, a[3] * v[3]],
        [a[1] * v[1], -a[0] * v[0], 0.0, a[4] * v[4], -a[3] * v[3], 0.0],
    ])
    Cv = (C_RB + C_A) @ v

    # Damping  D(v)·v = D_L·v + D_Q·|v|·v
    Dv = p.linear_damping * v + p.quadratic_damping * np.abs(v) * v

    # Restoring force g(η): body-frame gravity + body-frame buoyancy.
    R = R_zyx(*eta[3:6])
    weight_world = np.array([0.0, 0.0, -p.mass * GRAVITY])
    buoy_world = np.array([0.0, 0.0,  RHO_WATER * GRAVITY * p.volume])
    weight_body = R.T @ weight_world
    buoy_body = R.T @ buoy_world
    # Both forces resolved at COG (origin). Buoyancy also produces a torque
    # because it acts at the COB which is `cob_offset` above COG along body +Z.
    cob_body = np.array([0.0, 0.0, p.cob_offset])
    buoy_torque_body = np.cross(cob_body, buoy_body)
    g_force_body = weight_body + buoy_body
    g_eta = np.concatenate([g_force_body, buoy_torque_body])

    # Newton-Euler:  M v̇ = τ − C(v)v − D(v)v + g(η)
    # (g_eta as defined here is the *applied* gravity-buoyancy wrench in body
    # frame, hence + on the RHS. In the textbook Fossen form g(η) sits on the
    # LHS as a restoring term, which is the same equation with the opposite
    # sign convention.)
    rhs = tau - Cv - Dv + g_eta
    vdot = M_inv_diag * rhs
    return vdot


def step_fossen(
    eta: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
    p: FossenIntegratorParams,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single Euler step. RK4 didn't measurably help at dt=1/60 in spot-check
    rollouts; the dominant error term is the M_A · v̇ feedback that Fossen
    handles via LP-filter at the Isaac Sim integration boundary, and that's
    not present in the body-frame integrator (M_A is just lumped into M).

    Returns:
        eta_next, v_next.
    """
    vdot = fossen_vdot(eta, v, tau, p)
    v_next = v + dt * vdot
    # Pose kinematics: pos via R·v_lin, euler via T(η)·v_ang.
    R = R_zyx(*eta[3:6])
    pos_dot = R @ v[:3]
    eul_dot = T_euler_zyx(eta[3], eta[4]) @ v[3:]
    eta_next = eta.copy()
    eta_next[:3] += dt * pos_dot
    eta_next[3:] += dt * eul_dot
    # Do NOT wrap yaw. EDMDc fits a linear regression in the state, and a
    # discontinuity at ±π (yaw_t = 3.13, yaw_{t+1} = -3.10) injects 2π-magnitude
    # outliers that break the fit. Let yaw accumulate; the dictionary's
    # cos/sin terms handle the actual dynamics, and the bare-yaw row in the
    # state is only used for navigation reference tracking.
    return eta_next, v_next


def state_from_eta_v(eta: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Concatenate (η, ν) into the 12-dim WORLD-frame state.

    Kept for backward-compatibility with archived world-frame artifacts and
    multi-step prediction routines that operate in world coords. New code
    fitting the body-relative dictionary should call
    `state_from_body_relative` instead.
    """
    return np.concatenate([eta, v])


def state_from_eta_v_R(eta: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pack (η_world, ν_body) into the 18-dim **R-state**.

    Layout: ``[px, py, pz,
               R[0,0], R[0,1], R[0,2],
               R[1,0], R[1,1], R[1,2],
               R[2,0], R[2,1], R[2,2],
               u, v, w, p, q, r]`` (R row-major).

    This is the EDMDc dictionary's input from May-2026 onward. The 9
    rotation-matrix entries are smooth and bounded in [−1, 1]; the legacy
    Euler-angle bare-features (`SLOT_ATT_VEL_ANG`'s ``q·yaw`` etc.)  are
    replaced with R-element × body-velocity products in the new dictionary.
    """
    eta = np.asarray(eta, dtype=float).reshape(6)
    v = np.asarray(v, dtype=float).reshape(6)
    R = R_zyx(float(eta[3]), float(eta[4]), float(eta[5]))
    out = np.empty(18, dtype=float)
    out[0:3] = eta[0:3]
    out[3:12] = R.reshape(-1)
    out[12:18] = v
    return out


def step_state(
    x: np.ndarray,
    tau: np.ndarray,
    p: FossenIntegratorParams,
    dt: float,
) -> np.ndarray:
    """Convenience: take a 12-dim WORLD-frame state vector and return the next."""
    eta = x[:6].copy()
    v = x[6:].copy()
    eta_next, v_next = step_fossen(eta, v, tau, p, dt)
    return state_from_eta_v(eta_next, v_next)


# ---------------------------------------------------------------------------
# Body-relative state wrappers
#
# The EDMDc dictionary (`EDMDc.edmdc.lift`) now expects the 12-dim state
# expressed *relative to a virtual target* with target heading
# (yaw_target) zeroed out:
#
#   x_body_rel = [
#       p_target_frame (3),     # ROV pos in target's yaw-only frame
#       roll, pitch, yaw_rel,   # roll/pitch unchanged, yaw - yaw_target
#       u, v, w,                # body-frame linear vel (unchanged)
#       p, q, r,                # body-frame angular vel (unchanged)
#   ]
#
# The integrator below operates in WORLD coords (η world pose, ν body vel),
# so we transform world-frame ROV pose into target-frame pose before lifting.
# Because the target is at rest, applying the same wrench world-frame and
# stepping (η, ν) in world coords, then re-transforming, gives the same
# answer as stepping a body-relative state directly — physics is invariant
# under SE(2) translations and rotations about the gravity axis (yaw).
# ---------------------------------------------------------------------------

def _yaw_only_R(yaw: float) -> np.ndarray:
    """3×3 rotation about world Z (the gravity axis)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def state_from_body_relative(
    eta_world: np.ndarray,
    v_body: np.ndarray,
    target_pos_world: np.ndarray = np.zeros(3),
    target_yaw_world: float = 0.0,
    wrap_yaw: bool = False,
) -> np.ndarray:
    """Pack a (world-frame η, body-frame ν, virtual target) tuple into the
    12-dim BODY-RELATIVE state used by the new EDMDc dictionary.

    Args:
        eta_world: (6,) world-frame pose [x, y, z, roll, pitch, yaw_world].
        v_body:    (6,) body-frame velocity [u, v, w, p, q, r] (unchanged).
        target_pos_world: (3,) target world position (default origin).
        target_yaw_world: target heading about world Z (default 0).
        wrap_yaw: if True, wrap `yaw_rel` into `[-π, π]`. Default False so
            that this function can be called per-step inside training data
            collection without injecting 2π discontinuities at the wrap
            boundary into the EDMDc regression — `step_fossen` deliberately
            keeps yaw unwrapped, and the lifted-trig features (cos/sin of
            roll/pitch only) are insensitive to yaw_rel offsets by 2π. Set
            True at fresh-state assembly points (e.g. computing yaw_rel from
            an instantaneous world-frame measurement at deploy time).

    Returns:
        x: (12,) body-relative state suitable for `dictionary.lift`.
    """
    eta_world = np.asarray(eta_world, float).reshape(6)
    v_body = np.asarray(v_body, float).reshape(6)
    target_pos_world = np.asarray(target_pos_world, float).reshape(3)

    # Target-frame ROV position: rotate (rov - target) by R_z(-yaw_target).
    delta_world = eta_world[:3] - target_pos_world
    R_target_T = _yaw_only_R(-target_yaw_world)
    p_rel = R_target_T @ delta_world

    yaw_rel = float(eta_world[5]) - float(target_yaw_world)
    if wrap_yaw:
        yaw_rel = _wrap_pi(yaw_rel)

    x = np.empty(12, dtype=float)
    x[:3] = p_rel
    x[3] = eta_world[3]      # roll (gravity-anchored)
    x[4] = eta_world[4]      # pitch
    x[5] = yaw_rel
    x[6:] = v_body
    return x


def step_state_body_relative(
    x_rel: np.ndarray,
    tau: np.ndarray,
    p: FossenIntegratorParams,
    dt: float,
) -> np.ndarray:
    """Single Euler step for a BODY-RELATIVE state vector.

    Because the dynamics are SE(2)-invariant about world Z, we can equivalently:
      (a) lift the body-relative state to world coords (target at origin,
          yaw_target=0 by construction so this is identity),
      (b) step Fossen in world coords,
      (c) re-pack as body-relative.

    With the target permanently at the origin and heading 0 (the EDMDc
    training-time choice — see the `data.py` IC sampler), step (a)/(c) are
    no-ops and this function reduces to the world-frame integrator. Kept as
    a separate name so callers can express intent.
    """
    eta = x_rel[:6].copy()
    v = x_rel[6:].copy()
    eta_next, v_next = step_fossen(eta, v, tau, p, dt)
    # Re-anchor: target at origin, yaw_target = 0 — pass through.
    return state_from_body_relative(eta_next, v_next)


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    yaml_path = os.path.join(project_root, "assets", "BlueROVHeavy", "BlueROVHeavy.yaml")
    p = FossenIntegratorParams.from_yaml(yaml_path, mass_kg=13.5)

    print(f"M_total diag = {p.M_total_diag().round(3).tolist()} (kg or kg·m²)")
    print(f"buoyancy = {RHO_WATER * GRAVITY * p.volume:.2f} N "
          f"vs weight = {p.mass * GRAVITY:.2f} N "
          f"=> net vertical = {RHO_WATER * GRAVITY * p.volume - p.mass * GRAVITY:+.2f} N")

    # Static test: zero state, zero wrench should give v̇ ≈ small nonzero
    # (body-frame net buoyancy is approx -1.4 N → v̇_w ≈ -1.4/M_eff, tiny).
    eta0 = np.zeros(6)
    v0 = np.zeros(6)
    tau0 = np.zeros(6)
    vdot = fossen_vdot(eta0, v0, tau0, p)
    print(f"\nStatic (rest) v̇ = {vdot.round(4).tolist()}")
    assert abs(vdot[2]) < 0.5, "vertical accel should be ≤ 0.5 m/s² (net buoyancy / m)"

    # Forward thrust 30 N, 1 s rollout — terminal vel ≈ √(F/D_Q[0]) at steady
    # state; should approach but not exceed.
    print(f"\n30 N surge for 1 s @ dt = 1/60:")
    eta, v = eta0.copy(), v0.copy()
    tau_surge = np.array([30.0, 0, 0, 0, 0, 0])
    for k in range(60):
        eta, v = step_fossen(eta, v, tau_surge, p, 1.0 / 60.0)
    v_terminal_predict = np.sqrt(30.0 / p.quadratic_damping[0])
    print(f"  v at 1 s = {v[:3].round(3).tolist()} (m/s)")
    print(f"  predicted v_terminal = {v_terminal_predict:.3f} m/s "
          f"(actual converged value, may not reach in 1 s)")
    print(f"  position at 1 s = {eta[:3].round(3).tolist()} (m)")
    print("[fossen_integrator] smoke OK.")
