"""Fossen 6-DOF underwater vehicle hydrodynamics (standalone, NumPy-only).

Computes the external wrench (force + torque, world frame) that a rigid-body
simulator must apply to a vehicle so that, combined with the simulator's own
Newton-Euler integration, the full Fossen equations of motion are reproduced:

    M_RB·v̇ + C_RB(v)·v + M_A·v̇ + C_A(v)·v + D(v)·v + g(η) = τ_user

The rigid-body simulator (e.g. PhysX) handles M_RB·v̇, C_RB(v)·v, and gravity
on its own; this module contributes the underwater-specific terms:

    τ_hydro_body = -M_A·v̇  -  C_A(v)·v  -  D(v)·v
    τ_buoy_world = ρ g V · ẑ   (applied at COB, produces torque if offset)

Conventions:
    - Body frame: X forward, Y starboard, Z up
    - Velocity v = [u v w p q r] (linear x,y,z then angular x,y,z), body frame
    - Added mass is diagonal: M_A = diag(X_u̇, Y_v̇, Z_ẇ, K_ṗ, M_q̇, N_ṙ)
    - Damping is diagonal: D = D_linear + D_quadratic·|v|
    - Quaternion input follows Isaac Sim convention: (w, x, y, z)

The added-mass reaction -M_A·v̇ requires v̇, which creates an implicit loop
with the applied force. We break it with a backward finite difference:
v̇_t ≈ (v_t - v_{t-1}) / dt. This introduces a one-step lag but is stable
at 60 Hz for BlueROV-scale dynamics.
"""

from dataclasses import dataclass
import numpy as np
import yaml

RHO_WATER = 997.0
GRAVITY = 9.81


@dataclass
class FossenParams:
    volume: float                   # m^3 displaced by the hull
    cob_offset: float               # m, COB offset from COG along +Z body
    added_mass: np.ndarray          # (6,) diagonal of M_A
    linear_damping: np.ndarray      # (6,) diagonal of D_L
    quadratic_damping: np.ndarray   # (6,) diagonal of D_Q

    @classmethod
    def from_yaml(cls, path: str) -> "FossenParams":
        with open(path, "r") as f:
            p = yaml.safe_load(f)
        h = p["hydro_coef"]
        return cls(
            volume=float(p["volume"]),
            cob_offset=float(p["coBM"]),
            added_mass=np.asarray(h["added_mass"], dtype=np.float64),
            linear_damping=np.asarray(h["linear_damping"], dtype=np.float64),
            quadratic_damping=np.asarray(h["quadratic_damping"], dtype=np.float64),
        )


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Isaac Sim quaternion convention: (w, x, y, z). Returns 3x3 R such that
    v_world = R @ v_body."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def quat_wxyz_to_euler_zyx(q: np.ndarray) -> tuple[float, float, float]:
    """Quaternion (w, x, y, z) -> (roll, pitch, yaw) ZYX intrinsic Euler.

    Singular when |pitch| approaches pi/2; safe for our pool poses where
    pitch stays well below pi/2.
    """
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))
    sinp = 2.0 * (w * y - z * x)
    sinp = float(np.clip(sinp, -1.0, 1.0))
    pitch = float(np.arcsin(sinp))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


class Fossen:
    """Per-step Fossen wrench calculator. One instance per vehicle.

    Set enable_added_mass=True to include the -M_A·v̇ reaction term. The v̇
    estimate is a backward difference (v_t - v_{t-1})/dt, which is numerically
    UNSTABLE when M_A[i] / M_RB > 1 on any axis — true for BlueROV heave.
    Stable use requires an implicit solve (v̇ is coupled to the applied force).
    Default is False: without this term Coriolis, damping, and buoyancy still
    reproduce correct qualitative underwater behavior; only transient
    accelerations deviate by the added-mass fraction.
    """

    def __init__(
        self,
        params: FossenParams,
        enable_added_mass: bool = False,
        enable_angular_coriolis: bool = False,
        added_mass_lp_alpha: float | None = 0.3,
        water_surface_z: float | None = None,
        hull_half_height: float = 0.1,
    ):
        """
        Args:
            added_mass_lp_alpha: First-order low-pass filter coefficient for the
                v̇ estimate used in the -M_A·v̇ term. None = raw backward
                difference (unstable when M_A/M_RB > 1; documented behavior
                of the original implementation). Float in (0, 1] applies
                v̇_filtered = (1-alpha)·v̇_filtered_prev + alpha·v̇_raw.
                Default 0.3 matches MarineGym's stabilization trick (see
                marinegym/robots/drone/underwaterVehicle.py:230-234) and lets
                `enable_added_mass=True` run without blowing up on BlueROV
                heave (M_A_z/m ≈ 1.4 > 1). Smaller alpha = more smoothing.
                Only used when enable_added_mass=True.
            water_surface_z: If set, buoyancy is gated by the ROV's world z.
                Above `water_surface_z + hull_half_height` the buoyancy force
                and its torque are zero; below `water_surface_z - hull_half_height`
                they are at full magnitude; linear interpolation between.
                None = always-on buoyancy (no surface).
            hull_half_height: Half of the vertical extent of the submergible
                body, in meters. Used to smooth the buoyancy step transition.
                Default 0.1 m (BlueROV is ~0.2 m tall).
        """
        self.p = params
        self.buoyancy_magnitude = RHO_WATER * GRAVITY * params.volume
        self._v_body_prev = np.zeros(6)
        self._vdot_filtered = np.zeros(6)
        self._enable_added_mass = enable_added_mass
        self._added_mass_lp_alpha = added_mass_lp_alpha
        # The angular part of added-mass Coriolis, `cross(ma_ang, ang_b)`,
        # couples ω_y ↔ ω_x through ω_z (asymmetric added-mass tensor). For a
        # slender body at nonzero yaw rate this gives positive-feedback
        # tumbling that decoupled roll/pitch PID cannot catch. Disabled by
        # default for controlled-ROV stability; set True for high-fidelity
        # system-ID data where you want the real hydrodynamic gyro effect.
        self._enable_angular_coriolis = enable_angular_coriolis
        self.water_surface_z = water_surface_z
        self.hull_half_height = hull_half_height

    def reset(self) -> None:
        """Call after teleporting / respawning to zero out the vdot estimate."""
        self._v_body_prev = np.zeros(6)
        self._vdot_filtered = np.zeros(6)

    def _submerged_fraction(self, body_z_world: float | None) -> float:
        """Fraction of the hull below the water surface, in [0, 1]. Returns
        1.0 if no water surface is configured (always fully submerged)."""
        if self.water_surface_z is None or body_z_world is None:
            return 1.0
        dz = self.water_surface_z - body_z_world     # positive = below surface
        if dz >= self.hull_half_height:
            return 1.0
        if dz <= -self.hull_half_height:
            return 0.0
        return 0.5 + 0.5 * (dz / self.hull_half_height)

    def compute_wrench_world(
        self,
        quat_wxyz: np.ndarray,       # (4,) orientation
        lin_vel_world: np.ndarray,   # (3,)
        ang_vel_world: np.ndarray,   # (3,)
        user_wrench_body: np.ndarray,  # (6,) [Fx Fy Fz Tx Ty Tz] in body frame
        dt: float,
        body_z_world: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (force_world[3], torque_world[3]) to apply at the COG.

        If `body_z_world` is given and `self.water_surface_z` is set, the
        buoyancy term is scaled by the fraction of the hull below the surface.
        """
        R = quat_wxyz_to_rotmat(quat_wxyz)

        # World -> body velocities
        lin_b = R.T @ lin_vel_world
        ang_b = R.T @ ang_vel_world
        v_b = np.concatenate([lin_b, ang_b])

        # Optional added-mass reaction: -M_A · v̇.
        # Backward-difference v̇ is unstable for BlueROV heave (M_A/m > 1) when
        # used raw; the optional first-order LP filter on v̇ (alpha=0.3 default)
        # tames the high-frequency content and matches MarineGym's approach.
        if self._enable_added_mass:
            vdot_raw = (v_b - self._v_body_prev) / dt
            if self._added_mass_lp_alpha is None:
                vdot_used = vdot_raw
            else:
                a = self._added_mass_lp_alpha
                self._vdot_filtered = (1.0 - a) * self._vdot_filtered + a * vdot_raw
                vdot_used = self._vdot_filtered
            F_added = -self.p.added_mass * vdot_used
        else:
            F_added = np.zeros(6)
        self._v_body_prev = v_b.copy()

        # Coriolis/centripetal of added mass: -C_A(v)·v, for diagonal M_A:
        #   linear  = (M_A_lin · v_lin) × v_ang
        #   angular = (M_A_lin · v_lin) × v_lin + (M_A_ang · v_ang) × v_ang
        ma_lin = self.p.added_mass[:3] * lin_b
        ma_ang = self.p.added_mass[3:] * ang_b
        C_lin = np.cross(ma_lin, ang_b)
        # Angular term: the Munk moment `cross(ma_lin, lin_b)` always on (it's
        # the real physical destabilizer at forward speed for slender bodies).
        # `cross(ma_ang, ang_b)` (angular-angular gyro term) is gated —
        # destabilizing positive-feedback loop under sustained yaw; disabled
        # by default. See class docstring.
        C_ang = np.cross(ma_lin, lin_b)
        if self._enable_angular_coriolis:
            C_ang = C_ang + np.cross(ma_ang, ang_b)
        F_coriolis = np.concatenate([C_lin, C_ang])

        # Damping: -(D_L + D_Q·|v|) · v
        F_damping = -(self.p.linear_damping * v_b
                      + self.p.quadratic_damping * np.abs(v_b) * v_b)

        # Body-frame total wrench, then rotate to world
        wrench_b = F_added + F_coriolis + F_damping + user_wrench_body
        force_world = R @ wrench_b[:3]
        torque_world = R @ wrench_b[3:]

        # Buoyancy: up in world, torque from COB offset (body +Z) rotated to world.
        # Gated by submerged fraction if a water surface was configured.
        sub = self._submerged_fraction(body_z_world)
        buoy_force = sub * np.array([0.0, 0.0, self.buoyancy_magnitude])
        cob_world = R @ np.array([0.0, 0.0, self.p.cob_offset])
        buoy_torque = np.cross(cob_world, buoy_force)
        force_world = force_world + buoy_force
        torque_world = torque_world + buoy_torque

        return force_world, torque_world


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "assets", "BlueROV", "BlueROV.yaml")
    params = FossenParams.from_yaml(path)
    print(f"BlueROV Fossen params:")
    print(f"  volume       = {params.volume:.4f} m^3")
    print(f"  buoyancy     = {RHO_WATER*GRAVITY*params.volume:.2f} N")
    print(f"  COB offset   = {params.cob_offset} m")
    print(f"  added mass   = {params.added_mass}")
    print(f"  lin damping  = {params.linear_damping}")
    print(f"  quad damping = {params.quadratic_damping}")

    # Quick sanity run: vehicle at rest, identity orientation, push forward 1N,
    # expect small damping force opposite direction (zero at rest) and buoyancy up.
    fossen = Fossen(params)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    user = np.zeros(6); user[0] = 1.0
    f, t = fossen.compute_wrench_world(q, np.zeros(3), np.zeros(3), user, 1/60)
    print(f"\nStatic test (rest, +1N surge): force={f}, torque={t}")
    print("  (Z should be +buoyancy, X should be +1, Y should be 0)")
