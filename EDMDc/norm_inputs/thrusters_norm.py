"""Blue Robotics T200 thruster model + thrust allocation — NumPy, no Isaac Sim deps.

The T200 is the single-motor unit used on both the BlueROV2 and BlueROV2 Heavy;
only the *count* and *layout* differ between the two vehicles. This module
handles one thruster per instance with:

  command (in [-1, 1])  →  throttle (first-order lag)
                        →  target RPM (piecewise linear, with deadband)
                        →  RPM (first-order lag, saturated to ±max)
                        →  thrust force in Newtons (polynomial thrust curve)

Formula constants are from MarineGym's `marinegym/actuators/t200.py` which in
turn were fit from Blue Robotics tank data.

Also provides `ThrustAllocator`: given a desired 6-DOF body wrench
[Fx Fy Fz Tx Ty Tz], returns per-thruster commands that best realize it
(pseudo-inverse of the 6×N thrust configuration matrix, clipped to [-1, 1]).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ThrusterConfig:
    """Per-vehicle thruster configuration loaded from YAML.

    num_rotors:       N
    positions:        (N, 3) body-frame locations where thrust is applied
    thrust_axes:      (N, 3) body-frame unit vectors, force direction per thruster
    directions:       (N,) spin direction ±1 (affects reaction torque; +1 CCW viewed from +axis)
    force_constants:  (N,) — passthrough to T200 thrust curve scaling
    moment_constants: (N,) — passthrough (reaction torque scaling)
    max_rotation_velocities: (N,) — RPM cap
    time_constants:   (N,) — RPM first-order lag (s)
    """

    num_rotors: int
    positions: np.ndarray
    thrust_axes: np.ndarray
    directions: np.ndarray
    force_constants: np.ndarray
    moment_constants: np.ndarray
    max_rotation_velocities: np.ndarray
    time_constants: np.ndarray

    @classmethod
    def from_yaml_dict(cls, d: dict) -> "ThrusterConfig":
        rc = d["rotor_configuration"]
        n = int(rc["num_rotors"])
        positions = np.asarray(rc["positions"], dtype=np.float64).reshape(n, 3)
        axes = np.asarray(rc["thrust_axes"], dtype=np.float64).reshape(n, 3)
        # Normalize axes to unit length
        axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-9)
        return cls(
            num_rotors=n,
            positions=positions,
            thrust_axes=axes,
            directions=np.asarray(rc["directions"], dtype=np.float64),
            force_constants=np.asarray(rc["force_constants"], dtype=np.float64),
            moment_constants=np.asarray(rc["moment_constants"], dtype=np.float64),
            max_rotation_velocities=np.asarray(rc["max_rotation_velocities"], dtype=np.float64),
            time_constants=np.asarray(rc["time_constants"], dtype=np.float64),
        )


class T200Group:
    """Batched T200 motor model for all thrusters on one vehicle.

    State (per-thruster): throttle, rpm. `step(commands, dt)` integrates one
    tick and returns per-thruster thrust force (Newtons) and reaction moment
    (Newton-meters) about the thruster's spin axis.
    """

    # Throttle dynamics (MarineGym T200 line 23-24 and 37-39)
    _TAU_UP = 0.43
    _TAU_DOWN = 0.43

    # Throttle→target-RPM piecewise linear (MarineGym T200 line 42-44)
    _DEADBAND = 0.075
    _POS_SLOPE = 3.6599e3
    _POS_INTERCEPT = 3.4521e2
    _NEG_SLOPE = 3.4944e3
    _NEG_INTERCEPT = -4.3350e2

    # RPM→thrust polynomial (MarineGym T200 line 51)
    # Units: thrust in kgf (multiplied by 9.81 to Newtons; force_constants/4.4e-7
    # is a per-thruster scaling from the YAML).
    _POS_A = 4.7368e-7
    _POS_B = -1.9275e-4
    _POS_C = 8.4452e-2
    _NEG_A = -3.8442e-7
    _NEG_B = -1.6186e-4
    _NEG_C = -3.9139e-2

    def __init__(self, config: ThrusterConfig, *, linear: bool = False,
                 max_thrust_n: float = 25.0):
        self.config = config
        self.n = config.num_rotors
        self.throttle = np.zeros(self.n)
        self.rpm = np.zeros(self.n)
        # Pool-test "unit thrust" mode: a static linear map command -> thrust,
        # bypassing the T200 throttle/RPM dynamics, deadband, and asymmetric
        # polynomial. command c in [-1, 1] -> thrust = max_thrust_n * c (N).
        self.linear = bool(linear)
        self.max_thrust_n = float(max_thrust_n)

    def reset(self) -> None:
        self.throttle[:] = 0.0
        self.rpm[:] = 0.0

    def step(self, commands: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance one tick.

        Args:
            commands: (N,) per-thruster targets in [-1, 1]
            dt:       physics timestep in seconds
        Returns:
            thrusts: (N,) Newtons — force along each thruster's thrust axis
            moments: (N,) N·m — reaction torque about the thruster spin axis
        """
        target = np.clip(commands, -1.0, 1.0)

        # Linear unit-thrust mode: static map, no throttle/RPM lag, no deadband.
        if self.linear:
            self.throttle[:] = target           # kept for viz / introspection
            self.rpm[:] = 0.0
            thrusts = self.max_thrust_n * target
            moments = thrusts * (-self.config.directions) * 0.0
            return thrusts, moments

        # Throttle first-order lag (same tau both directions; clamp for safety)
        tau = self._TAU_UP  # TAU_UP == TAU_DOWN in MarineGym; split if needed later
        self.throttle += np.clip(tau, 0.0, 1.0) * (target - self.throttle)

        # Piecewise linear target RPM with deadband
        target_rpm = np.where(
            self.throttle > self._DEADBAND,
            self._POS_SLOPE * self.throttle + self._POS_INTERCEPT,
            np.where(
                self.throttle < -self._DEADBAND,
                self._NEG_SLOPE * self.throttle + self._NEG_INTERCEPT,
                0.0,
            ),
        )

        # RPM first-order lag (per-thruster time constants)
        alpha = np.exp(-dt / self.config.time_constants)
        self.rpm = alpha * self.rpm + (1.0 - alpha) * target_rpm

        # Saturate RPM to ±max (use max_rotation_velocities as the ceiling)
        cap = self.config.max_rotation_velocities
        self.rpm = np.clip(self.rpm, -cap, cap)

        # Polynomial thrust curve (kgf), scaled by force_constants/4.4e-7 and g=9.81.
        # Note 4.4e-7 is the normalization constant in MarineGym's formula; ratios
        # above 1 scale thrust proportionally for stronger motors.
        rpm_sq = self.rpm * self.rpm
        pos = self._POS_A * rpm_sq + self._POS_B * self.rpm + self._POS_C
        neg = self._NEG_A * rpm_sq + self._NEG_B * self.rpm + self._NEG_C
        thrust_kgf_unit = np.where(self.rpm > 0.0, pos, neg)
        # Zero out the phantom thrust when both throttle is in deadband and RPM
        # has decayed to near-stopped. Without this the polynomial's nonzero
        # intercept (-0.039 kgf on the neg branch) shows up as steady reverse
        # thrust when the motor is commanded off.
        idle = (np.abs(self.throttle) <= self._DEADBAND) & (np.abs(self.rpm) < 10.0)
        thrust_kgf_unit = np.where(idle, 0.0, thrust_kgf_unit)
        thrusts = (self.config.force_constants / 4.4e-7) * 9.81 * thrust_kgf_unit

        # Reaction moment about the spin axis — proportional to thrust and spin
        # direction. MarineGym currently scales this by 0 in the T200 module
        # (disabled). Keeping it disabled to match MarineGym behavior; flip to
        # a nonzero scale if you want reaction torques modeled.
        moments = thrusts * (-self.config.directions) * 0.0

        return thrusts, moments

    @classmethod
    def thrust_to_throttle(
        cls,
        thrust_N: float,
        force_constant: float,
        max_rpm: float,
    ) -> float:
        """Static steady-state inverse of the forward pipeline.

        Given the thrust (N) we want a single T200 to deliver at steady state,
        return the throttle in [-1, 1] that produces it. Used by
        `ThrustAllocator.allocate` to compensate for the piecewise-polynomial
        thrust curve so that `desired_wrench ≈ realized_wrench` in the valid
        operating range.

        Three physical limits the inverse cannot defeat — it just reports them
        honestly:
          1. Sub-deadband: if the desired thrust is smaller than what the
             motor produces at `|throttle| = DEADBAND + ε`, return 0. The motor
             simply cannot deliver that tiny force.
          2. Saturation: if the desired thrust exceeds the polynomial's value
             at `|throttle| = 1` or `|rpm| = max_rpm`, return ±1.
          3. First-order lag (dynamic): steady-state only. During a step
             transient, realized thrust lags behind. Not our problem here.
        """
        if thrust_N == 0.0:
            return 0.0

        # thrust_N = (force_constant / 4.4e-7) * 9.81 * thrust_kgf
        thrust_kgf = thrust_N * 4.4e-7 / (9.81 * force_constant)

        if thrust_kgf > 0.0:
            A, B, C = cls._POS_A, cls._POS_B, cls._POS_C
            slope, intercept = cls._POS_SLOPE, cls._POS_INTERCEPT
            # Minimum throttle that produces nonzero thrust is strictly > DEADBAND.
            # Use a tiny epsilon so we land on the actual monotonic segment.
            throttle_lo = cls._DEADBAND + 1e-6
            rpm_lo = slope * throttle_lo + intercept
            rpm_hi = min(slope * 1.0 + intercept, max_rpm)
            # Ensure ordering: rpm_lo < rpm_hi for positive branch
            rpm_min, rpm_max = rpm_lo, rpm_hi
        else:
            A, B, C = cls._NEG_A, cls._NEG_B, cls._NEG_C
            slope, intercept = cls._NEG_SLOPE, cls._NEG_INTERCEPT
            # Negative-throttle valid range (throttles in [-1, -DEADBAND - ε])
            # maps to RPMs in [-max_rpm, slope*(-DEADBAND-ε) + intercept].
            throttle_hi = -(cls._DEADBAND + 1e-6)
            rpm_hi = slope * throttle_hi + intercept
            rpm_lo = max(slope * (-1.0) + intercept, -max_rpm)
            rpm_min, rpm_max = rpm_lo, rpm_hi

        # Deliverable kgf range at the endpoints of the valid rpm segment.
        kgf_at_min = A * rpm_min * rpm_min + B * rpm_min + C
        kgf_at_max = A * rpm_max * rpm_max + B * rpm_max + C
        kgf_lo, kgf_hi = min(kgf_at_min, kgf_at_max), max(kgf_at_min, kgf_at_max)

        # Sub-deadband or opposite sign fallthrough → snap to 0.
        if thrust_kgf > 0.0 and thrust_kgf < kgf_lo:
            return 0.0
        if thrust_kgf < 0.0 and thrust_kgf > kgf_hi:
            return 0.0
        # Saturation → ±1.
        if thrust_kgf > 0.0 and thrust_kgf >= kgf_hi:
            return 1.0
        if thrust_kgf < 0.0 and thrust_kgf <= kgf_lo:
            return -1.0

        # Solve A·rpm² + B·rpm + (C − thrust_kgf) = 0 for rpm.
        c_shift = C - thrust_kgf
        disc = B * B - 4.0 * A * c_shift
        if disc < 0.0:
            return 0.0
        sqrt_disc = float(np.sqrt(disc))
        r1 = (-B + sqrt_disc) / (2.0 * A)
        r2 = (-B - sqrt_disc) / (2.0 * A)
        # Pick the root inside the monotonic segment.
        if rpm_min <= r1 <= rpm_max:
            rpm = r1
        elif rpm_min <= r2 <= rpm_max:
            rpm = r2
        else:
            # Bracketing should have caught this — last-resort clamp.
            rpm = float(np.clip(r1 if abs(r1) < abs(r2) else r2, rpm_min, rpm_max))

        throttle = (rpm - intercept) / slope
        # Snap if we land on (or below) the deadband — matches forward-model
        # zero-out behaviour exactly.
        if abs(throttle) <= cls._DEADBAND:
            return 0.0
        return float(np.clip(throttle, -1.0, 1.0))


class ThrustAllocator:
    """Solve for per-thruster commands given a desired 6-DOF body wrench.

    Builds a 6×N configuration matrix T where column i is the wrench
    contribution of a unit thrust on thruster i:
        T[:3, i] = thrust_axis_i                    (linear force)
        T[3:, i] = position_i × thrust_axis_i       (torque arm)

    `allocate` uses two stages:

      1. Linear pseudo-inverse of the thrust-config matrix to get the
         per-rotor force (Newtons) needed to realize the requested wrench.
      2. Per-rotor inverse of the T200 thrust curve (`T200Group.thrust_to_throttle`)
         to map that Newton value to a throttle command in [-1, 1]. This
         compensates the polynomial nonlinearity + piecewise-linear
         throttle→RPM map, so at steady state the realized wrench matches
         the desired wrench (within the physical limits: deadband and
         saturation).

    Limits the inverse cannot defeat:
      - Sub-deadband thrust requests snap to 0 (T200 can't deliver).
      - Saturation clips throttle to ±1.
      - First-order throttle/RPM lag means transients still differ from
        steady state for ~0.5 s after a step input.
    """

    def __init__(self, config: ThrusterConfig, max_thrust_per_motor: float = 50.0,
                 *, linear: bool = False):
        self.config = config
        # In linear mode this is the per-motor thrust at unit command (|c|=1),
        # so a command maps to thrust = max_thrust * c. In T200 mode it is
        # retained for logging only (the polynomial's saturation governs limits).
        self.max_thrust = max_thrust_per_motor
        # Pool-test "unit thrust" allocation: per-rotor command = force_N / 25,
        # clipped to [-1, 1] — the exact inverse of the linear T200Group map.
        self.linear = bool(linear)
        N = config.num_rotors
        T = np.zeros((6, N), dtype=np.float64)
        for i in range(N):
            ax = config.thrust_axes[i]
            pos = config.positions[i]
            T[:3, i] = ax
            T[3:, i] = np.cross(pos, ax)
        self.T = T
        # Pseudo-inverse: maps wrench → per-thruster thrust (N values in Newtons)
        self.T_pinv = np.linalg.pinv(T)

    def allocate(self, wrench_body: np.ndarray) -> np.ndarray:
        """Desired wrench [Fx Fy Fz Tx Ty Tz] (body frame) → commands in [-1, 1]."""
        thrusts_desired_N = self.T_pinv @ wrench_body          # (N,) Newtons
        if self.linear:
            # Linear unit-thrust inverse: command = force / max_thrust, clipped.
            # If the inscribed APRBS box is respected, no clipping occurs.
            return np.clip(thrusts_desired_N / self.max_thrust, -1.0, 1.0)
        commands = np.empty(self.config.num_rotors, dtype=np.float64)
        for i in range(self.config.num_rotors):
            commands[i] = T200Group.thrust_to_throttle(
                float(thrusts_desired_N[i]),
                float(self.config.force_constants[i]),
                float(self.config.max_rotation_velocities[i]),
            )
        return commands


def sum_to_wrench(
    config: ThrusterConfig,
    thrusts_per_rotor: np.ndarray,
) -> np.ndarray:
    """Given per-thruster forces (Newtons, signed — +ve is along thrust_axis),
    return the equivalent 6-DOF body wrench [Fx Fy Fz Tx Ty Tz].

    Used to replace MarineGym's per-rotor force application: we compute the
    net wrench and apply it once at base_link COG. Physically equivalent
    because PhysX's articulation solver would do the same superposition
    via the rotor joints.
    """
    wrench = np.zeros(6)
    for i in range(config.num_rotors):
        F = thrusts_per_rotor[i] * config.thrust_axes[i]       # body-frame force
        r = config.positions[i]
        wrench[:3] += F
        wrench[3:] += np.cross(r, F)
    return wrench


if __name__ == "__main__":
    # Sanity check: a symmetric +X wrench should yield symmetric commands on
    # a simple 4-corner layout.
    import os, yaml
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "assets", "BlueROV", "BlueROV.yaml")
    with open(path) as f:
        d = yaml.safe_load(f)
    cfg = ThrusterConfig.from_yaml_dict(d)
    alloc = ThrustAllocator(cfg)
    t200 = T200Group(cfg)

    wrench = np.array([10.0, 0, 0, 0, 0, 0])        # 10 N forward
    cmds = alloc.allocate(wrench)
    print(f"Commands for +10 N surge: {cmds}")
    # Step T200 for a second to let it settle
    dt = 1.0 / 60.0
    for _ in range(60):
        thrusts, _ = t200.step(cmds, dt)
    print(f"After 1 s, per-thruster thrust: {thrusts}")
    net = sum_to_wrench(cfg, thrusts)
    print(f"Net wrench: {net}")
    print(f"Desired vs realized X force: {wrench[0]:.2f} vs {net[0]:.2f}")
