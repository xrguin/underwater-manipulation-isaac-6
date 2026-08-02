"""ArduSub-STABILIZE-style roll/pitch leveling for the simulator.

The real-robot launcher in ``EDMDc_bluerov/control_v2/keyboard_stabilize.py``
leaves surge, sway, heave, and yaw with the operator/controller while ArduSub
levels roll and pitch.  In particular, STABILIZE is *not* a depth-hold mode:
the attitude correction is differential vertical thrust with zero collective
command before motor saturation.

This module mirrors that ownership boundary without depending on Isaac Sim:

* roll/pitch targets are fixed trim angles (zero by default);
* an angle-P outer loop produces body-rate targets;
* a rate-P inner loop produces normalized motor-mixer corrections;
* only vertical thrusters receive those corrections;
* the caller's existing eight-motor command remains the source of collective
  heave, planar motion, and yaw.

The onboard attitude parameters were not recorded by the real-robot control
repository, so the simulator gains below are explicit, conservative defaults
rather than claimed copies of a particular flight-controller parameter file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .thrusters import ThrusterConfig


@dataclass(frozen=True)
class ArduSubStabilizeConfig:
    """Configuration for :class:`ArduSubStabilizer`.

    ``angle_p`` has units rad/s per rad. ``rate_p`` converts a body-rate error
    in rad/s to normalized differential motor command.  ``max_correction``
    bounds the combined roll/pitch correction on every vertical motor.
    """

    angle_p: tuple[float, float] = (4.5, 4.5)
    rate_p: tuple[float, float] = (0.20, 0.20)
    max_rate_rad_s: tuple[float, float] = (
        np.deg2rad(60.0), np.deg2rad(60.0))
    max_correction: float = 0.35
    trim_roll_rad: float = 0.0
    trim_pitch_rad: float = 0.0


@dataclass(frozen=True)
class StabilizeStatus:
    """Last controller result, useful for HUDs and regression tests."""

    angle_error_rad: np.ndarray
    target_rate_rad_s: np.ndarray
    rate_error_rad_s: np.ndarray
    axis_output: np.ndarray
    motor_correction: np.ndarray
    desaturation_scale: float


class ArduSubStabilizer:
    """Add roll/pitch leveling to an existing eight-motor command.

    The motor patterns are derived from the configured thruster geometry
    rather than hard-coded rotor numbers. A positive mixer axis produces a
    positive body torque about that axis. Each pattern is mean-centered over
    the vertical motors, so attitude control has exactly zero collective
    *command* before saturation (real T200 forward/reverse asymmetry can still
    produce a small transient net force, as on the physical vehicle).
    """

    def __init__(
        self,
        thruster_config: ThrusterConfig,
        config: ArduSubStabilizeConfig | None = None,
    ):
        self.config = config or ArduSubStabilizeConfig()
        self._n = int(thruster_config.num_rotors)
        axes = np.asarray(thruster_config.thrust_axes, dtype=float)
        positions = np.asarray(thruster_config.positions, dtype=float)

        vertical = np.flatnonzero(
            (np.abs(axes[:, 2]) > 0.9)
            & (np.linalg.norm(axes[:, :2], axis=1) < 0.2)
        )
        if vertical.size < 4:
            raise ValueError(
                "ArduSub STABILIZE requires at least four vertical thrusters; "
                f"found {vertical.size}"
            )
        self.vertical_indices = vertical

        # Torque contribution per unit signed motor effort. The T200 curve is
        # nonlinear, but its sign is monotonic, so geometry gives the correct
        # firmware-style differential pattern.
        torque_per_effort = np.cross(positions[vertical], axes[vertical])
        torque_rp = torque_per_effort[:, :2].T  # (roll,pitch) x motors
        if np.linalg.matrix_rank(torque_rp) < 2:
            raise ValueError(
                "Vertical-thruster geometry cannot independently control "
                "roll and pitch"
            )

        patterns = np.linalg.pinv(torque_rp)  # motors x (roll,pitch)
        patterns -= patterns.mean(axis=0, keepdims=True)
        for axis in range(2):
            peak = float(np.max(np.abs(patterns[:, axis])))
            if peak <= 1e-12:
                raise ValueError("Degenerate roll/pitch motor-mixer pattern")
            patterns[:, axis] /= peak
            # Guard against an unexpected pseudo-inverse sign convention.
            response = torque_rp @ patterns[:, axis]
            if response[axis] < 0.0:
                patterns[:, axis] *= -1.0
        self._patterns = patterns

        z2 = np.zeros(2, dtype=float)
        zn = np.zeros(self._n, dtype=float)
        self.last_status = StabilizeStatus(
            angle_error_rad=z2.copy(),
            target_rate_rad_s=z2.copy(),
            rate_error_rad_s=z2.copy(),
            axis_output=z2.copy(),
            motor_correction=zn,
            desaturation_scale=1.0,
        )

    @property
    def mixer_patterns(self) -> np.ndarray:
        """Vertical-motor patterns, columns ``[roll, pitch]``."""

        return self._patterns.copy()

    def reset(self) -> None:
        """Clear diagnostic state (the default P/P controller has no memory)."""

        z2 = np.zeros(2, dtype=float)
        self.last_status = StabilizeStatus(
            angle_error_rad=z2.copy(),
            target_rate_rad_s=z2.copy(),
            rate_error_rad_s=z2.copy(),
            axis_output=z2.copy(),
            motor_correction=np.zeros(self._n, dtype=float),
            desaturation_scale=1.0,
        )

    def mix(
        self,
        base_commands: np.ndarray,
        *,
        roll_rad: float,
        pitch_rad: float,
        p_rad_s: float,
        q_rad_s: float,
    ) -> np.ndarray:
        """Return motor commands with differential roll/pitch correction.

        ``base_commands`` already contains the calibrated ArduSub allocation
        for ``[surge, sway, heave, yaw]``. This method never creates a depth or
        yaw target.
        """

        base = np.asarray(base_commands, dtype=float).reshape(self._n)
        measured = np.array(
            [roll_rad, pitch_rad, p_rad_s, q_rad_s], dtype=float)
        if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(base)):
            raise ValueError("STABILIZE received non-finite state or command")

        cfg = self.config
        angle = measured[:2]
        rates = measured[2:]
        trim = np.array([cfg.trim_roll_rad, cfg.trim_pitch_rad], dtype=float)
        angle_p = np.asarray(cfg.angle_p, dtype=float)
        rate_p = np.asarray(cfg.rate_p, dtype=float)
        rate_limit = np.asarray(cfg.max_rate_rad_s, dtype=float)

        angle_error = trim - angle
        target_rate = np.clip(
            angle_p * angle_error, -rate_limit, rate_limit)
        rate_error = target_rate - rates
        axis_output = rate_p * rate_error

        vertical_correction = self._patterns @ axis_output
        correction_peak = float(np.max(np.abs(vertical_correction)))
        correction_scale = 1.0
        max_correction = float(cfg.max_correction)
        if max_correction <= 0.0:
            vertical_correction[:] = 0.0
        elif correction_peak > max_correction:
            correction_scale = max_correction / correction_peak
            vertical_correction *= correction_scale
            axis_output *= correction_scale

        correction = np.zeros(self._n, dtype=float)
        correction[self.vertical_indices] = vertical_correction
        combined = base + correction

        # Firmware-style global desaturation on the vertical bank preserves
        # the relative collective/attitude mix and never touches planar/yaw
        # thrusters. At ordinary commands this scale is exactly one.
        vertical_total = combined[self.vertical_indices]
        peak = float(np.max(np.abs(vertical_total)))
        desaturation_scale = 1.0
        if peak > 1.0:
            desaturation_scale = 1.0 / peak
            combined[self.vertical_indices] *= desaturation_scale
        combined = np.clip(combined, -1.0, 1.0)

        self.last_status = StabilizeStatus(
            angle_error_rad=angle_error.copy(),
            target_rate_rad_s=target_rate.copy(),
            rate_error_rad_s=rate_error.copy(),
            axis_output=axis_output.copy(),
            motor_correction=correction.copy(),
            desaturation_scale=desaturation_scale,
        )
        return combined


__all__ = [
    "ArduSubStabilizeConfig",
    "ArduSubStabilizer",
    "StabilizeStatus",
]
