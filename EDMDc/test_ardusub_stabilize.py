"""Focused tests for the simulator's ArduSub STABILIZE compatibility layer."""
from __future__ import annotations

import unittest

import numpy as np

from .ardusub_stabilize import (
    ArduSubStabilizeConfig,
    ArduSubStabilizer,
)
from .thrusters import ThrusterConfig


def _heavy_config() -> ThrusterConfig:
    positions = np.array([
        [0.156, -0.111, -0.04],
        [0.156, 0.111, -0.04],
        [-0.156, -0.111, -0.04],
        [-0.156, 0.111, -0.04],
        [0.120, -0.218, 0.0],
        [0.120, 0.218, 0.0],
        [-0.120, -0.218, 0.0],
        [-0.120, 0.218, 0.0],
    ])
    axes = np.array([
        [0.707, 0.707, 0.0],
        [0.707, -0.707, 0.0],
        [-0.707, 0.707, 0.0],
        [-0.707, -0.707, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    return ThrusterConfig(
        num_rotors=8,
        positions=positions,
        thrust_axes=axes,
        directions=np.ones(8),
        force_constants=np.full(8, 4.4e-7),
        moment_constants=np.zeros(8),
        max_rotation_velocities=np.full(8, 3900.0),
        time_constants=np.full(8, 0.01),
    )


class ArduSubStabilizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tcfg = _heavy_config()
        self.controller = ArduSubStabilizer(self.tcfg)
        self.zero = np.zeros(8)

    def _command_torque(self, commands: np.ndarray) -> np.ndarray:
        torque = np.zeros(3)
        for pos, axis, command in zip(
                self.tcfg.positions, self.tcfg.thrust_axes, commands):
            torque += np.cross(pos, axis * command)
        return torque

    def test_geometry_mixer_has_zero_collective_and_correct_signs(self):
        patterns = self.controller.mixer_patterns
        np.testing.assert_allclose(patterns.mean(axis=0), 0.0, atol=1e-12)

        vertical = self.controller.vertical_indices
        for axis in range(2):
            commands = np.zeros(8)
            commands[vertical] = patterns[:, axis]
            response = self._command_torque(commands)[:2]
            self.assertGreater(response[axis], 0.0)
            self.assertAlmostEqual(response[1 - axis], 0.0, places=12)

    def test_level_and_stationary_does_not_change_manual_command(self):
        base = np.array([0.1, -0.2, 0.3, -0.4, 0.2, 0.2, 0.2, 0.2])
        actual = self.controller.mix(
            base, roll_rad=0.0, pitch_rad=0.0,
            p_rad_s=0.0, q_rad_s=0.0)
        np.testing.assert_array_equal(actual, base)

    def test_positive_angles_generate_restoring_roll_and_pitch_torque(self):
        roll_cmd = self.controller.mix(
            self.zero, roll_rad=np.deg2rad(15.0), pitch_rad=0.0,
            p_rad_s=0.0, q_rad_s=0.0)
        self.assertLess(self._command_torque(roll_cmd)[0], 0.0)

        pitch_cmd = self.controller.mix(
            self.zero, roll_rad=0.0, pitch_rad=np.deg2rad(15.0),
            p_rad_s=0.0, q_rad_s=0.0)
        self.assertLess(self._command_torque(pitch_cmd)[1], 0.0)

    def test_attitude_correction_does_not_own_collective_heave(self):
        base = np.zeros(8)
        base[4:] = 0.25
        actual = self.controller.mix(
            base, roll_rad=np.deg2rad(10.0),
            pitch_rad=np.deg2rad(-8.0),
            p_rad_s=0.0, q_rad_s=0.0)

        # Horizontal (planar/yaw) motors are never touched.
        np.testing.assert_array_equal(actual[:4], base[:4])
        # No saturation in this case: mean vertical command is still exactly
        # the operator's direct-heave command.
        self.assertAlmostEqual(float(actual[4:].mean()), 0.25, places=12)
        self.assertAlmostEqual(
            float(self.controller.last_status.motor_correction[4:].sum()),
            0.0, places=12)

    def test_combined_correction_and_heave_are_bounded(self):
        config = ArduSubStabilizeConfig(
            rate_p=(1.0, 1.0), max_correction=0.8)
        controller = ArduSubStabilizer(self.tcfg, config)
        base = np.zeros(8)
        base[4:] = 0.9
        actual = controller.mix(
            base, roll_rad=0.8, pitch_rad=-0.8,
            p_rad_s=1.0, q_rad_s=-1.0)
        self.assertLessEqual(float(np.max(np.abs(actual))), 1.0)
        self.assertLess(controller.last_status.desaturation_scale, 1.0)

    def test_no_depth_or_yaw_state_enters_the_controller(self):
        # The public call has only roll, pitch, p, and q. This also protects
        # against accidentally adding a hidden depth/yaw hold later.
        import inspect
        params = inspect.signature(self.controller.mix).parameters
        self.assertNotIn("depth", params)
        self.assertNotIn("yaw", params)
        self.assertNotIn("r_rad_s", params)


if __name__ == "__main__":
    unittest.main()
