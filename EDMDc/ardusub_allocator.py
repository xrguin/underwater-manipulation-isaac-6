"""Static ArduSub-style 4-axis allocator.

The runtime control input is a normalized stick command

    [surge, sway, heave, yaw] in [-1, 1]^4

matching ArduSub MANUAL_CONTROL axes x, y, z, r.  This module uses the
calibration produced by ``ardusub_check.py`` to map those four commands to the
eight BlueROV Heavy thruster commands in this repo's YAML order.  No live SITL
connection is needed in the simulation loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

COMMAND_DIM = 4
COMMAND_NAMES: tuple[str, ...] = ("surge", "sway", "heave", "yaw")
WRENCH_ACTIVE_IDX: tuple[int, ...] = (0, 1, 2, 5)
Z_NEUTRAL = 500.0


def _default_calib_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "ardusub_calib.npz"


@dataclass(frozen=True)
class ArduSubCalib:
    """Saved ArduSub calibration.

    ``demand`` maps a physical 4-axis wrench [Fx, Fy, Fz, Tz] to firmware motor
    commands.  ``k`` maps that physical wrench to MANUAL_CONTROL units, so
    dividing by the stick ranges gives the direct normalized-stick-to-motor map.
    """

    perm: np.ndarray
    sign: np.ndarray
    k: np.ndarray
    demand: np.ndarray
    gain: float
    throttle_gain: float

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ArduSubCalib":
        calib_path = Path(path) if path is not None else _default_calib_path()
        data = np.load(calib_path)
        if "demand" not in data.files:
            raise ValueError(
                f"{calib_path} has no demand matrix; rerun ardusub_check.py"
            )
        return cls(
            perm=data["perm"].astype(int),
            sign=data["sign"].astype(np.float64),
            k=data["k"].astype(np.float64),
            demand=data["demand"].astype(np.float64),
            gain=float(data["gain"]) if "gain" in data.files else 0.5,
            throttle_gain=(
                float(data["throttle_gain"])
                if "throttle_gain" in data.files else 1.0
            ),
        )


class StaticArduSubAllocator:
    """Map normalized 4-axis ArduSub commands to thruster commands."""

    def __init__(self, calib: ArduSubCalib | None = None):
        self.calib = calib if calib is not None else ArduSubCalib.load()
        self.last_scale = 1.0

        # demand columns are per Newton/Nm.  A unit normalized command is
        # equivalent to 1000/k for x, y, r and 500/k for heave z.
        authority = np.array(
            [
                1000.0 / abs(self.calib.k[0]),
                1000.0 / abs(self.calib.k[1]),
                500.0 / abs(self.calib.k[2]),
                1000.0 / abs(self.calib.k[3]),
            ],
            dtype=np.float64,
        )
        self.authority = authority
        self._stick_to_fw = self.calib.demand * authority.reshape(1, 4)

    def allocate(self, command4: np.ndarray) -> np.ndarray:
        """Return 8 thruster commands in YAML order.

        Args:
            command4: [surge, sway, heave, yaw] normalized to [-1, 1].
        """
        u = np.asarray(command4, dtype=np.float64).reshape(COMMAND_DIM)
        u = np.clip(u, -1.0, 1.0)
        cmd_fw = self._stick_to_fw @ u
        max_abs = float(np.max(np.abs(cmd_fw)))
        self.last_scale = 1.0
        if max_abs > 1.0:
            self.last_scale = 1.0 / max_abs
            cmd_fw = cmd_fw * self.last_scale

        cmd_yaml = np.zeros(8, dtype=np.float64)
        cmd_yaml[self.calib.perm] = self.calib.sign * cmd_fw
        return np.clip(cmd_yaml, -1.0, 1.0)

    def command_to_wrench4(self, command4: np.ndarray) -> np.ndarray:
        """Approximate steady-state active wrench before T200 motor lag."""
        u = np.asarray(command4, dtype=np.float64).reshape(COMMAND_DIM)
        return np.clip(u, -1.0, 1.0) * self.authority

    def command_to_wrench6(self, command4: np.ndarray) -> np.ndarray:
        wrench = np.zeros(6, dtype=np.float64)
        wrench[list(WRENCH_ACTIVE_IDX)] = self.command_to_wrench4(command4)
        return wrench


def command4_from_wrench6(wrench6: np.ndarray,
                          calib: ArduSubCalib | None = None) -> np.ndarray:
    """Convert [Fx, Fy, Fz, Tx, Ty, Tz] to normalized ArduSub command."""
    c = calib if calib is not None else ArduSubCalib.load()
    authority = np.array(
        [1000.0 / abs(c.k[0]), 1000.0 / abs(c.k[1]),
         500.0 / abs(c.k[2]), 1000.0 / abs(c.k[3])],
        dtype=np.float64,
    )
    w = np.asarray(wrench6, dtype=np.float64)
    return np.clip(w[list(WRENCH_ACTIVE_IDX)] / authority, -1.0, 1.0)
