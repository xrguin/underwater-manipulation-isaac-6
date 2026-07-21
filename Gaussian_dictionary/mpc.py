"""MPC adapter for the four-state Gaussian EDMDc model.

The public controller interface mirrors the existing EDMDc MPC: ``step`` may
receive a full six-velocity measurement/reference and returns the normalized
four-axis ArduSub command ``[surge, sway, heave, yaw]``.  Internally it tracks
only ``[u, v, w, r]`` because those are the states learned by this model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from .gaussian_edmdc import (
    CONTROL_DIM,
    STATE_DIM,
    STATE_INDICES,
    GaussianEDMDcModel,
    select_controlled_state,
)


@dataclass
class GaussianMPCConfig:
    """Configuration for four-state velocity tracking."""

    N: int = 30
    dt: float = 1.0 / 60.0
    # State axes [u, v, w, r].
    Q_diag: Tuple[float, ...] = (1000.0, 1000.0, 1000.0, 500.0)
    Q_N_diag: Tuple[float, ...] = (5000.0, 5000.0, 5000.0, 2500.0)
    # Command axes [surge, sway, heave, yaw].
    R_diag: Tuple[float, ...] = (1e-3, 1e-3, 1e-3, 1e-3)
    u_min: Tuple[float, ...] = (-1.0, -1.0, -1.0, -1.0)
    u_max: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    du_min: Tuple[float, ...] = (-0.25, -0.25, -0.25, -0.25)
    du_max: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    osqp_eps_abs: float = 1e-5
    osqp_eps_rel: float = 1e-5
    osqp_max_iter: int = 10_000
    osqp_polish: bool = False


def _validate_vector(value: Tuple[float, ...], length: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (length,):
        raise ValueError(f"{name} must contain {length} values; got {arr.shape}")
    return arr


def _reference_matrix(reference: np.ndarray, horizon: int) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim == 1:
        ref = select_controlled_state(ref)
        ref = np.tile(ref[:, None], (1, horizon + 1))
    elif ref.ndim == 2:
        if ref.shape[0] == 6:
            ref = ref[list(STATE_INDICES), :]
        elif ref.shape[0] != STATE_DIM:
            raise ValueError(
                f"reference must have {STATE_DIM} or 6 rows; got {ref.shape}"
            )
    else:
        raise ValueError(f"reference must be a vector or matrix; got {ref.shape}")
    if ref.shape != (STATE_DIM, horizon + 1):
        raise ValueError(
            f"reference must be ({STATE_DIM},), (6,), ({STATE_DIM}, {horizon + 1}), "
            f"or (6, {horizon + 1}); got {ref.shape}"
        )
    return ref


class GaussianEDMDcMPC:
    """Condensed OSQP controller using a trained Gaussian EDMDc model."""

    def __init__(
        self,
        model: GaussianEDMDcModel,
        config: GaussianMPCConfig | None = None,
    ) -> None:
        try:
            import osqp
            import scipy.sparse as sp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "GaussianEDMDcMPC requires osqp and scipy; install them in the "
                "same environment used by the current EDMDc MPC"
            ) from exc

        self._osqp = osqp
        self._sp = sp
        self.model = model
        self.A = model.A
        self.B = model.B
        self.cfg = config or GaussianMPCConfig(dt=model.dt if np.isfinite(model.dt) else 1.0 / 60.0)
        self.N = int(self.cfg.N)
        if self.N < 1:
            raise ValueError("MPC horizon N must be at least 1")
        self.d = model.dimension
        self.m = CONTROL_DIM

        self.C = np.zeros((STATE_DIM, self.d), dtype=np.float64)
        self.C[:, :STATE_DIM] = np.eye(STATE_DIM)
        self.u_prev = np.zeros(self.m, dtype=np.float64)
        self.last_solver_status = "not_solved"
        self.solve_count = 0
        self.solve_fail_count = 0
        self.solve_time_total_s = 0.0

        self._build_prediction_matrices()
        self._build_cost()
        self._build_constraints()
        self._problem = osqp.OSQP()
        self._problem.setup(
            P=sp.csc_matrix(np.triu(self._P)),
            q=np.zeros(self.N * self.m, dtype=np.float64),
            A=self._constraint_matrix,
            l=self._lower_base.copy(),
            u=self._upper_base.copy(),
            verbose=False,
            warm_start=True,
            polish=bool(self.cfg.osqp_polish),
            eps_abs=float(self.cfg.osqp_eps_abs),
            eps_rel=float(self.cfg.osqp_eps_rel),
            max_iter=int(self.cfg.osqp_max_iter),
        )

    @classmethod
    def from_npz(
        cls,
        model_path: str | Path,
        config: GaussianMPCConfig | None = None,
    ) -> "GaussianEDMDcMPC":
        return cls(GaussianEDMDcModel.load(model_path), config=config)

    def _build_prediction_matrices(self) -> None:
        output_dim = STATE_DIM * (self.N + 1)
        self._Sx = np.zeros((output_dim, self.d), dtype=np.float64)
        self._Su = np.zeros((output_dim, self.N * self.m), dtype=np.float64)
        powers = [np.eye(self.d, dtype=np.float64)]
        for _ in range(self.N):
            powers.append(powers[-1] @ self.A)
        for i in range(self.N + 1):
            row = slice(i * STATE_DIM, (i + 1) * STATE_DIM)
            self._Sx[row] = self.C @ powers[i]
            for j in range(i):
                column = slice(j * self.m, (j + 1) * self.m)
                self._Su[row, column] = self.C @ powers[i - 1 - j] @ self.B

    def _build_cost(self) -> None:
        q = _validate_vector(self.cfg.Q_diag, STATE_DIM, "Q_diag")
        q_terminal = _validate_vector(self.cfg.Q_N_diag, STATE_DIM, "Q_N_diag")
        r = _validate_vector(self.cfg.R_diag, self.m, "R_diag")
        q_bar = np.concatenate([np.tile(q, self.N), q_terminal])
        r_bar = np.tile(r, self.N)
        weighted_su = q_bar[:, None] * self._Su
        self._reference_gain = 2.0 * (self._Su.T * q_bar[None, :])
        self._P = 2.0 * (self._Su.T @ weighted_su + np.diag(r_bar))
        self._P = 0.5 * (self._P + self._P.T)

    def _build_constraints(self) -> None:
        u_min = _validate_vector(self.cfg.u_min, self.m, "u_min")
        u_max = _validate_vector(self.cfg.u_max, self.m, "u_max")
        du_min = _validate_vector(self.cfg.du_min, self.m, "du_min")
        du_max = _validate_vector(self.cfg.du_max, self.m, "du_max")
        if np.any(u_min > u_max) or np.any(du_min > du_max):
            raise ValueError("minimum constraints must not exceed maximum constraints")

        n_commands = self.N * self.m
        identity = np.eye(n_commands, dtype=np.float64)
        difference = np.zeros((n_commands, n_commands), dtype=np.float64)
        for i in range(self.N):
            rows = slice(i * self.m, (i + 1) * self.m)
            difference[rows, rows] = np.eye(self.m)
            if i > 0:
                previous = slice((i - 1) * self.m, i * self.m)
                difference[rows, previous] = -np.eye(self.m)
        self._constraint_matrix = self._sp.csc_matrix(np.vstack([identity, difference]))
        self._lower_base = np.concatenate([np.tile(u_min, self.N), np.tile(du_min, self.N)])
        self._upper_base = np.concatenate([np.tile(u_max, self.N), np.tile(du_max, self.N)])

    def step(self, nu_measured: np.ndarray, nu_ref: np.ndarray) -> np.ndarray:
        """Solve one step and return normalized ``[surge,sway,heave,yaw]``."""
        x = select_controlled_state(np.asarray(nu_measured, dtype=np.float64))
        if x.ndim != 1:
            raise ValueError("nu_measured must be one 4-D or 6-D velocity vector")
        reference = _reference_matrix(nu_ref, self.N).T.reshape(-1)
        z0 = self.model.lift(x)
        linear = self._reference_gain @ (self._Sx @ z0 - reference)

        lower = self._lower_base.copy()
        upper = self._upper_base.copy()
        first_rate = self.N * self.m
        lower[first_rate:first_rate + self.m] += self.u_prev
        upper[first_rate:first_rate + self.m] += self.u_prev

        start = time.perf_counter()
        self._problem.update(q=linear, l=lower, u=upper)
        result = self._problem.solve()
        self.solve_time_total_s += time.perf_counter() - start
        self.solve_count += 1
        self.last_solver_status = str(result.info.status)
        if result.x is None or result.info.status_val not in (1, 2):
            self.solve_fail_count += 1
            command = np.zeros(self.m, dtype=np.float64)
        else:
            command = np.asarray(result.x[:self.m], dtype=np.float64)
            command = np.clip(command, self.cfg.u_min, self.cfg.u_max)
        self.u_prev = command.copy()
        return command

    def reset(self) -> None:
        self.u_prev = np.zeros(self.m, dtype=np.float64)

    def observe(self, nu_measured: np.ndarray) -> None:
        """Compatibility no-op; this model has no delay-state history."""
        _ = nu_measured
