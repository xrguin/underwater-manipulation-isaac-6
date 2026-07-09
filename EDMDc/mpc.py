"""EDMDc-MPC velocity tracking controller.

Loads a trained EDMDc model and solves a quadratic program each control step to
drive the body velocity nu toward a reference command. Inputs are normalized
ArduSub commands [surge, sway, heave, yaw] in [-1,1].

Plain EDMDc models use the 34-D lifted state directly. HODMDc+ARX delay models
are converted to a linear companion form whose state contains lifted velocity
history and delayed inputs, giving them the same MPC interface.

The QP is solved with the condensed direct OSQP formulation: predicted states
are eliminated analytically, leaving only the N * 4 command sequence as
decision variables, with a persistent warm-started OSQP workspace whose
matrices are factorized once at setup. (The legacy sparse CVXPY formulation
was removed after benchmarking ~300x slower on the memory-augmented models;
see EDMDc/data/plots/mpc_solver_benchmark/ for the comparison numbers.)

Cost function (v1):

    J = sum_{i=0}^{N-1}  || C z_i - nu_ref(i) ||_Q^2          (stage tracking)
      +                  || C z_N - nu_ref(N) ||_{Q_N}^2      (terminal tracking)
      + sum_{i=0}^{N-1}  || u_i ||_R^2                        (control effort)

    s.t.   z_{i+1} = A z_i + B u_i,  z_0 = Psi(nu_measured)
           u_min <= u_i <= u_max
           du_min <= u_i - u_{i-1} <= du_max     (rate is hard-constrained, not penalized)

Usage:
    from EDMDc.mpc import EDMDcMPC, MPCConfig
    mpc = EDMDcMPC.from_npz("EDMDc/model/edmdc_<TS>.npz")
    u_cmd = mpc.step(nu_measured, nu_ref)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import osqp
    import scipy.sparse as sp
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "osqp and scipy are required for EDMDc.mpc. Install with:\n"
        "  pip install osqp scipy"
    ) from e

from .edmdc import CONTROL_DIM, DICT_DIM, NU_DIM, NU_NAMES, lift
from .edmdc_delay import MODEL_KIND as DELAY_MODEL_KIND


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MPCConfig:
    """Knobs for the EDMDc-MPC. Defaults are tuned for the BlueROV Heavy with
    the 34-feature monomial dictionary and 4-axis ArduSub command input.
    """
    # Horizon + timing
    N:  int   = 30                              # prediction horizon (steps)
    dt: float = 1.0 / 60.0                      # sampling time [s]

    # Cost weights (diag entries on the 6 axes)
    # State axes:  [u,  v,  w,  p,  q,  r]   -- track these aggressively.
    Q_diag:    Tuple = (1000.0, 1000.0, 1000.0, 100.0, 100.0, 500.0)
    Q_N_diag:  Tuple = (5000.0, 5000.0, 5000.0, 500.0, 500.0, 2500.0)
    # Input axes: [surge, sway, heave, yaw], normalized ArduSub command.
    R_diag:    Tuple = (1e-3, 1e-3, 1e-3, 1e-3)

    # Box constraints on normalized command.
    u_min:  Tuple = (-1.0, -1.0, -1.0, -1.0)
    u_max:  Tuple = ( 1.0,  1.0,  1.0,  1.0)

    # Rate constraints per control step.
    du_min: Tuple = (-0.25, -0.25, -0.25, -0.25)
    du_max: Tuple = ( 0.25,  0.25,  0.25,  0.25)

    # Solver backend. Only the condensed direct OSQP formulation remains;
    # "auto" is kept as an accepted alias so existing CLI flags keep working.
    solver_backend: str = "condensed_osqp"
    osqp_eps_abs: float = 1e-5
    osqp_eps_rel: float = 1e-5
    osqp_max_iter: int = 10000
    osqp_polish: bool = False


def _resolve_backend(name: str) -> str:
    backend = str(name).lower().replace("-", "_")
    if backend in {"auto", "condensed", "direct", "direct_osqp", "condensed_osqp"}:
        return "condensed_osqp"
    if backend in {"cvxpy", "cvxpy_osqp"}:
        raise ValueError(
            "the CVXPY backend was removed; use solver_backend='condensed_osqp'"
        )
    raise ValueError(
        f"solver_backend must be auto or condensed_osqp; got {name!r}"
    )


def _reference_matrix(nu_ref: np.ndarray, horizon: int) -> np.ndarray:
    ref = np.asarray(nu_ref, dtype=np.float64)
    if ref.ndim == 1:
        ref = np.tile(ref[:, None], (1, horizon + 1))
    if ref.shape != (NU_DIM, horizon + 1):
        raise ValueError(
            f"nu_ref must be ({NU_DIM},) or ({NU_DIM}, N+1={horizon + 1});"
            f" got {ref.shape}"
        )
    return ref


class _CondensedOSQPSolver:
    """Small dense-control QP for the linear EDMDc MPC.

    Eliminates the predicted lifted states analytically, leaving only the
    N * m command sequence as OSQP decision variables. This is equivalent to
    the CVXPY formulation but avoids repeatedly solving for hundreds of lifted
    state variables in the memory-augmented model.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                 config: MPCConfig):
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.cfg = config
        self.N = int(config.N)
        self.d = int(self.A.shape[0])
        self.m = int(self.B.shape[1])
        self.n_u = self.N * self.m

        self._build_prediction_matrices()
        self._build_cost_matrices()
        self._build_constraint_matrices()

        self.prob = osqp.OSQP()
        self.prob.setup(
            P=sp.csc_matrix(np.triu(self.P)),
            q=np.zeros(self.n_u, dtype=np.float64),
            A=self.A_cons,
            l=self.l_base.copy(),
            u=self.u_base.copy(),
            verbose=False,
            warm_start=True,
            polish=bool(config.osqp_polish),
            eps_abs=float(config.osqp_eps_abs),
            eps_rel=float(config.osqp_eps_rel),
            max_iter=int(config.osqp_max_iter),
        )
        self.last_status = "not_solved"

    def _build_prediction_matrices(self) -> None:
        N, d, m = self.N, self.d, self.m
        y_dim = NU_DIM * (N + 1)
        self.Sx = np.zeros((y_dim, d), dtype=np.float64)
        self.Su = np.zeros((y_dim, N * m), dtype=np.float64)

        A_powers = [np.eye(d, dtype=np.float64)]
        for _ in range(N):
            A_powers.append(A_powers[-1] @ self.A)

        for i in range(N + 1):
            row = slice(i * NU_DIM, (i + 1) * NU_DIM)
            self.Sx[row, :] = self.C @ A_powers[i]
            for j in range(i):
                col = slice(j * m, (j + 1) * m)
                self.Su[row, col] = self.C @ A_powers[i - 1 - j] @ self.B

    def _build_cost_matrices(self) -> None:
        N, m = self.N, self.m
        q_diag = np.asarray(self.cfg.Q_diag, dtype=np.float64)
        qn_diag = np.asarray(self.cfg.Q_N_diag, dtype=np.float64)
        r_diag = np.asarray(self.cfg.R_diag, dtype=np.float64)
        qbar_diag = np.concatenate([np.tile(q_diag, N), qn_diag])
        rbar_diag = np.tile(r_diag, N)

        weighted_su = qbar_diag[:, None] * self.Su
        self.H_ref = 2.0 * self.Su.T @ np.diag(qbar_diag)
        self.P = 2.0 * (self.Su.T @ weighted_su + np.diag(rbar_diag))
        # Symmetrize to suppress tiny roundoff asymmetry before sparse setup.
        self.P = 0.5 * (self.P + self.P.T)

    def _build_constraint_matrices(self) -> None:
        N, m = self.N, self.m
        u_min = np.asarray(self.cfg.u_min, dtype=np.float64)
        u_max = np.asarray(self.cfg.u_max, dtype=np.float64)
        du_min = np.asarray(self.cfg.du_min, dtype=np.float64)
        du_max = np.asarray(self.cfg.du_max, dtype=np.float64)

        I = np.eye(N * m, dtype=np.float64)
        D = np.zeros((N * m, N * m), dtype=np.float64)
        for i in range(N):
            rows = slice(i * m, (i + 1) * m)
            D[rows, i * m:(i + 1) * m] = np.eye(m)
            if i > 0:
                D[rows, (i - 1) * m:i * m] = -np.eye(m)

        self.A_cons = sp.csc_matrix(np.vstack([I, D]))
        self.l_base = np.concatenate([np.tile(u_min, N), np.tile(du_min, N)])
        self.u_base = np.concatenate([np.tile(u_max, N), np.tile(du_max, N)])

    def solve(self, x0: np.ndarray, nu_ref: np.ndarray,
              u_prev: np.ndarray) -> np.ndarray | None:
        ref = _reference_matrix(nu_ref, self.N).T.reshape(-1)
        q = self.H_ref @ (self.Sx @ np.asarray(x0, dtype=np.float64) - ref)

        l = self.l_base.copy()
        u = self.u_base.copy()
        rate_start = self.N * self.m
        l[rate_start:rate_start + self.m] += u_prev
        u[rate_start:rate_start + self.m] += u_prev

        try:
            self.prob.update(q=q, l=l, u=u)
            res = self.prob.solve()
        except Exception as exc:  # pragma: no cover - solver-dependent
            self.last_status = f"exception:{type(exc).__name__}"
            return None

        self.last_status = str(res.info.status)
        if res.x is None or res.info.status_val not in (1, 2):
            return None
        # OSQP without polish can violate bounds by ~eps; keep the applied
        # command strictly inside the box.
        u0 = np.asarray(res.x[:self.m], dtype=np.float64)
        return np.clip(u0, self.cfg.u_min, self.cfg.u_max)


# ============================================================================
# Controller
# ============================================================================

class EDMDcMPC:
    """Linear MPC built on a trained EDMDc (A, B) lifted predictor."""

    def __init__(self, A: np.ndarray, B: np.ndarray,
                 config: MPCConfig | None = None):
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        if self.A.shape != (DICT_DIM, DICT_DIM):
            raise ValueError(
                f"A must be ({DICT_DIM}, {DICT_DIM}); got {self.A.shape}"
            )
        if self.B.shape[0] != DICT_DIM or self.B.shape[1] != CONTROL_DIM:
            raise ValueError(
                f"B must be ({DICT_DIM}, {CONTROL_DIM}); got {self.B.shape}"
            )
        self.d  = self.A.shape[0]
        self.m  = self.B.shape[1]
        self.cfg = config or MPCConfig()

        # Output selector C picks nu (indices 1..6) out of Psi.
        self.C = np.zeros((NU_DIM, self.d), dtype=np.float64)
        self.C[:, 1:1 + NU_DIM] = np.eye(NU_DIM)

        # Previous applied control (for rate term)
        self.u_prev = np.zeros(self.m, dtype=np.float64)

        self.solver_backend = _resolve_backend(self.cfg.solver_backend)
        self.last_solver_status = "not_solved"
        self.solve_count = 0
        self.solve_fail_count = 0
        self.solve_time_total_s = 0.0
        self._direct_solver = _CondensedOSQPSolver(
            self.A, self.B, self.C, self.cfg
        )

    # ------------------------------------------------------------------ I/O

    @classmethod
    def from_npz(cls, model_path: str | Path,
                 config: MPCConfig | None = None) -> "EDMDcMPC":
        m = np.load(model_path, allow_pickle=True)
        if "kind" in m.files and str(m["kind"]) == DELAY_MODEL_KIND:
            return DelayEDMDcMPC.from_npz(model_path, config=config)
        return cls(m["A"], m["B"], config=config)

    # ----------------------------------------------------------------- step

    def step(self,
             nu_measured: np.ndarray,
             nu_ref:      np.ndarray) -> np.ndarray:
        """Solve one MPC step and return the 4-axis command for this instant.

        Args:
            nu_measured: (6,) current body velocity.
            nu_ref:      either (6,) constant reference or (6, N+1) trajectory.

        Returns:
            u0: (4,) normalized ArduSub command to send to the plant.
        """
        # Lift current state
        z0 = lift(np.asarray(nu_measured, dtype=np.float64))
        if z0.ndim == 2:
            z0 = z0[0]

        ref = _reference_matrix(nu_ref, self.cfg.N)

        t_solve = time.perf_counter()
        u0 = self._direct_solver.solve(z0, ref, self.u_prev)
        self.solve_time_total_s += time.perf_counter() - t_solve
        self.solve_count += 1
        self.last_solver_status = self._direct_solver.last_status
        if u0 is None:
            self.solve_fail_count += 1
            u0 = np.zeros(self.m, dtype=np.float64)
        self.u_prev = u0.copy()
        return u0

    def reset(self) -> None:
        """Clear u_prev so a fresh trajectory starts from zero command."""
        self.u_prev = np.zeros(self.m, dtype=np.float64)

    def observe(self, nu_measured: np.ndarray) -> None:
        """Update any controller history on non-solve ticks.

        Plain EDMDc has no internal measurement history, so this is a no-op.
        Delay-model MPC overrides it.
        """
        _ = nu_measured


class DelayEDMDcMPC:
    """Companion-form MPC for HODMDc+ARX memory models.

    The trained model is

        z(k+1) = sum_i A_i z(k-i) + sum_j B_j u(k-j)

    where z is the 34-D EDMDc lift. The MPC state is

        x(k) = [z(k), z(k-1), ..., z(k-ds),
                u(k-1), ..., u(k-di)].

    The first input block B_0 multiplies the MPC decision variable u(k); the
    remaining delayed-input blocks are part of the companion state.
    """

    model_kind = DELAY_MODEL_KIND

    def __init__(self, A_list: list[np.ndarray], B_list: list[np.ndarray],
                 config: MPCConfig | None = None):
        if not A_list or not B_list:
            raise ValueError("delay MPC requires at least one A block and one B block")
        self.A_list = [np.asarray(Ai, dtype=np.float64) for Ai in A_list]
        self.B_list = [np.asarray(Bj, dtype=np.float64) for Bj in B_list]
        self.ds = len(self.A_list) - 1
        self.di = len(self.B_list) - 1
        self.base_d = DICT_DIM
        self.m = CONTROL_DIM
        for i, Ai in enumerate(self.A_list):
            if Ai.shape != (DICT_DIM, DICT_DIM):
                raise ValueError(
                    f"A{i} must be ({DICT_DIM}, {DICT_DIM}); got {Ai.shape}"
                )
        for j, Bj in enumerate(self.B_list):
            if Bj.shape != (DICT_DIM, CONTROL_DIM):
                raise ValueError(
                    f"B{j} must be ({DICT_DIM}, {CONTROL_DIM}); got {Bj.shape}"
                )

        self.cfg = config or MPCConfig()
        self.d = DICT_DIM * (self.ds + 1) + CONTROL_DIM * self.di
        self.A, self.B = self._companion_matrices()

        self.C = np.zeros((NU_DIM, self.d), dtype=np.float64)
        self.C[:, 1:1 + NU_DIM] = np.eye(NU_DIM)

        self.u_prev = np.zeros(self.m, dtype=np.float64)
        self._z_hist: list[np.ndarray] | None = None
        self._u_hist: list[np.ndarray] = [
            np.zeros(self.m, dtype=np.float64) for _ in range(self.di)
        ]

        self.solver_backend = _resolve_backend(self.cfg.solver_backend)
        self.last_solver_status = "not_solved"
        self.solve_count = 0
        self.solve_fail_count = 0
        self.solve_time_total_s = 0.0
        self._direct_solver = _CondensedOSQPSolver(
            self.A, self.B, self.C, self.cfg
        )

    @classmethod
    def from_npz(cls, model_path: str | Path,
                 config: MPCConfig | None = None) -> "DelayEDMDcMPC":
        m = np.load(model_path, allow_pickle=True)
        if "kind" not in m.files or str(m["kind"]) != DELAY_MODEL_KIND:
            raise ValueError(f"{model_path} is not a {DELAY_MODEL_KIND} model")
        ds = int(m["state_delay"])
        di = int(m["input_delay"])
        A_list = [m[f"A{i}"] for i in range(ds + 1)]
        B_list = [m[f"B{j}"] for j in range(di + 1)]
        return cls(A_list, B_list, config=config)

    def _companion_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        d0 = DICT_DIM
        m = CONTROL_DIM
        z_blocks = self.ds + 1
        u_off = d0 * z_blocks
        A = np.zeros((self.d, self.d), dtype=np.float64)
        B = np.zeros((self.d, m), dtype=np.float64)

        for i, Ai in enumerate(self.A_list):
            A[:d0, i * d0:(i + 1) * d0] = Ai
        B[:d0, :] = self.B_list[0]
        for j in range(1, self.di + 1):
            A[:d0, u_off + (j - 1) * m:u_off + j * m] = self.B_list[j]

        for i in range(self.ds):
            A[(i + 1) * d0:(i + 2) * d0, i * d0:(i + 1) * d0] = np.eye(d0)

        if self.di > 0:
            B[u_off:u_off + m, :] = np.eye(m)
            for j in range(self.di - 1):
                src = u_off + j * m
                dst = u_off + (j + 1) * m
                A[dst:dst + m, src:src + m] = np.eye(m)

        return A, B

    def _advance_history(self, nu_measured: np.ndarray) -> np.ndarray:
        z_now = lift(np.asarray(nu_measured, dtype=np.float64))
        if self._z_hist is None:
            self._z_hist = [z_now.copy() for _ in range(self.ds + 1)]
        else:
            self._z_hist.insert(0, z_now.copy())
            del self._z_hist[self.ds + 1:]

        if self.di > 0:
            self._u_hist.insert(0, self.u_prev.copy())
            del self._u_hist[self.di:]

        return np.concatenate(self._z_hist + self._u_hist)

    def step(self,
             nu_measured: np.ndarray,
             nu_ref: np.ndarray) -> np.ndarray:
        x0 = self._advance_history(nu_measured)

        ref = _reference_matrix(nu_ref, self.cfg.N)

        t_solve = time.perf_counter()
        u0 = self._direct_solver.solve(x0, ref, self.u_prev)
        self.solve_time_total_s += time.perf_counter() - t_solve
        self.solve_count += 1
        self.last_solver_status = self._direct_solver.last_status
        if u0 is None:
            self.solve_fail_count += 1
            u0 = np.zeros(self.m, dtype=np.float64)
        self.u_prev = u0.copy()
        return u0

    def observe(self, nu_measured: np.ndarray) -> None:
        """Advance memory on held-control ticks when MPC is decimated."""
        self._advance_history(nu_measured)

    def reset(self) -> None:
        self.u_prev = np.zeros(self.m, dtype=np.float64)
        self._z_hist = None
        self._u_hist = [np.zeros(self.m, dtype=np.float64)
                        for _ in range(self.di)]
