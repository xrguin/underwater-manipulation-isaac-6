"""LQR controller built on the trained 34-feature EDMDc model.

Mirror of ``EDMDc.lqr_38`` but for the 34-D ν-only dictionary (no sin/cos
of roll, pitch). A is (34, 34), B is (34, 4), with inputs equal to normalized
ArduSub commands [surge, sway, heave, yaw] in [-1,1].

Library exports:
  * Config dataclass ``LQRConfig``.
  * ``EDMDcLQR`` class with ``from_npz()`` and ``step()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from scipy.linalg import solve_discrete_are
except ImportError as e:  # pragma: no cover
    raise SystemExit("scipy is required for EDMDc.lqr (DARE solver).") from e

from .edmdc import CONTROL_DIM, CONTROL_NAMES, DICT_DIM, NU_DIM, NU_NAMES, lift


@dataclass
class LQRConfig:
    """Knobs for the 34-feature EDMDc-LQR. Mirrors LQRConfig38 settings."""
    active_axes: Tuple = (0, 1, 2, 3)
    Q_nu_diag:  Tuple = (1000.0, 1000.0, 1000.0, 100.0, 100.0, 500.0)
    Q_aux_reg:  float = 1e-6
    R_diag:     Tuple = (1e-4, 1e-4, 1e-4, 1e-3)
    u_min:      Tuple = (-1.0, -1.0, -1.0, -1.0)
    u_max:      Tuple = ( 1.0,  1.0,  1.0,  1.0)


class EDMDcLQR:
    """Discrete-time LQR for the 34-feature EDMDc plant + saturation clip."""

    def __init__(self, A: np.ndarray, B: np.ndarray,
                 config: LQRConfig | None = None):
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        if self.A.shape != (DICT_DIM, DICT_DIM):
            raise ValueError(f"A must be ({DICT_DIM},{DICT_DIM}); got {self.A.shape}")
        if self.B.shape[0] != DICT_DIM or self.B.shape[1] != CONTROL_DIM:
            raise ValueError(
                f"B must be ({DICT_DIM},{CONTROL_DIM}); got {self.B.shape}"
            )
        self.cfg = config or LQRConfig()

        self.active_axes = list(self.cfg.active_axes)
        self.n_active    = len(self.active_axes)

        self.B_active = self.B[:, self.active_axes].copy()

        C_nu = np.zeros((NU_DIM, DICT_DIM), dtype=np.float64)
        C_nu[:, 1:1 + NU_DIM] = np.eye(NU_DIM)
        Q_nu  = np.diag(self.cfg.Q_nu_diag)
        Q_eff = C_nu.T @ Q_nu @ C_nu + self.cfg.Q_aux_reg * np.eye(DICT_DIM)
        Q_eff = 0.5 * (Q_eff + Q_eff.T)

        R_eff = np.diag(self.cfg.R_diag)
        if R_eff.shape[0] != self.n_active:
            raise ValueError(
                f"R_diag length {R_eff.shape[0]} != active_axes count {self.n_active}")

        P = solve_discrete_are(self.A, self.B_active, Q_eff, R_eff)
        BtP = self.B_active.T @ P
        self.K = np.linalg.solve(R_eff + BtP @ self.B_active, BtP @ self.A)
        self.P = P

        eigs_cl = np.linalg.eigvals(self.A - self.B_active @ self.K)
        self.cl_max_abs = float(np.max(np.abs(eigs_cl)))

        self.u_min = np.asarray(self.cfg.u_min, dtype=np.float64)
        self.u_max = np.asarray(self.cfg.u_max, dtype=np.float64)

    @classmethod
    def from_npz(cls, model_path: str | Path,
                 config: LQRConfig | None = None) -> "EDMDcLQR":
        m = np.load(model_path, allow_pickle=True)
        if "dict_dim" in m.files and int(m["dict_dim"]) != DICT_DIM:
            raise ValueError(
                f"Model {model_path} has dict_dim={int(m['dict_dim'])}; "
                f"expected {DICT_DIM}. Use EDMDc.lqr_38 for 38-feature models."
            )
        return cls(m["A"], m["B"], config=config)

    def step(self,
             nu_measured: np.ndarray,
             nu_ref:      np.ndarray,
             ) -> np.ndarray:
        """One control tick. Returns 4-D normalized ArduSub command.

        34-D dictionary lifts only ν (no attitude), so this signature drops
        roll/pitch vs EDMDcLQR38.step. Provided for interface parity in a
        backstep wrapper.
        """
        psi      = lift(np.asarray(nu_measured, dtype=np.float64))
        psi_ref  = lift(np.asarray(nu_ref, dtype=np.float64))
        delta    = psi - psi_ref
        u_active = -self.K @ delta

        u_full = np.zeros(self.B.shape[1], dtype=np.float64)
        for i, axis in enumerate(self.active_axes):
            u_full[axis] = u_active[i]
        np.clip(u_full, self.u_min, self.u_max, out=u_full)
        return u_full

    def reset(self) -> None:
        pass


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="LQR sanity check on a trained 34-feature EDMDc model.",
    )
    ap.add_argument("model", type=Path, help="edmdc_gripper_*.npz (34-D)")
    args = ap.parse_args()

    lqr = EDMDcLQR.from_npz(args.model)
    print(f"[lqr-34] loaded {args.model}")
    print(f"[lqr-34] K shape: {lqr.K.shape}  (active axes: "
          f"{[CONTROL_NAMES[i] for i in lqr.active_axes]})")
    print(f"[lqr-34] closed-loop max|λ| = {lqr.cl_max_abs:.6f}  "
          f"({'STABLE' if lqr.cl_max_abs < 1.0 else 'UNSTABLE'})")
    u1 = lqr.step(np.zeros(6), nu_ref=np.zeros(6))
    print(f"[lqr-34] u(rest, zero ref) = "
          f"{[f'{x:+.3f}' for x in u1]}  (expect ~0)")
    u2 = lqr.step(np.zeros(6), nu_ref=np.array([0.4, 0, 0, 0, 0, 0]))
    print(f"[lqr-34] u(surge ref 0.4) = "
          f"{[f'{x:+.3f}' for x in u2]}  (expect Fx > 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
