"""EDMDc dictionary + training (Tikhonov-regularized least squares).

Library exports (used by both this module's CLI and EDMDc.evaluate_edmdc):

  * NU_DIM, NU_NAMES, DICT_DIM     dictionary shape constants
  * lift(nu) -> Psi                34-feature monomial dictionary
  * decode_nu(Psi) -> nu           inverse (slices the linear-feature rows)
  * feature_names() -> list[str]   per-feature human-readable names
  * fit_edmdc(nu, nu_next, tau)    Tikhonov-regularized closed-form solve
  * rollout(A, B, nu0, tau_seq)    multi-step lifted rollout

Dictionary captures the Fossen nonlinearities under task-DOF excitation:

  * constant:               1                        (1 feature)
  * linear:                 nu_i                     (6 features)
  * quadratic monomials:    nu_i * nu_j, i <= j      (21 features)
  * quadratic damping:      |nu_i| * nu_i            (6 features)

Total dictionary dim = 1 + 6 + 21 + 6 = 34.

Loss function (Tikhonov):

    min_{A,B}  ||Z_+ - A Z - B T||_F^2  +  lambda (||A||_F^2 + ||B||_F^2)

Closed-form: [A  B] = Z_+ Phi^T (Phi Phi^T + lambda I)^{-1}, Phi = [Z; U].

Usage:
    # Train on the 200-trajectory dataset
    python -m EDMDc.edmdc EDMDc/data/numpy/fossen_<TS>.npz

    # Evaluate a trained model on held-out test data (separate module)
    python -m EDMDc.evaluate_edmdc \\
        EDMDc/model/edmdc_<TS>.npz \\
        EDMDc/data/numpy/fossen_<test-TS>.npz
"""
from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from ._common import project_root


# ============================================================================
# Dictionary
# ============================================================================

NU_DIM = 6
NU_NAMES: tuple[str, ...] = ("u", "v", "w", "p", "q", "r")
_QUAD_PAIRS = list(combinations_with_replacement(range(NU_DIM), 2))   # 21 pairs
DICT_DIM = 1 + NU_DIM + len(_QUAD_PAIRS) + NU_DIM                     # = 34

# Control channels are normalized ArduSub MANUAL_CONTROL axes.
CONTROL_DIM = 4
CONTROL_NAMES: tuple[str, ...] = ("surge", "sway", "heave", "yaw")
ACTIVE_DOF: tuple[int, ...] = (0, 1, 2, 3)


def lift(nu: np.ndarray) -> np.ndarray:
    """Lift body velocity nu through the dictionary.

    Args:
        nu: (N, 6) or (6,) body velocity.

    Returns:
        Psi: (N, 34) or (34,) — matches input rank.
    """
    nu_arr = np.asarray(nu, dtype=np.float64)
    squeeze = nu_arr.ndim == 1
    if squeeze:
        nu_arr = nu_arr[None, :]
    if nu_arr.shape[-1] != NU_DIM:
        raise ValueError(f"lift expects last dim = {NU_DIM}, got shape {nu_arr.shape}")
    N = nu_arr.shape[0]

    ones   = np.ones((N, 1), dtype=np.float64)
    linear = nu_arr                                                         # (N, 6)
    quad   = np.stack(
        [nu_arr[:, i] * nu_arr[:, j] for i, j in _QUAD_PAIRS], axis=1,
    )                                                                       # (N, 21)
    abs_q  = np.abs(nu_arr) * nu_arr                                        # (N, 6)

    Psi = np.hstack([ones, linear, quad, abs_q])                            # (N, 34)
    return Psi[0] if squeeze else Psi


def decode_nu(Psi: np.ndarray) -> np.ndarray:
    """Recover nu from lifted state by slicing the linear-feature rows."""
    return Psi[..., 1:1 + NU_DIM]


def feature_names() -> list[str]:
    """Human-readable name for each of the 34 features."""
    names = ["1"]
    names += list(NU_NAMES)
    for i, j in _QUAD_PAIRS:
        if i == j:
            names.append(f"{NU_NAMES[i]}^2")
        else:
            names.append(f"{NU_NAMES[i]}*{NU_NAMES[j]}")
    for n in NU_NAMES:
        names.append(f"|{n}|*{n}")
    return names


# ============================================================================
# Training
# ============================================================================

def fit_edmdc(
    nu: np.ndarray, nu_next: np.ndarray, tau: np.ndarray,
    lam: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Tikhonov-regularized closed-form fit of A, B."""
    Z      = lift(nu).T          # (d, N)
    Z_plus = lift(nu_next).T     # (d, N)
    T      = tau.T               # (m, N)
    d, m = Z.shape[0], T.shape[0]
    Phi = np.vstack([Z, T])      # (d + m, N)
    G   = Phi @ Phi.T + lam * np.eye(d + m)
    rhs = Z_plus @ Phi.T         # (d, d + m)
    sol = np.linalg.solve(G.T, rhs.T).T   # = rhs @ G^{-1}
    A, B = sol[:, :d], sol[:, d:]
    return A, B


def fit_edmdc_bilinear(
    nu: np.ndarray, nu_next: np.ndarray, tau: np.ndarray,
    lam: float = 1e-3, active: tuple[int, ...] = ACTIVE_DOF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Koopman Bilinear Realization (KBR) fit — paper eq. (7).

        z_+ = A z + B u + sum_j H_j z u_j,      z = lift(nu)

    For a control-affine system (Fossen: nu_dot = f(nu) + M^{-1} tau, constant
    gain) the lifted dynamics are genuinely bilinear: d/dt psi(nu) =
    grad psi(nu) . M^{-1} tau couples the input with the (nu-dependent)
    gradient of every nonlinear feature.  The plain `fit_edmdc` collapses that
    coupling into a constant B (KLR), valid only at the features' linearization
    point.  This fit recovers the H block.

    Args:
        nu, nu_next: (N, 6) body-velocity snapshot pairs.
        tau:         (N, m) control input; default m=4 ArduSub command axes.
        lam:         Tikhonov regularization.
        active:      wrench indices that carry the bilinear term.

    Returns:
        A: (d, d), B: (d, m), H: (d, d, p) with p = len(active);
        H[:, :, j] is the matrix multiplying z for active channel j.
    """
    Z      = lift(nu)                                  # (N, d)
    Z_plus = lift(nu_next).T                           # (d, N)
    U      = np.asarray(tau, dtype=np.float64)         # (N, m)
    Ua     = U[:, list(active)]                        # (N, p)
    N, d = Z.shape
    m, p = U.shape[1], Ua.shape[1]

    # Bilinear features z_i * u_j, flattened column index i*p + j.
    ZU  = (Z[:, :, None] * Ua[:, None, :]).reshape(N, d * p)   # (N, d*p)
    Phi = np.hstack([Z, U, ZU]).T                             # (d + m + d*p, N)
    G   = Phi @ Phi.T + lam * np.eye(Phi.shape[0])
    sol = np.linalg.solve(G.T, (Z_plus @ Phi.T).T).T          # (d, d+m+d*p)

    A = sol[:, :d]
    B = sol[:, d:d + m]
    H = sol[:, d + m:].reshape(d, d, p)
    return A, B, H


def rollout_bilinear(
    A: np.ndarray, B: np.ndarray, H: np.ndarray,
    nu0: np.ndarray, tau_seq: np.ndarray,
    active: tuple[int, ...] = ACTIVE_DOF,
) -> np.ndarray:
    """Multi-step KBR rollout (mirror of `rollout` for the bilinear model).

    Returns nu_hat: (H_steps + 1, 6) with nu_hat[0] = nu0.
    """
    H_steps = tau_seq.shape[0]
    d = A.shape[0]
    idx = list(active)
    Psi = np.empty((H_steps + 1, d), dtype=np.float64)
    Psi[0] = lift(nu0)
    for k in range(H_steps):
        z = Psi[k]
        u = tau_seq[k]
        bil = np.einsum("oij,i,j->o", H, z, u[idx])
        Psi[k + 1] = A @ z + B @ u + bil
    return decode_nu(Psi)


def _select_complete(data, nu, nu_next, tau, traj_idx, step_idx):
    """Filter to trajectories whose max step_idx >= n_steps (completed APRBS)."""
    n_steps = int(round(float(data["episode_seconds"]) / float(data["dt"])))
    keep = set()
    for t in np.unique(traj_idx):
        if int(step_idx[traj_idx == t].max()) >= n_steps:
            keep.add(int(t))
    mask = np.isin(traj_idx, list(keep))
    return nu[mask], nu_next[mask], tau[mask], len(keep)


def _train(args: argparse.Namespace) -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    d = np.load(args.npz)
    nu      = d["X"]
    nu_next = d["X_next"]
    wrench_kind = getattr(args, "wrench", "commanded")
    if wrench_kind == "realized":
        if "U_realized" not in d.files:
            raise SystemExit(
                f"[train] --wrench realized requested but {args.npz.name} has no "
                "'U_realized' column (collected before the realized-wrench change). "
                "Re-collect with the current collect_isaac.py, or use --wrench commanded."
            )
        tau = d["U_realized"]
    else:
        tau = d["U"]
    input_units = str(d["input_units"]) if "input_units" in d.files else "unknown"
    print(f"[train] loaded {nu.shape[0]:,} snapshots from {args.npz.name} "
          f"(input = {wrench_kind}, units={input_units}, dim={tau.shape[1]})")

    if args.only_complete:
        nu, nu_next, tau, n_kept = _select_complete(
            d, nu, nu_next, tau, d["traj_idx"], d["step_idx"],
        )
        print(f"[train] only-complete: {n_kept} trajs, {nu.shape[0]:,} snapshots")

    print(f"[train] fitting EDMDc: dict_dim={DICT_DIM}, lam={args.lam}")
    A, B = fit_edmdc(nu, nu_next, tau, lam=args.lam)
    print(f"[train] A shape: {A.shape}, B shape: {B.shape}")

    # One-step training residual.
    Z      = lift(nu).T
    Z_plus = lift(nu_next).T
    Z_pred = A @ Z + B @ tau.T
    lifted_rmse = float(np.sqrt(np.mean((Z_plus - Z_pred) ** 2)))
    nu_pred = decode_nu(Z_pred.T)
    nu_err = nu_next - nu_pred
    rmse_axis = np.sqrt(np.mean(nu_err ** 2, axis=0))
    print(f"[train] one-step lifted RMSE: {lifted_rmse:.4e}")
    print(f"[train] one-step nu RMSE per axis (training data):")
    for n, r in zip(NU_NAMES, rmse_axis):
        print(f"          {n}: {r:.4e}")

    # Spectral stability.
    eigs = np.linalg.eigvals(A)
    max_abs = float(np.max(np.abs(eigs)))
    stab = "STABLE (rho(A) <= 1)" if max_abs <= 1.0 + 1e-9 else "UNSTABLE (rho(A) > 1)"
    print(f"[train] eig(A): max |lambda| = {max_abs:.4f}  -> {stab}")

    # Save.
    if args.out is None:
        out_path = project_root() / "EDMDc" / "model" / f"edmdc_{ts}.npz"
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        A=A, B=B,
        lam=np.float64(args.lam),
        dict_dim=np.int32(DICT_DIM),
        nu_dim=np.int32(NU_DIM),
        feature_names=np.array(feature_names()),
        input_dim=np.int32(tau.shape[1]),
        input_names=(
            d["input_names"] if "input_names" in d.files
            else np.array(CONTROL_NAMES[:tau.shape[1]])
        ),
        input_units=input_units,
        source_npz=str(args.npz),
        input_kind=str(getattr(args, "wrench", "commanded")),
        only_complete=bool(args.only_complete),
        rmse_lifted=np.float64(lifted_rmse),
        rmse_nu_per_axis=rmse_axis.astype(np.float64),
        eig_max_abs=np.float64(max_abs),
        eigs=eigs.astype(np.complex128),
    )
    sz_kb = out_path.stat().st_size / 1e3
    print(f"[train] wrote {out_path}  ({sz_kb:.1f} KB)")
    return 0


# ============================================================================
# Rollout (library helper — used by EDMDc.evaluate_edmdc and any MPC driver)
# ============================================================================

def rollout(A: np.ndarray, B: np.ndarray, nu0: np.ndarray, tau_seq: np.ndarray) -> np.ndarray:
    """Multi-step lifted rollout from nu0 with control sequence tau_seq.

    Returns nu_hat: (H + 1, 6) where nu_hat[0] = nu0 and nu_hat[k] is the
    k-step prediction.
    """
    H = tau_seq.shape[0]
    d = A.shape[0]
    Psi = np.empty((H + 1, d), dtype=np.float64)
    Psi[0] = lift(nu0)
    for k in range(H):
        Psi[k + 1] = A @ Psi[k] + B @ tau_seq[k]
    return decode_nu(Psi)


# ============================================================================
# CLI (train only — evaluation lives in EDMDc/evaluate_edmdc.py)
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("npz", type=Path,
                    help=".npz file from collect_isaac or collect_fossen.")
    ap.add_argument("--lam", type=float, default=1e-3,
                    help="Tikhonov regularization (default 1e-3). Use 0 for pure LS.")
    ap.add_argument("--wrench", choices=("commanded", "realized"),
                    default="commanded",
                    help="Which control-input column to train on. 'commanded' "
                         "(default) = the 4-axis ArduSub command U. "
                         "'realized' = the realized 6-DOF body wrench U_realized "
                         "(includes the allocator's parasitic Tx/Ty); yields a "
                         "true-vehicle model. Requires data collected with the "
                         "U_realized column.")
    ap.add_argument("--only-complete", action="store_true",
                    help="Restrict training to trajectories that finished APRBS.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output model .npz path. Default: EDMDc/model/edmdc_<TS>.npz.")
    args = ap.parse_args()
    return _train(args)


if __name__ == "__main__":
    raise SystemExit(main())
