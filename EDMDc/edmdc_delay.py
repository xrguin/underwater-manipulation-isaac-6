"""HODMDc + ARX: history-embedded EDMDc (delayed lifted states + delayed inputs).

Experimental sibling of ``EDMDc.edmdc``. Reuses the exact 34-feature monomial
dictionary (``edmdc.lift``) but learns a *higher-order* Koopman-with-control
model that mixes delayed lifted states and delayed inputs:

    psi(k+1) ~= sum_{i=0..ds} A_i psi(nu(k-i))  +  sum_{j=0..di} B_j u(k-j)

with ``ds`` = state-delay depth and ``di`` = input-delay depth. ds=di=0
recovers ordinary EDMDc (``EDMDc.edmdc``) exactly.

Motivation: the LS-multicollinearity failure (see
``debug_journal_gripper.md`` Bug #3 / memory ``edmdc-ls-multicollinearity``)
cannot be fixed by adding snapshots in the same velocity window. A delayed
model gives the regression access to acceleration information
(nu(k) - nu(k-1)), so it can distinguish "low velocity, accelerating from
rest" from "low velocity, steady-state with drag balancing input" -- two
regimes with identical instantaneous nu but different damping behaviour.

This module is *prediction-accuracy only*: it does not build the companion
form needed for LQR/MPC. Use ``EDMDc.validate_delay`` to compare K-step
held-out RMSE against the 34-D / 38-D baselines.

Loss (Tikhonov, identical structure to edmdc.fit_edmdc):

    min ||Z_+ - [A_0..A_ds B_0..B_di] Phi||_F^2 + lam ||.||_F^2
    Phi = [psi(k); psi(k-1); ...; psi(k-ds); u(k); u(k-1); ...; u(k-di)]

Closed-form: W = Z_+ Phi^T (Phi Phi^T + lam I)^{-1}, then slice W into the
A_i and B_j blocks.

Usage:
    # Train an ARX(ds=3, di=1) model on a collection dataset
    python -m EDMDc.edmdc_delay EDMDc/data/with_gripper/isaac_gripper_<TS>.npz \\
        --state-delay 3 --input-delay 1

    # Compare against the 34-D baseline on held-out APRBS
    python -m EDMDc.validate_delay --K 30 \\
        EDMDc/model/edmdc_gripper_34_<TS>.npz \\
        EDMDc/model/edmdc_delay_<TS>.npz
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from ._common import project_root
from .edmdc import DICT_DIM, NU_DIM, NU_NAMES, decode_nu, feature_names, lift

MODEL_KIND = "hodmdc_arx"


# ============================================================================
# Window construction (respects trajectory boundaries)
# ============================================================================

def _ordered_traj_indices(traj_idx: np.ndarray, step_idx: np.ndarray | None,
                          tr) -> np.ndarray:
    """Global row indices for trajectory ``tr``, ordered by step_idx if given."""
    idxs = np.where(traj_idx == tr)[0]
    if step_idx is not None:
        idxs = idxs[np.argsort(step_idx[idxs])]
    return idxs


def build_windows(
    X: np.ndarray, U: np.ndarray, X_next: np.ndarray,
    traj_idx: np.ndarray, step_idx: np.ndarray | None,
    ds: int, di: int,
):
    """Collect global row indices for every valid (current, lag) tuple.

    A sample at local position ``p`` in a trajectory is valid iff it has
    ``max(ds, di)`` predecessors in the *same* trajectory. The target is
    ``X_next[g]`` (the recorded next state), so no contiguity assumption is
    needed for the target; history lags use the step-ordered indices.

    Returns:
        cur:       (M,)        global index of each current sample k
        state_lag: (ds+1, M)   state_lag[i] = global index of k-i
        input_lag: (di+1, M)   input_lag[j] = global index of k-j
    """
    d0 = max(ds, di)
    cur: list[int] = []
    state_lag: list[list[int]] = [[] for _ in range(ds + 1)]
    input_lag: list[list[int]] = [[] for _ in range(di + 1)]

    for tr in np.unique(traj_idx):
        idxs = _ordered_traj_indices(traj_idx, step_idx, tr)
        for p in range(d0, len(idxs)):
            cur.append(int(idxs[p]))
            for i in range(ds + 1):
                state_lag[i].append(int(idxs[p - i]))
            for j in range(di + 1):
                input_lag[j].append(int(idxs[p - j]))

    return (
        np.asarray(cur, dtype=np.int64),
        np.asarray(state_lag, dtype=np.int64),
        np.asarray(input_lag, dtype=np.int64),
    )


# ============================================================================
# Training
# ============================================================================

def fit_delay(
    X: np.ndarray, U: np.ndarray, X_next: np.ndarray,
    traj_idx: np.ndarray, step_idx: np.ndarray | None,
    ds: int, di: int, lam: float = 1e-3,
):
    """Tikhonov closed-form fit of the HODMDc+ARX model.

    Returns:
        A_list: list of (ds+1) arrays, each (DICT_DIM, DICT_DIM)
        B_list: list of (di+1) arrays, each (DICT_DIM, input_dim)
        n_samples: number of regression rows used
    """
    cur, state_lag, input_lag = build_windows(
        X, U, X_next, traj_idx, step_idx, ds, di)
    if cur.size == 0:
        raise ValueError(
            f"no valid windows for ds={ds}, di={di}; trajectories too short")

    d = DICT_DIM
    m = int(U.shape[1])
    M = cur.size

    Z_plus = lift(X_next[cur]).T                       # (d, M)

    # State-lag blocks: stack lift(X[k-i]) for i = 0..ds  -> (d*(ds+1), M)
    Z_blocks = [lift(X[state_lag[i]]).T for i in range(ds + 1)]
    # Input-lag blocks: stack U[k-j] for j = 0..di        -> (m*(di+1), M)
    U_blocks = [U[input_lag[j]].T for j in range(di + 1)]

    Phi = np.vstack(Z_blocks + U_blocks)               # (d*(ds+1)+m*(di+1), M)
    p = Phi.shape[0]
    G = Phi @ Phi.T + lam * np.eye(p)
    rhs = Z_plus @ Phi.T                               # (d, p)
    W = np.linalg.solve(G.T, rhs.T).T                  # = rhs @ G^{-1}, (d, p)

    A_list = [W[:, i * d:(i + 1) * d] for i in range(ds + 1)]
    off = d * (ds + 1)
    B_list = [W[:, off + j * m: off + (j + 1) * m] for j in range(di + 1)]
    return A_list, B_list, M


# ============================================================================
# Rollout (history-aware)
# ============================================================================

def rollout_delay(
    A_list, B_list,
    nu_hist: np.ndarray, u_past: np.ndarray, tau_seq: np.ndarray,
) -> np.ndarray:
    """Multi-step rollout of the HODMDc+ARX model.

    Args:
        A_list:  list of (ds+1) state matrices (DICT_DIM, DICT_DIM).
        B_list:  list of (di+1) input matrices (DICT_DIM, NU_DIM).
        nu_hist: (ds+1, 6) initial state history;
                 nu_hist[0] = nu(start), nu_hist[i] = nu(start-i).
        u_past:  (di, m) past inputs before start;
                 u_past[j] = u(start-1-j).  Pass shape (0, m) if di == 0.
        tau_seq: (K, m) forward inputs u(start), u(start+1), ..., u(start+K-1).

    Returns:
        nu_hat: (K+1, 6); nu_hat[0] = nu(start), nu_hat[t] = t-step prediction.
    """
    ds = len(A_list) - 1
    di = len(B_list) - 1
    K = tau_seq.shape[0]

    # psi_hist[i] = psi(start - i), most-recent first.
    psi_hist = [lift(nu_hist[i]) for i in range(ds + 1)]

    def u_at(m: int) -> np.ndarray:
        """Input at absolute forward offset m from start (m may be negative)."""
        if m >= 0:
            return tau_seq[m]
        return u_past[-m - 1]

    nu_hat = np.empty((K + 1, NU_DIM), dtype=np.float64)
    nu_hat[0] = nu_hist[0]

    for t in range(K):
        psi_next = np.zeros(DICT_DIM, dtype=np.float64)
        for i in range(ds + 1):
            psi_next += A_list[i] @ psi_hist[i]
        for j in range(di + 1):
            psi_next += B_list[j] @ u_at(t - j)
        nu_hat[t + 1] = decode_nu(psi_next)
        psi_hist.insert(0, psi_next)
        psi_hist.pop()                                 # keep length ds+1

    return nu_hat


# ============================================================================
# CLI (train)
# ============================================================================

def _train(args: argparse.Namespace) -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    d = np.load(args.npz)
    X = d["X"]; U = d["U"]; X_next = d["X_next"]
    traj_idx = d["traj_idx"]
    step_idx = d["step_idx"] if "step_idx" in d.files else None
    input_units = str(d["input_units"]) if "input_units" in d.files else "unknown"
    print(f"[train] loaded {X.shape[0]:,} snapshots from {Path(args.npz).name}")
    print(f"[train] input units = {input_units}, input_dim={U.shape[1]}")
    print(f"[train] HODMDc+ARX: state_delay={args.state_delay}, "
          f"input_delay={args.input_delay}, lam={args.lam}")

    A_list, B_list, M = fit_delay(
        X, U, X_next, traj_idx, step_idx,
        ds=args.state_delay, di=args.input_delay, lam=args.lam)
    print(f"[train] fitted on {M:,} windows; "
          f"{len(A_list)} A-blocks {A_list[0].shape}, "
          f"{len(B_list)} B-blocks {B_list[0].shape}")

    # One-step training residual (history-aware, single step).
    cur, state_lag, input_lag = build_windows(
        X, U, X_next, traj_idx, step_idx, args.state_delay, args.input_delay)
    Z_plus = lift(X_next[cur]).T
    Z_pred = np.zeros_like(Z_plus)
    for i, Ai in enumerate(A_list):
        Z_pred += Ai @ lift(X[state_lag[i]]).T
    for j, Bj in enumerate(B_list):
        Z_pred += Bj @ U[input_lag[j]].T
    nu_pred = decode_nu(Z_pred.T)
    nu_err = X_next[cur] - nu_pred
    rmse_axis = np.sqrt(np.mean(nu_err ** 2, axis=0))
    lifted_rmse = float(np.sqrt(np.mean((Z_plus - Z_pred) ** 2)))
    print(f"[train] one-step lifted RMSE: {lifted_rmse:.4e}")
    print(f"[train] one-step nu RMSE per axis (training data):")
    for n, r in zip(NU_NAMES, rmse_axis):
        print(f"          {n}: {r:.4e}")

    # Spectral stability of the companion form A_hat (informational; not used
    # for control here). A_hat is block-companion of dim DICT_DIM*(ds+1).
    ds = args.state_delay
    dco = DICT_DIM * (ds + 1)
    A_hat = np.zeros((dco, dco), dtype=np.float64)
    A_hat[:DICT_DIM, :] = np.hstack(A_list)
    if ds >= 1:
        A_hat[DICT_DIM:, :DICT_DIM * ds] = np.eye(DICT_DIM * ds)
    eigs = np.linalg.eigvals(A_hat)
    max_abs = float(np.max(np.abs(eigs)))
    stab = "STABLE (rho<=1)" if max_abs <= 1.0 + 1e-9 else "UNSTABLE (rho>1)"
    print(f"[train] companion eig: max |lambda| = {max_abs:.4f}  -> {stab}")

    out_path = (Path(args.out) if args.out
                else project_root() / "EDMDc" / "model" / f"edmdc_delay_{ts}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_kw = dict(
        kind=MODEL_KIND,
        state_delay=np.int32(args.state_delay),
        input_delay=np.int32(args.input_delay),
        lam=np.float64(args.lam),
        dict_dim=np.int32(DICT_DIM),
        nu_dim=np.int32(NU_DIM),
        input_dim=np.int32(U.shape[1]),
        input_names=(
            d["input_names"] if "input_names" in d.files
            else np.array([f"u{i}" for i in range(U.shape[1])])
        ),
        input_units=input_units,
        feature_names=np.array(feature_names()),
        source_npz=str(args.npz),
        rmse_lifted=np.float64(lifted_rmse),
        rmse_nu_per_axis=rmse_axis.astype(np.float64),
        companion_eig_max_abs=np.float64(max_abs),
    )
    for i, Ai in enumerate(A_list):
        save_kw[f"A{i}"] = Ai
    for j, Bj in enumerate(B_list):
        save_kw[f"B{j}"] = Bj
    np.savez_compressed(out_path, **save_kw)
    sz_kb = out_path.stat().st_size / 1e3
    print(f"[train] wrote {out_path}  ({sz_kb:.1f} KB)")
    return 0


def load_delay_model(model_path: str | Path):
    """Load an edmdc_delay model. Returns (A_list, B_list, ds, di)."""
    m = np.load(model_path, allow_pickle=True)
    if "kind" not in m.files or str(m["kind"]) != MODEL_KIND:
        raise ValueError(f"{model_path} is not a {MODEL_KIND} model")
    ds = int(m["state_delay"]); di = int(m["input_delay"])
    A_list = [m[f"A{i}"] for i in range(ds + 1)]
    B_list = [m[f"B{j}"] for j in range(di + 1)]
    return A_list, B_list, ds, di


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("npz", type=Path,
                    help="Training dataset .npz (collect_isaac{,_38} output).")
    ap.add_argument("--state-delay", type=int, default=3,
                    help="State-delay depth ds (number of past nu lags). Default 3.")
    ap.add_argument("--input-delay", type=int, default=3,
                    help="Input-delay depth di (number of past u lags). Default 3.")
    ap.add_argument("--lam", type=float, default=1e-3,
                    help="Tikhonov regularization (default 1e-3).")
    ap.add_argument("--out", type=str, default=None,
                    help="Output model path. Default EDMDc/model/edmdc_delay_<TS>.npz.")
    args = ap.parse_args()
    return _train(args)


if __name__ == "__main__":
    raise SystemExit(main())
