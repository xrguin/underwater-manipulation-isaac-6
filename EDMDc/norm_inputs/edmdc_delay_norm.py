"""HODMDc + ARX trained on NORMALIZED [-1,1] control inputs (norm_inputs).

Thin entry point reusing the window builder and Tikhonov LS solve from
``EDMDc.edmdc_delay`` (no duplicated math) but training on the normalized
command ``U`` from ``EDMDc.norm_inputs.collect_norm``.

As with the plain EDMDc case, the normalized-to-Newton map is a single per-axis
scale, and the ARX model is linear in the delayed inputs, so each fitted
``B_j`` block is the Newton ``B_j`` with its columns rescaled by
``symmetric_caps``. The ARX construction is therefore unchanged in structure;
only the input columns are rescaled. The equivalent Newton blocks are saved
alongside the normalized ones.

Usage:
    python -m EDMDc.norm_inputs.edmdc_delay_norm \\
        EDMDc/norm_inputs/data/norm_gripper_<TS>.npz \\
        --state-delay 3 --input-delay 1
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from EDMDc._common import project_root
from EDMDc.edmdc import DICT_DIM, NU_DIM, NU_NAMES, decode_nu, feature_names, lift
from EDMDc.edmdc_delay import MODEL_KIND, build_windows, fit_delay


def _train(args: argparse.Namespace) -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    d = np.load(args.npz, allow_pickle=True)
    X = d["X"]; U = d["U"]; X_next = d["X_next"]
    traj_idx = d["traj_idx"]
    step_idx = d["step_idx"] if "step_idx" in d.files else None
    units = str(d["input_units"]) if "input_units" in d.files else "unknown"
    caps = (d["symmetric_caps"] if "symmetric_caps" in d.files
            else np.ones(U.shape[1]))
    print(f"[train] loaded {X.shape[0]:,} snapshots from {Path(args.npz).name}")
    print(f"[train] input units = {units}")
    if units not in ("per_motor_thrust", "normalized_[-1,1]"):
        print("[train] WARNING: dataset has unexpected input_units.")
    print(f"[train] HODMDc+ARX: state_delay={args.state_delay}, "
          f"input_delay={args.input_delay}, lam={args.lam}")

    A_list, B_list, M = fit_delay(
        X, U, X_next, traj_idx, step_idx,
        ds=args.state_delay, di=args.input_delay, lam=args.lam)
    print(f"[train] fitted on {M:,} windows; "
          f"{len(A_list)} A-blocks {A_list[0].shape}, "
          f"{len(B_list)} B-blocks {B_list[0].shape}")

    # Equivalent Newton B blocks (column-wise rescale by per-axis scale).
    nz = np.asarray(caps, dtype=float) != 0.0
    B_newton_list = []
    for Bj in B_list:
        Bn = Bj.copy()
        Bn[:, nz] = Bj[:, nz] / np.asarray(caps, dtype=float)[nz]
        B_newton_list.append(Bn)

    # One-step training residual (history-aware).
    cur, state_lag, input_lag = build_windows(
        X, U, X_next, traj_idx, step_idx, args.state_delay, args.input_delay)
    Z_plus = lift(X_next[cur]).T
    Z_pred = np.zeros_like(Z_plus)
    for i, Ai in enumerate(A_list):
        Z_pred += Ai @ lift(X[state_lag[i]]).T
    for j, Bj in enumerate(B_list):
        Z_pred += Bj @ U[input_lag[j]].T
    nu_pred = decode_nu(Z_pred.T)
    rmse_axis = np.sqrt(np.mean((X_next[cur] - nu_pred) ** 2, axis=0))
    lifted_rmse = float(np.sqrt(np.mean((Z_plus - Z_pred) ** 2)))
    print(f"[train] one-step lifted RMSE: {lifted_rmse:.4e}")
    print("[train] one-step nu RMSE per axis (training data):")
    for n, r in zip(NU_NAMES, rmse_axis):
        print(f"          {n}: {r:.4e}")

    out_path = (Path(args.out) if args.out
                else project_root() / "EDMDc" / "norm_inputs" / "model"
                / f"edmdc_delay_norm_{ts}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kw = dict(
        kind=MODEL_KIND,
        state_delay=np.int32(args.state_delay),
        input_delay=np.int32(args.input_delay),
        lam=np.float64(args.lam),
        dict_dim=np.int32(DICT_DIM),
        nu_dim=np.int32(NU_DIM),
        feature_names=np.array(feature_names()),
        source_npz=str(args.npz),
        input_units="normalized_[-1,1]",
        symmetric_caps=np.asarray(caps, dtype=float),
        rmse_lifted=np.float64(lifted_rmse),
        rmse_nu_per_axis=rmse_axis.astype(np.float64),
    )
    for i, Ai in enumerate(A_list):
        save_kw[f"A{i}"] = Ai
    for j, Bj in enumerate(B_list):
        save_kw[f"B{j}"] = Bj
        save_kw[f"B{j}_newton"] = B_newton_list[j]
    np.savez_compressed(out_path, **save_kw)
    print(f"[train] wrote {out_path}  ({out_path.stat().st_size/1e3:.1f} KB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, help="normalized dataset from collect_norm.")
    ap.add_argument("--state-delay", type=int, default=3)
    ap.add_argument("--input-delay", type=int, default=1)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default=None)
    return _train(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
