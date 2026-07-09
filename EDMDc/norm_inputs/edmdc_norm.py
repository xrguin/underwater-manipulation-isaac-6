"""EDMDc trained on NORMALIZED [-1,1] control inputs (norm_inputs experiment).

Thin entry point that reuses the exact 34-feature dictionary and Tikhonov LS
solve from ``EDMDc.edmdc`` (no duplicated regression math) but trains on the
normalized command ``U`` produced by ``EDMDc.norm_inputs.collect_norm``.

Because the normalized-to-Newton map is a single per-axis scale
(``u_newton = symmetric_caps * u_norm``) and EDMDc is linear in the input, the
normalized model is *exactly* the Newton model with its input columns rescaled:

    z_+ = A z + B_norm u_norm ,   B_norm[:, i] = symmetric_caps[i] * B_newton[:, i]

So EDMDc is unchanged in structure; only ``B`` is rescaled. This script reports
both ``B_norm`` (as fitted) and the equivalent ``B_newton`` so the result can be
compared directly against a Newton-trained model and used for control.

Usage:
    python -m EDMDc.norm_inputs.edmdc_norm \\
        EDMDc/norm_inputs/data/norm_gripper_<TS>.npz
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from EDMDc._common import project_root
from EDMDc.edmdc import (
    DICT_DIM, NU_DIM, NU_NAMES, ACTIVE_DOF,
    decode_nu, feature_names, fit_edmdc, lift,
)


def _train(args: argparse.Namespace) -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    d = np.load(args.npz, allow_pickle=True)
    nu = d["X"]
    nu_next = d["X_next"]
    wrench_kind = getattr(args, "wrench", "commanded")
    tau = d["U_realized"] if wrench_kind == "realized" else d["U"]

    units = str(d["input_units"]) if "input_units" in d.files else "unknown"
    caps = (d["symmetric_caps"] if "symmetric_caps" in d.files
            else np.ones(tau.shape[1]))
    print(f"[train] loaded {nu.shape[0]:,} snapshots from {Path(args.npz).name}")
    print(f"[train] input = {wrench_kind} U, units = {units}")
    if units not in ("per_motor_thrust", "normalized_[-1,1]"):
        print("[train] WARNING: dataset has unexpected input_units; this "
              "trainer assumes normalized/per-motor U (see collect_norm.py).")
    print(f"[train] |U| range per axis: min={tau.min(0)}, max={tau.max(0)}")

    print(f"[train] fitting EDMDc: dict_dim={DICT_DIM}, lam={args.lam}")
    A, B = fit_edmdc(nu, nu_next, tau, lam=args.lam)
    print(f"[train] A shape: {A.shape}, B shape: {B.shape}")

    # Equivalent Newton-input B (column-wise rescale by the per-axis scale).
    B_newton = B.copy()
    nz = np.asarray(caps, dtype=float) != 0.0
    B_newton[:, nz] = B[:, nz] / np.asarray(caps, dtype=float)[nz]

    # One-step training residual.
    Z = lift(nu).T
    Z_plus = lift(nu_next).T
    Z_pred = A @ Z + B @ tau.T
    lifted_rmse = float(np.sqrt(np.mean((Z_plus - Z_pred) ** 2)))
    nu_pred = decode_nu(Z_pred.T)
    rmse_axis = np.sqrt(np.mean((nu_next - nu_pred) ** 2, axis=0))
    print(f"[train] one-step lifted RMSE: {lifted_rmse:.4e}")
    print("[train] one-step nu RMSE per axis (training data):")
    for n, r in zip(NU_NAMES, rmse_axis):
        print(f"          {n}: {r:.4e}")

    eigs = np.linalg.eigvals(A)
    max_abs = float(np.max(np.abs(eigs)))
    stab = "STABLE (rho(A) <= 1)" if max_abs <= 1.0 + 1e-9 else "UNSTABLE (rho(A) > 1)"
    print(f"[train] eig(A): max |lambda| = {max_abs:.4f}  -> {stab}")
    print(f"[train] active-DOF B_norm column norms (Fx,Fy,Fz,Tz): "
          f"{np.round(np.linalg.norm(B[:, list(ACTIVE_DOF)], axis=0), 4)}")

    out_path = (Path(args.out) if args.out
                else project_root() / "EDMDc" / "norm_inputs" / "model"
                / f"edmdc_norm_{ts}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        A=A, B=B, B_newton=B_newton,
        symmetric_caps=np.asarray(caps, dtype=float),
        input_units="normalized_[-1,1]",
        lam=np.float64(args.lam),
        dict_dim=np.int32(DICT_DIM),
        nu_dim=np.int32(NU_DIM),
        feature_names=np.array(feature_names()),
        source_npz=str(args.npz),
        input_kind=str(wrench_kind),
        rmse_lifted=np.float64(lifted_rmse),
        rmse_nu_per_axis=rmse_axis.astype(np.float64),
        eig_max_abs=np.float64(max_abs),
        eigs=eigs.astype(np.complex128),
    )
    print(f"[train] wrote {out_path}  ({out_path.stat().st_size/1e3:.1f} KB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, help="normalized dataset from collect_norm.")
    ap.add_argument("--lam", type=float, default=1e-3,
                    help="Tikhonov regularization (default 1e-3).")
    ap.add_argument("--wrench", choices=("commanded", "realized"),
                    default="commanded",
                    help="Train on commanded U (default) or realized U_realized.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output model path. Default "
                         "EDMDc/norm_inputs/model/edmdc_norm_<TS>.npz.")
    return _train(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
