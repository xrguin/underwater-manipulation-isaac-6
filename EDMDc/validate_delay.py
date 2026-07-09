"""K-step held-out RMSE comparison: HODMDc+ARX delay models vs EDMDc baselines.

Standard system-ID metric (same as ``EDMDc.validate_heldout_sweep``): for
every valid starting index inside a held-out APRBS trajectory, roll out K
steps from the recorded inputs and compare the K-step prediction to ground
truth. Aggregates RMSE per nu axis over all (trajectory, start) pairs.

Handles two model kinds transparently:
  * plain EDMDc (``EDMDc.edmdc`` / ``edmdc_38``): keys ``A``, ``B``.
  * HODMDc+ARX (``EDMDc.edmdc_delay``): kind == "hodmdc_arx".

For a delay model with depths (ds, di), the first ``max(ds, di)`` samples of
each trajectory are skipped (no history available) -- so all models are
compared on the SAME set of (trajectory, start) pairs, namely those valid for
the deepest delay among the models. This keeps the comparison apples-to-apples.

Usage:
    python -m EDMDc.validate_delay --K 30 \\
        EDMDc/data/min_data_sweep/heldout_aprbs.npz \\
        EDMDc/model/edmdc_gripper_34_<TS>.npz \\
        EDMDc/model/edmdc_delay_<TS>.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .edmdc import NU_NAMES, rollout
from .edmdc_delay import MODEL_KIND, load_delay_model, rollout_delay


def _model_depths(model_path: Path) -> tuple[int, int]:
    """(ds, di) for a model; (0, 0) for a plain EDMDc model."""
    m = np.load(model_path, allow_pickle=True)
    if "kind" in m.files and str(m["kind"]) == MODEL_KIND:
        return int(m["state_delay"]), int(m["input_delay"])
    return 0, 0


def _ordered(traj_idx, step_idx, tr):
    idxs = np.where(traj_idx == tr)[0]
    if step_idx is not None:
        idxs = idxs[np.argsort(step_idx[idxs])]
    return idxs


def kstep_rmse(model_path: Path, X, U, traj_idx, step_idx, K: int,
               warmup: int) -> np.ndarray | None:
    """K-step prediction RMSE per nu axis. ``warmup`` start offset is shared
    across all models so the start set is identical."""
    if not model_path.exists():
        return None
    m = np.load(model_path, allow_pickle=True)
    is_delay = "kind" in m.files and str(m["kind"]) == MODEL_KIND
    if is_delay:
        A_list, B_list, ds, di = load_delay_model(model_path)
    else:
        A, B = m["A"], m["B"]
        ds = di = 0

    err_blocks: list[np.ndarray] = []
    for tr in np.unique(traj_idx):
        idxs = _ordered(traj_idx, step_idx, tr)
        L = len(idxs)
        for p in range(warmup, L - K):
            g = idxs[p]
            tau = U[idxs[p:p + K]]
            truth = X[idxs[p + K]]
            if is_delay:
                nu_hist = X[idxs[p - np.arange(ds + 1)]]          # (ds+1, 6)
                if di > 0:
                    u_past = U[idxs[p - 1 - np.arange(di)]]        # (di, 6)
                else:
                    u_past = np.zeros((0, U.shape[1]))
                pred = rollout_delay(A_list, B_list, nu_hist, u_past, tau)
            else:
                pred = rollout(A, B, X[g], tau)
            err_blocks.append((pred[K] - truth).reshape(1, -1))

    if not err_blocks:
        return None
    errors = np.vstack(err_blocks)
    finite = np.isfinite(errors).all(axis=1)
    if not finite.any():
        return np.full(len(NU_NAMES), np.inf)
    return np.sqrt(np.mean(errors[finite] ** 2, axis=0))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("heldout", type=Path, help="Held-out APRBS dataset .npz.")
    ap.add_argument("models", type=Path, nargs="+",
                    help="One or more model .npz (plain EDMDc or edmdc_delay).")
    ap.add_argument("--K", type=int, default=30,
                    help="Prediction horizon in steps (default 30 = 0.5 s @ 60 Hz).")
    args = ap.parse_args()

    d = np.load(args.heldout)
    X = d["X"]; U = d["U"]
    traj_idx = d["traj_idx"]
    step_idx = d["step_idx"] if "step_idx" in d.files else None
    n_traj = len(np.unique(traj_idx))
    print(f"[validate] {X.shape[0]:,} snapshots / {n_traj} trajectories, K={args.K}")

    # Shared warmup = deepest delay among models, so start sets match exactly.
    warmup = max(max(_model_depths(mp)) for mp in args.models)
    print(f"[validate] shared warmup offset = {warmup} (deepest model delay)\n")

    rows = []
    for mp in args.models:
        ds, di = _model_depths(mp)
        rmse = kstep_rmse(mp, X, U, traj_idx, step_idx, args.K, warmup)
        rows.append((mp.name, ds, di, rmse))

    hdr = f"{'model':<42} {'ds':>2} {'di':>2} " + " ".join(f"{n:>9}" for n in NU_NAMES) + f" {'mean':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, ds, di, rmse in rows:
        if rmse is None:
            print(f"{name:<42} {ds:>2} {di:>2}  (failed to load / no windows)")
            continue
        cells = " ".join(f"{v:9.4e}" for v in rmse)
        print(f"{name[:42]:<42} {ds:>2} {di:>2} {cells} {np.mean(rmse):9.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
