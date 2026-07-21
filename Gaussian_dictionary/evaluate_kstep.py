"""Sliding-window K-step prediction RMSE on held-out trajectories.

Primary comparison metric for the Gaussian vs 34-D dictionary study (the
20 Hz analog of the old ``validate_heldout_sweep`` convention).  For every
valid start index in every contiguous trajectory segment, roll the model
forward K steps in lifted space (no re-lifting, matching each model's
canonical ``rollout``) and score the K-step-ahead error on ``[u, v, w, r]``.

Unlike ``Gaussian_dictionary.evaluate`` (one from-start rollout per
trajectory), every sample of the held-out file anchors a window, so the
estimate covers the whole state distribution rather than the initial
transient.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from .gaussian_edmdc import (
    STATE_INDICES,
    STATE_NAMES,
    GaussianEDMDcModel,
    load_dataset_arrays,
    project_root,
)

DIVERGE_LIMIT = 1e6


def contiguous_segments(traj_idx: np.ndarray, step_idx: np.ndarray) -> list[np.ndarray]:
    """Row-index arrays over which step_idx increments by exactly 1."""
    segments: list[np.ndarray] = []
    for trajectory in np.unique(traj_idx):
        rows = np.flatnonzero(traj_idx == trajectory)
        rows = rows[np.argsort(step_idx[rows], kind="stable")]
        if rows.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(step_idx[rows]) != 1)
        segments.extend(np.split(rows, breaks + 1))
    return [s for s in segments if s.size > 0]


def kstep_errors(
    lift_fn,
    decode_sel,
    A: np.ndarray,
    B: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    truth4: np.ndarray,
    segments: list[np.ndarray],
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, int]:
    """Return (n_starts, 4) K-step errors and the diverged-rollout count."""
    starts = np.concatenate([
        seg[: seg.size - horizon + 1: stride]
        for seg in segments if seg.size >= horizon
    ]) if any(seg.size >= horizon for seg in segments) else np.empty(0, dtype=np.int64)
    if starts.size == 0:
        return np.empty((0, len(STATE_NAMES))), 0

    Z = lift_fn(states[starts])
    for j in range(horizon):
        Z = Z @ A.T + controls[starts + j] @ B.T
    prediction = decode_sel(Z)
    finite = np.all(np.isfinite(prediction), axis=1) & (
        np.max(np.abs(prediction), axis=1) < DIVERGE_LIMIT
    )
    errors = prediction[finite] - truth4[starts + horizon - 1][finite]
    return errors, int(starts.size - int(finite.sum()))


def _rows(model, horizon, dt, errors, n_diverged, axis_scale):
    out = []
    if errors.shape[0]:
        axis_rmse = np.sqrt(np.mean(errors ** 2, axis=0))
        norm_err = errors / axis_scale[None, :]
        total = float(np.sqrt(np.mean(errors ** 2)))
        total_norm = float(np.sqrt(np.mean(norm_err ** 2)))
    else:
        axis_rmse = np.full(len(STATE_NAMES), np.nan)
        total = total_norm = float("nan")
    for axis, value, scale in zip(STATE_NAMES, axis_rmse, axis_scale):
        out.append({
            "model": model, "metric": "kstep", "horizon_steps": horizon,
            "horizon_seconds": horizon * dt, "axis": axis,
            "rmse": float(value), "rmse_norm": float(value / scale),
            "n_samples": int(errors.shape[0]), "n_diverged": n_diverged,
        })
    out.append({
        "model": model, "metric": "kstep", "horizon_steps": horizon,
        "horizon_seconds": horizon * dt, "axis": "total",
        "rmse": total, "rmse_norm": total_norm,
        "n_samples": int(errors.shape[0]), "n_diverged": n_diverged,
    })
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Gaussian model .npz")
    parser.add_argument("data", type=Path, help="Held-out collector dataset .npz")
    parser.add_argument("--compare-34d", type=Path, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20, 40, 100])
    parser.add_argument("--stride", type=int, default=1, help="Start-index stride (default 1 = every sample)")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model = GaussianEDMDcModel.load(args.model)

    with np.load(args.data, allow_pickle=False) as data:
        state4, _, control = load_dataset_arrays(data, model.input_kind)
        full_state = np.asarray(data["X"], dtype=np.float64)
        truth4 = np.asarray(data["X_next"], dtype=np.float64)[:, list(STATE_INDICES)]
        traj_idx = np.asarray(data["traj_idx"], dtype=np.int64)
        step_idx = np.asarray(data["step_idx"], dtype=np.int64)
        dt = float(np.asarray(data["dt"]).item()) if "dt" in data.files else model.dt

    segments = contiguous_segments(traj_idx, step_idx)
    axis_scale = np.std(state4, axis=0)
    axis_scale = np.where(axis_scale > 1e-12, axis_scale, 1.0)
    print(f"[kstep] {state4.shape[0]:,} pairs, {len(segments)} contiguous segments, "
          f"dt={dt:.4f}s, axis std={np.round(axis_scale, 4)}")

    candidates = [(
        "gaussian",
        model.lift,
        lambda Z: Z[:, :len(STATE_NAMES)],
        model.A, model.B, state4,
    )]
    if args.compare_34d is not None:
        from EDMDc.edmdc import lift as lift34

        with np.load(args.compare_34d, allow_pickle=True) as m34:
            A34 = np.asarray(m34["A"], dtype=np.float64)
            B34 = np.asarray(m34["B"], dtype=np.float64)
        if A34.shape != (34, 34) or B34.shape[1] != control.shape[1]:
            raise ValueError(f"unexpected 34-D model shapes A{A34.shape} B{B34.shape}")
        candidates.append((
            "edmdc_34d",
            lift34,
            lambda Z: Z[:, [1 + i for i in STATE_INDICES]],
            A34, B34, full_state,
        ))

    rows = []
    for horizon in sorted(args.horizons):
        for name, lift_fn, decode_sel, A, B, states in candidates:
            errors, n_div = kstep_errors(
                lift_fn, decode_sel, A, B, states, control, truth4,
                segments, horizon, args.stride,
            )
            rows.extend(_rows(name, horizon, dt, errors, n_div, axis_scale))
        # Hold-last baseline: predict x[s + K] = x[s].
        starts = np.concatenate([
            seg[: seg.size - horizon + 1: args.stride]
            for seg in segments if seg.size >= horizon
        ]) if any(seg.size >= horizon for seg in segments) else np.empty(0, dtype=np.int64)
        hold_err = (state4[starts] - truth4[starts + horizon - 1]) if starts.size else np.empty((0, 4))
        rows.extend(_rows("hold_last", horizon, dt, hold_err, 0, axis_scale))

    print("\nK-step RMSE on [u, v, w, r] (total | normalized)")
    print("model        K (s)         RMSE     nRMSE  n_starts  diverged")
    print("-----------  -----------  --------  ------  --------  --------")
    for row in rows:
        if row["axis"] != "total":
            continue
        print(f"{str(row['model']):11s}  {int(row['horizon_steps']):3d} "
              f"({float(row['horizon_seconds']):5.2f}s)  {float(row['rmse']):8.5f}  "
              f"{float(row['rmse_norm']):6.3f}  {int(row['n_samples']):8d}  "
              f"{int(row['n_diverged']):8d}")

    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = project_root() / "Gaussian_dictionary" / "results" / f"kstep_{stamp}.csv"
    else:
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "metric", "horizon_steps", "horizon_seconds",
                  "axis", "rmse", "rmse_norm", "n_samples", "n_diverged"]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[kstep] saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
