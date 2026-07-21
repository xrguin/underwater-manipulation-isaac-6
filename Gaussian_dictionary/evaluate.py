"""Evaluate the Gaussian model and optionally compare it with the 34-D EDMDc.

The comparison is fair on the four modeled axes ``[u, v, w, r]`` and uses the
same held-out trajectories, initial states, inputs, and rollout horizons.
Results are written to CSV for later plotting or reporting.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .gaussian_edmdc import (
    STATE_INDICES,
    STATE_NAMES,
    GaussianEDMDcModel,
    load_dataset_arrays,
    project_root,
    select_controlled_state,
)


PredictOne = Callable[[np.ndarray, np.ndarray], np.ndarray]
Rollout = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _metric_rows(
    model_name: str,
    metric: str,
    error: np.ndarray,
    *,
    horizon: int,
    dt: float,
) -> list[dict[str, object]]:
    err = np.asarray(error, dtype=np.float64)
    if err.ndim == 1:
        err = err[None, :]
    rows: list[dict[str, object]] = []
    axis_rmse = np.sqrt(np.mean(err ** 2, axis=0))
    for axis, value in zip(STATE_NAMES, axis_rmse):
        rows.append({
            "model": model_name,
            "metric": metric,
            "horizon_steps": horizon,
            "horizon_seconds": horizon * dt,
            "axis": axis,
            "rmse": float(value),
            "n_samples": int(err.shape[0]),
        })
    rows.append({
        "model": model_name,
        "metric": metric,
        "horizon_steps": horizon,
        "horizon_seconds": horizon * dt,
        "axis": "total",
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "n_samples": int(err.shape[0]),
    })
    return rows


def _evaluate_model(
    model_name: str,
    predict_one: PredictOne,
    rollout: Rollout,
    state: np.ndarray,
    state_next: np.ndarray,
    control: np.ndarray,
    full_state: np.ndarray,
    full_state_next: np.ndarray,
    traj_idx: np.ndarray,
    horizons: list[int],
    dt: float,
    *,
    uses_full_state: bool,
) -> list[dict[str, object]]:
    one_input = full_state if uses_full_state else state
    one_truth = select_controlled_state(full_state_next) if uses_full_state else state_next
    one_prediction = predict_one(one_input, control)
    rows = _metric_rows(
        model_name, "one_step", one_prediction - one_truth, horizon=1, dt=dt,
    )

    errors: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    for trajectory in np.unique(traj_idx):
        selected = traj_idx == trajectory
        x = full_state[selected] if uses_full_state else state[selected]
        x_next = full_state_next[selected] if uses_full_state else state_next[selected]
        u = control[selected]
        if x.shape[0] == 0:
            continue
        prediction = rollout(x[0], u)
        truth = np.vstack([select_controlled_state(x[0]), select_controlled_state(x_next)])
        for h in horizons:
            if h < prediction.shape[0]:
                errors[h].append(select_controlled_state(prediction[h]) - truth[h])

    for h in horizons:
        if errors[h]:
            rows.extend(_metric_rows(
                model_name, "rollout", np.asarray(errors[h]), horizon=h, dt=dt,
            ))
    return rows


def _hold_predict(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    del control
    return select_controlled_state(state)


def _hold_rollout(state0: np.ndarray, controls: np.ndarray) -> np.ndarray:
    x0 = select_controlled_state(state0)
    return np.tile(x0, (controls.shape[0] + 1, 1))


def _load_34d_predictor(
    path: Path,
) -> tuple[PredictOne, Rollout, int]:
    from EDMDc.edmdc import decode_nu, lift, rollout as rollout_34d

    with np.load(path, allow_pickle=True) as model:
        A = np.asarray(model["A"], dtype=np.float64)
        B = np.asarray(model["B"], dtype=np.float64)
    if A.shape != (34, 34):
        raise ValueError(f"comparison model must have A shape (34, 34); got {A.shape}")

    def predict(state: np.ndarray, control: np.ndarray) -> np.ndarray:
        z_next = lift(state) @ A.T + control @ B.T
        return select_controlled_state(decode_nu(z_next))

    def roll(state0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        return select_controlled_state(rollout_34d(A, B, state0, controls))

    return predict, roll, int(B.shape[1])


def _input_for_34d(data: np.lib.npyio.NpzFile, gaussian_control: np.ndarray, input_dim: int) -> np.ndarray:
    if input_dim == gaussian_control.shape[1]:
        return gaussian_control
    if input_dim == 6 and "U_realized" in data.files:
        realized = np.asarray(data["U_realized"], dtype=np.float64)
        if realized.ndim == 2 and realized.shape[1] == 6:
            return realized
    raise ValueError(
        f"34-D model expects {input_dim} inputs, but the selected Gaussian input has "
        f"{gaussian_control.shape[1]} and no compatible dataset array is available"
    )


def _print_summary(rows: list[dict[str, object]]) -> None:
    print("\nmodel comparison (total RMSE on [u, v, w, r])")
    print("model                    metric       horizon      RMSE")
    print("-----------------------  -----------  -------  ---------")
    for row in rows:
        if row["axis"] != "total":
            continue
        print(
            f"{str(row['model']):23s}  {str(row['metric']):11s}  "
            f"{int(row['horizon_steps']):7d}  {float(row['rmse']):9.5f}"
        )


def _write_csv(rows: list[dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "metric", "horizon_steps", "horizon_seconds",
        "axis", "rmse", "n_samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Gaussian model .npz")
    parser.add_argument("data", type=Path, help="Held-out EDMDc collector dataset")
    parser.add_argument("--compare-34d", type=Path, default=None, help="Optional current 34-D EDMDc model")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 25, 50, 100])
    parser.add_argument("--max-trajs", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Comparison CSV output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model = GaussianEDMDcModel.load(args.model)
    with np.load(args.data, allow_pickle=False) as data:
        state, state_next, control = load_dataset_arrays(data, model.input_kind)
        full_state = np.asarray(data["X"], dtype=np.float64)
        full_state_next = np.asarray(data["X_next"], dtype=np.float64)
        traj_idx = np.asarray(data["traj_idx"], dtype=np.int64) if "traj_idx" in data.files else np.zeros(state.shape[0], dtype=np.int64)
        dt = float(np.asarray(data["dt"]).item()) if "dt" in data.files else model.dt

        if args.max_trajs is not None:
            keep_ids = np.unique(traj_idx)[: args.max_trajs]
            mask = np.isin(traj_idx, keep_ids)
            state, state_next, control = state[mask], state_next[mask], control[mask]
            full_state, full_state_next, traj_idx = full_state[mask], full_state_next[mask], traj_idx[mask]

        rows = _evaluate_model(
            "gaussian", model.predict, model.rollout,
            state, state_next, control, full_state, full_state_next, traj_idx,
            args.horizons, dt, uses_full_state=False,
        )
        rows.extend(_evaluate_model(
            "hold_last", _hold_predict, _hold_rollout,
            state, state_next, control, full_state, full_state_next, traj_idx,
            args.horizons, dt, uses_full_state=False,
        ))
        if args.compare_34d is not None:
            predict_34, rollout_34, input_dim = _load_34d_predictor(args.compare_34d)
            control_34 = _input_for_34d(data, control, input_dim)
            if args.max_trajs is not None:
                control_34 = control_34[mask]
            rows.extend(_evaluate_model(
                "edmdc_34d", predict_34, rollout_34,
                state, state_next, control_34, full_state, full_state_next, traj_idx,
                args.horizons, dt, uses_full_state=True,
            ))

    _print_summary(rows)
    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = project_root() / "Gaussian_dictionary" / "results" / f"comparison_{stamp}.csv"
    else:
        out = args.out
    _write_csv(rows, out)
    print(f"\n[evaluate] saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
