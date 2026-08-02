"""Causal lifted-history ARX model for the four-state Gaussian dictionary.

The collector pair contract is ``X_next[k] == x[k+1]``.  Consequently, the
regressor whose newest block is ``g_k = [phi(x_k), u_k]`` must be trained
against ``X_next[k]``.  This module keeps that alignment explicit and provides
the archive layout consumed by the real-robot ARX controller.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gaussian_edmdc import (
    CONTROL_NAMES,
    STATE_INDICES,
    GaussianEDMDcModel,
    project_root,
    select_controlled_state,
)


MODEL_KIND = "arx_lifted_history"
BASE_KIND = "gaussian_edmdc_4state"


def contiguous_segments(
    traj_idx: np.ndarray, step_idx: np.ndarray,
) -> list[np.ndarray]:
    """Return ordered row arrays with unit-incrementing step indices."""
    segments: list[np.ndarray] = []
    for trajectory in np.unique(traj_idx):
        rows = np.flatnonzero(traj_idx == trajectory)
        rows = rows[np.argsort(step_idx[rows], kind="stable")]
        if rows.size:
            breaks = np.flatnonzero(np.diff(step_idx[rows]) != 1) + 1
            segments.extend(np.split(rows, breaks))
    return [segment for segment in segments if segment.size]


def history_rows(segments: list[np.ndarray], m: int) -> np.ndarray:
    """Rows with ``m-1`` causal predecessors inside the same segment."""
    usable = [segment[m - 1:] for segment in segments if segment.size >= m]
    return np.concatenate(usable) if usable else np.empty(0, dtype=np.int64)


@dataclass
class ARXHistoryModel:
    base: GaussianEDMDcModel
    m: int
    W: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    clip: np.ndarray
    dt: float
    ridge_alpha: float = 1.0

    def __post_init__(self) -> None:
        self.W = np.asarray(self.W, dtype=np.float64)
        self.mu = np.asarray(self.mu, dtype=np.float64)
        self.sd = np.asarray(self.sd, dtype=np.float64)
        self.clip = np.asarray(self.clip, dtype=np.float64)
        self.n_g = self.base.dimension + len(CONTROL_NAMES)
        if self.m < 1:
            raise ValueError("m must be positive")
        if self.W.shape != (4, self.m * self.n_g + 1):
            raise ValueError(f"unexpected W shape {self.W.shape}")
        if self.mu.shape != (self.n_g,) or self.sd.shape != (self.n_g,):
            raise ValueError("mu/sd shape does not match lifted feature block")
        if self.clip.shape != (4,) or np.any(self.sd <= 0.0):
            raise ValueError("clip must be (4,) and sd must be positive")

    def features(self, state4: np.ndarray, control4: np.ndarray) -> np.ndarray:
        state = select_controlled_state(state4)
        control = np.asarray(control4, dtype=np.float64)
        if state.ndim == 1:
            return np.concatenate([self.base.lift(state), control])
        return np.hstack([self.base.lift(state), control])

    def predict_history(self, newest_first: list[np.ndarray]) -> np.ndarray:
        """Predict x[k+1] from raw feature blocks g[k], ..., g[k-m+1]."""
        if len(newest_first) != self.m:
            raise ValueError(f"expected {self.m} history blocks")
        standardized = [
            (np.asarray(block, dtype=np.float64) - self.mu) / self.sd
            for block in newest_first
        ]
        first = standardized[0]
        if first.ndim == 1:
            regressor = np.concatenate([*standardized, [1.0]])
            prediction = self.W @ regressor
        else:
            regressor = np.hstack([
                *standardized, np.ones((first.shape[0], 1)),
            ])
            prediction = regressor @ self.W.T
        return np.clip(prediction, -self.clip, self.clip)

    def endpoint_rollout(
        self,
        state4: np.ndarray,
        controls: np.ndarray,
        prior_history: list[np.ndarray],
    ) -> np.ndarray:
        """Roll multiple origins to the endpoint of their control sequences."""
        x = select_controlled_state(state4).copy()
        history = [np.asarray(block, dtype=np.float64).copy()
                   for block in prior_history]
        if len(history) != self.m - 1:
            raise ValueError(f"expected {self.m - 1} prior history blocks")
        for j in range(controls.shape[1]):
            g_current = self.features(x, controls[:, j])
            x = self.predict_history([g_current, *history])
            if self.m > 1:
                history = [g_current, *history[:-1]]
        return x

    def rollout(
        self,
        state0: np.ndarray,
        controls: np.ndarray,
        prior_history: list[np.ndarray],
    ) -> np.ndarray:
        """Single-origin recursive rollout including the supplied initial state."""
        controls = np.asarray(controls, dtype=np.float64)
        x = select_controlled_state(state0).copy()
        history = [np.asarray(block, dtype=np.float64).copy()
                   for block in prior_history]
        if len(history) != self.m - 1:
            raise ValueError(f"expected {self.m - 1} prior history blocks")
        output = np.empty((controls.shape[0] + 1, 4), dtype=np.float64)
        output[0] = x
        for j, control in enumerate(controls):
            g_current = self.features(x, control)
            x = self.predict_history([g_current, *history])
            output[j + 1] = x
            if self.m > 1:
                history = [g_current, *history[:-1]]
        return output

    def save(self, path: Path, base_model_path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            kind=np.array(MODEL_KIND),
            base_dictionary=np.array(BASE_KIND),
            base_model_path=np.array(str(base_model_path.resolve())),
            m=np.array(self.m),
            W=self.W,
            mu=self.mu,
            sd=self.sd,
            clip=self.clip,
            out_axes=np.asarray(STATE_INDICES, dtype=np.int64),
            scored_cols=np.arange(4, dtype=np.int64),
            lift_dim=np.array(self.base.dimension),
            dt=np.array(self.dt),
            ridge_alpha=np.array(self.ridge_alpha),
            input_names=np.asarray(CONTROL_NAMES),
            feature_layout=np.array(
                "g = [phi(x), u], X = [g_k..g_k-m+1, 1] standardized; "
                "target = X_next[k]"
            ),
        )
        return path

    @classmethod
    def load(
        cls, path: Path, base_model_path: Path | None = None,
    ) -> "ARXHistoryModel":
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["kind"]) != MODEL_KIND:
                raise ValueError(f"{path} is not an {MODEL_KIND} archive")
            saved_base = Path(str(archive["base_model_path"]))
            base_path = base_model_path if base_model_path is not None else saved_base
            return cls(
                base=GaussianEDMDcModel.load(base_path),
                m=int(archive["m"]),
                W=archive["W"],
                mu=archive["mu"],
                sd=archive["sd"],
                clip=archive["clip"],
                dt=float(archive["dt"]),
                ridge_alpha=float(archive["ridge_alpha"]),
            )


def fit_arx_history(
    base: GaussianEDMDcModel,
    data,
    *,
    m: int = 10,
    ridge_alpha: float = 1.0,
) -> tuple[ARXHistoryModel, np.ndarray]:
    """Fit the correctly aligned causal ARX model and return its fit rows."""
    state = select_controlled_state(np.asarray(data["X"], dtype=np.float64))
    target = select_controlled_state(np.asarray(data["X_next"], dtype=np.float64))
    control = np.asarray(data["U"], dtype=np.float64)
    segments = contiguous_segments(data["traj_idx"], data["step_idx"])
    rows = history_rows(segments, m)
    if rows.size == 0:
        raise ValueError("dataset has no rows with enough causal history")

    raw = np.hstack([base.lift(state), control])
    mu = raw[rows].mean(axis=0)
    sd = np.maximum(raw[rows].std(axis=0), 1e-6)
    standardized = (raw - mu) / sd
    design = np.hstack([
        *[standardized[rows - lag] for lag in range(m)],
        np.ones((rows.size, 1)),
    ])
    penalty = ridge_alpha * np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    W = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target[rows],
    ).T
    clip = 2.0 * np.max(np.abs(target[rows]), axis=0)
    model = ARXHistoryModel(
        base=base, m=m, W=W, mu=mu, sd=sd, clip=clip,
        dt=float(data["dt"]), ridge_alpha=ridge_alpha,
    )
    return model, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("base_model", type=Path)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    base = GaussianEDMDcModel.load(args.base_model)
    with np.load(args.data, allow_pickle=False) as data:
        model, rows = fit_arx_history(
            base, data, m=args.m, ridge_alpha=args.ridge_alpha,
        )
        raw = np.hstack([
            base.lift(select_controlled_state(data["X"])), data["U"],
        ])
        history = [raw[rows - lag] for lag in range(model.m)]
        prediction = model.predict_history(history)
        truth = select_controlled_state(data["X_next"])[rows]
        rmse = np.sqrt(np.mean((prediction - truth) ** 2, axis=0))

    out = args.out or (
        project_root() / "Gaussian_dictionary" / "model"
        / f"arx{args.m}_stab_free20_gauss_2rbf.npz"
    )
    model.save(out, args.base_model)
    print(f"[arx-fit] rows={rows.size:,}, aligned target=X_next[k]")
    print(f"[arx-fit] training one-step RMSE [u,v,w,r]={rmse.round(6).tolist()}")
    print(f"[arx-fit] saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
