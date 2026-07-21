"""Barbalata-style Gaussian dictionary and additive EDMDc trainer.

The full simulator remains six-DOF.  This module selects the task velocities

    x = [u, v, w, r]

from each full body-velocity snapshot and learns

    z[k+1] = A z[k] + B command[k]

with

    z = [u, v, w, r, phi_1(x), phi_2(x)].

Every Gaussian observable uses all four selected velocities.  The number of
RBFs is configurable, although two is the paper-style default.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


FULL_NU_DIM = 6
STATE_INDICES: tuple[int, ...] = (0, 1, 2, 5)
STATE_NAMES: tuple[str, ...] = ("u", "v", "w", "r")
STATE_DIM = len(STATE_INDICES)

CONTROL_DIM = 4
CONTROL_NAMES: tuple[str, ...] = ("surge", "sway", "heave", "yaw")

DEFAULT_N_RBFS = 2
DICT_DIM = STATE_DIM + DEFAULT_N_RBFS
MODEL_KIND = "gaussian_edmdc_4state"


def project_root() -> Path:
    """Return the repository root without depending on the existing package."""
    return Path(__file__).resolve().parents[1]


def select_controlled_state(nu: np.ndarray) -> np.ndarray:
    """Select ``[u, v, w, r]`` from a 6-D velocity, or validate a 4-D state.

    This function never changes or constrains the simulator's roll/pitch rates.
    It only defines the state observed by the reduced Koopman model.
    """
    arr = np.asarray(nu, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError(f"state must be rank 1 or 2; got shape {arr.shape}")
    if arr.shape[-1] == FULL_NU_DIM:
        return arr[..., list(STATE_INDICES)]
    if arr.shape[-1] == STATE_DIM:
        return arr
    raise ValueError(
        f"state must end in {FULL_NU_DIM} full velocities or {STATE_DIM} "
        f"controlled velocities; got shape {arr.shape}"
    )


def _as_samples(x: np.ndarray, expected_dim: int, name: str) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=np.float64)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"{name} must have shape ({expected_dim},) or (N, {expected_dim}); got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr, squeeze


@dataclass(frozen=True)
class GaussianDictionary:
    """Scaling, centers, and widths for Gaussian RBF observables.

    Centers and widths live in standardized state coordinates.  A single
    observable is

        phi_j(x) = exp(-0.5 * sum_i ((x_i - c_ji) / sigma_j)^2).

    Thus each ``phi_j`` is one scalar feature that jointly uses all four state
    components; it is not one Gaussian per velocity.
    """

    centers: np.ndarray
    widths: np.ndarray
    state_mean: np.ndarray
    state_scale: np.ndarray

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=np.float64)
        widths = np.asarray(self.widths, dtype=np.float64)
        mean = np.asarray(self.state_mean, dtype=np.float64)
        scale = np.asarray(self.state_scale, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != STATE_DIM:
            raise ValueError(f"centers must be (n_rbfs, {STATE_DIM}); got {centers.shape}")
        if centers.shape[0] < 1:
            raise ValueError("at least one Gaussian RBF is required")
        if widths.shape != (centers.shape[0],):
            raise ValueError(f"widths must be ({centers.shape[0]},); got {widths.shape}")
        if mean.shape != (STATE_DIM,) or scale.shape != (STATE_DIM,):
            raise ValueError(f"state_mean/state_scale must each be ({STATE_DIM},)")
        if np.any(widths <= 0.0) or np.any(scale <= 0.0):
            raise ValueError("all RBF widths and state scales must be positive")
        for name, value in (
            ("centers", centers), ("widths", widths),
            ("state_mean", mean), ("state_scale", scale),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN or infinite values")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "widths", widths)
        object.__setattr__(self, "state_mean", mean)
        object.__setattr__(self, "state_scale", scale)

    @property
    def n_rbfs(self) -> int:
        return int(self.centers.shape[0])

    @property
    def dimension(self) -> int:
        return STATE_DIM + self.n_rbfs

    def rbf(self, state: np.ndarray) -> np.ndarray:
        """Evaluate only the Gaussian observables."""
        x, squeeze = _as_samples(select_controlled_state(state), STATE_DIM, "state")
        standardized = (x - self.state_mean) / self.state_scale
        delta = standardized[:, None, :] - self.centers[None, :, :]
        squared_radius = np.sum(delta * delta, axis=2)
        values = np.exp(-0.5 * squared_radius / (self.widths[None, :] ** 2))
        return values[0] if squeeze else values

    def lift(self, state: np.ndarray) -> np.ndarray:
        """Return ``[u, v, w, r, phi_1, ..., phi_L]``."""
        x, squeeze = _as_samples(select_controlled_state(state), STATE_DIM, "state")
        phi = self.rbf(x)
        if phi.ndim == 1:
            phi = phi[None, :]
        lifted = np.hstack([x, phi])
        return lifted[0] if squeeze else lifted

    def feature_names(self) -> tuple[str, ...]:
        return STATE_NAMES + tuple(f"gaussian_{i + 1}" for i in range(self.n_rbfs))


def _initial_centers_kmeans_pp(x: np.ndarray, n_centers: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((n_centers, x.shape[1]), dtype=np.float64)
    centers[0] = x[int(rng.integers(x.shape[0]))]
    closest_sq = np.sum((x - centers[0]) ** 2, axis=1)
    for j in range(1, n_centers):
        total = float(np.sum(closest_sq))
        if total <= np.finfo(np.float64).eps:
            centers[j] = x[int(rng.integers(x.shape[0]))]
        else:
            centers[j] = x[int(rng.choice(x.shape[0], p=closest_sq / total))]
        closest_sq = np.minimum(closest_sq, np.sum((x - centers[j]) ** 2, axis=1))
    return centers


def _fit_centers(
    standardized: np.ndarray,
    n_rbfs: int,
    method: str,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if method == "random":
        replace = standardized.shape[0] < n_rbfs
        indices = rng.choice(standardized.shape[0], size=n_rbfs, replace=replace)
        centers = standardized[indices].copy()
    elif method == "kmeans":
        centers = _initial_centers_kmeans_pp(standardized, n_rbfs, rng)
        for _ in range(max_iter):
            dist_sq = np.sum((standardized[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dist_sq, axis=1)
            updated = centers.copy()
            for j in range(n_rbfs):
                members = standardized[labels == j]
                if members.size:
                    updated[j] = np.mean(members, axis=0)
                else:
                    updated[j] = standardized[int(np.argmax(np.min(dist_sq, axis=1)))]
            if np.allclose(updated, centers, rtol=1e-7, atol=1e-9):
                centers = updated
                break
            centers = updated
    else:
        raise ValueError(f"center_method must be 'kmeans' or 'random'; got {method!r}")

    dist_sq = np.sum((standardized[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dist_sq, axis=1)
    return centers, labels


def fit_dictionary(
    state: np.ndarray,
    n_rbfs: int = DEFAULT_N_RBFS,
    *,
    center_method: str = "kmeans",
    seed: int = 7,
    width_scale: float = 1.0,
    max_iter: int = 100,
    max_center_samples: int = 100_000,
) -> GaussianDictionary:
    """Fit standardized Gaussian centers and robust cluster-based widths."""
    x, _ = _as_samples(select_controlled_state(state), STATE_DIM, "state")
    if n_rbfs < 1:
        raise ValueError("n_rbfs must be at least 1")
    if width_scale <= 0.0:
        raise ValueError("width_scale must be positive")
    if x.shape[0] < 2:
        raise ValueError("at least two state samples are required")

    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (x - mean) / scale

    if standardized.shape[0] > max_center_samples:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(standardized.shape[0], size=max_center_samples, replace=False)
        center_data = standardized[chosen]
    else:
        center_data = standardized

    centers, labels = _fit_centers(
        center_data, n_rbfs=n_rbfs, method=center_method, seed=seed, max_iter=max_iter,
    )
    distances = np.linalg.norm(center_data - centers[labels], axis=1)
    positive = distances[distances > 1e-12]
    global_width = float(np.median(positive)) if positive.size else 1.0
    global_width = max(global_width, 0.25)

    widths = np.empty(n_rbfs, dtype=np.float64)
    for j in range(n_rbfs):
        local = distances[(labels == j) & (distances > 1e-12)]
        widths[j] = float(np.median(local)) if local.size else global_width
    widths = np.maximum(widths * width_scale, 1e-6)
    return GaussianDictionary(centers, widths, mean, scale)


def fit_edmdc(
    state: np.ndarray,
    state_next: np.ndarray,
    control: np.ndarray,
    dictionary: GaussianDictionary,
    *,
    lam: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit additive EDMDc with a numerically stable ridge least-squares solve."""
    x, _ = _as_samples(select_controlled_state(state), STATE_DIM, "state")
    x_next, _ = _as_samples(select_controlled_state(state_next), STATE_DIM, "state_next")
    u, _ = _as_samples(control, CONTROL_DIM, "control")
    if not (x.shape[0] == x_next.shape[0] == u.shape[0]):
        raise ValueError("state, state_next, and control must have equal sample counts")
    if lam < 0.0:
        raise ValueError("lam must be non-negative")

    z = dictionary.lift(x)
    z_next = dictionary.lift(x_next)
    design = np.hstack([z, u])
    if lam > 0.0:
        identity = np.sqrt(lam) * np.eye(design.shape[1], dtype=np.float64)
        design_aug = np.vstack([design, identity])
        target_aug = np.vstack([z_next, np.zeros((design.shape[1], z_next.shape[1]))])
    else:
        design_aug = design
        target_aug = z_next
    coefficients, *_ = np.linalg.lstsq(design_aug, target_aug, rcond=None)
    d = dictionary.dimension
    return coefficients[:d].T, coefficients[d:].T


@dataclass
class GaussianEDMDcModel:
    """Serializable four-state Gaussian EDMDc predictor."""

    A: np.ndarray
    B: np.ndarray
    dictionary: GaussianDictionary
    input_kind: str = "commanded"
    dt: float = float("nan")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=np.float64)
        self.B = np.asarray(self.B, dtype=np.float64)
        d = self.dictionary.dimension
        if self.A.shape != (d, d):
            raise ValueError(f"A must be ({d}, {d}); got {self.A.shape}")
        if self.B.shape != (d, CONTROL_DIM):
            raise ValueError(f"B must be ({d}, {CONTROL_DIM}); got {self.B.shape}")

    @property
    def dimension(self) -> int:
        return self.dictionary.dimension

    def lift(self, state: np.ndarray) -> np.ndarray:
        return self.dictionary.lift(state)

    @staticmethod
    def decode(lifted: np.ndarray) -> np.ndarray:
        arr = np.asarray(lifted, dtype=np.float64)
        return arr[..., :STATE_DIM]

    def predict(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        x = select_controlled_state(state)
        u = np.asarray(control, dtype=np.float64)
        if x.ndim == 1:
            if u.shape != (CONTROL_DIM,):
                raise ValueError(f"control must be ({CONTROL_DIM},); got {u.shape}")
            return self.decode(self.A @ self.lift(x) + self.B @ u)
        u_samples, _ = _as_samples(u, CONTROL_DIM, "control")
        if u_samples.shape[0] != x.shape[0]:
            raise ValueError("state and control must have equal sample counts")
        z_next = self.lift(x) @ self.A.T + u_samples @ self.B.T
        return self.decode(z_next)

    def rollout(self, state0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        u, _ = _as_samples(controls, CONTROL_DIM, "controls")
        x0 = select_controlled_state(state0)
        if x0.ndim != 1:
            raise ValueError("state0 must be a single 4-D or 6-D state")
        z = np.empty((u.shape[0] + 1, self.dimension), dtype=np.float64)
        z[0] = self.lift(x0)
        for k in range(u.shape[0]):
            z[k + 1] = self.A @ z[k] + self.B @ u[k]
        return self.decode(z)

    def predict_full_nu(self, nu: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Predict controlled axes and carry measured ``p,q`` through unchanged.

        Carrying ``p,q`` is an adapter convention, not a claim that they are
        zero or predicted by this reduced model.
        """
        full, squeeze = _as_samples(nu, FULL_NU_DIM, "nu")
        u, _ = _as_samples(control, CONTROL_DIM, "control")
        if squeeze and u.shape[0] != 1:
            raise ValueError("a single velocity requires a single control")
        if full.shape[0] != u.shape[0]:
            raise ValueError("nu and control must have equal sample counts")
        result = full.copy()
        result[:, list(STATE_INDICES)] = self.predict(full, u)
        return result[0] if squeeze else result

    def diagnostics(self, state: np.ndarray, state_next: np.ndarray, control: np.ndarray) -> dict[str, Any]:
        truth = select_controlled_state(state_next)
        error = self.predict(state, control) - truth
        zero_drift = self.predict(np.zeros(STATE_DIM), np.zeros(CONTROL_DIM))
        return {
            "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(self.A)))),
            "one_step_rmse_total": float(np.sqrt(np.mean(error ** 2))),
            "one_step_rmse_axes": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
            "zero_input_zero_state_drift": zero_drift.tolist(),
            "zero_input_zero_state_drift_norm": float(np.linalg.norm(zero_drift)),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            kind=np.array(MODEL_KIND),
            A=self.A,
            B=self.B,
            centers=self.dictionary.centers,
            widths=self.dictionary.widths,
            state_mean=self.dictionary.state_mean,
            state_scale=self.dictionary.state_scale,
            state_indices=np.asarray(STATE_INDICES, dtype=np.int64),
            state_names=np.asarray(STATE_NAMES),
            control_names=np.asarray(CONTROL_NAMES),
            feature_names=np.asarray(self.dictionary.feature_names()),
            n_rbfs=np.array(self.dictionary.n_rbfs),
            input_kind=np.array(self.input_kind),
            dt=np.array(self.dt),
            metadata_json=np.array(json.dumps(self.metadata, sort_keys=True)),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "GaussianEDMDcModel":
        with np.load(path, allow_pickle=False) as data:
            kind = str(np.asarray(data["kind"]).item())
            if kind != MODEL_KIND:
                raise ValueError(f"expected model kind {MODEL_KIND!r}; found {kind!r}")
            indices = tuple(int(v) for v in np.asarray(data["state_indices"]).tolist())
            if indices != STATE_INDICES:
                raise ValueError(f"unsupported state indices {indices}; expected {STATE_INDICES}")
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            dictionary = GaussianDictionary(
                centers=data["centers"],
                widths=data["widths"],
                state_mean=data["state_mean"],
                state_scale=data["state_scale"],
            )
            return cls(
                A=data["A"], B=data["B"], dictionary=dictionary,
                input_kind=str(np.asarray(data["input_kind"]).item()),
                dt=float(np.asarray(data["dt"]).item()), metadata=metadata,
            )


def _select_inputs(data: Mapping[str, np.ndarray], input_kind: str) -> np.ndarray:
    if input_kind == "commanded":
        if "U" not in data:
            raise KeyError("dataset is missing commanded input array 'U'")
        control = np.asarray(data["U"], dtype=np.float64)
    elif input_kind == "realized":
        if "U_realized" not in data:
            raise KeyError("dataset is missing realized wrench array 'U_realized'")
        realized = np.asarray(data["U_realized"], dtype=np.float64)
        if realized.ndim != 2:
            raise ValueError(f"U_realized must be rank 2; got {realized.shape}")
        if realized.shape[1] == FULL_NU_DIM:
            control = realized[:, list(STATE_INDICES)]
        elif realized.shape[1] == CONTROL_DIM:
            control = realized
        else:
            raise ValueError("U_realized must contain 6-D wrench or four task-axis inputs")
    else:
        raise ValueError(f"input_kind must be 'commanded' or 'realized'; got {input_kind!r}")
    control, _ = _as_samples(control, CONTROL_DIM, "control")
    return control


def load_dataset_arrays(
    data: Mapping[str, np.ndarray], input_kind: str = "commanded",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load selected model arrays from the existing EDMDc dataset schema."""
    if "X" not in data or "X_next" not in data:
        raise KeyError("dataset must contain X and X_next")
    state = select_controlled_state(np.asarray(data["X"], dtype=np.float64))
    state_next = select_controlled_state(np.asarray(data["X_next"], dtype=np.float64))
    control = _select_inputs(data, input_kind)
    if not (state.shape[0] == state_next.shape[0] == control.shape[0]):
        raise ValueError("X, X_next, and selected input have unequal sample counts")
    return state, state_next, control


def _complete_mask(data: Mapping[str, np.ndarray]) -> np.ndarray:
    required = ("traj_idx", "step_idx", "episode_seconds", "dt")
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"--only-complete requires dataset fields: {', '.join(missing)}")
    traj = np.asarray(data["traj_idx"])
    step = np.asarray(data["step_idx"])
    n_steps = int(round(float(np.asarray(data["episode_seconds"]).item()) / float(np.asarray(data["dt"]).item())))
    keep = [int(t) for t in np.unique(traj) if int(np.max(step[traj == t])) >= n_steps]
    return np.isin(traj, keep)


def _train(args: argparse.Namespace) -> int:
    with np.load(args.npz, allow_pickle=False) as data:
        state, state_next, control = load_dataset_arrays(data, args.wrench)
        mask = _complete_mask(data) if args.only_complete else np.ones(state.shape[0], dtype=bool)
        state, state_next, control = state[mask], state_next[mask], control[mask]
        dt = float(np.asarray(data["dt"]).item()) if "dt" in data else float("nan")
        source_fields = set(data.files)

    print(
        f"[gaussian] loaded {state.shape[0]:,} snapshots; state={STATE_NAMES}, "
        f"input={args.wrench}, n_rbfs={args.n_rbfs}"
    )
    dictionary = fit_dictionary(
        state, n_rbfs=args.n_rbfs, center_method=args.center_method,
        seed=args.seed, width_scale=args.width_scale,
    )
    A, B = fit_edmdc(state, state_next, control, dictionary, lam=args.lam)
    metadata: dict[str, Any] = {
        "source_dataset": str(args.npz.resolve()),
        "source_fields": sorted(source_fields),
        "ridge_lambda": float(args.lam),
        "center_method": args.center_method,
        "center_seed": int(args.seed),
        "width_scale": float(args.width_scale),
        "n_training_snapshots": int(state.shape[0]),
        "simulator_state_remains_6d": True,
        "unmodeled_axes": ["p", "q"],
    }
    model = GaussianEDMDcModel(A, B, dictionary, input_kind=args.wrench, dt=dt, metadata=metadata)
    diagnostics = model.diagnostics(state, state_next, control)
    model.metadata["training_diagnostics"] = diagnostics

    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = project_root() / "Gaussian_dictionary" / "model" / f"gaussian_edmdc_{stamp}.npz"
    else:
        out = args.out
    model.save(out)
    print(f"[gaussian] A {A.shape}, B {B.shape}")
    print(f"[gaussian] one-step training RMSE: {diagnostics['one_step_rmse_total']:.6g}")
    print(f"[gaussian] spectral radius(A): {diagnostics['spectral_radius']:.6g}")
    print(f"[gaussian] saved: {out}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="EDMDc collector dataset containing X, X_next, and U")
    parser.add_argument("--out", type=Path, default=None, help="Output model .npz")
    parser.add_argument("--lam", type=float, default=1e-3, help="Ridge penalty (default: 1e-3)")
    parser.add_argument("--n-rbfs", type=int, default=DEFAULT_N_RBFS, help="Number of Gaussian observables (default: 2)")
    parser.add_argument(
        "--center-method", choices=("kmeans", "random"), default="kmeans",
        help="kmeans is robust; random more closely reproduces randomly placed paper RBFs",
    )
    parser.add_argument("--seed", type=int, default=7, help="Reproducible center seed")
    parser.add_argument("--width-scale", type=float, default=1.0, help="Multiplier for fitted RBF widths")
    parser.add_argument("--wrench", choices=("commanded", "realized"), default="commanded")
    parser.add_argument("--only-complete", action="store_true", help="Discard truncated collector trajectories")
    return parser


def main(argv: list[str] | None = None) -> int:
    return _train(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
