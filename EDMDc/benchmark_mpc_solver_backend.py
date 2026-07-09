"""Benchmark the EDMDc-MPC condensed direct OSQP solver.

Times the per-step solve cost of the condensed OSQP backend on the trained
EDMDc and memory-augmented models. This benchmark does not boot Isaac Sim; it
measures controller solve time only.

The legacy CVXPY formulation was removed from EDMDc.mpc; the historical
comparison numbers (~300x slower on the 148-D memory model) are preserved in
EDMDc/data/plots/mpc_solver_benchmark/mpc_solver_backend_benchmark.csv from
the 2026-06-29 run.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from .mpc import EDMDcMPC, MPCConfig


PROJECT = Path(__file__).resolve().parents[1]
MODELS = (
    ("gripper EDMDc", PROJECT / "EDMDc" / "model" / "ardusub_edmdc.npz"),
    ("gripper memory h=3", PROJECT / "EDMDc" / "model" / "edmdc_delay.npz"),
    ("cube EDMDc", PROJECT / "EDMDc" / "model" / "ardusub_edmdc_cube.npz"),
    ("cube memory h=3", PROJECT / "EDMDc" / "model" / "edmdc_delay_cube.npz"),
)


def _sample_inputs(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    nu = rng.normal(0.0, 0.05, size=(n, 6))
    lo = np.array([-0.4, -0.3, -0.12, 0.0, 0.0, -0.45])
    hi = np.array([0.4, 0.3, 0.12, 0.0, 0.0, 0.45])
    ref = rng.uniform(lo, hi, size=(n, 6))
    return nu, ref


def _time_steps(ctrl: EDMDcMPC, nu: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.empty(len(nu), dtype=np.float64)
    u = np.empty((len(nu), ctrl.m), dtype=np.float64)
    for i, (nu_i, ref_i) in enumerate(zip(nu, ref)):
        t0 = time.perf_counter()
        u[i] = ctrl.step(nu_i, ref_i)
        times[i] = time.perf_counter() - t0
    return times, u


def run_benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for label, model_path in MODELS:
        nu_warm, ref_warm = _sample_inputs(rng, args.warmup)
        nu, ref = _sample_inputs(rng, args.samples)

        cfg = MPCConfig(
            N=args.mpc_N,
            solver_backend="condensed_osqp",
            osqp_eps_abs=args.osqp_eps,
            osqp_eps_rel=args.osqp_eps,
            osqp_max_iter=args.osqp_max_iter,
        )
        ctrl = EDMDcMPC.from_npz(model_path, config=cfg)

        _time_steps(ctrl, nu_warm, ref_warm)
        times, _ = _time_steps(ctrl, nu, ref)

        rows.append({
            "model": label,
            "backend": "condensed_osqp",
            "state_dim": ctrl.d,
            "samples": args.samples,
            "mean_ms": 1e3 * float(np.mean(times)),
            "p95_ms": 1e3 * float(np.percentile(times, 95)),
            "max_ms": 1e3 * float(np.max(times)),
            "last_status": ctrl.last_solver_status,
        })
    return rows


def write_outputs(rows: list[dict[str, object]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "mpc_condensed_osqp_benchmark.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = [r["model"] for r in rows]
    mean_ms = [r["mean_ms"] for r in rows]
    p95_ms = [r["p95_ms"] for r in rows]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width / 2, mean_ms, width, label="mean")
    ax.bar(x + width / 2, p95_ms, width, label="p95")
    ax.set_ylabel("Solve time per MPC step (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path = out_dir / "mpc_condensed_osqp_benchmark.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return csv_path, png_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mpc-N", type=int, default=5)
    ap.add_argument("--osqp-eps", type=float, default=1e-5)
    ap.add_argument("--osqp-max-iter", type=int, default=10000)
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT / "EDMDc" / "data" / "plots" / "mpc_solver_benchmark")
    args = ap.parse_args()

    rows = run_benchmark(args)
    csv_path, png_path = write_outputs(rows, args.out_dir)
    print(f"[mpc-bench] csv -> {csv_path}")
    print(f"[mpc-bench] plot -> {png_path}")
    for row in rows:
        print(
            f"[mpc-bench] {row['model']} (d={row['state_dim']}): "
            f"mean={row['mean_ms']:.3f} ms, p95={row['p95_ms']:.3f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
