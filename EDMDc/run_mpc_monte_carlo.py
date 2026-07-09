"""Paired Monte Carlo MPC navigation experiment.

Runs the four cases:
  * gripper only + EDMDc
  * gripper only + EDMDc + memory h=3
  * gripper + cube + EDMDc
  * gripper + cube + EDMDc + memory h=3

The same waypoint task is reused across controllers and physical scenarios for
each trial index. Results are appended to a CSV after every case, so long runs
can be resumed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


CASES = (
    {
        "case": "no_cube_edmdc",
        "scenario": "with_gripper",
        "controller": "edmdc",
        "model": PROJECT / "EDMDc" / "model" / "ardusub_edmdc.npz",
        "payload_cube": False,
    },
    {
        "case": "no_cube_memory_h3",
        "scenario": "with_gripper",
        "controller": "memory_h3",
        "model": PROJECT / "EDMDc" / "model" / "edmdc_delay.npz",
        "payload_cube": False,
    },
    {
        "case": "cube_edmdc",
        "scenario": "with_gripper_cube",
        "controller": "edmdc",
        "model": PROJECT / "EDMDc" / "model" / "ardusub_edmdc_cube.npz",
        "payload_cube": True,
    },
    {
        "case": "cube_memory_h3",
        "scenario": "with_gripper_cube",
        "controller": "memory_h3",
        "model": PROJECT / "EDMDc" / "model" / "edmdc_delay_cube.npz",
        "payload_cube": True,
    },
)


CSV_FIELDS = (
    "run_id", "trial_idx", "task_seed", "difficulty", "case",
    "scenario", "controller", "payload_cube", "model", "output_npz",
    "log_path", "returncode", "elapsed_wall_s", "success", "failure_reason",
    "reached", "n_waypoints", "completion_time_s", "final_wp_error_m",
    "mean_wp_error_m", "vel_rmse", "cmd_rms", "cmd_sat_frac",
    "max_abs_roll_pitch_deg", "max_xy_radius_m", "min_z_m", "max_z_m",
    "mpc_N", "mpc_decimation", "mpc_backend", "mpc_solve_count",
    "mpc_solve_fail_count", "mpc_solve_time_total_s", "mpc_solve_time_mean_ms",
    "duration_s", "arrival_tol_m",
    "waypoints_json",
)


def _latest_case_output(tag: str, payload_cube: bool,
                        data_root: Path) -> Path | None:
    subdir = "with_gripper_cube" if payload_cube else "with_gripper"
    out_dir = data_root / subdir / "backstep_mpc"
    matches = sorted(out_dir.glob(f"*_{tag}.npz"))
    return matches[-1] if matches else None


def _task_waypoints(task_seed: int, difficulty: str) -> np.ndarray:
    rng = np.random.default_rng(task_seed)
    if difficulty == "easy":
        x = rng.uniform(3.0, 4.2)
        y = rng.uniform(1.8, 2.8)
        z_mid = rng.uniform(-1.45, -0.85)
    elif difficulty == "medium":
        x = rng.uniform(3.8, 5.2)
        y = rng.uniform(2.4, 3.4)
        z_mid = rng.uniform(-1.65, -0.65)
    elif difficulty == "hard":
        x = rng.uniform(4.6, 6.0)
        y = rng.uniform(3.0, 4.1)
        z_mid = rng.uniform(-1.8, -0.45)
    else:
        raise ValueError(f"unknown difficulty {difficulty!r}")

    side = -1.0 if rng.random() < 0.5 else 1.0
    z0 = rng.uniform(-1.35, -0.65)
    z1 = z_mid
    z2 = rng.uniform(-1.15, -0.45)
    wp = np.array([
        [x, side * y, z0],
        [x + rng.uniform(-0.3, 0.35), -side * y, z1],
        [rng.uniform(-0.35, 0.35), rng.uniform(-0.45, 0.45), z2],
    ], dtype=np.float64)
    return np.round(wp, 3)


def _completed_keys(csv_path: Path) -> set[tuple[int, str]]:
    if not csv_path.exists():
        return set()
    done: set[tuple[int, str]] = set()
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("returncode") == "0" and row.get("output_npz"):
                done.add((int(row["trial_idx"]), row["case"]))
    return done


def _append_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    if exists:
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                old_rows = list(reader)
                tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
                with tmp_path.open("w", newline="") as tmp:
                    writer = csv.DictWriter(tmp, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    for old in old_rows:
                        writer.writerow({k: old.get(k, "") for k in CSV_FIELDS})
                tmp_path.replace(csv_path)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _summarize_npz(
    npz_path: Path,
    *,
    success_final_error: float,
    max_roll_pitch_deg: float,
    max_xy_radius: float,
    min_z: float,
    max_z: float,
) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    t = data["t"]
    e_wp = data["e_wp"]
    eta = data["eta"]
    pos = data["pos"]
    u = data["u"]
    nu = data["nu"]
    nu_ref = data["nu_ref"]
    arrivals = data["arrival_steps"]

    reached = int(np.sum(arrivals >= 0))
    n_waypoints = int(arrivals.size)
    all_reached = bool(reached == n_waypoints)
    completion_time = float(t[int(arrivals[-1])]) if all_reached else math.nan
    final_wp_error = float(np.linalg.norm(e_wp[-1]))
    mean_wp_error = float(np.mean(np.linalg.norm(e_wp, axis=1)))
    track_axes = (0, 1, 2, 5)
    vel_rmse = float(np.sqrt(np.mean((nu[:, track_axes] - nu_ref[:, track_axes]) ** 2)))
    cmd_rms = float(np.sqrt(np.mean(u ** 2)))
    cmd_sat_frac = float(np.mean(np.abs(u) >= 0.98))
    max_rp = float(np.rad2deg(np.max(np.abs(eta[:, :2]))))
    xy_radius = float(np.max(np.linalg.norm(pos[:, :2], axis=1)))
    z_low = float(np.min(pos[:, 2]))
    z_high = float(np.max(pos[:, 2]))
    mpc_solve_count = int(data["mpc_solve_count"]) if "mpc_solve_count" in data else 0
    mpc_solve_fail_count = (
        int(data["mpc_solve_fail_count"]) if "mpc_solve_fail_count" in data else 0
    )
    mpc_solve_time_total = (
        float(data["mpc_solve_time_total_s"])
        if "mpc_solve_time_total_s" in data else math.nan
    )
    mpc_solve_time_mean_ms = (
        float(data["mpc_solve_time_mean_s"]) * 1e3
        if "mpc_solve_time_mean_s" in data else math.nan
    )

    boundary_ok = (
        xy_radius <= max_xy_radius
        and z_low >= min_z
        and z_high <= max_z
    )
    success = (
        all_reached
        and final_wp_error < success_final_error
        and max_rp <= max_roll_pitch_deg
        and boundary_ok
    )
    if success:
        failure_reason = "success"
    elif not all_reached:
        failure_reason = "missed_waypoint"
    elif final_wp_error >= success_final_error:
        failure_reason = "final_error"
    elif max_rp > max_roll_pitch_deg:
        failure_reason = "attitude"
    else:
        failure_reason = "workspace"

    return {
        "success": int(success),
        "failure_reason": failure_reason,
        "reached": reached,
        "n_waypoints": n_waypoints,
        "completion_time_s": completion_time,
        "final_wp_error_m": final_wp_error,
        "mean_wp_error_m": mean_wp_error,
        "vel_rmse": vel_rmse,
        "cmd_rms": cmd_rms,
        "cmd_sat_frac": cmd_sat_frac,
        "max_abs_roll_pitch_deg": max_rp,
        "max_xy_radius_m": xy_radius,
        "min_z_m": z_low,
        "max_z_m": z_high,
        "mpc_backend": (
            str(data["mpc_solver_backend"]) if "mpc_solver_backend" in data else ""
        ),
        "mpc_solve_count": mpc_solve_count,
        "mpc_solve_fail_count": mpc_solve_fail_count,
        "mpc_solve_time_total_s": mpc_solve_time_total,
        "mpc_solve_time_mean_ms": mpc_solve_time_mean_ms,
    }


def _run_case(args: argparse.Namespace, case: dict[str, Any],
              trial_idx: int, task_seed: int, difficulty: str,
              waypoints: np.ndarray) -> dict[str, Any]:
    tag = f"{args.run_id}_{case['case']}_trial{trial_idx:04d}"
    log_path = args.out_dir / "logs" / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        "-m", "EDMDc.backstep_edmdc_mpc",
        "--model", str(case["model"]),
        "--duration", str(args.duration),
        "--mpc-N", str(args.mpc_N),
        "--mpc-decimation", str(args.mpc_decimation),
        "--mpc-backend", str(args.mpc_backend),
        "--mpc-osqp-eps", str(args.mpc_osqp_eps),
        "--mpc-osqp-max-iter", str(args.mpc_osqp_max_iter),
        "--arrival-tol", str(args.arrival_tol),
        "--tag", tag,
        "--out-root", str(args.data_root),
        "--headless",
        "--no-video",
        "--no-live-plot",
        "--no-diagnostic-plot",
        "--no-floor-grid",
        "--waypoints",
        *[f"{v:.3f}" for v in waypoints.reshape(-1)],
    ]
    if args.mpc_osqp_polish:
        cmd.append("--mpc-osqp-polish")
    if case["payload_cube"]:
        cmd.append("--payload-cube")
    cmd.extend(args.extra_args)

    base_row: dict[str, Any] = {
        "run_id": args.run_id,
        "trial_idx": trial_idx,
        "task_seed": task_seed,
        "difficulty": difficulty,
        "case": case["case"],
        "scenario": case["scenario"],
        "controller": case["controller"],
        "payload_cube": int(bool(case["payload_cube"])),
        "model": str(case["model"]),
        "log_path": str(log_path),
        "mpc_N": args.mpc_N,
        "mpc_decimation": args.mpc_decimation,
        "mpc_backend": args.mpc_backend,
        "duration_s": args.duration,
        "arrival_tol_m": args.arrival_tol,
        "waypoints_json": json.dumps(waypoints.tolist()),
    }

    t0 = time.time()
    with log_path.open("w") as log:
        log.write("[mc] " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    row = dict(base_row)
    row["returncode"] = proc.returncode
    row["elapsed_wall_s"] = elapsed

    out = _latest_case_output(tag, bool(case["payload_cube"]), args.data_root)
    if proc.returncode == 0 and out is not None:
        row["output_npz"] = str(out)
        row.update(_summarize_npz(
            out,
            success_final_error=args.success_final_error,
            max_roll_pitch_deg=args.max_roll_pitch_deg,
            max_xy_radius=args.max_xy_radius,
            min_z=args.min_z,
            max_z=args.max_z,
        ))
    else:
        row["output_npz"] = "" if out is None else str(out)
        row["success"] = 0
        row["failure_reason"] = "process_failed" if proc.returncode else "missing_output"
    return row


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials-per-case", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--run-id", default=f"mc_{ts}")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--mpc-N", type=int, default=5)
    ap.add_argument("--mpc-decimation", type=int, default=2)
    ap.add_argument("--mpc-backend", default="condensed_osqp",
                    choices=("auto", "condensed_osqp"))
    ap.add_argument("--mpc-osqp-eps", type=float, default=1e-5)
    ap.add_argument("--mpc-osqp-max-iter", type=int, default=10000)
    ap.add_argument("--mpc-osqp-polish", action=argparse.BooleanOptionalAction,
                    default=False)
    ap.add_argument("--arrival-tol", type=float, default=0.15)
    ap.add_argument("--success-final-error", type=float, default=0.20)
    ap.add_argument("--max-roll-pitch-deg", type=float, default=35.0)
    ap.add_argument("--max-xy-radius", type=float, default=8.0)
    ap.add_argument("--min-z", type=float, default=-3.0)
    ap.add_argument("--max-z", type=float, default=0.25)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--data-root", type=Path,
                    default=PROJECT / "EDMDc" / "data",
                    help="Root passed to backstep_edmdc_mpc as --out-root; "
                         "per-trial npz land under it.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Directory for the metrics CSV, logs, and analysis. "
                         "Default: <data-root>/mpc_monte_carlo.")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--analyze", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("extra_args", nargs=argparse.REMAINDER,
                    help="Optional extra args after '--' passed to every case.")
    args = ap.parse_args()
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    if args.out_dir is None:
        args.out_dir = args.data_root / "mpc_monte_carlo"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.run_id}_metrics.csv"
    done = _completed_keys(csv_path) if args.resume else set()

    print(f"[mc] run_id={args.run_id}")
    print(f"[mc] trials_per_case={args.trials_per_case}, total planned="
          f"{args.trials_per_case * len(CASES)}")
    print(f"[mc] csv={csv_path}")

    difficulties = ("easy", "medium", "hard")
    for trial_idx in range(args.trials_per_case):
        difficulty = difficulties[trial_idx % len(difficulties)]
        task_seed = int(args.seed + 1009 * trial_idx)
        waypoints = _task_waypoints(task_seed, difficulty)
        print(f"[mc] trial {trial_idx:04d}/{args.trials_per_case - 1:04d} "
              f"seed={task_seed} difficulty={difficulty} "
              f"waypoints={waypoints.reshape(-1).tolist()}")
        for case in CASES:
            key = (trial_idx, case["case"])
            if key in done:
                print(f"[mc]   skip {case['case']} (already complete)")
                continue
            print(f"[mc]   run {case['case']}")
            row = _run_case(args, case, trial_idx, task_seed, difficulty, waypoints)
            _append_row(csv_path, row)
            print(f"[mc]   done {case['case']} rc={row['returncode']} "
                  f"success={row.get('success')} reason={row.get('failure_reason')} "
                  f"elapsed={float(row['elapsed_wall_s']):.1f}s")

    if args.analyze:
        cmd = [
            args.python,
            "-m", "EDMDc.analyze_mpc_monte_carlo",
            str(csv_path),
            "--out-dir", str(args.out_dir / f"{args.run_id}_plots"),
        ]
        subprocess.run(cmd, cwd=PROJECT, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
