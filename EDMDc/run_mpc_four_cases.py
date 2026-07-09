"""Run the four MPC experiment cases and generate comparison plots."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .plot_mpc_four_cases import make_plots


PROJECT = Path(__file__).resolve().parents[1]


CASES = (
    {
        "name": "no_cube_edmdc",
        "model": PROJECT / "EDMDc" / "model" / "ardusub_edmdc.npz",
        "payload_cube": False,
    },
    {
        "name": "no_cube_memory",
        "model": PROJECT / "EDMDc" / "model" / "edmdc_delay.npz",
        "payload_cube": False,
    },
    {
        "name": "cube_edmdc",
        "model": PROJECT / "EDMDc" / "model" / "ardusub_edmdc_cube.npz",
        "payload_cube": True,
    },
    {
        "name": "cube_memory",
        "model": PROJECT / "EDMDc" / "model" / "edmdc_delay_cube.npz",
        "payload_cube": True,
    },
)


def _latest_case_output(tag: str, payload_cube: bool, data_root: Path) -> Path:
    subdir = "with_gripper_cube" if payload_cube else "with_gripper"
    out_dir = data_root / subdir / "backstep_mpc"
    matches = sorted(out_dir.glob(f"*_{tag}.npz"))
    if not matches:
        raise FileNotFoundError(f"no output .npz found for tag {tag} in {out_dir}")
    return matches[-1]


def _run_case(args: argparse.Namespace, case: dict, run_id: str) -> Path:
    tag = f"{run_id}_{case['name']}"
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
        "--tag", tag,
        "--out-root", str(args.data_root),
    ]
    if args.mpc_osqp_polish:
        cmd.append("--mpc-osqp-polish")
    if args.headless:
        cmd.append("--headless")
    if args.no_video:
        cmd.append("--no-video")
    if args.no_live_plot:
        cmd.append("--no-live-plot")
    if case["payload_cube"]:
        cmd.append("--payload-cube")
    cmd.extend(args.extra_args)

    print(f"[mpc-four] running {case['name']}")
    print("[mpc-four] " + " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT, check=True)
    out = _latest_case_output(tag, bool(case["payload_cube"]), args.data_root)
    print(f"[mpc-four] output {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=120.0,
                    help="Seconds per case after settle. Default 120.")
    ap.add_argument("--mpc-N", type=int, default=5,
                    help="MPC horizon passed to backstep_edmdc_mpc. Default 5.")
    ap.add_argument("--mpc-decimation", type=int, default=2,
                    help="Solve every N ticks and hold command. Default 2.")
    ap.add_argument("--mpc-backend", default="condensed_osqp",
                    choices=("auto", "condensed_osqp"),
                    help="MPC solver backend passed to backstep_edmdc_mpc "
                         "(condensed direct OSQP; auto is an alias).")
    ap.add_argument("--mpc-osqp-eps", type=float, default=1e-5,
                    help="Absolute/relative OSQP tolerance for condensed backend.")
    ap.add_argument("--mpc-osqp-max-iter", type=int, default=10000,
                    help="Maximum OSQP iterations for condensed backend.")
    ap.add_argument("--mpc-osqp-polish", action="store_true",
                    help="Enable OSQP polishing for condensed backend.")
    ap.add_argument("--python", default=sys.executable,
                    help="Python executable used for subprocess runs.")
    ap.add_argument("--headless", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--no-video", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--no-live-plot", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Directory for the four-case comparison plots. "
                         "Default: <data-root>/plots/mpc_four_cases.")
    ap.add_argument("--data-root", type=Path,
                    default=PROJECT / "EDMDc" / "data",
                    help="Root passed to backstep_edmdc_mpc as --out-root; "
                         "per-case npz/pngs land under it.")
    ap.add_argument("extra_args", nargs=argparse.REMAINDER,
                    help="Optional extra args after '--' passed to each case.")
    args = ap.parse_args()
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    if args.out_dir is None:
        args.out_dir = args.data_root / "plots" / "mpc_four_cases"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs: list[Path] = []
    for case in CASES:
        outputs.append(_run_case(args, case, run_id))

    plot_paths = make_plots(outputs, args.out_dir)
    manifest = {
        "run_id": run_id,
        "mpc_backend": args.mpc_backend,
        "mpc_osqp_eps": args.mpc_osqp_eps,
        "mpc_osqp_max_iter": args.mpc_osqp_max_iter,
        "mpc_osqp_polish": bool(args.mpc_osqp_polish),
        "cases": [
            {"name": case["name"], "model": str(case["model"]),
             "payload_cube": bool(case["payload_cube"]), "output": str(out)}
            for case, out in zip(CASES, outputs)
        ],
        "plots": [str(p) for p in plot_paths],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / f"mpc_four_case_manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[mpc-four] manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
