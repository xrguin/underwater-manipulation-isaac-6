"""Plot and summarize the four MPC experimental cases.

Expected cases:
  1. no-cube baseline EDMDc
  2. no-cube memory HODMDc+ARX
  3. payload-cube baseline EDMDc
  4. payload-cube memory HODMDc+ARX

Each input is a ``backstep_edmdc_mpc.py`` output ``.npz`` file.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .edmdc import NU_NAMES
from .edmdc_delay import MODEL_KIND as DELAY_MODEL_KIND
from ._plot_style import FIG_W, apply as apply_plot_style


TRACK_AXES = (0, 1, 2, 5)
CONTROL_NAMES = ("surge", "sway", "heave", "yaw")


def _scalar_str(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    return str(data[key].item() if data[key].shape == () else data[key])


def _scalar_bool(data: np.lib.npyio.NpzFile, key: str, default: bool = False) -> bool:
    if key not in data.files:
        return default
    return bool(data[key].item() if data[key].shape == () else data[key])


def _case_label(path: Path, data: np.lib.npyio.NpzFile) -> str:
    scenario = "cube" if _scalar_bool(data, "payload_cube") else "no cube"
    kind = _scalar_str(data, "model_kind", "edmdc")
    method = "memory" if kind == DELAY_MODEL_KIND else "EDMDc"
    return f"{scenario} / {method}"


def _load_case(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    t = data["t"]
    nu = data["nu"]
    nu_ref = data["nu_ref"]
    pos = data["pos"]
    u = data["u"]
    e_wp = data["e_wp"]
    arrivals = data["arrival_steps"]
    err = nu[:, TRACK_AXES] - nu_ref[:, TRACK_AXES]
    axis_rmse = np.sqrt(np.mean(err ** 2, axis=0))
    return {
        "path": path,
        "label": _case_label(path, data),
        "tag": _scalar_str(data, "tag", path.stem),
        "scenario": _scalar_str(data, "scenario", ""),
        "model_kind": _scalar_str(data, "model_kind", "edmdc"),
        "payload_cube": _scalar_bool(data, "payload_cube"),
        "t": t,
        "nu": nu,
        "nu_ref": nu_ref,
        "pos": pos,
        "u": u,
        "e_wp": e_wp,
        "waypoints": data["waypoints"],
        "target_idx": data["target_idx"],
        "arrival_steps": arrivals,
        "axis_rmse": axis_rmse,
        "vel_rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean_wp_error": float(np.mean(np.linalg.norm(e_wp, axis=1))),
        "final_wp_error": float(np.linalg.norm(e_wp[-1])),
        "reached": int(np.sum(arrivals >= 0)),
        "n_waypoints": int(arrivals.size),
        "cmd_rms": float(np.sqrt(np.mean(u ** 2))),
        "cmd_sat_frac": float(np.mean(np.abs(u) >= 0.98)),
    }


def _write_metrics(cases: list[dict], out_dir: Path) -> Path:
    out = out_dir / "mpc_four_case_metrics.csv"
    fields = [
        "label", "path", "scenario", "model_kind", "payload_cube",
        "vel_rmse", "mean_wp_error", "final_wp_error",
        "reached", "n_waypoints", "cmd_rms", "cmd_sat_frac",
    ]
    fields += [f"rmse_{NU_NAMES[i]}" for i in TRACK_AXES]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = {k: case[k] for k in fields if k in case}
            row["path"] = str(case["path"])
            for name, rmse in zip([NU_NAMES[i] for i in TRACK_AXES], case["axis_rmse"]):
                row[f"rmse_{name}"] = float(rmse)
            writer.writerow(row)
    return out


def _set_axes_equal(ax, xyz: np.ndarray, waypoints: np.ndarray) -> None:
    pts = np.vstack([xyz, waypoints])
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    radius = max(radius, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_velocity(cases: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(cases), len(TRACK_AXES),
                             figsize=(FIG_W, 1.35 * len(cases) + 1.0),
                             sharex=False)
    if len(cases) == 1:
        axes = axes[None, :]
    for r, case in enumerate(cases):
        for c, axis in enumerate(TRACK_AXES):
            ax = axes[r, c]
            ax.plot(case["t"], case["nu"][:, axis], color="C0", lw=1.0,
                    label="measured" if r == 0 and c == 0 else None)
            ax.plot(case["t"], case["nu_ref"][:, axis], color="C3", lw=0.9,
                    ls="--", label="ref" if r == 0 and c == 0 else None)
            ax.set_title(f"{case['label']} - {NU_NAMES[axis]}", fontsize=8)
            ax.set_xlabel("t [s]")
            ax.set_ylabel("vel")
            ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / "mpc_four_case_velocity_tracking.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_controls(cases: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(cases), len(CONTROL_NAMES),
                             figsize=(FIG_W, 1.35 * len(cases) + 1.0),
                             sharex=False, sharey=True)
    if len(cases) == 1:
        axes = axes[None, :]
    for r, case in enumerate(cases):
        for c, name in enumerate(CONTROL_NAMES):
            ax = axes[r, c]
            ax.plot(case["t"], case["u"][:, c], color=f"C{c}", lw=1.0)
            ax.axhline(1.0, color="k", lw=0.5, ls=":")
            ax.axhline(-1.0, color="k", lw=0.5, ls=":")
            ax.set_ylim(-1.1, 1.1)
            ax.set_title(f"{case['label']} - {name}", fontsize=8)
            ax.set_xlabel("t [s]")
            ax.set_ylabel("cmd")
            ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / "mpc_four_case_controls.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_3d(cases: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(FIG_W, 6.2))
    for i, case in enumerate(cases, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        pos = case["pos"]
        wp = case["waypoints"]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="C0", lw=1.4,
                label="vehicle")
        ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], color="C3", lw=1.0,
                ls="--", marker="*", label="waypoints")
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color="C2", s=20,
                   label="start")
        ax.set_title(case["label"], fontsize=9)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        _set_axes_equal(ax, pos, wp)
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    out = out_dir / "mpc_four_case_3d_trajectories.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_metrics(cases: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    labels = [
        ("cube" if c["payload_cube"] else "NC")
        + "\n"
        + ("mem" if c["model_kind"] == DELAY_MODEL_KIND else "EDMDc")
        for c in cases
    ]
    x = np.arange(len(cases))
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 3.0))

    axes[0].bar(x, [c["vel_rmse"] for c in cases], color="C0")
    axes[0].set_title("velocity tracking RMSE")
    axes[0].set_ylabel("RMSE")

    axes[1].bar(x, [c["final_wp_error"] for c in cases], color="C3")
    axes[1].set_title("final waypoint error")
    axes[1].set_ylabel("m")

    sat_pct = [100.0 * c["cmd_sat_frac"] for c in cases]
    axes[2].bar(x, sat_pct, color="C4")
    axes[2].set_title("command saturation")
    axes[2].set_ylabel("% samples |u| >= 0.98")
    axes[2].set_ylim(0.0, max(1.0, 1.1 * max(sat_pct)))

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = out_dir / "mpc_four_case_tracking_metrics.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def make_plots(npz_paths: list[Path], out_dir: Path) -> list[Path]:
    apply_plot_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [_load_case(Path(p)) for p in npz_paths]
    outputs = [
        _write_metrics(cases, out_dir),
        _plot_velocity(cases, out_dir),
        _plot_controls(cases, out_dir),
        _plot_3d(cases, out_dir),
        _plot_metrics(cases, out_dir),
    ]
    return outputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", type=Path, nargs="+",
                    help="Four backstep_edmdc_mpc output .npz files.")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("EDMDc/data/plots/mpc_four_cases"))
    args = ap.parse_args()
    outputs = make_plots(args.npz, args.out_dir)
    for out in outputs:
        print(f"[mpc-plot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
