"""Scenario-panel MPC plots: velocity tracking left, 3D trajectory right."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ._plot_style import FIG_W, apply as apply_plot_style
from .edmdc import NU_NAMES


TRACK_AXES = (0, 1, 2, 5)


def _load(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "path": path,
        "t": data["t"],
        "nu": data["nu"],
        "nu_ref": data["nu_ref"],
        "pos": data["pos"],
        "waypoints": data["waypoints"],
        "arrival_steps": data["arrival_steps"],
        "mpc_N": int(data["mpc_N"]),
        "model_kind": str(data["model_kind"]),
        "payload_cube": bool(data["payload_cube"]),
    }


def _case_paths_from_manifest(manifest_path: Path) -> dict[str, Path]:
    manifest = json.loads(manifest_path.read_text())
    return {case["name"]: Path(case["output"]) for case in manifest["cases"]}


def _memory_label(case: dict) -> str:
    # The shipped memory models are HODMDc+ARX with state history h=3.
    return "EDMDc + memory h=3"


def _set_axes_equal(ax, *point_sets: np.ndarray) -> None:
    pts = np.vstack(point_sets)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    radius = max(radius, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_scenario(edmdc: dict, memory: dict, title: str, out_path: Path,
                   edmdc_color: str, memory_color: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _ = title
    fig = plt.figure(figsize=(FIG_W, 5.2))
    outer = fig.add_gridspec(
        1, 2, width_ratios=(1.0, 1.35), left=0.07, right=0.98,
        bottom=0.08, top=0.86, wspace=0.20,
    )
    left = outer[0].subgridspec(len(TRACK_AXES), 1, hspace=0.12)

    for row, axis in enumerate(TRACK_AXES):
        ax = fig.add_subplot(left[row, 0])
        ax.plot(edmdc["t"], edmdc["nu"][:, axis], color=edmdc_color, lw=1.15,
                alpha=0.45, zorder=2,
                label=None)
        ax.plot(memory["t"], memory["nu"][:, axis], color=memory_color, lw=1.15,
                alpha=0.45, zorder=3,
                label=None)
        ax.plot(edmdc["t"], edmdc["nu_ref"][:, axis], color=edmdc_color, lw=1.6,
                ls=(0, (5, 2)), alpha=1.0, zorder=4, label=None)
        ax.plot(memory["t"], memory["nu_ref"][:, axis], color=memory_color, lw=1.6,
                ls=(0, (1.2, 1.8)), alpha=1.0, zorder=5, label=None)
        ax.set_ylabel(NU_NAMES[axis])
        ax.grid(alpha=0.3)
        if row < len(TRACK_AXES) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("t [s]")

    ax3d = fig.add_subplot(outer[1], projection="3d")
    ax3d.plot(edmdc["pos"][:, 0], edmdc["pos"][:, 1], edmdc["pos"][:, 2],
              color=edmdc_color, lw=2.2, label=None)
    ax3d.plot(memory["pos"][:, 0], memory["pos"][:, 1], memory["pos"][:, 2],
              color=memory_color, lw=2.2, ls="-.", label=None)
    wp = edmdc["waypoints"]
    ax3d.plot(wp[:, 0], wp[:, 1], wp[:, 2], color="black", lw=1.4,
              ls="--", marker="*", markersize=8, markeredgecolor="black",
              markeredgewidth=0.6, label=None)
    start = edmdc["pos"][0]
    ax3d.scatter(start[0], start[1], start[2], color="black", s=46,
                 edgecolors="black", linewidths=0.6, label=None)
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.view_init(elev=24, azim=-58)
    _set_axes_equal(ax3d, edmdc["pos"], memory["pos"], wp)

    handles = [
        Line2D([0], [0], color=edmdc_color, lw=2.0, label="EDMDc measured / path"),
        Line2D([0], [0], color=edmdc_color, lw=1.6, ls=(0, (5, 2)),
               label="EDMDc ref"),
        Line2D([0], [0], color=memory_color, lw=2.0, ls="-.",
               label=f"{_memory_label(memory)} measured / path"),
        Line2D([0], [0], color=memory_color, lw=1.6, ls=(0, (1.2, 1.8)),
               label=f"{_memory_label(memory)} ref"),
        Line2D([0], [0], color="black", lw=1.4, ls="--", marker="*",
               label="waypoints"),
        Line2D([0], [0], color="black", marker="o", lw=0, label="start"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=Path("EDMDc/data/plots/mpc_four_cases/"
                                 "mpc_four_case_manifest_20260629_030420.json"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("EDMDc/data/plots/mpc_four_cases"))
    args = ap.parse_args()

    apply_plot_style()
    paths = _case_paths_from_manifest(args.manifest)
    no_cube_edmdc = _load(paths["no_cube_edmdc"])
    no_cube_memory = _load(paths["no_cube_memory"])
    cube_edmdc = _load(paths["cube_edmdc"])
    cube_memory = _load(paths["cube_memory"])

    outputs = [
        args.out_dir / "mpc_gripper_only_velocity_3d_edmdc_memory_h3.png",
        args.out_dir / "mpc_gripper_cube_velocity_3d_edmdc_memory_h3.png",
    ]
    # Match the modeling-accuracy rollouts: gripper-only blue/red, cube green/orange.
    _plot_scenario(no_cube_edmdc, no_cube_memory,
                   "Gripper only: body nu vs backstep ref and 3D trajectory",
                   outputs[0], edmdc_color="#1f77b4", memory_color="#d62728")
    _plot_scenario(cube_edmdc, cube_memory,
                   "Gripper + cube: body nu vs backstep ref and 3D trajectory",
                   outputs[1], edmdc_color="#2ca02c", memory_color="#ff7f0e")
    for out in outputs:
        print(f"[scenario-panels] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
