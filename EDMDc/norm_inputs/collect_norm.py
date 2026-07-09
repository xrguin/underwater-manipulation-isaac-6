"""Collect APRBS snapshots with NORMALIZED [-1,1] control inputs (norm_inputs).

Self-contained copy of ``EDMDc.collect_isaac`` for the ``norm_inputs``
experiment. Behavioural change: the recorded control input ``U`` is the 6-DOF
body wrench **normalized per axis to [-1, 1]** (4-DOF task mask), not a Newton
wrench. Excitation samples the inscribed box of the achievable polytope (the same
realizable region the shipping pipeline uses).

Pipeline per step:
    APRBS in [-1,1] (active axes)  --denorm (x SYMMETRIC_CAPS)-->  wrench (N)
        --ThrustAllocator(linear=True)-->  per-thruster cmds c in [-1,1]
        --T200Group(linear=True)-->  per-rotor thrust = 25*c (N)
        --sum_to_wrench-->  realized body wrench (N)
        --fossen.compute_wrench_world-->  PhysX.
    Recorded:  U = realized wrench normalized to [-1,1];  C = c.

The inscribed box + linear unit-thrust keep every sample realizable (no deadband
/ lag / saturation), so ``U_recorded = U_realized`` — a clean LS fit. See
``norm_aprbs.py`` for the normalization math.

See ``EDMDc/norm_inputs/norm_aprbs.py`` for the normalization conventions and
``EDMDc/norm_inputs/README.md`` for the experiment rationale.

Invoke as: ``python -m EDMDc.norm_inputs.collect_norm`` (under the marinegym /
isaac-sim env, see CLAUDE.md).

Mirrors `collect_fossen.py` step-for-step but the dynamics generator is
PhysX (via `RigidPrimView.apply_forces_and_torques_at_pos` + `world.step`)
instead of the NumPy forward-Euler integrator. The Fossen hydrodynamic
corrections (buoyancy, damping, Coriolis, added-mass reaction) are applied
externally via the top-level `fossen.compute_wrench_world` — same physics
as `core.fossen_integrator`, just routed to PhysX.

By default (**thruster pipeline**, matches `bluerov_demo.py`):
desired APRBS wrench (body frame) → `ThrustAllocator` → `T200Group.step`
→ `sum_to_wrench` realized wrench → `fossen.compute_wrench_world`.

Pass **`--direct-wrench`** to apply the APRBS wrench directly (matches
NumPy `collect_fossen` / legacy Isaac parity for Fossen-vs-PhysX A/B).

With the **thruster pipeline** and a **visible Kit window** (no ``--headless``), per-rotor thrust arrows draw by default (same as ``bluerov_demo``). ``--direct-wrench`` or ``--headless`` disables them.

Optional **`--spawn-flat-pool`**: skip random IC poses; teleport to
**(spawn-x-world, 0, −1)** with **yaw_rel = 0** and **zero velocity** (still
consumes RNG for APRBS parity). Default **--settle-steps 0** skips pre-roll.

Each trajectory ends with a **drain phase** (matches ``collect_fossen.py``):
``tau = 0`` is applied until ``‖v‖ < DRAIN_VEL_TOL`` or ``DRAIN_MAX_STEPS``
steps elapse. Drain snapshots are stored with ``U = 0``.

Snapshot schema: **U** = APRBS after clipping (or zero during drain).


Sequences and ICs are still `(seed, trajectory_index)`-deterministic via
shared `aprbs_sequence` / `sample_initial_eta_v`.

Usage:
    # GUI: one short trajectory, surge-only APRBS (no --headless).
    ~/miniconda3/envs/marinegym/bin/python -m EDMDc.collect_isaac \\
        --n 1 --episode-s 8 --fx-only

    ~/miniconda3/envs/marinegym/bin/python -m EDMDc.collect_isaac \\
        --n 5 --episode-s 1.0 --seed 0 --headless --out /tmp/isaac_smoke.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Inscribed-box normalized [-1,1] conventions. norm_aprbs imports only numpy +
# isaac-free collect_fossen helpers, so it is safe before SimulationApp().
from .norm_aprbs import (
    SYMMETRIC_CAPS,
    NORM_CMD_MAX_POS,
    NORM_CMD_MAX_NEG,
    denorm_to_newton,
    norm_from_newton,
    aprbs_sequence_normalized,
)

# Run timestamp — stamped into default --out and auto-video filenames so
# successive runs don't overwrite. Override with --out / --video to pin paths.
_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Normalized [-1,1] command limits. The recorded control input U is the 6-DOF
# wrench normalized per axis to [-1, 1] (active axes Fx, Fy, Fz, Tz; Tx, Ty = 0).
# APRBS is clipped to the unit box; conversion to Newtons for the sim is
# u_newton = SYMMETRIC_CAPS * u_norm (inscribed-box excitation; see norm_aprbs).
WRENCH_CMD_MAX_POS = NORM_CMD_MAX_POS.copy()      # [1, 1, 1, 0, 0, 1]
WRENCH_CMD_MAX_NEG = NORM_CMD_MAX_NEG.copy()
# Per-motor unit-thrust scale: command c in [-1,1] -> thrust = 25*c (N), linear.
MAX_THRUST_PER_MOTOR_N = 25.0

# Drain phase: after APRBS ends, apply tau = 0 and keep recording until the
# body velocity decays to rest (or DRAIN_MAX_STEPS as a safety cap). Mirrors
# the drain logic in collect_fossen.py.
DRAIN_VEL_TOL = 1e-2      # ‖v‖ threshold (mixed m/s + rad/s units)
DRAIN_MAX_STEPS = 600     # safety cap (~10 s at dt = 1/60)


def _project_root() -> Path:
    """Repo root: two parents up from EDMDc/norm_inputs/."""
    return Path(__file__).resolve().parents[2]


def _clip_wrench_command(u_raw: np.ndarray) -> np.ndarray:
    u = np.asarray(u_raw, dtype=np.float64).copy()
    np.minimum(u, WRENCH_CMD_MAX_POS, out=u)
    np.maximum(u, -WRENCH_CMD_MAX_NEG, out=u)
    return u


# Step-response specs (--n-step). The model trained on APRBS-only data
# under-predicts sustained single-axis steady state (see debug_journal_gripper.md
# Bug #2). These deterministic 4×3×2 = 24 step trajectories hold a single
# axis at a fixed amplitude for the full episode, teaching steady-state
# response on each task axis at three magnitudes and both signs.
_STEP_AXES = (0, 1, 2, 5)            # Fx, Fy, Fz, Tz (task DOFs)
_STEP_AMP_FRACS = (0.3, 0.6, 0.9)    # small / medium / large
_STEP_SIGNS = (+1.0, -1.0)


def _step_trajectory_specs() -> list[tuple[int, float]]:
    """Deterministic 4 × 3 × 2 = 24 (axis_idx, amplitude) specs. Amplitude
    uses WRENCH_CMD_MAX_POS for + sign and WRENCH_CMD_MAX_NEG for −."""
    specs: list[tuple[int, float]] = []
    for axis in _STEP_AXES:
        for frac in _STEP_AMP_FRACS:
            for sign in _STEP_SIGNS:
                cap = WRENCH_CMD_MAX_POS[axis] if sign > 0 else WRENCH_CMD_MAX_NEG[axis]
                specs.append((int(axis), float(sign * frac * cap)))
    return specs


# Argparse FIRST so --help works without booting Isaac Sim.
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, dest="n_trajectories")
    ap.add_argument("--episode-s", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mass-kg", type=float, default=13.5)
    ap.add_argument("--headless", action="store_true",
                    help="Run without opening Isaac Sim viewports.")
    ap.add_argument("--settle-steps", type=int, default=0,
                    help="Extra zero-wrench Physics ticks after each teleport "
                         "before APRBS. Default 0 = skip settling (no pre-roll "
                         "drop/drift). Increase if PhysX needs warm-up.")
    ap.add_argument("--settle-no-reset", action="store_true",
                    help="Do NOT teleport back to the IC after the settle phase, "
                         "so APRBS starts from the SETTLED state (gripper "
                         "equilibrium, damped velocity). Default behavior resets "
                         "to the IC for APRBS bit-parity; use this for settled-"
                         "start evaluation datasets.")
    ap.add_argument(
        "--spawn-flat-pool",
        action="store_true",
        help="Fixed spawn: world pose (spawn-x-world, 0, -1) with roll=pitch=yaw=0 "
             "and zero body velocity instead of randomized IC. A discarded "
             "sample_initial_eta_v() keeps APRBS wrench bit-identical to the same "
             "seed without this flag; use --fx-only for surge-only APRBS.",
    )
    ap.add_argument(
        "--spawn-x-world",
        type=float,
        default=0.0,
        help="World x (m) when --spawn-flat-pool is set (default 0).",
    )
    ap.add_argument(
        "--fx-only",
        action="store_true",
        help="APRBS on Fx only (matches NumPy fx_only training). Overrides "
             "the default 4-DOF task mask.",
    )
    ap.add_argument(
        "--continuous-aprbs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DEFAULT ON. Continuous-uniform APRBS sampled inside the "
             "max-volume inscribed box of the 4-DOF achievable wrench polytope "
             "(per-rotor cap 25 N). Caps: Fx ±22.57, Fy ±23.47, "
             "Fz +88.17/−69.15, Tz ±6.59. Every sample is realizable so "
             "U_recorded matches the realized wrench (no allocator-saturation "
             "bias). Pass --no-continuous-aprbs to fall back to the legacy "
             "discrete-amplitude APRBS at full polytope-vertex caps (Fx ±65, "
             "Fy ±37.5, Fz +102/−80, Tz ±17), which causes ~80%% saturation "
             "distortion in the recorded U.",
    )
    ap.add_argument(
        "--hold-max-tz", type=int, default=None,
        help="Override APRBS_HOLD_MAX_STEPS for Tz (yaw torque) only. "
             "Default (None) uses the global 30 steps (0.5 s @ dt=1/60). "
             "Yaw has zero linear damping in the YAML, so r grows slowly at "
             "low magnitudes; longer Tz holds (e.g. 120 = 2 s) let r reach "
             "steady state during training and improve the model's high-r "
             "prediction. Applies in --continuous-aprbs mode only.",
    )
    ap.add_argument(
        "--hold-max-fz", type=int, default=None,
        help="Override APRBS_HOLD_MAX_STEPS for Fz (heave force) only. "
             "Same rationale as --hold-max-tz but for heave. Default (None) "
             "uses the global 30 steps.",
    )
    ap.add_argument(
        "--out", type=str, default="__AUTO__",
        help="Output .npz path. Default: EDMDc/data/simulator/isaac_<TS>.npz; "
             "with --gripper, defaults to "
             "EDMDc/data/with_gripper/isaac_gripper_<TS>.npz.",
    )
    ap.add_argument(
        "--video", type=str, nargs="?", default=None, const="__AUTO__",
        metavar="PATH",
        help="Record chase-cam MP4 (60 FPS, 640x480). Pass --video with no "
             "argument to use the auto-named default in EDMDc/recording/; "
             "pass --video /some/path.mp4 to override.",
    )
    ap.add_argument(
        "--direct-wrench",
        action="store_true",
        help="Skip T200 + allocator; pass APRBS body wrench straight to "
             "Fossen (legacy match to NumPy collect_fossen physics path). "
             "Default is YAML thruster layout + bluerov_demo-style pipeline.",
    )
    ap.add_argument(
        "--neutral-buoyancy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Override Fossen volume so ρgV exactly cancels m·g. Default ON "
             "so drain trajectories reach rest cleanly. Pass --no-neutral-buoyancy "
             "to keep the YAML's actual mass/volume mismatch (terminal sink).",
    )
    ap.add_argument(
        "--gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach the Newton Gripper to base_link via FixedJoint at body "
             "offset (0.148, 0, -0.10) m. Adjusts masses for composite "
             "neutral buoyancy (ROV→13.233 kg, gripper→0.524 kg) and "
             "injects gripper buoyancy (2.52 N up) at the gripper position "
             "each step. Output filename auto-tagged with `gripper_`. "
             "Default ON. Use --no-gripper for bare-ROV collection.",
    )
    ap.add_argument(
        "--infinite-env",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bump the truncation envelope so the ROV can drift without "
             "hitting walls or floor (TRUNCATE_RADIUS=500 m, POOL_Z_HALF=100 m). "
             "Keeps environment.usd loaded for water-surface reference. "
             "Default ON.",
    )
    ap.add_argument(
        "--n-step", type=int, default=0,
        help="Append N step-response trajectories AFTER the APRBS phase. Each "
             "step traj holds a single axis (Fx/Fy/Fz/Tz) at fixed amplitude "
             "for the full episode. Max 24 (4 axes × 3 amps × 2 signs); the "
             "first N are taken from the deterministic spec table. Default 0.",
    )
    return ap


args = _build_argparser().parse_args()

# Resolve "--out __AUTO__" sentinel: auto-tag with "gripper_" when applicable
# and route into the matching data subdir so plots / models stay grouped.
if args.out == "__AUTO__":
    _tag = "norm_gripper" if bool(args.gripper) else "norm"
    args.out = str(
        _project_root() / "EDMDc" / "norm_inputs" / "data"
        / f"{_tag}_{_RUN_TS}.npz"
    )

# Resolve "--video (no path)" sentinel to a default in EDMDc/recording/.
if args.video == "__AUTO__":
    args.video = str(
        _project_root() / "EDMDc" / "recording"
        / f"isaac_seed{args.seed}_{_RUN_TS}.mp4"
    )

# ffmpeg resolution must happen BEFORE SimulationApp() sanitises PATH
# (matches bluerov_demo.py line 113-123). Without this, the post-run
# transcode falls into the "ffmpeg not found" branch and leaves a raw
# mp4v file that may not play in modern browsers / video apps.
FFMPEG_PATH: str | None = None
if args.video:
    import shutil as _shutil
    FFMPEG_PATH = (_shutil.which("ffmpeg")
                   or next((p for p in ("/home/xzha/miniconda3/bin/ffmpeg",
                                         "/usr/bin/ffmpeg",
                                         "/usr/local/bin/ffmpeg")
                            if os.path.exists(p)), None))
    if FFMPEG_PATH is None:
        print("[isaac] WARNING: ffmpeg not found on PATH — MP4 will stay mp4v "
              "and may not play in browsers.")
    else:
        print(f"[isaac] ffmpeg pre-resolved at {FFMPEG_PATH}")

# SimulationApp must be created before any omni.isaac.* import. Mirrors
# bluerov_demo.py / mission_demo.py.
import isaacsim  # noqa: F401
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": bool(args.headless),
    "width": 1280, "height": 720, "renderer": "RayTracedLighting",
})

import carb  # noqa: E402
from pxr import UsdGeom, UsdPhysics, Gf  # noqa: E402

# Disable the persistent grid overlay (cosmetic, not required headless).
_settings = carb.settings.get_settings()
for _key in ("/persistent/app/viewport/grid/enabled",
             "/app/viewport/grid/enabled"):
    try:
        _settings.set(_key, False)
    except Exception:
        pass

from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: E402
from omni.isaac.core.articulations import Articulation  # noqa: E402
from omni.isaac.core.prims import RigidPrimView  # noqa: E402
from omni.isaac.sensor import Camera  # noqa: E402

# Project-root + EDMDc/ on sys.path so this module survives Isaac's PYTHONPATH
# reset. Relative `.foo` imports work when invoked as `python -m EDMDc.collect_isaac`;
# the sys.path entries are a fallback for unusual launchers.
_PROJECT_ROOT = _project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml as _yaml_coll  # noqa: E402

from ..fossen import (  # noqa: E402
    Fossen, FossenParams, RHO_WATER as _RHO_WATER,
    quat_wxyz_to_rotmat, quat_wxyz_to_euler_zyx,
)
# Local linear-mode thruster model (linear=True unit-thrust map: thrust = 25*c).
from .thrusters_norm import (  # noqa: E402
    ThrusterConfig,
    T200Group,
    ThrustAllocator,
    sum_to_wrench,
)
from ..thrust_viz import ThrustVizDrawer  # noqa: E402

from ..collect_fossen import (  # noqa: E402
    CollectionConfig,
    axis_mask_for,
    sample_initial_eta_v,
    trajectory_rng,
)


# ---- constants matching mission_demo / bluerov_demo --------------------
PROJECT_DIR = str(_PROJECT_ROOT)
VEHICLE = "BlueROVHeavy"
ASSET_DIR = os.path.join(PROJECT_DIR, "assets", VEHICLE)
VEHICLE_USD = os.path.join(ASSET_DIR, f"{VEHICLE}.usd")
VEHICLE_YAML = os.path.join(ASSET_DIR, f"{VEHICLE}.yaml")
ENV_USD = os.path.join(PROJECT_DIR, "assets", "environment.usd")
WATER_SURFACE_Z = -0.2

# Truncation envelope (re-exported from collect_fossen.py).
from ..collect_fossen import (  # noqa: E402
    PITCH_LIMIT, POOL_Z_HALF, TRUNCATE_RADIUS_M,
)

# Normalized mode: caps are the unit box (set above); the clip to [-1, 1] is a
# no-op because aprbs_sequence_normalized already samples the unit box. The
# inscribed-box excitation guarantees realizability (cf. collect_isaac).

# --- Newton Gripper constants (active when --gripper) ----------------------
# Body-frame offset from base_link to gripper base_link. Matches teleop_demo.py.
GRIPPER_USD = os.path.join(PROJECT_DIR, "assets", "NewtonGripper", "NewtonGripper.usd")
GRIPPER_ATTACH_OFFSET_BODY = np.array([0.148, 0.0, -0.10], dtype=float)
GRIPPER_MASS_KG = 0.524        # dry mass (kg)
GRIPPER_NET_WEIGHT_IN_WATER_KG = 0.267   # gripper's weight-in-water
# Per-step buoyancy injected at the gripper position (gravity comes from PhysX
# via the FixedJoint composite, buoyancy must be injected explicitly):
GRIPPER_BUOYANCY_N = (GRIPPER_MASS_KG - GRIPPER_NET_WEIGHT_IN_WATER_KG) * 9.81

# Relax tipover envelope when gripper is on: static pitch is ~17 deg from the
# gripper's forward offset COG/COB. 60 deg gives ~15 deg of headroom over the
# worst-observed APRBS dynamic excursion (~45 deg, see debug_journal_gripper.md
# Bug #1) without approaching the cos(pitch) = 0 Euler-rate singularity at 90.
if bool(args.gripper):
    PITCH_LIMIT = float(np.deg2rad(60.0))

# Push envelope walls far enough that random APRBS drift doesn't hit them.
# environment.usd stays loaded for the water-surface reference; we just stop
# truncating on geometry-collision proxies.
if bool(args.infinite_env):
    TRUNCATE_RADIUS_M = 500.0
    POOL_Z_HALF = 100.0


def _euler_to_quat_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX intrinsic Euler → wxyz quaternion. Inverse of
    `quat_wxyz_to_euler_zyx` in mission_target.py."""
    cr = np.cos(roll * 0.5);  sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5); sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5);   sy = np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=float)


def _read_state(base_view) -> tuple[np.ndarray, np.ndarray]:
    """Read PhysX state and pack to (eta_world_12, v_body_6) — same convention
    as the Fossen NumPy integrator's per-step state."""
    pos_w, quat_w = base_view.get_world_poses()
    lin_w = base_view.get_linear_velocities()
    ang_w = base_view.get_angular_velocities()
    pos = np.asarray(pos_w[0], dtype=float).reshape(3)
    quat = np.asarray(quat_w[0], dtype=float).reshape(4)
    lin = np.asarray(lin_w[0], dtype=float).reshape(3)
    ang = np.asarray(ang_w[0], dtype=float).reshape(3)
    roll, pitch, yaw = quat_wxyz_to_euler_zyx(quat)
    eta = np.array([pos[0], pos[1], pos[2], roll, pitch, yaw], dtype=float)
    R = quat_wxyz_to_rotmat(quat)
    v_body = np.empty(6, dtype=float)
    v_body[0:3] = R.T @ lin
    v_body[3:6] = R.T @ ang
    return eta, v_body


def _set_state(base_view, eta: np.ndarray, v_body: np.ndarray) -> None:
    """Teleport the ROV to (eta, v_body) and zero the velocities afterwards
    if v_body specifies world-frame zero. v_body is body-frame; we rotate
    it back to world frame for Isaac's set_velocities API."""
    pos = np.asarray(eta[0:3], dtype=float).reshape(1, 3)
    quat = _euler_to_quat_zyx(float(eta[3]), float(eta[4]), float(eta[5]))
    base_view.set_world_poses(positions=pos, orientations=quat.reshape(1, 4))
    R = quat_wxyz_to_rotmat(quat)
    lin_w = (R @ v_body[0:3]).reshape(1, 3)
    ang_w = (R @ v_body[3:6]).reshape(1, 3)
    velocities = np.concatenate([lin_w, ang_w], axis=1)  # (1, 6)
    base_view.set_velocities(velocities=velocities)


CAM_RESOLUTION = (640, 480)
CHASE_OFFSET_WORLD = np.array([-5.0, 0.0, 2.0])   # 5 m back, 2 m up of spawn


def _look_at_quat_wxyz(cam_pos: np.ndarray, target_pos: np.ndarray,
                       world_up: tuple = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion (w,x,y,z) orienting a USD camera at cam_pos to look at
    target_pos. USD convention: local -Z is forward, local +Y is up.
    Lifted from bluerov_demo._look_at_quat_wxyz."""
    cam_pos = np.asarray(cam_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    up = np.asarray(world_up, dtype=float)
    fwd = target_pos - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(up, z); x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([0.25 / s,
                         (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s])
    if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def _setup_chase_camera(stage):
    """Define a /World/chase_camera prim with translate + orient ops.
    Returns (cam_prim, translate_op) so the caller can update position
    each step. Orientation set once at init (look-at toward world origin)."""
    cam_path = "/World/chase_camera"
    cam_prim = UsdGeom.Camera.Define(stage, cam_path)
    cam_prim.CreateFocalLengthAttr(18.5)
    cam_prim.CreateHorizontalApertureAttr(20.955)
    cam_prim.CreateVerticalApertureAttr(20.955 * CAM_RESOLUTION[1] / CAM_RESOLUTION[0])
    cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
    cam_xform = UsdGeom.Xformable(cam_prim.GetPrim())
    cam_xform.ClearXformOpOrder()
    translate_op = cam_xform.AddTranslateOp()
    orient_op = cam_xform.AddOrientOp()
    chase_start = np.array([0.0, 0.0, -1.0]) + CHASE_OFFSET_WORLD
    chase_quat = _look_at_quat_wxyz(chase_start, np.array([0.0, 0.0, -1.0]))
    translate_op.Set(Gf.Vec3d(float(chase_start[0]),
                              float(chase_start[1]),
                              float(chase_start[2])))
    orient_op.Set(Gf.Quatf(float(chase_quat[0]), float(chase_quat[1]),
                           float(chase_quat[2]), float(chase_quat[3])))
    return cam_path, translate_op


def _in_envelope(x_euler: np.ndarray) -> bool:
    """Truncate when ROV leaves the envelope or tips over in roll/pitch.
    Same checks as collect_fossen._in_envelope."""
    px, py, pz = x_euler[:3]
    horizontal_r = float(np.hypot(px, py))
    return (
        horizontal_r <= TRUNCATE_RADIUS_M
        and abs(pz) <= POOL_Z_HALF
        and abs(x_euler[3]) < PITCH_LIMIT  # roll  phi  (tipover guard)
        and abs(x_euler[4]) < PITCH_LIMIT  # pitch theta (tipover + gimbal-safe)
    )


def main() -> int:
    cfg = CollectionConfig(
        n_trajectories=int(args.n_trajectories),
        episode_seconds=float(args.episode_s),
        dt=float(args.dt),
        seed=int(args.seed),
        mass_kg=float(args.mass_kg),
        yaml_path=VEHICLE_YAML,
        task_dof=4,  # hardcoded: Fx, Fy, Fz, Tz only (matches collect_fossen)
        fx_only=bool(args.fx_only),
    )
    n_steps = int(round(cfg.episode_seconds / cfg.dt))
    mask = axis_mask_for(cfg.task_dof, fx_only=cfg.fx_only)
    print(f"[isaac] task_dof={cfg.task_dof}; APRBS axes active = "
          f"{[bool(m) for m in mask]} (Fx, Fy, Fz, Tx, Ty, Tz)")
    print(f"[isaac] wrench clip +max (Fx..Tz) = {WRENCH_CMD_MAX_POS.tolist()}")
    print(f"[isaac] wrench clip −max (Fx..Tz) = {WRENCH_CMD_MAX_NEG.tolist()}")
    if bool(args.spawn_flat_pool):
        print(
            f"[isaac] spawn-flat-pool ON: world_pose= "
            f"({float(args.spawn_x_world):g}, 0, -1) yaw=0 · v_body=0"
        )
    # --- Fossen plant params (same numbers as the NumPy integrator) ----
    fparams = FossenParams.from_yaml(VEHICLE_YAML)
    if bool(args.neutral_buoyancy):
        fparams.volume = float(cfg.mass_kg) / _RHO_WATER
        print(f"[isaac] [neutral-buoyancy] volume override → "
              f"{fparams.volume:.6f} m³ (buoyancy "
              f"{_RHO_WATER * 9.81 * fparams.volume:.2f} N == "
              f"m·g {cfg.mass_kg * 9.81:.2f} N)")
    fossen = Fossen(
        fparams,
        enable_added_mass=True, added_mass_lp_alpha=0.3,
        water_surface_z=WATER_SURFACE_Z, hull_half_height=0.1,
    )

    use_thrusters = not bool(args.direct_wrench)
    t200 = None
    allocator = None
    thruster_cfg_o = None
    if use_thrusters:
        with open(VEHICLE_YAML) as _yf:
            _vcfg = _yaml_coll.safe_load(_yf)
        thruster_cfg_o = ThrusterConfig.from_yaml_dict(_vcfg)
        t200 = T200Group(
            thruster_cfg_o, linear=True, max_thrust_n=MAX_THRUST_PER_MOTOR_N,
        )
        allocator = ThrustAllocator(
            thruster_cfg_o,
            max_thrust_per_motor=MAX_THRUST_PER_MOTOR_N,
            linear=True,
        )
        print(
            f"[isaac] thruster path (LINEAR unit-thrust): "
            f"{thruster_cfg_o.num_rotors} rotors "
            f"(ThrustAllocator + T200 → sum_to_wrench), "
            f"thrust = {MAX_THRUST_PER_MOTOR_N:.1f}*cmd N (no lag/deadband)"
        )
    else:
        print("[isaac] --direct-wrench: APRBS body wrench applied without "
              "allocator / T200")

    # --- World + scene ----------------------------------------------------
    world = World(stage_units_in_meters=1.0, physics_dt=cfg.dt, backend="numpy")
    add_reference_to_stage(usd_path=ENV_USD, prim_path="/World/Environment")
    add_reference_to_stage(usd_path=VEHICLE_USD, prim_path="/World/Vehicle")

    # --- Optional Newton Gripper attach (Pattern A: FixedJoint composite) ----
    # Gripper USD has no ArticulationRootAPI, so PhysX merges it into the ROV's
    # articulation via the runtime FixedJoint. PhysX then computes the
    # composite COG / inertia automatically; we only have to adjust per-body
    # masses so the composite weight-in-water stays neutral.
    if bool(args.gripper):
        import omni.usd
        add_reference_to_stage(usd_path=GRIPPER_USD, prim_path="/World/Gripper")
        _stage = omni.usd.get_context().get_stage()

        _gripper_base_prim = _stage.GetPrimAtPath("/World/Gripper/base_link")
        UsdPhysics.MassAPI(_gripper_base_prim).GetMassAttr().Set(GRIPPER_MASS_KG)

        # Composite-neutral: cut ROV mass by the gripper's weight-in-water so
        # the total dry mass stays at cfg.mass_kg and the buoyancy + gripper
        # buoyancy injection still balances gravity. See plan §Patches step 6.
        _adjusted_rov_mass = float(cfg.mass_kg) - GRIPPER_NET_WEIGHT_IN_WATER_KG
        _base_prim = _stage.GetPrimAtPath("/World/Vehicle/base_link")
        UsdPhysics.MassAPI(_base_prim).GetMassAttr().Set(_adjusted_rov_mass)

        # Place gripper at body-frame offset from the ROV spawn (level, so
        # body == world). Once the joint is active PhysX maintains rigidity.
        _spawn_world = np.array([0.0, 0.0, -1.0]) + GRIPPER_ATTACH_OFFSET_BODY
        _gxform = UsdGeom.Xformable(_stage.GetPrimAtPath("/World/Gripper"))
        _gxform.ClearXformOpOrder()
        _gxform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in _spawn_world]))
        _gxform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        _joint = UsdPhysics.FixedJoint.Define(_stage, "/World/gripper_attach_joint")
        _joint.CreateBody0Rel().SetTargets(["/World/Vehicle/base_link"])
        _joint.CreateBody1Rel().SetTargets(["/World/Gripper/base_link"])
        _joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(*[float(v) for v in GRIPPER_ATTACH_OFFSET_BODY])
        )
        _joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        _joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        _joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        print(
            f"[isaac] gripper attached: mass={GRIPPER_MASS_KG:.3f} kg, "
            f"ROV mass→{_adjusted_rov_mass:.3f} kg, "
            f"offset={GRIPPER_ATTACH_OFFSET_BODY.tolist()} m, "
            f"buoyancy={GRIPPER_BUOYANCY_N:.3f} N (injected per-step)"
        )

    rov = Articulation(
        prim_path="/World/Vehicle", name=VEHICLE.lower(),
        position=np.array([0.0, 0.0, -1.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    world.scene.add(rov)
    world.reset()
    base_view = RigidPrimView(
        prim_paths_expr="/World/Vehicle/base_link", name="base_link_view",
    )
    world.scene.add(base_view)
    world.reset()

    # Optional MP4 chase-camera recording.
    isaac_cam = None
    chase_translate_op = None
    video_writer = None
    raw_video_path = None
    if args.video:
        import cv2
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        cam_path, chase_translate_op = _setup_chase_camera(stage)
        isaac_cam = Camera(prim_path=cam_path, resolution=CAM_RESOLUTION)
        isaac_cam.initialize()
        # Write raw mp4v first (Isaac's bundled OpenCV ffmpeg silently falls
        # back from libx264/avc1 to mp4v anyway), then transcode to H.264 at
        # close via system ffmpeg. Same pattern as perception.OverlayRenderer.
        out_video = Path(args.video)
        out_video.parent.mkdir(parents=True, exist_ok=True)
        raw_video_path = str(out_video.with_suffix(".raw.mp4"))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            raw_video_path, fourcc, 1.0 / cfg.dt, CAM_RESOLUTION,
        )
        if not video_writer.isOpened():
            print(f"[isaac] WARNING: cv2.VideoWriter failed to open {raw_video_path}; "
                  "video recording disabled.")
            video_writer = None
        else:
            print(f"[isaac] recording chase-cam (raw mp4v) to {raw_video_path} "
                  f"@ {1.0/cfg.dt:.0f} FPS, will transcode to H.264 → {args.video}")

    print(f"[isaac] {VEHICLE} loaded; running {cfg.n_trajectories} × {cfg.episode_seconds:.1f}s "
          f"@ dt={cfg.dt:.4f} (n_steps={n_steps}/traj). seed={cfg.seed}")

    thrust_viz: ThrustVizDrawer | None = None
    if use_thrusters and (not bool(args.headless)):
        thrust_viz = ThrustVizDrawer(
            thruster_cfg_o,
            max_thrust_n=float(MAX_THRUST_PER_MOTOR_N),
        )
        print("[isaac] thrust-vector viz ON (default; per-rotor 3D arrows)")
    elif use_thrusters and bool(args.headless):
        print("[isaac] thrust-vector viz OFF (--headless)")
    elif not use_thrusters:
        print("[isaac] thrust-vector viz OFF (--direct-wrench)")

    X_list: list[np.ndarray] = []
    U_list: list[np.ndarray] = []
    # Realized 6-DOF body wrench (sum_to_wrench output) recorded in lockstep
    # with U_list. With the 4-DOF thruster allocation this carries the
    # parasitic Tx/Ty the commanded U omits; lets a downstream trainer fit a
    # true-vehicle model (input = realized wrench) instead of the composite
    # commanded model. Saved always — it is free (already computed each step).
    U_realized_list: list[np.ndarray] = []
    C_list: list[np.ndarray] = []          # per-thruster commands in [-1, 1]
    Xn_list: list[np.ndarray] = []
    traj_list: list[int] = []
    step_list: list[int] = []
    traj_kind_list: list[int] = []   # 0 = APRBS, 1 = step-response
    n_truncated = 0

    # Step-response trajectories disabled here (the K-step held-out metric uses
    # APRBS only). --n-step is ignored.
    if int(args.n_step) > 0:
        print("[isaac] NOTE: --n-step ignored in norm_inputs mode.")
    n_step_traj = 0
    step_specs = []
    total_n = cfg.n_trajectories

    for traj in range(total_n):
        is_step_traj = traj >= cfg.n_trajectories
        rng = trajectory_rng(cfg.seed, traj)
        if bool(args.spawn_flat_pool):
            sample_initial_eta_v(rng, cfg)
            eta_init = np.asarray(
                [float(args.spawn_x_world), 0.0, -1.0, 0.0, 0.0, 0.0],
                dtype=float,
            )
            v_init = np.zeros(6, dtype=float)
        else:
            eta_init, v_init = sample_initial_eta_v(rng, cfg)
        # Normalized inscribed-box APRBS in [-1,1] on the active axes.
        from ..collect_fossen import APRBS_HOLD_MAX_STEPS as _DFLT_HMAX
        hold_max_arr = np.full(6, int(_DFLT_HMAX), dtype=int)
        if args.hold_max_tz is not None:
            hold_max_arr[5] = int(args.hold_max_tz)
        if args.hold_max_fz is not None:
            hold_max_arr[2] = int(args.hold_max_fz)
        U_seq = aprbs_sequence_normalized(
            rng, n_steps, axis_mask=mask, hold_max_steps=hold_max_arr,
        )

        # Reset to IC + clear Fossen internal state.
        _set_state(base_view, eta_init, v_init)
        fossen.reset()
        if t200 is not None:
            t200.reset()
        if int(args.settle_steps) > 0:
            for _ in range(int(args.settle_steps)):
                base_view.apply_forces_and_torques_at_pos(
                    forces=np.zeros((1, 3)), torques=np.zeros((1, 3)),
                    is_global=True,
                )
                world.step(render=not bool(args.headless))

            if not bool(args.settle_no_reset):
                _set_state(base_view, eta_init, v_init)
                fossen.reset()
                if t200 is not None:
                    t200.reset()

        # Read x_t = state at the IC. Stored state is body velocity only.
        eta, v_body = _read_state(base_view)
        if not _in_envelope(eta):
            n_truncated += 1
            continue

        for k in range(n_steps):
            u_norm = U_seq[k]                           # normalized wrench [-1,1]
            u_newton = denorm_to_newton(u_norm)         # Newtons for the sim
            pos_w, quat_w = base_view.get_world_poses()
            lin_w = base_view.get_linear_velocities()
            ang_w = base_view.get_angular_velocities()
            thrusts_per_rotor: np.ndarray | None = None
            c_cmd = np.zeros(8, dtype=float)
            if allocator is None or t200 is None or thruster_cfg_o is None:
                user_wb = u_newton
            else:
                c_cmd = allocator.allocate(u_newton)    # per-thruster cmds [-1,1]
                thrusts_per_rotor, _ = t200.step(c_cmd, cfg.dt)
                user_wb = sum_to_wrench(thruster_cfg_o, thrusts_per_rotor)
            # U recorded for EDMDc = realized wrench normalized to [-1, 1].
            u_cmd = norm_from_newton(user_wb)

            if thrust_viz is not None and thrusts_per_rotor is not None:
                thrust_viz.update(pos_w[0], quat_w[0], thrusts_per_rotor)

            # Fossen wrench in WORLD frame (buoyancy + damping + Coriolis +
            # added-mass + thrust-realized user wrench, all rotated to world).
            force_w, torque_w = fossen.compute_wrench_world(
                quat_wxyz=np.asarray(quat_w[0], dtype=float),
                lin_vel_world=np.asarray(lin_w[0], dtype=float),
                ang_vel_world=np.asarray(ang_w[0], dtype=float),
                user_wrench_body=user_wb,
                dt=cfg.dt,
                body_z_world=float(pos_w[0][2]),
            )

            # Inject gripper buoyancy at the gripper's world position. Gravity
            # is already provided by PhysX (gripper has its own mass via the
            # composite); we only need the buoyancy half of g(η) here.
            if bool(args.gripper):
                R_body = quat_wxyz_to_rotmat(np.asarray(quat_w[0], dtype=float))
                base_pos_w = np.asarray(pos_w[0], dtype=float)
                gripper_pos_w = base_pos_w + R_body @ GRIPPER_ATTACH_OFFSET_BODY
                extra_force = np.array([0.0, 0.0, GRIPPER_BUOYANCY_N])
                extra_torque = np.cross(gripper_pos_w - base_pos_w, extra_force)
                force_w = force_w + extra_force
                torque_w = torque_w + extra_torque

            base_view.apply_forces_and_torques_at_pos(
                forces=np.asarray(force_w, dtype=float).reshape(1, 3),
                torques=np.asarray(torque_w, dtype=float).reshape(1, 3),
                is_global=True,
            )
            # Force render when --video is set so the chase cam has fresh
            # pixels even under --headless.
            world.step(render=(not bool(args.headless)) or (video_writer is not None))

            # Update chase-cam translation to follow the ROV.
            if chase_translate_op is not None:
                chase_pos = np.asarray(pos_w[0], dtype=float) + CHASE_OFFSET_WORLD
                chase_translate_op.Set(Gf.Vec3d(
                    float(chase_pos[0]), float(chase_pos[1]), float(chase_pos[2]),
                ))
            # Grab a frame for the MP4.
            if video_writer is not None and isaac_cam is not None:
                rgba = isaac_cam.get_rgba()
                if rgba is not None and rgba.size > 0:
                    # cv2 expects BGR.
                    bgr = rgba[..., :3][..., ::-1]
                    try:
                        video_writer.write(np.ascontiguousarray(bgr))
                    except Exception:
                        pass

            eta_next, v_body_next = _read_state(base_view)
            if not _in_envelope(eta_next):
                n_truncated += 1
                print(
                    f"[isaac] traj {traj}: TRUNCATED at step k={k} "
                    f"(active phase), pose=("
                    f"x={eta_next[0]:.2f}, y={eta_next[1]:.2f}, z={eta_next[2]:.2f}, "
                    f"roll={np.rad2deg(eta_next[3]):.1f}°, "
                    f"pitch={np.rad2deg(eta_next[4]):.1f}°, "
                    f"yaw={np.rad2deg(eta_next[5]):.1f}°)"
                )
                break
            if not np.all(np.isfinite(v_body_next)):
                n_truncated += 1
                print(f"[isaac] traj {traj}: TRUNCATED at step k={k} "
                      f"(non-finite v_body): {v_body_next}")
                break

            X_list.append(v_body.copy())
            U_list.append(u_cmd.copy())              # wrench, per-motor units
            U_realized_list.append(u_cmd.copy())     # == U (no saturation)
            C_list.append(np.asarray(c_cmd, dtype=float).copy())  # thruster cmds
            Xn_list.append(v_body_next.copy())
            traj_list.append(traj)
            step_list.append(k)
            traj_kind_list.append(1 if is_step_traj else 0)

            v_body = v_body_next

        # ---- Drain phase: tau = 0 until ‖v‖ < DRAIN_VEL_TOL ----
        zero_wrench = np.zeros(6, dtype=float)
        for d in range(DRAIN_MAX_STEPS):
            pos_w, quat_w = base_view.get_world_poses()
            lin_w = base_view.get_linear_velocities()
            ang_w = base_view.get_angular_velocities()
            c_drain = (allocator.allocate(np.zeros(6)) if allocator is not None
                       else np.zeros(8, dtype=float))   # all thrusters off
            thrusts_per_rotor_drain: np.ndarray | None = None
            if t200 is None or thruster_cfg_o is None:
                user_wb = np.zeros(6, dtype=float)
            else:
                thrusts_per_rotor_drain, _ = t200.step(c_drain, cfg.dt)
                user_wb = sum_to_wrench(thruster_cfg_o, thrusts_per_rotor_drain)

            if thrust_viz is not None and thrusts_per_rotor_drain is not None:
                thrust_viz.update(pos_w[0], quat_w[0], thrusts_per_rotor_drain)

            force_w, torque_w = fossen.compute_wrench_world(
                quat_wxyz=np.asarray(quat_w[0], dtype=float),
                lin_vel_world=np.asarray(lin_w[0], dtype=float),
                ang_vel_world=np.asarray(ang_w[0], dtype=float),
                user_wrench_body=user_wb,
                dt=cfg.dt,
                body_z_world=float(pos_w[0][2]),
            )
            base_view.apply_forces_and_torques_at_pos(
                forces=np.asarray(force_w, dtype=float).reshape(1, 3),
                torques=np.asarray(torque_w, dtype=float).reshape(1, 3),
                is_global=True,
            )
            world.step(render=(not bool(args.headless)) or (video_writer is not None))

            if chase_translate_op is not None:
                chase_pos = np.asarray(pos_w[0], dtype=float) + CHASE_OFFSET_WORLD
                chase_translate_op.Set(Gf.Vec3d(
                    float(chase_pos[0]), float(chase_pos[1]), float(chase_pos[2]),
                ))
            if video_writer is not None and isaac_cam is not None:
                rgba = isaac_cam.get_rgba()
                if rgba is not None and rgba.size > 0:
                    bgr = rgba[..., :3][..., ::-1]
                    try:
                        video_writer.write(np.ascontiguousarray(bgr))
                    except Exception:
                        pass

            eta_next, v_body_next = _read_state(base_view)
            if not _in_envelope(eta_next):
                n_truncated += 1
                break
            if not np.all(np.isfinite(v_body_next)):
                n_truncated += 1
                break

            X_list.append(v_body.copy())
            U_list.append(zero_wrench.copy())                      # 0 wrench
            U_realized_list.append(norm_from_newton(user_wb).copy())
            C_list.append(np.asarray(c_drain, dtype=float).copy())
            Xn_list.append(v_body_next.copy())
            traj_list.append(traj)
            step_list.append(n_steps + d)
            traj_kind_list.append(1 if is_step_traj else 0)

            v_body = v_body_next
            if float(np.linalg.norm(v_body_next)) < DRAIN_VEL_TOL:
                break

        if (traj + 1) % max(1, total_n // 10) == 0:
            print(f"[isaac] traj {traj+1}/{total_n}  "
                  f"snapshots={len(X_list):,}  truncated={n_truncated}")

    X = np.asarray(X_list, dtype=float)
    U = np.asarray(U_list, dtype=float)
    U_realized = np.asarray(U_realized_list, dtype=float)
    C = np.asarray(C_list, dtype=float)
    Xn = np.asarray(Xn_list, dtype=float)
    traj_idx = np.asarray(traj_list, dtype=np.int32)
    step_idx = np.asarray(step_list, dtype=np.int32)
    traj_kind = np.asarray(traj_kind_list, dtype=np.int8)
    n_kept = total_n - n_truncated
    print(f"[isaac] kept {n_kept}/{total_n} trajectories "
          f"({n_truncated} truncated by envelope/tipover); "
          f"{len(X):,} total snapshots "
          f"(seed={cfg.seed}, episode={cfg.episode_seconds:.1f}s, "
          f"APRBS={cfg.n_trajectories}, step={n_step_traj})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, X=X, U=U, U_realized=U_realized, C=C, X_next=Xn,
        traj_idx=traj_idx, step_idx=step_idx,
        traj_kind=traj_kind,
        seed=cfg.seed, dt=cfg.dt,
        episode_seconds=cfg.episode_seconds,
        n_trajectories=cfg.n_trajectories,
        n_step_trajectories=n_step_traj,
        task_dof=cfg.task_dof,
        fx_only=cfg.fx_only,
        use_thrusters=use_thrusters,
        wrench_cmd_max_pos=WRENCH_CMD_MAX_POS,
        wrench_cmd_max_neg=WRENCH_CMD_MAX_NEG,
        max_thrust_per_motor=MAX_THRUST_PER_MOTOR_N,
        # --- normalized [-1,1] / inscribed-box metadata (norm_inputs) ---
        # U, U_realized are the 6-DOF wrench normalized per axis to [-1, 1]
        # (C = per-thruster commands in [-1,1]). Recover Newtons via:
        #   U_newton = symmetric_caps * U.
        input_units="normalized_[-1,1]",
        symmetric_caps=SYMMETRIC_CAPS,
        per_motor_thrust_n=np.float64(MAX_THRUST_PER_MOTOR_N),
        linear_unit_thrust=True,
        thrust_viz=bool(thrust_viz is not None),
        spawn_flat_pool=bool(args.spawn_flat_pool),
        spawn_x_world=np.float64(args.spawn_x_world),
        drain_vel_tol=np.float64(DRAIN_VEL_TOL),
        drain_max_steps=np.int32(DRAIN_MAX_STEPS),
    )
    sz_mb = out_path.stat().st_size / 1e6
    print(f"[isaac] wrote {out_path}  ({sz_mb:.2f} MB)")

    # Finalize video writer + H.264 transcode.
    if video_writer is not None:
        video_writer.release()
        if raw_video_path and FFMPEG_PATH and Path(raw_video_path).exists():
            import subprocess
            ok = False
            for encoder in ("libx264", "libopenh264"):
                cmd = [FFMPEG_PATH, "-y", "-loglevel", "error",
                       "-i", raw_video_path,
                       "-c:v", encoder, "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", str(args.video)]
                try:
                    r = subprocess.run(cmd, capture_output=True)
                    if r.returncode == 0:
                        ok = True
                        print(f"[isaac] transcoded → {args.video} ({encoder})")
                        Path(raw_video_path).unlink(missing_ok=True)
                        break
                except Exception:
                    pass
            if not ok:
                print(f"[isaac] H.264 transcode failed; raw mp4v left at {raw_video_path}")
        elif raw_video_path:
            print(f"[isaac] no ffmpeg; raw mp4v left at {raw_video_path}")

    if thrust_viz is not None:
        thrust_viz.close()

    simulation_app.close()

    # Auto-plot APRBS input distribution after Kit is fully shut down (avoids
    # any matplotlib-vs-Kit GUI conflict).
    try:
        from ..plot_inputs import plot_distribution
        plot_distribution([out_path], labels=[out_path.stem])
    except Exception as e:
        print(f"[isaac] WARNING: input-distribution plot failed ({e}); "
              f"run `python -m EDMDc.plot_inputs {out_path}` manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
