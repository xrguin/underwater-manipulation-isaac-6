"""Shared Isaac Sim scaffolding for the EDMDc gripper experiments.

The four scripts ``mpc_isaac_gripper_38``, ``open_loop_inputs_gripper_38``,
``lqr_isaac_gripper_38`` / ``mpc_velocity_eval_38`` and
``cascade_nav_38`` / ``cascade_nav_38_lqr`` each carry ~200 lines of
identical Isaac/Fossen/thruster/chase-cam boilerplate. This module
factors that out so each script can focus on what it actually does
(control law, reference signal, logging).

Three classes:

  * ``GripperScene`` — boots Isaac, builds the BlueROV-Heavy + Newton
    Gripper composite, owns Fossen + T200 + ArduSub allocator. Exposes
    ``read_state()``, ``apply_wrench(u_cmd4)``, ``settle(...)``,
    ``reset_to_spawn()``, ``close()``.

  * ``ChaseCam``  — chase camera at a fixed world-frame offset plus
    optional H.264 (libopenh264) / cv2 mp4v video recording. The video
    pipe can be started and stopped independently of scene lifetime so
    multi-scenario eval scripts can rotate output files.

  * ``FPVCam``    — body-frame onboard camera (forward-facing).

Plus free helpers ``look_at_quat_wxyz``, ``quat_mul`` and the module-level
``resolve_ffmpeg()`` (must be called BEFORE ``SimulationApp()`` because
Isaac sanitises PATH at boot).

Usage skeleton::

    from EDMDc.isaac_scene import GripperScene, ChaseCam, resolve_ffmpeg

    FFMPEG_PATH = resolve_ffmpeg()        # before any Isaac import
    scene = GripperScene(dt=1/60, mass_kg=13.5, headless=False)
    chase = ChaseCam(scene, ffmpeg_path=FFMPEG_PATH)
    chase.start_recording(Path("run.mp4"))
    scene.settle(max_s=5.0)

    for k in range(N):
        s = scene.read_state()
        u = controller.step(s.nu, s.roll, s.pitch, ref)
        scene.apply_wrench(u)
        chase.follow_and_record()

    chase.close(); scene.close()
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


# ============================================================================
# Constants — match the values previously scattered across scripts
# ============================================================================

GRIPPER_OFFSET     = np.array([0.148, 0.0, -0.10], dtype=float)
GRIPPER_MASS_KG    = 0.524
GRIPPER_NET_WET_KG = 0.267
GRIPPER_BUOY_N     = (GRIPPER_MASS_KG - GRIPPER_NET_WET_KG) * 9.81

# --- 1 kg payload cube (opt-in via GripperScene(payload_cube=True)) ---------
# A 1 kg negatively-buoyant cube welded to the gripper jaws (second FixedJoint).
# Matches collect_isaac_cube.py / teleop_demo_cube.py: the asset ships as a
# 0.1 m / 2 kg cube with explicit (2 kg) inertia, so we override mass to 1 kg,
# clear the baked inertia/COM (PhysX recomputes), and rescale to a 0.08 m edge
# so 1 kg is genuinely negatively buoyant. Buoyancy (~5.01 N up) is injected
# per-tick; the ROV mass is NOT compensated, so the composite sinks (net ~4.8 N
# down) — the intended payload disturbance.
CUBE_REL_PATH       = ("assets", "PickupCube", "PickupCube.usd")
CUBE_MASS_KG        = 1.0
CUBE_SIDE_M         = 0.08                              # rescaled edge (density ~1953 kg/m³)
CUBE_RENDER_SCALE   = 0.05 * (CUBE_SIDE_M / 0.10)       # -> 0.04 (ship renders 0.1 m at scale 0.05)
CUBE_OFFSET_FROM_GRIPPER = np.array([0.1, 0.0, -0.0], dtype=float)  # at the jaws
CUBE_BUOY_N         = 997.0 * 9.81 * (CUBE_SIDE_M ** 3)  # ≈ 5.01 N up (997 = fossen.RHO_WATER)
CUBE_OFFSET         = GRIPPER_OFFSET + CUBE_OFFSET_FROM_GRIPPER  # cube center from base_link

DEFAULT_SPAWN = np.array([0.0, 0.0, -0.2], dtype=float)

# Centered trim (sim analog of the physical foam re-trim after mounting
# the gripper): put the composite COG and the effective COB on the
# vehicle's vertical center axis (x = y = 0), keeping their heights so
# the metacentric separation (BM ~ 11.8 mm) and passive roll/pitch
# stability are unchanged. Two closed-form pieces (see show_cog_cob):
#   * an authored base-link COM cancels the gripper's horizontal MASS
#     moment      -> composite COG at x=y=0;
#   * a hull-COB x/y trim cancels the gripper's horizontal BUOYANCY
#     moment      -> effective COB at x=y=0.
# Untrimmed, the misalignment gives a +13.7 deg passive pitch equilibrium.


# ============================================================================
# Module-level helpers (pure math, no Isaac dependency)
# ============================================================================

def look_at_quat_wxyz(cam_pos: np.ndarray, target_pos: np.ndarray,
                      world_up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
                      ) -> np.ndarray:
    """Quaternion (w, x, y, z) orienting a USD camera at ``cam_pos`` to
    look at ``target_pos``. USD convention: local -Z is forward, local
    +Y is up. Lifted unchanged from the duplicated copies across scripts.
    """
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


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def resolve_ffmpeg() -> Optional[str]:
    """Locate the ``ffmpeg`` binary BEFORE ``SimulationApp()`` is called.

    Isaac Sim sanitises ``PATH`` at boot, so ``shutil.which("ffmpeg")``
    inside the post-boot environment may fail. Callers should invoke this
    at the top of their script (after CLI parsing, before any
    ``import isaacsim``) and pass the result into ``ChaseCam``.
    """
    import shutil
    return (shutil.which("ffmpeg")
            or next((p for p in ("/home/xzha/miniconda3/bin/ffmpeg",
                                 "/usr/bin/ffmpeg",
                                 "/usr/local/bin/ffmpeg")
                     if os.path.exists(p)), None))


# ============================================================================
# State dataclass — returned by GripperScene.read_state()
# ============================================================================

@dataclass
class SceneState:
    """Snapshot of the BlueROV-Heavy state at one tick.

    Attributes
    ----------
    pos_w : (3,) ndarray
        World-frame COG position.
    quat_wxyz : (4,) ndarray
        World-frame orientation as (w, x, y, z).
    R_body : (3, 3) ndarray
        Body-to-world rotation matrix.
    roll, pitch, yaw : float
        Euler angles (ZYX convention) in radians.
    nu : (6,) ndarray
        Body-frame velocity ``[u, v, w, p, q, r]``.
    lin_w : (3,) ndarray
        World-frame linear velocity.
    ang_w : (3,) ndarray
        World-frame angular velocity.
    """
    pos_w:     np.ndarray
    quat_wxyz: np.ndarray
    R_body:    np.ndarray
    roll:      float
    pitch:     float
    yaw:       float
    nu:        np.ndarray
    lin_w:     np.ndarray
    ang_w:     np.ndarray


# ============================================================================
# GripperScene — owns the entire Isaac/Fossen/thrusters pipeline
# ============================================================================

class GripperScene:
    """BlueROV-Heavy + Newton Gripper composite scene with Fossen
    hydrodynamics and T200 thrusters.

    Constructor boots Isaac, builds the stage, attaches the gripper via
    UsdPhysics.FixedJoint with composite-neutral mass adjustment, and
    constructs the Fossen + thruster pipeline. After construction the
    body sits at ``spawn_pos`` with attitude (0, 0, 0); the body will
    naturally drift to its equilibrium pitch (~17.7°) under zero wrench
    — call :meth:`settle` to bring it to rest before recording.

    Parameters
    ----------
    dt : float
        Physics time step (seconds).
    mass_kg : float, default 13.5
        BlueROV-Heavy mass. The gripper's net wet weight is subtracted
        internally so the composite is neutral-buoyant when
        ``neutral_buoyancy=True``.
    headless : bool, default False
        Disable Isaac GUI. Recording still works (``ChaseCam`` flips a
        force-render flag).
    neutral_buoyancy : bool, default True
        Override Fossen's displacement so ρgV = mg.
    retrim_level : bool, default True
        Apply the centered trim: composite COG and effective COB both on
        the vertical center axis (x=y=0, heights unchanged, BM intact),
        so the gripper composite floats level like the re-trimmed
        physical vehicle. ``False`` restores the un-trimmed configuration
        (passive pitch equilibrium ~13.7 deg) that the pre-trim
        datasets/models assume. The trim targets the bare gripper
        composite — a payload cube's moment is intentionally NOT
        trimmed away.
    payload_cube : bool, default False
        Weld a 1 kg negatively-buoyant cube to the gripper jaws (second
        FixedJoint). The cube's mass/inertia/COG are real in PhysX and its
        buoyancy is injected per-tick by :meth:`apply_wrench`, giving a net
        ~4.8 N down disturbance (the ROV mass is NOT compensated). See the
        ``CUBE_*`` module constants.
    spawn_pos : array-like, default (0, 0, -0.2)
        World-frame spawn position. The default drops the vehicle in at
        the water surface (WATER_SURFACE_Z = -0.2) like a real tank
        deployment; the neutral-buoyant body sinks to full submersion
        (center ~ -0.3) as the surface buoyancy taper fades. Fits both
        the tank env (floor -0.8) and the legacy pools.
    vehicle : str, default "BlueROVHeavy"
        Vehicle name used to locate ``assets/<vehicle>/<vehicle>.usd``
        and ``.yaml``.
    env_usd : str | None, default None
        Path to the environment USD. ``None`` resolves to
        ``assets/environment.usd``.
    """

    def __init__(self, *,
                 dt: float,
                 mass_kg: float = 13.5,
                 headless: bool = False,
                 neutral_buoyancy: bool = True,
                 retrim_level: bool = True,
                 payload_cube: bool = False,
                 cube_mass_kg: float = CUBE_MASS_KG,
                 cube_buoy_N: float | None = None,
                 spawn_pos: np.ndarray = DEFAULT_SPAWN,
                 vehicle: str = "BlueROVHeavy",
                 env_usd: Optional[str] = None,
                 width: int = 1280,
                 height: int = 720):
        self._dt = float(dt)
        self._headless = bool(headless)
        self._spawn = np.asarray(spawn_pos, dtype=float).copy()
        self._has_cube = bool(payload_cube)
        self._cube_mass = float(cube_mass_kg)
        # Cube buoyancy: default = volume-based (negatively buoyant payload).
        # Override (e.g. = mass*g) to make the cube neutrally buoyant, which
        # isolates the cube's mass/inertia change from its net force/torque.
        self._cube_buoy_N = (CUBE_BUOY_N if cube_buoy_N is None
                             else float(cube_buoy_N))
        self._force_render = False     # ChaseCam sets True while recording

        # ----- Boot Isaac --------------------------------------------------
        import isaacsim  # noqa: F401
        from isaacsim import SimulationApp
        self._sim = SimulationApp({
            "headless": self._headless,
            "width": width, "height": height,
            "renderer": "RayTracedLighting",
        })

        # ----- Isaac/USD imports (only available after SimulationApp()) ----
        import carb  # noqa: E402
        from pxr import UsdGeom, UsdPhysics, Gf  # noqa: E402
        import omni.usd  # noqa: E402
        from omni.isaac.core import World  # noqa: E402
        from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: E402
        from omni.isaac.core.articulations import Articulation  # noqa: E402
        from omni.isaac.core.prims import RigidPrimView  # noqa: E402

        # Stash module refs that ChaseCam / FPVCam will reuse.
        self._UsdGeom = UsdGeom
        self._UsdPhysics = UsdPhysics
        self._Gf = Gf
        self._omni_usd = omni_usd_mod = omni.usd

        # Disable viewport grid (cleaner recordings).
        _settings = carb.settings.get_settings()
        for _key in ("/persistent/app/viewport/grid/enabled",
                     "/app/viewport/grid/enabled"):
            try: _settings.set(_key, False)
            except Exception: pass

        # ----- Project-local imports (Fossen, thrusters, helpers) ----------
        PROJECT = Path(__file__).resolve().parents[1]
        if str(PROJECT) not in sys.path:
            sys.path.insert(0, str(PROJECT))
        import yaml as _yaml  # noqa: E402
        from .fossen import (  # noqa: E402
            Fossen, FossenParams, RHO_WATER as _RHO_WATER,
            GRAVITY as _GRAVITY,
            quat_wxyz_to_rotmat, quat_wxyz_to_euler_zyx,
        )
        from .thrusters import (  # noqa: E402
            ThrusterConfig, T200Group, sum_to_wrench,
        )
        from .ardusub_allocator import (  # noqa: E402
            COMMAND_DIM, StaticArduSubAllocator, command4_from_wrench6,
        )
        # Stash for use in apply_wrench / read_state.
        self._quat_to_rotmat = quat_wxyz_to_rotmat
        self._quat_to_euler  = quat_wxyz_to_euler_zyx
        self._sum_to_wrench  = sum_to_wrench
        self._command_dim = COMMAND_DIM
        self._command4_from_wrench6 = command4_from_wrench6

        # ----- Asset paths -------------------------------------------------
        ASSET_DIR    = PROJECT / "assets" / vehicle
        VEHICLE_USD  = str(ASSET_DIR / f"{vehicle}.usd")
        VEHICLE_YAML = str(ASSET_DIR / f"{vehicle}.yaml")
        ENV_USD      = env_usd or str(PROJECT / "assets" / "environment.usd")
        GRIPPER_USD  = str(PROJECT / "assets" / "NewtonGripper" / "NewtonGripper.usd")

        # ----- Fossen + thrusters -----------------------------------------
        fparams = FossenParams.from_yaml(VEHICLE_YAML)
        if neutral_buoyancy:
            fparams.volume = float(mass_kg) / _RHO_WATER
        if retrim_level:
            B_hull = _RHO_WATER * _GRAVITY * fparams.volume
            fparams.cob_x = -GRIPPER_BUOY_N * GRIPPER_OFFSET[0] / B_hull
            fparams.cob_y = -GRIPPER_BUOY_N * GRIPPER_OFFSET[1] / B_hull
            print(f"[scene] centered trim: hull COB ({fparams.cob_x*1000:+.2f}, "
                  f"{fparams.cob_y*1000:+.2f}, {fparams.cob_offset*1000:+.1f}) mm "
                  f"(COG/COB on center axis; retrim_level=False for legacy)")
        self._fossen = Fossen(fparams,
                              enable_added_mass=True,
                              added_mass_lp_alpha=0.3,
                              water_surface_z=-0.2,
                              hull_half_height=0.1)
        with open(VEHICLE_YAML) as _yf:
            _vcfg = _yaml.safe_load(_yf)
        self._tcfg = ThrusterConfig.from_yaml_dict(_vcfg)
        self._t200 = T200Group(self._tcfg)
        self._allocator = StaticArduSubAllocator()

        # ----- World + stage ----------------------------------------------
        self._world = World(stage_units_in_meters=1.0,
                            physics_dt=self._dt, backend="numpy")
        add_reference_to_stage(usd_path=ENV_USD, prim_path="/World/Environment")
        add_reference_to_stage(usd_path=VEHICLE_USD, prim_path="/World/Vehicle")
        add_reference_to_stage(usd_path=GRIPPER_USD, prim_path="/World/Gripper")
        self._stage = omni_usd_mod.get_context().get_stage()

        # Hide the decorative target cube baked onto the pool floor by
        # environment.usd (/Environment/Target/Cube) — a static marker, not used
        # by physics or logic. Deactivating removes it from render + collision.
        _target_prim = self._stage.GetPrimAtPath("/World/Environment/Target")
        if _target_prim.IsValid():
            _target_prim.SetActive(False)

        # Composite-neutral mass adjustment.
        UsdPhysics.MassAPI(self._stage.GetPrimAtPath("/World/Gripper/base_link"))\
            .GetMassAttr().Set(GRIPPER_MASS_KG)
        adjusted_rov_mass = float(mass_kg) - GRIPPER_NET_WET_KG
        UsdPhysics.MassAPI(self._stage.GetPrimAtPath("/World/Vehicle/base_link"))\
            .GetMassAttr().Set(adjusted_rov_mass)
        if retrim_level:
            # Authored base-link COM cancels the gripper's horizontal mass
            # moment, putting the composite COG on the center axis (the z
            # moment is left alone so COG/COB heights — and BM — stay put).
            # NOTE: apply_wrench applies Fossen forces at the base origin
            # explicitly, so an off-origin COM stays consistent with the
            # torque bookkeeping.
            com_trim = -GRIPPER_MASS_KG * GRIPPER_OFFSET / adjusted_rov_mass
            UsdPhysics.MassAPI(self._stage.GetPrimAtPath("/World/Vehicle/base_link"))\
                .CreateCenterOfMassAttr().Set(
                    Gf.Vec3f(float(com_trim[0]), float(com_trim[1]), 0.0))
            print(f"[scene] centered trim: base COM authored at "
                  f"({com_trim[0]*1000:+.2f}, {com_trim[1]*1000:+.2f}, +0.00) mm")

        # Position the gripper at the offset before the FixedJoint snaps it.
        _gxform = UsdGeom.Xformable(self._stage.GetPrimAtPath("/World/Gripper"))
        _gxform.ClearXformOpOrder()
        _gxform.AddTranslateOp().Set(
            Gf.Vec3d(*(self._spawn + GRIPPER_OFFSET).tolist()))
        _gxform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        # FixedJoint: weld gripper at the body offset.
        _joint = UsdPhysics.FixedJoint.Define(self._stage,
                                              "/World/gripper_attach_joint")
        _joint.CreateBody0Rel().SetTargets(["/World/Vehicle/base_link"])
        _joint.CreateBody1Rel().SetTargets(["/World/Gripper/base_link"])
        _joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*GRIPPER_OFFSET.tolist()))
        _joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        _joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        _joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        # --- Optional 1 kg payload cube: weld to the gripper jaws -----------
        # PickupCube.usd has no ArticulationRootAPI, so PhysX folds it into the
        # composite. Mass overridden 2 kg -> 1 kg, baked inertia/COM cleared
        # (PhysX recomputes), visual+collider rescaled to a 0.08 m edge. NO
        # ROV-mass compensation — the cube's net ~4.8 N down is the payload.
        if self._has_cube:
            CUBE_USD = str(PROJECT.joinpath(*CUBE_REL_PATH))
            add_reference_to_stage(usd_path=CUBE_USD, prim_path="/World/Cube")
            _cube_base = self._stage.GetPrimAtPath("/World/Cube/base_link")
            _cube_mass_api = UsdPhysics.MassAPI(_cube_base)
            _cube_mass_api.GetMassAttr().Set(self._cube_mass)
            if _cube_mass_api.GetDiagonalInertiaAttr().HasAuthoredValue():
                _cube_mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            if _cube_mass_api.GetCenterOfMassAttr().HasAuthoredValue():
                _cube_mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            for _sub in ("visuals/cube", "collider"):
                _cgprim = self._stage.GetPrimAtPath(f"/World/Cube/base_link/{_sub}")
                for _op in UsdGeom.Xformable(_cgprim).GetOrderedXformOps():
                    if _op.GetOpName() == "xformOp:scale":
                        _op.Set(Gf.Vec3f(CUBE_RENDER_SCALE,
                                         CUBE_RENDER_SCALE, CUBE_RENDER_SCALE))
            _cxform = UsdGeom.Xformable(self._stage.GetPrimAtPath("/World/Cube"))
            _cxform.ClearXformOpOrder()
            _cxform.AddTranslateOp().Set(
                Gf.Vec3d(*(self._spawn + CUBE_OFFSET).tolist()))
            _cxform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            _cjoint = UsdPhysics.FixedJoint.Define(self._stage,
                                                   "/World/cube_attach_joint")
            _cjoint.CreateBody0Rel().SetTargets(["/World/Gripper/base_link"])
            _cjoint.CreateBody1Rel().SetTargets(["/World/Cube/base_link"])
            _cjoint.CreateLocalPos0Attr().Set(
                Gf.Vec3f(*CUBE_OFFSET_FROM_GRIPPER.tolist()))
            _cjoint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            _cjoint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            _cjoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        # Articulation + RigidPrimView (two world.reset() per Isaac quirk).
        self._rov = Articulation(prim_path="/World/Vehicle",
                                 name=vehicle.lower(),
                                 position=self._spawn.copy(),
                                 orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        self._world.scene.add(self._rov)
        self._world.reset()
        self._base_view = RigidPrimView(prim_paths_expr="/World/Vehicle/base_link",
                                        name="base_link_view")
        self._world.scene.add(self._base_view)
        self._world.reset()

        print(f"[scene] {vehicle} + gripper attached "
              f"(mass {adjusted_rov_mass:.3f} + {GRIPPER_MASS_KG:.3f} kg, "
              f"buoy {GRIPPER_BUOY_N:.3f} N)")
        if self._has_cube:
            print(f"[scene] payload cube welded to gripper: "
                  f"mass {self._cube_mass:.3f} kg, edge {CUBE_SIDE_M:.3f} m, "
                  f"buoy {self._cube_buoy_N:.2f} N up -> net "
                  f"{self._cube_mass*9.81 - self._cube_buoy_N:+.2f} N (down), "
                  f"offset {CUBE_OFFSET.tolist()} m from base_link")

    # ------------------------------------------------------------------
    # Public properties — for callers that need direct Isaac handles
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float: return self._dt

    @property
    def headless(self) -> bool: return self._headless

    @property
    def world(self):                     return self._world
    @property
    def stage(self):                     return self._stage
    @property
    def base_view(self):                 return self._base_view
    @property
    def thruster_config(self):           return self._tcfg
    @property
    def num_rotors(self) -> int:         return self._tcfg.num_rotors
    @property
    def simulation_app(self):            return self._sim
    @property
    def ardusub_authority(self) -> np.ndarray:
        return self._allocator.authority.copy()

    @property
    def force_render(self) -> bool: return self._force_render

    @force_render.setter
    def force_render(self, value: bool) -> None:
        """Set by ChaseCam during start_recording / stop_recording so
        physics still renders even in headless mode."""
        self._force_render = bool(value)

    # ------------------------------------------------------------------
    # Visual decoration: checkerboard floor grid for motion cues
    # ------------------------------------------------------------------

    def add_floor_grid(self, *,
                       floor_z: float = -1.95,
                       span: float = 20.0,
                       tile: float = 0.5,
                       colors: Tuple[Tuple[float, float, float],
                                     Tuple[float, float, float]] =
                           ((0.9, 0.9, 0.9), (0.20, 0.20, 0.20)),
                       z_thickness: float = 0.01) -> None:
        """Lay down a checkerboard of thin coloured tiles on the pool floor
        for motion visualisation. Visual-only — no collision, no mass.

        With the defaults the body sees ~0.8 tile-crossings per second at
        cruise speed (0.4 m/s ÷ 0.5 m tile), which is a clear but not
        strobey motion cue. The tiles are positioned 5 cm above the
        nominal floor at z = -2 m so they don't z-fight the pool USD.

        Parameters
        ----------
        floor_z : float
            World z of the tile centre. Default −1.95 (5 cm above the
            shallow-pool floor). Override for the deep-pool environment.
        span : float
            Extent of the grid (metres) in both x and y. 20 m covers the
            cascade-nav waypoint envelope plus margin.
        tile : float
            Edge length of each tile in metres.
        colors : 2-tuple of (r, g, b)
            Diffuse-display colour of the alternating tiles.
        z_thickness : float
            Cube height in metres. Keep small (≈ 1 cm) so the tile reads
            as flat but stays out of z-fight territory.
        """
        UsdGeom = self._UsdGeom
        Gf = self._Gf
        n_half = max(1, int(span / tile / 2))
        grid_root = "/World/FloorGrid"
        UsdGeom.Xform.Define(self._stage, grid_root)

        # UsdGeom.Cube has a built-in edge-length attribute (default 2 m).
        # We scale a unit cube (size = 1 m) by (tile, tile, z_thickness) to
        # get the thin tile shape — cleaner than mucking with the cube's
        # size attribute per-prim.
        n_tiles = 0
        for i in range(-n_half, n_half):
            for j in range(-n_half, n_half):
                color = colors[(i + j) % 2]
                path = f"{grid_root}/tile_{i + n_half}_{j + n_half}"
                cube = UsdGeom.Cube.Define(self._stage, path)
                cube.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(cube.GetPrim())
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(
                    (i + 0.5) * tile, (j + 0.5) * tile, floor_z))
                xf.AddScaleOp().Set(Gf.Vec3f(tile, tile, z_thickness))
                UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr(
                    [Gf.Vec3f(*color)])
                n_tiles += 1

        print(f"[scene] floor grid: {n_tiles} tiles, "
              f"{tile:g} m spacing, span ±{n_half*tile:g} m, "
              f"z = {floor_z:g} m")

    # ------------------------------------------------------------------
    # State read
    # ------------------------------------------------------------------

    def read_state(self) -> SceneState:
        pos_w, quat_w = self._base_view.get_world_poses()
        lin_w = self._base_view.get_linear_velocities()
        ang_w = self._base_view.get_angular_velocities()
        q = np.asarray(quat_w[0], dtype=float)
        R = self._quat_to_rotmat(q)
        lin = np.asarray(lin_w[0], dtype=float)
        ang = np.asarray(ang_w[0], dtype=float)
        nu = np.empty(6, dtype=np.float64)
        nu[0:3] = R.T @ lin
        nu[3:6] = R.T @ ang
        roll, pitch, yaw = self._quat_to_euler(q)
        return SceneState(
            pos_w=np.asarray(pos_w[0], dtype=float),
            quat_wxyz=q,
            R_body=R,
            roll=float(roll), pitch=float(pitch), yaw=float(yaw),
            nu=nu, lin_w=lin, ang_w=ang,
        )

    # ------------------------------------------------------------------
    # Per-tick physics: allocator -> t200 -> fossen -> gripper buoyancy
    # -> apply -> world.step
    # ------------------------------------------------------------------

    def apply_wrench(self, u_cmd: np.ndarray,
                     *, render: Optional[bool] = None) -> np.ndarray:
        """Push one ArduSub command through the full pipeline
        and step physics one tick.

        Parameters
        ----------
        u_cmd : (4,) array
            Normalized ArduSub command ``[surge, sway, heave, yaw]`` in
            ``[-1,1]``.  A legacy 6-D wrench is accepted and converted through
            the calibration authority for compatibility.
        render : bool | None
            Force the render flag passed to ``world.step``. ``None``
            (default) uses ``not headless OR force_render``.

        Returns
        -------
        thrusts : (num_rotors,) ndarray
            Realised per-rotor thrust (post-T200 first-order lag).
        """
        u_arr = np.asarray(u_cmd, dtype=float).reshape(-1)
        if u_arr.size == 6:
            u_arr = self._command4_from_wrench6(u_arr)
        elif u_arr.size != self._command_dim:
            raise ValueError(
                f"apply_wrench expects 4-D ArduSub command or legacy 6-D wrench; "
                f"got shape {np.asarray(u_cmd).shape}"
            )
        commands = self._allocator.allocate(u_arr)
        thrusts, _ = self._t200.step(commands, self._dt)
        user_wb = self._sum_to_wrench(self._tcfg, thrusts)

        pos_w, quat_w = self._base_view.get_world_poses()
        lin_w = self._base_view.get_linear_velocities()
        ang_w = self._base_view.get_angular_velocities()

        force_w, torque_w = self._fossen.compute_wrench_world(
            quat_wxyz=np.asarray(quat_w[0], dtype=float),
            lin_vel_world=np.asarray(lin_w[0], dtype=float),
            ang_vel_world=np.asarray(ang_w[0], dtype=float),
            user_wrench_body=user_wb,
            dt=self._dt,
            body_z_world=float(pos_w[0][2]),
        )

        # Gripper buoyancy injection at the gripper's world position.
        R_body = self._quat_to_rotmat(np.asarray(quat_w[0], dtype=float))
        base_pos_w = np.asarray(pos_w[0], dtype=float)
        gripper_pos_w = base_pos_w + R_body @ GRIPPER_OFFSET
        extra_F = np.array([0.0, 0.0, GRIPPER_BUOY_N])
        extra_T = np.cross(gripper_pos_w - base_pos_w, extra_F)
        force_w  = force_w  + extra_F
        torque_w = torque_w + extra_T

        # Payload cube buoyancy at the cube's world position (PhysX provides
        # its gravity via the composite; net mass*g - buoy is the payload).
        if self._has_cube:
            cube_pos_w = base_pos_w + R_body @ CUBE_OFFSET
            cube_F = np.array([0.0, 0.0, self._cube_buoy_N])
            cube_T = np.cross(cube_pos_w - base_pos_w, cube_F)
            force_w  = force_w  + cube_F
            torque_w = torque_w + cube_T

        # Apply at the base-link ORIGIN explicitly: every torque above is
        # computed about that point. Without `positions`, PhysX applies the
        # force at the link COM instead — correct only while COM == origin
        # (true for the stock asset, silently wrong once a COM is authored,
        # e.g. by show_cog_cob --centered).
        self._base_view.apply_forces_and_torques_at_pos(
            forces=np.asarray(force_w, dtype=float).reshape(1, 3),
            torques=np.asarray(torque_w, dtype=float).reshape(1, 3),
            positions=np.asarray(base_pos_w, dtype=float).reshape(1, 3),
            is_global=True,
        )
        if render is None:
            render = (not self._headless) or self._force_render
        self._world.step(render=render)
        return thrusts

    # ------------------------------------------------------------------
    # Settle helper
    # ------------------------------------------------------------------

    def settle(self, *, max_s: float = 5.0, min_s: float = 4.0,
               tol: float = 0.02,
               on_step: Optional[Any] = None) -> SceneState:
        """Run zero-wrench ticks until ``|nu| < tol`` AND ``min_s`` elapsed,
        or ``max_s`` seconds reached.

        Parameters
        ----------
        max_s, min_s : float
            Time bounds. ``max_s=5, min_s=4`` are good defaults for the
            gripper-attached BlueROV-Heavy (~2 s damping time).
        tol : float
            Magnitude threshold on the body-frame ν vector (mixed units —
            m/s and rad/s — but conservative at 0.02).
        on_step : callable | None
            Optional ``f(scene)`` invoked each tick (e.g. to follow the
            chase cam through the settle so the GUI doesn't look frozen).

        Returns
        -------
        state : SceneState
            Snapshot at the moment settling completes.
        """
        n_max = int(round(float(max_s) / self._dt))
        n_min = int(round(float(min_s) / self._dt))
        n_done = 0
        zeros6 = np.zeros(6, dtype=np.float64)
        for _s in range(n_max):
            self.apply_wrench(zeros6)
            if on_step is not None:
                on_step(self)
            n_done = _s + 1
            if n_done >= n_min:
                state = self.read_state()
                if float(np.linalg.norm(state.nu)) < float(tol):
                    print(f"[scene] settled in {n_done} steps "
                          f"({n_done*self._dt:.2f} s): "
                          f"|nu|={np.linalg.norm(state.nu):.4f}, "
                          f"roll={np.rad2deg(state.roll):+.2f} deg, "
                          f"pitch={np.rad2deg(state.pitch):+.2f} deg")
                    return state
        state = self.read_state()
        print(f"[scene] settle timeout at {n_done} steps "
              f"({n_done*self._dt:.2f} s): |nu|={np.linalg.norm(state.nu):.4f}")
        return state

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_to_spawn(self) -> None:
        """Teleport the body back to ``spawn_pos`` with attitude (1,0,0,0)
        and zero velocity. Resets Fossen + T200 internal state too. Used
        between trials in eval scripts."""
        self._base_view.set_world_poses(
            positions=self._spawn.reshape(1, 3),
            orientations=np.array([[1.0, 0.0, 0.0, 0.0]]))
        self._base_view.set_velocities(velocities=np.zeros((1, 6)))
        self._fossen.reset()
        self._t200.reset()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._sim.close()
        except Exception:
            pass


# ============================================================================
# ChaseCam — following camera + optional video recording
# ============================================================================

CHASE_DEFAULT_OFFSET    = np.array([-3.0, 0.0, 1.0], dtype=float)
CHASE_DEFAULT_RES       = (640, 480)
CHASE_DEFAULT_FOCAL_MM  = 18.5
CHASE_DEFAULT_APT_MM    = 20.955


class ChaseCam:
    """Following camera anchored at a fixed world-frame offset from the
    body. Optionally pipes frames into ffmpeg (libopenh264) or falls back
    to cv2 mp4v if ffmpeg isn't found.

    Recording is decoupled from camera lifetime: call
    :meth:`start_recording` to open a pipe, :meth:`stop_recording` to
    close it. Multi-scenario eval scripts can rotate files between trials.

    Parameters
    ----------
    scene : GripperScene
        The scene whose body this camera follows.
    offset_world : (3,) array, default (-5, 0, 2)
        World-frame offset of the camera from the body's COG.
    resolution : (W, H) tuple, default (640, 480)
    focal_length, aperture_h : float, mm
        Camera intrinsics in Isaac's mm units.
    ffmpeg_path : str | None
        Path to ffmpeg, as returned by ``resolve_ffmpeg()``. ``None``
        forces the cv2 mp4v fallback.
    set_active_in_viewport : bool
        Switch the GUI viewport to this camera. Only one camera (chase
        or FPV) should be set active.
    """

    def __init__(self, scene: GripperScene, *,
                 offset_world: np.ndarray = CHASE_DEFAULT_OFFSET,
                 resolution: Tuple[int, int] = CHASE_DEFAULT_RES,
                 focal_length: float = CHASE_DEFAULT_FOCAL_MM,
                 aperture_h: float = CHASE_DEFAULT_APT_MM,
                 ffmpeg_path: Optional[str] = None,
                 set_active_in_viewport: bool = True):
        self._scene = scene
        self._offset = np.asarray(offset_world, dtype=float).copy()
        self._resolution = tuple(resolution)
        self._ffmpeg_path = ffmpeg_path
        self._ffmpeg_proc = None
        self._cv2_writer = None
        self._cv2_raw_path: Optional[Path] = None
        self._recording_path: Optional[Path] = None

        # --- USD camera prim ----------------------------------------------
        UsdGeom = scene._UsdGeom
        Gf = scene._Gf
        self._cam_path = "/World/chase_camera"
        cam_prim = UsdGeom.Camera.Define(scene._stage, self._cam_path)
        cam_prim.CreateFocalLengthAttr(focal_length)
        cam_prim.CreateHorizontalApertureAttr(aperture_h)
        cam_prim.CreateVerticalApertureAttr(
            aperture_h * self._resolution[1] / self._resolution[0])
        cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))

        xform = UsdGeom.Xformable(cam_prim.GetPrim())
        xform.ClearXformOpOrder()
        self._translate_op = xform.AddTranslateOp()
        self._orient_op    = xform.AddOrientOp()

        # Initial pose: behind/above spawn, looking at spawn.
        cam_start = scene._spawn + self._offset
        cam_quat  = look_at_quat_wxyz(cam_start, scene._spawn)
        self._translate_op.Set(Gf.Vec3d(*[float(v) for v in cam_start]))
        self._orient_op.Set(Gf.Quatf(*[float(v) for v in cam_quat]))

        # Sensor camera for RGBA grabs.
        from omni.isaac.sensor import Camera as _IsaacCamera
        self._isaac_cam = _IsaacCamera(prim_path=self._cam_path,
                                       resolution=self._resolution)
        self._isaac_cam.initialize()

        if set_active_in_viewport and not scene.headless:
            try:
                from omni.kit.viewport.utility import get_active_viewport
                get_active_viewport().set_active_camera(self._cam_path)
            except Exception as e:
                print(f"[chase-cam] viewport switch failed: {e}")

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

    def start_recording(self, out_path: Path) -> None:
        """Open a new video pipe writing to ``out_path``. Forces the scene
        to render even in headless mode."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self._ffmpeg_path:
            import subprocess
            try:
                self._ffmpeg_proc = subprocess.Popen([
                    self._ffmpeg_path, "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", f"{self._resolution[0]}x{self._resolution[1]}",
                    "-r", str(1.0 / self._scene.dt),
                    "-i", "-",
                    "-c:v", "libopenh264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(out_path),
                ], stdin=subprocess.PIPE)
                print(f"[chase-cam] recording (H.264) -> {out_path.name}")
                self._recording_path = out_path
                self._scene.force_render = True
                return
            except Exception as e:
                print(f"[chase-cam] ffmpeg pipe failed ({e}); cv2 mp4v fallback")
                self._ffmpeg_proc = None
        # cv2 fallback — writes a raw mp4v file, transcoded on stop if
        # ffmpeg is available.
        import cv2
        raw_path = out_path.with_suffix(".raw.mp4")
        writer = cv2.VideoWriter(
            str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
            1.0 / self._scene.dt, self._resolution,
        )
        if not writer.isOpened():
            print(f"[chase-cam] cv2 writer failed for {out_path}")
            return
        self._cv2_writer = writer
        self._cv2_raw_path = raw_path
        self._recording_path = out_path
        self._scene.force_render = True
        print(f"[chase-cam] recording (cv2 mp4v raw) -> {raw_path.name}")

    def stop_recording(self) -> None:
        """Close the current video pipe. Transcodes the cv2 raw .mp4 to
        H.264 if ffmpeg is available, then deletes the raw."""
        if self._ffmpeg_proc is not None:
            try:
                if self._ffmpeg_proc.stdin is not None \
                        and not self._ffmpeg_proc.stdin.closed:
                    self._ffmpeg_proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            self._ffmpeg_proc.wait()
            print(f"[chase-cam] saved -> {self._recording_path.name}")
            self._ffmpeg_proc = None
        elif self._cv2_writer is not None:
            self._cv2_writer.release()
            self._cv2_writer = None
            if self._ffmpeg_path and self._cv2_raw_path is not None \
                    and self._cv2_raw_path.exists():
                import subprocess
                r = subprocess.run([
                    self._ffmpeg_path, "-y", "-loglevel", "error",
                    "-i", str(self._cv2_raw_path), "-c:v", "libopenh264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(self._recording_path),
                ], capture_output=True)
                if r.returncode == 0:
                    self._cv2_raw_path.unlink(missing_ok=True)
                    print(f"[chase-cam] saved -> {self._recording_path.name}")
                else:
                    print(f"[chase-cam] transcode failed; raw at "
                          f"{self._cv2_raw_path.name}")
            else:
                print(f"[chase-cam] saved (raw) -> {self._cv2_raw_path.name}")
            self._cv2_raw_path = None
        self._recording_path = None
        self._scene.force_render = False

    # ------------------------------------------------------------------
    # Per-tick: follow + (optionally) grab a frame
    # ------------------------------------------------------------------

    def follow_and_record(self) -> None:
        """Move the camera to ``body_pos + offset`` and, if recording is
        active, grab one RGBA frame and write it to the video pipe."""
        Gf = self._scene._Gf
        pos_w, _ = self._scene._base_view.get_world_poses()
        cp = np.asarray(pos_w[0], dtype=float) + self._offset
        self._translate_op.Set(Gf.Vec3d(float(cp[0]), float(cp[1]), float(cp[2])))

        if self._ffmpeg_proc is None and self._cv2_writer is None:
            return
        rgba = self._isaac_cam.get_rgba()
        if rgba is None or rgba.size == 0:
            return
        bgr = np.ascontiguousarray(rgba[..., :3][..., ::-1])
        try:
            if self._ffmpeg_proc is not None:
                self._ffmpeg_proc.stdin.write(bgr.tobytes())
            else:
                self._cv2_writer.write(bgr)
        except (BrokenPipeError, OSError):
            pass

    def close(self) -> None:
        if self._ffmpeg_proc is not None or self._cv2_writer is not None:
            self.stop_recording()


# ============================================================================
# FPVCam — body-frame forward-facing onboard camera
# ============================================================================

FPV_DEFAULT_OFFSET    = np.array([0.20, 0.0, 0.05], dtype=float)
# USD convention: local -Z forward, local +Y up. (0.5, 0.5, -0.5, -0.5)
# rotates so local -Z aligns with body +X and local +Y with body +Z.
FPV_DEFAULT_QUAT_WXYZ = np.array([0.5, 0.5, -0.5, -0.5], dtype=float)
FPV_DEFAULT_FOCAL_MM  = 2.15
FPV_DEFAULT_APT_MM    = 3.6


class FPVCam:
    """Body-frame forward-facing onboard camera. Rigidly follows the
    body (offset + orientation in body frame). Useful for immersive
    chase shots and FPV-style debugging.

    Parameters
    ----------
    scene : GripperScene
    body_offset : (3,) array
        Body-frame camera position. Default places it 0.20 m forward,
        0.05 m above the COG.
    body_quat_wxyz : (4,) array
        Body-frame camera orientation. Default points the camera along
        body +X (forward).
    focal_length, aperture_h : float, mm
        Camera intrinsics. Defaults give ~80° FOV.
    set_active_in_viewport : bool
        Switch the GUI viewport to this camera.
    """

    def __init__(self, scene: GripperScene, *,
                 body_offset: np.ndarray = FPV_DEFAULT_OFFSET,
                 body_quat_wxyz: np.ndarray = FPV_DEFAULT_QUAT_WXYZ,
                 focal_length: float = FPV_DEFAULT_FOCAL_MM,
                 aperture_h: float = FPV_DEFAULT_APT_MM,
                 set_active_in_viewport: bool = False):
        self._scene = scene
        self._body_offset = np.asarray(body_offset, dtype=float).copy()
        self._body_quat   = np.asarray(body_quat_wxyz, dtype=float).copy()

        UsdGeom = scene._UsdGeom
        Gf = scene._Gf
        self._cam_path = "/World/front_camera"
        prim = UsdGeom.Camera.Define(scene._stage, self._cam_path)
        prim.CreateFocalLengthAttr(focal_length)
        prim.CreateHorizontalApertureAttr(aperture_h)
        prim.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))

        xform = UsdGeom.Xformable(prim.GetPrim())
        xform.ClearXformOpOrder()
        self._translate_op = xform.AddTranslateOp()
        self._orient_op    = xform.AddOrientOp()
        self._translate_op.Set(Gf.Vec3d(0.3, 0.0, -0.9))   # placeholder
        self._orient_op.Set(Gf.Quatf(
            float(self._body_quat[0]), float(self._body_quat[1]),
            float(self._body_quat[2]), float(self._body_quat[3]),
        ))

        if set_active_in_viewport and not scene.headless:
            try:
                from omni.kit.viewport.utility import get_active_viewport
                get_active_viewport().set_active_camera(self._cam_path)
                print(f"[fpv-cam] viewport switched to {self._cam_path}")
            except Exception as e:
                print(f"[fpv-cam] viewport switch failed: {e}")

    def update(self) -> None:
        """Move + rotate the camera to follow the body. Call once per tick
        after physics has stepped."""
        Gf = self._scene._Gf
        pos_w, quat_w = self._scene._base_view.get_world_poses()
        R_body = self._scene._quat_to_rotmat(np.asarray(quat_w[0], dtype=float))
        world_pos  = np.asarray(pos_w[0], dtype=float) + R_body @ self._body_offset
        world_quat = quat_mul(np.asarray(quat_w[0], dtype=float),
                              self._body_quat)
        self._translate_op.Set(Gf.Vec3d(
            float(world_pos[0]), float(world_pos[1]), float(world_pos[2])
        ))
        self._orient_op.Set(Gf.Quatf(
            float(world_quat[0]), float(world_quat[1]),
            float(world_quat[2]), float(world_quat[3]),
        ))
