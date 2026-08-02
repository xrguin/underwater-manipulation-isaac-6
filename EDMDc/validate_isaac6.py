"""Focused Isaac Sim 6 migration validation for the BlueROV tank scene.

Run with Isaac Sim's bundled Python:

    /home/miaodong/Documents/isaac-sim-6.0/python.sh \
        -m EDMDc.validate_isaac6

Add ``--camera`` on a renderer-capable host to validate RTX camera pixels.
``--gui`` runs the checks with a visible RTX Real-Time viewport.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true",
                        help="Also require a finite 640x480 RGBA camera frame.")
    parser.add_argument("--camera-out", type=Path,
                        help="Optional path at which to save the camera frame.")
    parser.add_argument("--gui", action="store_true",
                        help="Run with a visible RTX Real-Time viewport.")
    parser.add_argument("--zero-seconds", type=float, default=8.0,
                        help="Duration of the zero-command settling check.")
    return parser.parse_args()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[isaac6-validation] PASS  {message}")


def _finite_state(np, state) -> bool:
    values = np.concatenate([
        state.pos_w, state.quat_wxyz, state.nu, state.lin_w, state.ang_w,
    ])
    return bool(np.all(np.isfinite(values)))


def main() -> int:
    import numpy as np

    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    tank_usd = project / "assets" / "environment_tank.usda"

    from .isaac_scene import ChaseCam, GRIPPER_OFFSET, GripperScene

    scene = GripperScene(
        dt=1.0 / 60.0,
        headless=not args.gui,
        env_usd=str(tank_usd),
    )
    # Core Experimental modules become importable only after GripperScene has
    # created SimulationApp.
    from .isaac6_compat import RigidPrimView
    chase = None
    try:
        stage = scene.stage

        # Asset composition and the authored runtime weld.
        required_prims = (
            "/World/Environment",
            "/World/Vehicle",
            "/World/Vehicle/base_link",
            "/World/Gripper",
            "/World/Gripper/base_link",
        )
        for path in required_prims:
            _check(stage.GetPrimAtPath(path).IsValid(), f"asset prim exists: {path}")

        joint = stage.GetPrimAtPath("/World/gripper_attach_joint")
        _check(joint.IsValid(), "gripper fixed joint exists")
        from pxr import UsdPhysics

        fixed = UsdPhysics.FixedJoint(joint)
        body0 = [str(path) for path in fixed.GetBody0Rel().GetTargets()]
        body1 = [str(path) for path in fixed.GetBody1Rel().GetTargets()]
        _check(body0 == ["/World/Vehicle/base_link"],
               "fixed joint body0 targets the vehicle")
        _check(body1 == ["/World/Gripper/base_link"],
               "fixed joint body1 targets the gripper")
        _check(scene._rov.impl.is_physics_tensor_entity_valid(),
               "vehicle articulation physics view is valid")

        state0 = scene.read_state()
        _check(_finite_state(np, state0), "initial rigid-body state is finite")

        # Reference 4.2 behavior: surface-spawn z≈-0.208 sinks and settles near
        # z≈-0.449 by eight simulated seconds.
        zero = np.zeros(4, dtype=float)
        z_history = [float(state0.pos_w[2])]
        steps = int(round(float(args.zero_seconds) / scene.dt))
        rollout_finite = True
        for _ in range(steps):
            thrusts = scene.apply_wrench(zero, render=False)
            state = scene.read_state()
            rollout_finite &= bool(np.all(np.isfinite(thrusts)))
            rollout_finite &= _finite_state(np, state)
            z_history.append(float(state.pos_w[2]))
        _check(rollout_finite, "zero-command rollout remains finite")

        z = np.asarray(z_history)
        z_start, z_end = float(z[0]), float(z[-1])
        _check(-0.27 < z_start < -0.15,
               f"zero-command start depth matches surface spawn (z={z_start:+.3f})")
        _check(-0.56 < z_end < -0.34,
               f"zero-command settles near 4.2 reference (z={z_end:+.3f})")
        _check(float(np.max(z)) <= z_start + 0.05,
               "zero-command buoyancy does not rise uncontrollably")
        tail = z[-min(60, z.size):]
        _check(float(np.ptp(tail)) < 0.04,
               f"zero-command depth is settling (last-second span={np.ptp(tail):.4f} m)")
        _check(abs(float(state.nu[2])) < 0.02,
               f"tank floor stops downward motion (w={state.nu[2]:+.4f} m/s)")

        # Observe the two bodies after the rollout, not just the USD joint
        # declaration, to ensure the fixed joint remained physically active.
        gripper_view = RigidPrimView(
            prim_paths_expr="/World/Gripper/base_link",
            name="isaac6_validation_gripper",
        )
        grip_pos, _ = gripper_view.get_world_poses()
        base_state = scene.read_state()
        rel_body = base_state.R_body.T @ (grip_pos[0] - base_state.pos_w)
        _check(np.allclose(rel_body, GRIPPER_OFFSET, atol=0.015),
               f"vehicle and gripper remain joined (offset={np.round(rel_body, 4)})")

        # Thruster commands, motor lag, and saturation remain bounded.
        mixed_command = np.ones(4, dtype=float)
        motor_commands = scene._allocator.allocate(mixed_command)
        _check(float(np.max(np.abs(motor_commands))) <= 1.0 + 1e-12,
               "ArduSub allocation saturates within normalized motor limits")
        for _ in range(60):
            thrusts = scene.apply_wrench(mixed_command, render=False)
        _check(np.all(np.isfinite(thrusts)), "realized thruster forces are finite")
        _check(float(np.max(np.abs(scene._t200.throttle))) <= 1.0 + 1e-12,
               "T200 throttle remains saturated within [-1, 1]")
        _check(np.all(np.abs(scene._t200.rpm)
                      <= scene._tcfg.max_rotation_velocities + 1e-9),
               "T200 RPM remains within YAML saturation")

        # Pause must not advance the T200 model or access an invalid physics
        # view.  State reads return the last finite snapshot until resume.
        import isaacsim.core.experimental.utils.app as app_utils

        before_pause = scene.read_state()
        throttle_before = scene._t200.throttle.copy()
        app_utils.pause()
        scene.pump_app_while_paused()
        _check(scene.timeline_paused, "timeline reports paused")
        paused_thrusts = scene.apply_wrench(np.ones(4), render=False)
        paused_state = scene.read_state()
        _check(np.array_equal(paused_thrusts, np.zeros(scene.num_rotors)),
               "paused timeline applies no force")
        _check(np.array_equal(scene._t200.throttle, throttle_before),
               "paused timeline does not advance T200 state")
        _check(np.array_equal(paused_state.pos_w, before_pause.pos_w),
               "paused state read returns the last finite pose")

        app_utils.play()
        app_utils.update_app()
        resumed_thrusts = scene.apply_wrench(zero, render=False)
        resumed_state = scene.read_state()
        _check(scene.timeline_playing, "timeline resumes")
        _check(np.all(np.isfinite(resumed_thrusts))
               and _finite_state(np, resumed_state),
               "physics interface is finite after resume")

        # Debug draw import/acquisition is part of the Isaac 6 namespace
        # migration.  Drawing itself is deliberately a no-op in headless mode.
        from isaacsim.util.debug_draw import _debug_draw

        draw = _debug_draw.acquire_debug_draw_interface()
        _check(draw is not None, "Isaac 6 debug-draw interface is available")

        if args.camera:
            chase = ChaseCam(scene, set_active_in_viewport=bool(args.gui))
            scene.force_render = True
            rgba = None
            for _ in range(60):
                scene.apply_wrench(zero, render=True)
                chase.follow_and_record()
                rgba = chase._isaac_cam.get_rgba()
                if rgba is not None and rgba.size:
                    break
            _check(rgba is not None and rgba.shape == (480, 640, 4),
                   f"RTX camera returns 640x480 RGBA "
                   f"(shape={None if rgba is None else rgba.shape})")
            _check(np.all(np.isfinite(rgba)), "camera output is finite")
            if args.camera_out:
                from PIL import Image

                args.camera_out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.asarray(rgba, dtype=np.uint8), "RGBA").save(
                    args.camera_out
                )
                print(f"[isaac6-validation] camera frame: {args.camera_out}")

        print(
            "[isaac6-validation] OVERALL PASS  "
            f"z {z_start:+.3f} -> {z_end:+.3f} m over "
            f"{float(args.zero_seconds):.1f} simulated seconds"
        )
        return 0
    finally:
        if chase is not None:
            chase.close()
        scene.close()


if __name__ == "__main__":
    raise SystemExit(main())
