"""Keyboard teleop for the BlueROV-Heavy + Newton Gripper in the tank env.

Drives the vehicle with the 4-DOF normalized ArduSub command
``[surge, sway, heave, yaw]`` (hold-to-move: command = ±gain while the key
is held, zero on release). The vehicle carries the centered trim by
default — COG and COB on the vertical center axis, BM intact, floats
level like the re-trimmed physical ROV (``--no-retrim`` for the legacy
~13.7 deg nose-up trim). The Isaac viewport shows a chase camera
(``C`` switches to the onboard FPV); three out-of-process live plots (ν 6-axis, 3D tank path,
per-thruster) stream alongside — the subprocess split avoids the
Qt-vs-Kit event-loop conflict (see ``EDMDc/live_plot_client.py``).

The default attitude mode matches the real
``control_v2.keyboard_stabilize`` launcher: roll/pitch are leveled with
differential vertical thrust, while heave remains direct (no depth hold) and
the four-command ``[surge, sway, heave, yaw]`` interface is unchanged. Use
``--attitude-mode manual`` to recover the unstabilized plant.

Key map (keyboard focus must be ON THE ISAAC WINDOW):

    W / S      surge forward / back
    A / D      sway left / right
    E / Q      heave up / down
    R / F      yaw left / right
    + / -      command gain up / down (0.1 .. 1.0)
    C          toggle viewport camera chase <-> FPV
    Space      panic: clear all held keys (zero command)
    Esc        quit

Run (GUI):

    /home/miaodong/Documents/isaac-sim-6.0/python.sh -m EDMDc.teleop_tank

Headless scripted check, no keyboard (verifies key-sign conventions and
wall collision empirically):

    /home/miaodong/Documents/isaac-sim-6.0/python.sh \
        -m EDMDc.teleop_tank --self-test
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Argparse FIRST — --help must work without booting Isaac.
# ---------------------------------------------------------------------------
import argparse
import time
from datetime import datetime
from pathlib import Path

ENV_FILES = {
    "tank":      "environment_tank.usda",
    "pool":      "environment.usd",
    "deep-pool": "environment_deep_pool.usd",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=sorted(ENV_FILES), default="tank",
                   help="Environment USD to load (default: tank).")
    p.add_argument("--dt", type=float, default=1.0 / 60.0,
                   help="Physics step (default 1/60 s).")
    p.add_argument("--gain", type=float, default=0.4,
                   help="Initial command gain in [0.1, 1.0] (default 0.4).")
    p.add_argument("--attitude-mode", choices=("stabilize", "manual"),
                   default="stabilize",
                   help="Roll/pitch control mode (default: stabilize). "
                        "STABILIZE does not hold depth or yaw.")
    p.add_argument("--window-seconds", type=float, default=20.0,
                   help="Rolling window of the 2D live plots (default 20 s).")
    p.add_argument("--no-plot-nu", action="store_true",
                   help="Disable the 6-axis velocity live plot.")
    p.add_argument("--no-plot-3d", action="store_true",
                   help="Disable the 3D path live plot.")
    p.add_argument("--no-plot-thrusters", action="store_true",
                   help="Disable the per-thruster live plot.")
    p.add_argument("--no-retrim", action="store_true",
                   help="Fly the legacy un-trimmed vehicle (passive pitch "
                        "equilibrium ~13.7 deg) instead of the level "
                        "re-trimmed configuration.")
    p.add_argument("--no-settle", action="store_true",
                   help="Skip the settle phase after the surface spawn.")
    p.add_argument("--max-minutes", type=float, default=30.0,
                   help="Hard session limit (default 30 min).")
    p.add_argument("--command-timeout", type=float, default=0.75,
                   help="Clear held commands after this many seconds without "
                        "a key press/repeat event (default 0.75; guards against "
                        "a lost KEY_RELEASE when menus take focus).")
    p.add_argument("--self-test", action="store_true",
                   help="Headless scripted run instead of keyboard input: "
                        "verifies command-sign conventions and wall collision, "
                        "then exits with a PASS/FAIL summary.")
    return p.parse_args()


def main() -> int:
    import numpy as np

    args = parse_args()
    PROJECT = Path(__file__).resolve().parents[1]
    env_usd = str(PROJECT / "assets" / ENV_FILES[args.env])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = PROJECT / "EDMDc" / "data" / "plots" / "teleop"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ----- Boot Isaac (inside GripperScene) --------------------------------
    from .isaac_scene import GripperScene, ChaseCam, FPVCam

    scene = GripperScene(dt=args.dt, headless=bool(args.self_test),
                         env_usd=env_usd,
                         retrim_level=not args.no_retrim,
                         ardusub_stabilize=args.attitude_mode == "stabilize")

    if args.self_test:
        return _self_test(scene, np)

    # ----- Cameras: chase cam in the viewport, FPV as the C-toggle ---------
    chase = ChaseCam(scene, offset_world=np.array([-2.5, 0.0, 1.2]),
                     set_active_in_viewport=True)
    fpv = FPVCam(scene, set_active_in_viewport=False)
    cam_names = ["chase", "FPV"]
    cam_paths = [chase._cam_path, fpv._cam_path]
    cam_state = {"active": 0}

    def _toggle_camera() -> None:
        cam_state["active"] ^= 1
        try:
            from omni.kit.viewport.utility import get_active_viewport
            get_active_viewport().set_active_camera(cam_paths[cam_state["active"]])
            print(f"[teleop] viewport -> {cam_names[cam_state['active']]}")
        except Exception as e:
            print(f"[teleop] camera toggle failed: {e}")

    # ----- Live plots (out-of-process) -------------------------------------
    from .live_plot_client import LivePlot2D, LivePlot3D, LivePlotThrusters

    plot_nu = None if args.no_plot_nu else LivePlot2D(
        dt=args.dt, title="teleop: body velocity nu",
        window_seconds=args.window_seconds, label_meas="Isaac",
        label_ref="zero",
        save_on_close=str(plot_dir / f"{ts}_nu.png"))
    plot_3d = None if args.no_plot_3d else LivePlot3D(
        title="teleop: tank path",
        save_on_close=str(plot_dir / f"{ts}_path.png"))
    plot_thr = None if args.no_plot_thrusters else LivePlotThrusters(
        dt=args.dt, title="teleop: thruster commands",
        window_seconds=args.window_seconds,
        n_thrusters=scene.num_rotors, max_thrust=25.0,
        save_on_close=str(plot_dir / f"{ts}_thrusters.png"))

    # ----- Keyboard (carb input on the Kit app window) ---------------------
    import carb.input
    import omni.appwindow

    KI = carb.input.KeyboardInput
    KET = carb.input.KeyboardEventType
    pressed: set = set()
    session = {
        "running": True,
        "gain": float(np.clip(args.gain, 0.1, 1.0)),
        "last_key_activity": time.monotonic(),
    }

    def _on_key(event, *_):
        session["last_key_activity"] = time.monotonic()
        if event.type in (KET.KEY_PRESS, KET.KEY_REPEAT):
            pressed.add(event.input)
            if event.type == KET.KEY_PRESS:           # edge-triggered actions
                if event.input == KI.ESCAPE:
                    session["running"] = False
                elif event.input == KI.C:
                    _toggle_camera()
                elif event.input == KI.SPACE:
                    pressed.clear()
                elif event.input in (KI.EQUAL, KI.NUMPAD_ADD):
                    session["gain"] = min(1.0, session["gain"] + 0.1)
                    print(f"[teleop] gain = {session['gain']:.1f}")
                elif event.input in (KI.MINUS, KI.NUMPAD_SUBTRACT):
                    session["gain"] = max(0.1, session["gain"] - 0.1)
                    print(f"[teleop] gain = {session['gain']:.1f}")
        elif event.type == KET.KEY_RELEASE:
            pressed.discard(event.input)
        return True

    appwin = omni.appwindow.get_default_app_window()
    keyboard = appwin.get_keyboard()
    inp = carb.input.acquire_input_interface()
    kb_sub = inp.subscribe_to_keyboard_events(keyboard, _on_key)

    # Held-key -> normalized [surge, sway, heave, yaw] command.
    # Signs from --self-test + piloted check: surge + = forward,
    # sway + = screen-LEFT as piloted (so A carries +), heave + = UP
    # (ArduSub throttle-stick convention, NOT NED Fz), yaw + = right.
    AXIS_KEYS = (
        (0, KI.W, +1.0), (0, KI.S, -1.0),      # surge
        (1, KI.A, +1.0), (1, KI.D, -1.0),      # sway  (A = left, D = right)
        (2, KI.E, +1.0), (2, KI.Q, -1.0),      # heave (E = up,   Q = down)
        (3, KI.F, +1.0), (3, KI.R, -1.0),      # yaw   (R = left, F = right)
    )

    # ----- Settle off the surface spawn, then hand over control ------------
    if not args.no_settle:
        print("[teleop] settling off the surface spawn ...")
        scene.settle(max_s=8.0, on_step=lambda s: (fpv.update(),
                                                   chase.follow_and_record()))

    print(f"[teleop] ready — attitude={args.attitude_mode}, "
          f"drive with WASD/QE/RF, gain {session['gain']:.1f} "
          f"(+/- to adjust), C toggles camera, Esc quits.")

    zeros6 = np.zeros(6)
    t = 0.0
    next_hud = 0.0
    max_s = float(args.max_minutes) * 60.0
    timeline_was_playing = scene.timeline_playing
    try:
        while (session["running"] and t < max_s
               and scene.simulation_app.is_running()):
            # Never touch the rigid-body physics view while the Kit timeline is
            # paused/stopped.  Pump one render-only update so menus and the
            # viewport stay responsive, clear latched input, and resume on the
            # next iteration only after the timeline reports PLAYING again.
            if not scene.timeline_playing:
                if timeline_was_playing:
                    print("[teleop] timeline paused/stopped — command cleared")
                pressed.clear()
                timeline_was_playing = False
                scene.pump_app_while_paused()
                continue
            if not timeline_was_playing:
                pressed.clear()
                session["last_key_activity"] = time.monotonic()
                print("[teleop] timeline resumed — physics view ready, command zero")
            timeline_was_playing = True

            # A renderer menu can consume KEY_RELEASE while the outer app still
            # owns focus.  Window focus plus a repeat-event deadline bounds any
            # stale command without changing normal hold-to-move behavior.
            focus_lost = not appwin.is_focused()
            command_stale = (
                bool(pressed)
                and float(args.command_timeout) > 0.0
                and time.monotonic() - session["last_key_activity"]
                    > float(args.command_timeout)
            )
            if focus_lost or command_stale:
                if pressed:
                    why = "window focus lost" if focus_lost else "key timeout"
                    print(f"[teleop] {why} — command cleared")
                pressed.clear()

            cmd = np.zeros(4)
            for axis, key, sign in AXIS_KEYS:
                if key in pressed:
                    cmd[axis] += sign
            cmd = np.clip(cmd, -1.0, 1.0) * session["gain"]

            thrusts = scene.apply_wrench(cmd)
            state = scene.read_state()
            fpv.update()
            chase.follow_and_record()

            if plot_nu is not None:
                plot_nu.push(t, state.nu, zeros6)
            if plot_3d is not None:
                plot_3d.push(t, state.pos_w)
            if plot_thr is not None:
                plot_thr.push(t, thrusts)

            if t >= next_hud:
                depth = -0.2 - float(state.pos_w[2])
                print(f"[teleop] t={t:7.1f}s pos=({state.pos_w[0]:+.2f}, "
                      f"{state.pos_w[1]:+.2f}, {state.pos_w[2]:+.2f}) "
                      f"depth={depth:+.2f} "
                      f"rp=({np.rad2deg(state.roll):+5.1f},"
                      f"{np.rad2deg(state.pitch):+5.1f}) "
                      f"yaw={np.rad2deg(state.yaw):+6.1f} "
                      f"cmd={np.round(cmd, 2).tolist()}")
                next_hud = t + 2.0
            t += args.dt
    finally:
        inp.unsubscribe_to_keyboard_events(keyboard, kb_sub)
        for plt_client in (plot_nu, plot_3d, plot_thr):
            if plt_client is not None:
                plt_client.close()
        scene.close()
    return 0


# ---------------------------------------------------------------------------
# Headless self-test: scripted bursts instead of keyboard.
# Verifies the sign conventions the key map claims, and that the tank
# wall actually stops the articulation (collision, not tunneling).
# ---------------------------------------------------------------------------

def _self_test(scene, np) -> int:
    results: list[tuple[str, bool, str]] = []

    def fresh():
        """Isolate each segment: teleport back to spawn, kill velocity,
        settle. Without this a prior burst leaks into the next test —
        e.g. a yaw burst leaves the vehicle spinning (no linear yaw
        damping), so a 'drive into the wall' segment circles instead."""
        scene.reset_to_spawn()
        return scene.settle(max_s=6.0, min_s=3.0)

    def burst(cmd4, seconds):
        for _ in range(int(round(seconds / scene.dt))):
            scene.apply_wrench(np.asarray(cmd4, dtype=float))
        return scene.read_state()

    # -- re-trim: settled attitude must be level (was ~13.7 deg untrimmed) --
    s_eq = fresh()
    ok = abs(np.rad2deg(s_eq.pitch)) < 2.0
    results.append(("re-trim: level equilibrium", ok,
                    f"pitch={np.rad2deg(s_eq.pitch):+.2f} deg"))

    # -- STABILIZE: a deliberate roll/pitch upset must return toward level --
    if scene.ardusub_stabilize_enabled:
        roll0 = np.deg2rad(15.0)
        pitch0 = np.deg2rad(-12.0)
        cr, sr = np.cos(roll0 / 2.0), np.sin(roll0 / 2.0)
        cp, sp = np.cos(pitch0 / 2.0), np.sin(pitch0 / 2.0)
        upset_quat_wxyz = np.array([
            cr * cp, sr * cp, cr * sp, -sr * sp,
        ])
        scene.reset_to_spawn()
        scene.base_view.set_world_poses(
            positions=s_eq.pos_w.reshape(1, 3),
            orientations=upset_quat_wxyz.reshape(1, 4))
        scene.base_view.set_velocities(velocities=np.zeros((1, 6)))
        max_motor_correction = 0.0
        for _ in range(int(round(4.0 / scene.dt))):
            scene.apply_wrench(np.zeros(4))
            status = scene.stabilize_status
            max_motor_correction = max(
                max_motor_correction,
                float(np.max(np.abs(status.motor_correction))))
        s_recovered = scene.read_state()
        rp_final_deg = np.rad2deg(
            [s_recovered.roll, s_recovered.pitch])
        ok = (
            np.max(np.abs(rp_final_deg)) < 3.0
            and max_motor_correction > 0.075
        )
        results.append((
            "STABILIZE recovers roll/pitch upset", ok,
            f"final=({rp_final_deg[0]:+.2f},{rp_final_deg[1]:+.2f}) deg, "
            f"max correction={max_motor_correction:.3f}",
        ))

    # -- surge sign: W should move forward (u > 0) --------------------------
    fresh()
    s1 = burst([0.5, 0, 0, 0], 2.0)
    ok = s1.nu[0] > 0.05
    results.append(("surge W -> u>0", ok, f"u={s1.nu[0]:+.3f} m/s"))

    fresh()
    s1 = burst([-0.5, 0, 0, 0], 2.0)
    ok = s1.nu[0] < -0.05
    results.append(("surge S -> u<0", ok, f"u={s1.nu[0]:+.3f} m/s"))

    # -- sway signs: A is body +y / piloted screen-left; D is body -y -------
    fresh()
    s1 = burst([0, +0.5, 0, 0], 2.0)
    ok = s1.nu[1] > 0.05
    results.append(("sway A -> v>0 (left)", ok, f"v={s1.nu[1]:+.3f} m/s"))

    fresh()
    s1 = burst([0, -0.5, 0, 0], 2.0)
    ok = s1.nu[1] < -0.05
    results.append(("sway D -> v<0 (right)", ok, f"v={s1.nu[1]:+.3f} m/s"))

    # -- heave sign: E maps to cmd +, expect UP (z increases) ---------------
    s0 = fresh()
    s1 = burst([0, 0, +0.5, 0], 1.5)
    dz = float(s1.pos_w[2] - s0.pos_w[2])
    ok = dz > 0.01
    results.append(("heave E(+) -> up", ok, f"dz={dz:+.3f} m"))

    s0 = fresh()
    s1 = burst([0, 0, -0.5, 0], 1.5)
    dz = float(s1.pos_w[2] - s0.pos_w[2])
    ok = dz < -0.01
    results.append(("heave Q(-) -> down", ok, f"dz={dz:+.3f} m"))

    # -- yaw sign: F maps to cmd +, expect r > 0 (yaw right) ----------------
    fresh()
    s1 = burst([0, 0, 0, 0.4], 1.5)
    ok = s1.nu[5] > 0.02
    results.append(("yaw F(+) -> r>0 (right)", ok, f"r={s1.nu[5]:+.3f} rad/s"))

    fresh()
    s1 = burst([0, 0, 0, -0.4], 1.5)
    ok = s1.nu[5] < -0.02
    results.append(("yaw R(-) -> r<0 (left)", ok, f"r={s1.nu[5]:+.3f} rad/s"))

    # -- wall collision: sustained surge must be stopped by the east wall ---
    # Moderate command: sustained surge >= ~0.5 pitches the vehicle over
    # (nose-down past -80 deg, thrust goes vertical, x stalls mid-tank —
    # measured here at 0.8; same sustained-cap dynamics as the CLAUDE.md
    # "sustained single-axis step" note). 0.3 cruises flat.
    fresh()
    print("[self-test] driving into the +x wall ...")
    x_hist = []
    hit = False
    for k in range(int(round(30.0 / scene.dt))):
        scene.apply_wrench(np.array([0.3, 0.0, 0.0, 0.0]))
        st = scene.read_state()
        x_hist.append(float(st.pos_w[0]))
        if k % 60 == 0:
            print(f"  [wall-run] t={k*scene.dt:4.1f}s "
                  f"x={st.pos_w[0]:+.3f} y={st.pos_w[1]:+.3f} "
                  f"z={st.pos_w[2]:+.3f} pitch={np.rad2deg(st.pitch):+5.1f} "
                  f"yaw={np.rad2deg(st.yaw):+6.1f} u={st.nu[0]:+.3f}")
        if len(x_hist) > 60 and abs(x_hist[-1] - x_hist[-61]) < 0.002:
            hit = True
            break
    x_final = x_hist[-1]
    # Must have actually reached the wall region (not just stalled or
    # circled), and must not have passed through it.
    ok = hit and (1.5 < x_final < 2.30)
    results.append(("wall stops ROV, no tunneling", ok,
                    f"x_final={x_final:+.3f} m (interior face +2.248)"))

    print("\n========== TELEOP SELF-TEST ==========")
    n_fail = 0
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:32s} {detail}")
        n_fail += (not ok)
    print(f"OVERALL: {'PASS' if n_fail == 0 else f'FAIL ({n_fail})'}")
    scene.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
