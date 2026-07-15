"""Probe / demo of xyz APRBS excitation in the small tank.

Yaw is HELD (not excited, not modeled) so the onboard camera keeps
facing the +x wall AprilTags. Two modes:

  * dither (default) — weak station-keeping PID toward tank center with
    the APRBS superimposed. The validated recipe (probe sweeps
    2026-07-12): xy amplitude 0.45, heave sign*U(0.20, 0.28), holds
    <= 0.4 s -> best coverage, pitch < ~20 deg, yaw error < 4 deg.
  * drift — pure APRBS, no position feedback (clean exogenous input;
    episodes end at the safety box after ~5-10 s).

Headless: episodes END at a box violation — survival time is the
measurement. GUI: adds chase cam + live plots (nu 6-axis and the 4
command channels) and runs CONTINUOUSLY — on a box violation the APRBS
pauses, a stronger PID re-centers the vehicle, then excitation resumes
(the same segment structure a real collection session uses); metrics
count exciting samples only.

Run:

    conda activate marinegym && python -m EDMDc.aprbs_tank_probe          # headless, 3 eps
    python -m EDMDc.aprbs_tank_probe --gui --seconds 180                  # watch it
"""
from __future__ import annotations

# Argparse FIRST — --help must work without booting Isaac.
import argparse
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["dither", "drift"], default="dither")
    p.add_argument("--amp", type=float, default=0.45,
                   help="Surge/sway APRBS amplitude, normalized (default 0.45).")
    p.add_argument("--amp-z", type=float, default=0.28,
                   help="Heave amplitude (default 0.28 — the z workspace is "
                        "~10x tighter than x and heave authority is higher).")
    p.add_argument("--hold", type=float, default=0.4,
                   help="APRBS hold-time upper bound [s]; lower = half of it.")
    p.add_argument("--seconds", type=float, default=60.0,
                   help="Episode length [s] (GUI runs it continuously).")
    p.add_argument("--gui", action="store_true",
                   help="Isaac GUI + chase cam + live plots; on a box "
                        "violation re-center and resume instead of ending.")
    return p.parse_args()


# Safety box on the ROV CENTER (interior minus hull/gripper margins),
# tank: |x|<2.248, |y|<1.105, floor top -0.8, surface -0.2.
BOX_X, BOX_Y = 1.90, 0.80
BOX_Z_MIN, BOX_Z_MAX = -0.62, -0.30
YAW_LIMIT_DEG = 25.0          # tag leaves FOV / grazing view beyond this
PITCH_LIMIT_DEG = 30.0        # tag leaves vertical FOV / pitch-over onset
CENTER = (0.0, 0.0, -0.47)
Z_FLOOR = 0.20                # T200 deadband: |heave| < ~0.2 produces no thrust
SEED = 7
EPISODES_HEADLESS = 3


def main() -> int:
    import numpy as np

    args = parse_args()
    PROJECT = Path(__file__).resolve().parents[1]
    out_dir = PROJECT / "EDMDc" / "data" / "tank_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_episodes = 1 if args.gui else EPISODES_HEADLESS

    from .isaac_scene import GripperScene

    scene = GripperScene(dt=1 / 60, headless=not args.gui,
                         env_usd=str(PROJECT / "assets" / "environment_tank.usda"),
                         spawn_pos=np.array(CENTER))
    dt = scene.dt

    chase = None
    plot_nu = plot_cmd = None
    plot_clock = {"t": 0.0}
    if args.gui:
        from .isaac_scene import ChaseCam
        chase = ChaseCam(scene, offset_world=np.array([-2.5, 0.0, 1.2]),
                         set_active_in_viewport=True)
        from .live_plot_client import LivePlot2D, LivePlotThrusters
        plot_png = PROJECT / "EDMDc" / "data" / "plots" / "tank_probe"
        plot_png.mkdir(parents=True, exist_ok=True)
        plot_nu = LivePlot2D(
            dt=dt, title="tank probe: body velocity nu",
            window_seconds=20.0, label_meas="Isaac", label_ref="zero",
            save_on_close=str(plot_png / f"{ts}_nu.png"))
        plot_cmd = LivePlotThrusters(
            dt=dt, title="tank probe cmd: [0]=surge [1]=sway [2]=heave [3]=yaw",
            window_seconds=20.0, n_thrusters=4, max_thrust=1.0,
            save_on_close=str(plot_png / f"{ts}_cmd.png"))

    def push_plots(st, cmd):
        if plot_nu is None:
            return
        plot_clock["t"] += dt
        plot_nu.push(plot_clock["t"], st.nu, np.zeros(6))
        plot_cmd.push(plot_clock["t"], cmd)

    def tick(cmd, st):
        scene.apply_wrench(cmd)
        if chase is not None:
            chase.follow_and_record()
        push_plots(st, cmd)

    def box_violation(pos, yaw, pitch):
        return (abs(pos[0]) > BOX_X or abs(pos[1]) > BOX_Y
                or not (BOX_Z_MIN < pos[2] < BOX_Z_MAX)
                or abs(np.rad2deg(yaw)) > YAW_LIMIT_DEG
                or abs(np.rad2deg(pitch)) > PITCH_LIMIT_DEG)

    def yaw_hold(st):
        return float(np.clip(-1.0 * st.yaw - 0.30 * st.nu[5], -0.30, 0.30))

    def center_pid(st, gain=1.0, clip=0.25):
        """Feedback toward tank center; `gain`/`clip` scale for recovery."""
        fb = np.zeros(3)
        fb[0] = np.clip(gain * 0.40 * (CENTER[0] - st.pos_w[0])
                        - 0.35 * st.nu[0], -clip, clip)
        fb[1] = np.clip(gain * 0.40 * (CENTER[1] - st.pos_w[1])
                        - 0.35 * st.nu[1], -clip, clip)
        fb[2] = np.clip(gain * 0.60 * (CENTER[2] - st.pos_w[2])
                        - 0.40 * st.nu[2], -clip * 1.2, clip * 1.2)
        return fb

    class Aprbs:
        """Continuous-amplitude APRBS, independent timer per axis. Heave
        draws sign*U(Z_FLOOR, amp_z): smaller magnitudes sit inside the
        T200 deadband and would be dead time on that axis."""

        def __init__(self, rng):
            self.rng = rng
            self.h = (0.5 * args.hold, args.hold)
            self.val = np.zeros(3)
            self.t_next = np.zeros(3)

        def step(self, t):
            for i in range(3):
                if t >= self.t_next[i]:
                    if i == 2:
                        lo = min(Z_FLOOR, args.amp_z)
                        mag = self.rng.uniform(lo, args.amp_z)
                        self.val[i] = mag * self.rng.choice((-1.0, 1.0))
                    else:
                        self.val[i] = self.rng.uniform(-args.amp, args.amp)
                    self.t_next[i] = t + self.rng.uniform(*self.h)
            return self.val.copy()

    def recovered(st):
        return (abs(st.pos_w[0]) < 0.25 and abs(st.pos_w[1]) < 0.25
                and abs(st.pos_w[2] - CENTER[2]) < 0.08
                and float(np.linalg.norm(st.nu[:3])) < 0.08)

    def run_episode(ep_idx):
        rng = np.random.default_rng(np.random.SeedSequence(SEED * 1000 + ep_idx))
        scene.reset_to_spawn()
        for _ in range(int(round(2.0 / dt))):        # brief settle at center
            tick(np.zeros(4), scene.read_state())
        aprbs = Aprbs(rng)
        n_max = int(round(args.seconds / dt))
        log = {k: [] for k in ("pos", "yaw", "pitch", "nu", "cmd")}
        excite_s, recover_s = 0.0, 0.0
        recovering = False
        for k in range(n_max):
            st = scene.read_state()
            if not recovering and box_violation(st.pos_w, st.yaw, st.pitch):
                if not args.gui:
                    break                             # headless: episode over
                recovering = True
            cmd = np.zeros(4)
            cmd[3] = yaw_hold(st)
            if recovering:
                cmd[:3] = center_pid(st, gain=3.0, clip=0.45)
                recover_s += dt
                if recovered(st):
                    recovering = False
            else:
                cmd[:3] = aprbs.step(k * dt)
                if args.mode == "dither":
                    cmd[:3] += center_pid(st)
                cmd[:3] = np.clip(cmd[:3], -0.9, 0.9)
                excite_s += dt
                log["pos"].append(st.pos_w.copy())
                log["yaw"].append(st.yaw)
                log["pitch"].append(st.pitch)
                log["nu"].append(st.nu.copy())
                log["cmd"].append(cmd.copy())
            tick(cmd, st)
        return excite_s, recover_s, {k: np.asarray(v) for k, v in log.items()}

    # ----- run ---------------------------------------------------------------
    eps = []
    for e in range(n_episodes):
        excite_s, recover_s, log = run_episode(e)
        nu = log["nu"]
        if len(nu) < 10:
            print(f"[probe] ep{e}: too short ({excite_s:.1f}s), skipped")
            continue
        eps.append({
            "excite_s": excite_s, "recover_s": recover_s,
            "std_uvw": nu[:, :3].std(axis=0),
            "yaw_max": float(np.rad2deg(np.abs(log["yaw"]).max())),
            "pitch_max": float(np.rad2deg(np.abs(log["pitch"]).max())),
        })
        print(f"[probe] ep{e}: excite={excite_s:5.1f}s recover={recover_s:4.1f}s "
              f"std_uvw={np.round(nu[:, :3].std(axis=0), 3)}")

    if eps:
        print("\n================== TANK APRBS PROBE ==================")
        print(f"mode={args.mode} amp={args.amp} amp_z={args.amp_z} "
              f"hold<={args.hold}s x {len(eps)} episodes")
        print(f"excite time   : {np.mean([m['excite_s'] for m in eps]):5.1f} s mean"
              + (f" ({np.mean([m['recover_s'] for m in eps]):4.1f} s recovering)"
                 if args.gui else "  (headless: ends at box)"))
        print(f"std u/v/w     : "
              f"{np.mean([m['std_uvw'][0] for m in eps]):.3f} / "
              f"{np.mean([m['std_uvw'][1] for m in eps]):.3f} / "
              f"{np.mean([m['std_uvw'][2] for m in eps]):.3f} m/s")
        print(f"yaw max       : {np.max([m['yaw_max'] for m in eps]):.1f} deg")
        print(f"pitch max     : {np.max([m['pitch_max'] for m in eps]):.1f} deg")
        out = out_dir / f"probe_{ts}.npz"
        np.savez(out, episodes=np.array(eps, dtype=object),
                 meta=dict(vars(args)))
        print(f"[probe] saved -> {out}")

    for c in (plot_nu, plot_cmd):
        if c is not None:
            c.close()
    scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
