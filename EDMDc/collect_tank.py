"""Collect EDMDc training data in the small tank (continuous dither).

The validated tank excitation (see ``EDMDc.aprbs_tank_probe``): weak
station-keeping PID + continuous-amplitude APRBS on surge/sway/heave,
yaw HELD toward the +x wall (AprilTag localization), automatic
re-center-and-resume when the safety box is approached. Recovery
segments are recorded but NEVER contribute training pairs as inputs.

Control input logged as U is the TOTAL applied normalized ArduSub
command (APRBS + station-keeping PID + clip) — the tank analog of the
"record the realized wrench" rule: the model must see what was actually
applied, not just the excitation term. Units match what MAVLink
RC-override takes on the real vehicle, and what ``EDMDc.edmdc`` trains
on (CONTROL_DIM = 4).

Output: two npz files (train / test, split by whole episodes) in
``EDMDc/data/tank/``, fields compatible with ``EDMDc.edmdc``:

    X (N,6) nu   U (N,4) cmd   X_next (N,6)   Eta (N,6) pose
    traj_idx, step_idx, dt, episode_seconds, input_units

Run:

    conda activate marinegym && python -m EDMDc.collect_tank
    python -m EDMDc.edmdc EDMDc/data/tank/tank_train_<TS>.npz
"""
from __future__ import annotations

# Argparse FIRST — --help must work without booting Isaac.
import argparse
from datetime import datetime
from pathlib import Path

# Excitation recipe + safety box: single source of truth in the probe.
from .aprbs_tank_probe import (
    BOX_X, BOX_Y, BOX_Z_MIN, BOX_Z_MAX, CENTER,
    PITCH_LIMIT_DEG, YAW_LIMIT_DEG, Z_FLOOR,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=8,
                   help="Total episodes; the last --test-episodes go to the "
                        "test npz (default 8).")
    p.add_argument("--test-episodes", type=int, default=2)
    p.add_argument("--seconds", type=float, default=120.0,
                   help="Episode length [s] (default 120).")
    p.add_argument("--amp", type=float, default=0.45)
    p.add_argument("--amp-z", type=float, default=0.28)
    p.add_argument("--hold", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--control-hz", type=int, default=60,
                   help="Control/sampling rate [Hz]. Physics always steps at "
                        "60 Hz; the command is computed every 60/control_hz "
                        "ticks and zero-order-held in between, and states are "
                        "logged at those control instants (rehearses the real "
                        "robot's ~20 Hz state-estimate rate). Must divide 60.")
    p.add_argument("--infinite", action="store_true",
                   help="Infinite-pool free drift instead of the tank: deep "
                        "pool env, spawn (0,0,-5) far from surface and floor, "
                        "NO station-keeping and NO yaw hold (fully exogenous "
                        "input), episode ends only on tipover (|roll|/|pitch| "
                        "> 60 deg) or the generous pool-interior guard. "
                        "Output goes to EDMDc/data/free/.")
    p.add_argument("--yaw-amp", type=float, default=0.0,
                   help="Yaw APRBS amplitude (uniform +/-). 0 (default) = "
                        "yaw held toward the tags (tank behavior); > 0 "
                        "excites yaw — fixes the r-row identifiability.")
    p.add_argument("--hold-min", type=float, default=None,
                   help="APRBS hold-time lower bound [s]. Default: half of "
                        "--hold.")
    return p.parse_args()


def main() -> int:
    import numpy as np

    args = parse_args()
    PROJECT = Path(__file__).resolve().parents[1]
    out_dir = PROJECT / "EDMDc" / "data" / "tank"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    from .isaac_scene import GripperScene

    if 60 % args.control_hz:
        raise SystemExit(f"--control-hz must divide 60, got {args.control_hz}")
    decim = 60 // args.control_hz

    if args.infinite:
        env_file, spawn = "environment_deep_pool.usd", (0.0, 0.0, -5.0)
        out_dir = PROJECT / "EDMDc" / "data" / "free"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = "free"
    else:
        env_file, spawn = "environment_tank.usda", CENTER
        stem = "tank"

    scene = GripperScene(dt=1 / 60, headless=True,
                         env_usd=str(PROJECT / "assets" / env_file),
                         spawn_pos=np.array(spawn))
    dt = scene.dt                       # physics step (1/60)
    dt_ctrl = decim * dt                # control/sampling step
    print(f"[collect-tank] control rate {args.control_hz} Hz "
          f"(decimation {decim}, dt_ctrl={dt_ctrl:.4f} s), "
          f"mode={'infinite free-drift' if args.infinite else 'tank dither'}, "
          f"yaw_amp={args.yaw_amp}")

    def box_violation(pos, yaw, pitch):
        return (abs(pos[0]) > BOX_X or abs(pos[1]) > BOX_Y
                or not (BOX_Z_MIN < pos[2] < BOX_Z_MAX)
                or abs(np.rad2deg(yaw)) > YAW_LIMIT_DEG
                or abs(np.rad2deg(pitch)) > PITCH_LIMIT_DEG)

    def yaw_hold(st):
        return float(np.clip(-1.0 * st.yaw - 0.30 * st.nu[5], -0.30, 0.30))

    def center_pid(st, gain=1.0, clip=0.25):
        fb = np.zeros(3)
        fb[0] = np.clip(gain * 0.40 * (CENTER[0] - st.pos_w[0])
                        - 0.35 * st.nu[0], -clip, clip)
        fb[1] = np.clip(gain * 0.40 * (CENTER[1] - st.pos_w[1])
                        - 0.35 * st.nu[1], -clip, clip)
        fb[2] = np.clip(gain * 0.60 * (CENTER[2] - st.pos_w[2])
                        - 0.40 * st.nu[2], -clip * 1.2, clip * 1.2)
        return fb

    def recovered(st):
        return (abs(st.pos_w[0]) < 0.25 and abs(st.pos_w[1]) < 0.25
                and abs(st.pos_w[2] - CENTER[2]) < 0.08
                and float(np.linalg.norm(st.nu[:3])) < 0.08)

    hold_lo = args.hold_min if args.hold_min is not None else 0.5 * args.hold

    class Aprbs:
        """4-channel continuous-amplitude APRBS ([surge, sway, heave, yaw]).
        Heave draws sign*U(Z_FLOOR, amp_z) to clear the ESC deadband; yaw
        is uniform +/- yaw_amp (0 -> channel stays silent)."""

        def __init__(self, rng):
            self.rng = rng
            self.h = (hold_lo, args.hold)
            self.val = np.zeros(4)
            self.t_next = np.zeros(4)

        def step(self, t):
            for i in range(4):
                if t >= self.t_next[i]:
                    if i == 2:
                        lo = min(Z_FLOOR, args.amp_z)
                        mag = self.rng.uniform(lo, args.amp_z)
                        self.val[i] = mag * self.rng.choice((-1.0, 1.0))
                    elif i == 3:
                        self.val[i] = (self.rng.uniform(-args.yaw_amp,
                                                        args.yaw_amp)
                                       if args.yaw_amp > 0 else 0.0)
                    else:
                        self.val[i] = self.rng.uniform(-args.amp, args.amp)
                    self.t_next[i] = t + self.rng.uniform(*self.h)
            return self.val.copy()

    def guard_infinite(st):
        """Tipover + generous deep-pool interior guard (walls at +/-15,
        floor -10, surface -0.2)."""
        return (abs(np.rad2deg(st.roll)) > 60.0
                or abs(np.rad2deg(st.pitch)) > 60.0
                or abs(st.pos_w[0]) > 13.0 or abs(st.pos_w[1]) > 13.0
                or not (-9.3 < st.pos_w[2] < -0.6))

    def run_episode(ep):
        """Return per-step arrays: nu, eta, cmd, excite flag."""
        rng = np.random.default_rng(np.random.SeedSequence((args.seed, ep)))
        scene.reset_to_spawn()
        for _ in range(int(round(2.0 / dt))):
            scene.apply_wrench(np.zeros(4))
        aprbs = Aprbs(rng)
        n_ctrl = int(round(args.seconds * args.control_hz))
        nus, etas, cmds, excite = [], [], [], []
        recovering = False
        for k in range(n_ctrl):
            st = scene.read_state()
            if args.infinite:
                if guard_infinite(st):
                    print(f"[collect-tank]   ep guard trip at t={k * dt_ctrl:.1f}s "
                          f"(pos={np.round(st.pos_w, 2)}, "
                          f"roll={np.rad2deg(st.roll):+.0f} "
                          f"pitch={np.rad2deg(st.pitch):+.0f} deg)")
                    break
                cmd = aprbs.step(k * dt_ctrl)          # fully exogenous, 4-axis
            else:
                if not recovering and box_violation(st.pos_w, st.yaw, st.pitch):
                    recovering = True
                cmd = np.zeros(4)
                cmd[3] = yaw_hold(st)
                if recovering:
                    cmd[:3] = center_pid(st, gain=3.0, clip=0.45)
                    if recovered(st):
                        recovering = False
                else:
                    cmd[:3] = np.clip(
                        aprbs.step(k * dt_ctrl)[:3] + center_pid(st),
                        -0.9, 0.9)
            nus.append(st.nu.copy())
            etas.append(np.array([*st.pos_w, st.roll, st.pitch, st.yaw]))
            cmds.append(cmd.copy())
            excite.append(not recovering)
            for _ in range(decim):          # zero-order hold across 60 Hz ticks
                scene.apply_wrench(cmd)
        return (np.asarray(nus), np.asarray(etas), np.asarray(cmds),
                np.asarray(excite, dtype=bool))

    def pairs_from_episode(nu, eta, cmd, excite, ep):
        """Training pairs (nu_k, cmd_k) -> nu_{k+1}. The input step k must
        be exciting; the successor may be any state (a boundary sample is
        still a physically valid transition under cmd_k)."""
        k_idx = np.where(excite[:-1])[0]
        return (nu[k_idx], cmd[k_idx], nu[k_idx + 1], eta[k_idx],
                np.full(len(k_idx), ep, dtype=np.int32),
                k_idx.astype(np.int32))

    def save(path, chunks, episodes):
        X, U, Xn, Eta, ti, si = (np.concatenate(c) for c in zip(*chunks))
        np.savez(
            path, X=X, U=U, X_next=Xn, Eta=Eta, traj_idx=ti, step_idx=si,
            dt=np.float64(dt_ctrl), sim_dt=np.float64(dt),
            control_hz=np.int32(args.control_hz),
            episode_seconds=np.float64(args.seconds),
            input_units="normalized_cmd4",
            aprbs_amp=np.float64(args.amp), aprbs_amp_z=np.float64(args.amp_z),
            aprbs_yaw_amp=np.float64(args.yaw_amp),
            aprbs_hold_max_s=np.float64(args.hold),
            aprbs_hold_min_s=np.float64(hold_lo),
            mode=stem,
            z_floor=np.float64(Z_FLOOR), episodes=np.int32(episodes),
        )
        print(f"[collect-tank] saved {X.shape[0]:,} pairs "
              f"({episodes} eps) -> {path.name}")

    train_chunks, test_chunks = [], []
    n_train_eps = args.episodes - args.test_episodes
    for ep in range(args.episodes):
        nu, eta, cmd, excite = run_episode(ep)
        chunk = pairs_from_episode(nu, eta, cmd, excite, ep)
        (train_chunks if ep < n_train_eps else test_chunks).append(chunk)
        print(f"[collect-tank] ep{ep}: {int(excite.sum()):,}/{len(excite):,} "
              f"exciting steps, |nu| max={np.linalg.norm(nu[:, :3], axis=1).max():.3f}")

    save(out_dir / f"{stem}_train_{ts}.npz", train_chunks, n_train_eps)
    save(out_dir / f"{stem}_test_{ts}.npz", test_chunks, args.test_episodes)
    scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
