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
    return p.parse_args()


def main() -> int:
    import numpy as np

    args = parse_args()
    PROJECT = Path(__file__).resolve().parents[1]
    out_dir = PROJECT / "EDMDc" / "data" / "tank"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    from .isaac_scene import GripperScene

    scene = GripperScene(dt=1 / 60, headless=True,
                         env_usd=str(PROJECT / "assets" / "environment_tank.usda"),
                         spawn_pos=np.array(CENTER))
    dt = scene.dt

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

    class Aprbs:
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

    def run_episode(ep):
        """Return per-step arrays: nu, eta, cmd, excite flag."""
        rng = np.random.default_rng(np.random.SeedSequence((args.seed, ep)))
        scene.reset_to_spawn()
        for _ in range(int(round(2.0 / dt))):
            scene.apply_wrench(np.zeros(4))
        aprbs = Aprbs(rng)
        n_max = int(round(args.seconds / dt))
        nus, etas, cmds, excite = [], [], [], []
        recovering = False
        for k in range(n_max):
            st = scene.read_state()
            if not recovering and box_violation(st.pos_w, st.yaw, st.pitch):
                recovering = True
            cmd = np.zeros(4)
            cmd[3] = yaw_hold(st)
            if recovering:
                cmd[:3] = center_pid(st, gain=3.0, clip=0.45)
                if recovered(st):
                    recovering = False
            else:
                cmd[:3] = np.clip(aprbs.step(k * dt) + center_pid(st), -0.9, 0.9)
            nus.append(st.nu.copy())
            etas.append(np.array([*st.pos_w, st.roll, st.pitch, st.yaw]))
            cmds.append(cmd.copy())
            excite.append(not recovering)
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
            dt=np.float64(dt), episode_seconds=np.float64(args.seconds),
            input_units="normalized_cmd4",
            aprbs_amp=np.float64(args.amp), aprbs_amp_z=np.float64(args.amp_z),
            aprbs_hold_max_s=np.float64(args.hold),
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

    save(out_dir / f"tank_train_{ts}.npz", train_chunks, n_train_eps)
    save(out_dir / f"tank_test_{ts}.npz", test_chunks, args.test_episodes)
    scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
