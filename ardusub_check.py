"""Bring-up + calibration harness for the ArduSub MANUAL-mode mixer (the GATE).

Run this BEFORE wiring ArduSub into nav/data_collection. It:

  1. Connects to ArduSub SITL (spawning it if needed), arms in MANUAL.
  2. Drives each MANUAL_CONTROL axis (x,y,z,r) one at a time and reads back the
     8 motor commands (SERVO_OUTPUT_RAW).
  3. Solves the firmware-motor -> YAML-thruster permutation + per-motor sign by
     matching each motor's 4-axis response to the YAML thrusters' ideal response
     (cosine similarity). No reliance on convention guesses.
  4. VERIFIES the resulting input->realized-wrench map is diagonal (each axis
     produces its own wrench, negligible cross-coupling), and extracts the
     per-axis gains. Asserts loudly if the map is not clean.
  5. Saves the calibration for ardusub_bridge.ArduSubMixer.
  6. Compares ArduSub-realized wrench vs the analytic inscribed-box allocation
     across an amplitude sweep -> quantifies how much saturation reallocation
     changes the effective input (open-loop diagnostic, working-style #4).

Usage:  python ardusub_check.py            (uses BlueROVHeavy.yaml)
"""

from __future__ import annotations

import os
import numpy as np
import yaml

from EDMDc.thrusters import ThrusterConfig, T200Group, ThrustAllocator, sum_to_wrench
from ardusub_bridge import (ArduSubLink, ArduSubCalib, Z_NEUTRAL, CALIB_PATH)

PROJECT = os.path.dirname(os.path.abspath(__file__))
VEHICLE_YAML = os.path.join(PROJECT, "assets", "BlueROVHeavy", "BlueROVHeavy.yaml")
MAX_THRUST_N = 25.0
AXES = ("Fx", "Fy", "Fz", "Tz")     # the 4 active wrench components
WRENCH_IDX = (0, 1, 2, 5)           # their indices in the 6-vector


def load_geometry():
    with open(VEHICLE_YAML) as f:
        tcfg = ThrusterConfig.from_yaml_dict(yaml.safe_load(f))
    t200 = T200Group(tcfg)
    alloc = ThrustAllocator(tcfg, max_thrust_per_motor=MAX_THRUST_N)
    return tcfg, t200, alloc


def realized_wrench(tcfg, t200, cmd_yaml):
    """8 commands in YAML order -> approximate steady 6-DOF body wrench."""
    t200.reset()
    thrusts = np.zeros(tcfg.num_rotors)
    for _ in range(180):
        thrusts, _ = t200.step(np.asarray(cmd_yaml), dt=1.0 / 60.0)
    return sum_to_wrench(tcfg, thrusts)


def solve_mapping(link, alloc, amp=300):
    """Drive each axis, match firmware motors to YAML thrusters.

    Returns (perm, sign, R) where R[i,a] is motor i's command response to a
    unit-ish push on axis a (x,y,z,r)."""
    # firmware response to each axis push (z uses the 500-neutral convention)
    pushes = {
        "Fx": dict(x=amp, y=0, z=Z_NEUTRAL, r=0),
        "Fy": dict(x=0, y=amp, z=Z_NEUTRAL, r=0),
        "Fz": dict(x=0, y=0, z=Z_NEUTRAL + amp, r=0),
        "Tz": dict(x=0, y=0, z=Z_NEUTRAL, r=amp),
    }
    neutral = link.drive_read(0, 0, Z_NEUTRAL, 0, hold=1.0)
    R = np.zeros((8, 4))
    for a, ax in enumerate(AXES):
        cmd = link.drive_read(hold=1.0, **pushes[ax])
        R[:, a] = cmd - neutral
        link.drive_read(0, 0, Z_NEUTRAL, 0, hold=0.3)       # relax to neutral

    # YAML thrusters' ideal response to a unit wrench on each axis (analytic).
    Y = np.zeros((8, 4))
    for a, wi in enumerate(WRENCH_IDX):
        w = np.zeros(6); w[wi] = 5.0
        Y[:, a] = alloc.allocate(w)

    # Normalize per-axis (column) so a motor's identity is set by WHICH thrusters
    # participate in each axis (the pattern), not by each axis's native command
    # scale (yaw needs far more thrust per unit wrench than surge/sway).
    def col_norm(M):
        n = np.linalg.norm(M, axis=0, keepdims=True)
        return M / (n + 1e-12)
    Rn, Yn = col_norm(R), col_norm(Y)

    # Match motor i -> thruster j by cosine of their 4-axis response vectors.
    perm = -np.ones(8, dtype=int)
    sign = np.zeros(8)
    used = set()
    cosines = np.zeros(8)
    for i in range(8):
        ri = Rn[i]
        best_j, best_c = -1, 0.0
        for j in range(8):
            if j in used:
                continue
            yj = Yn[j]
            denom = (np.linalg.norm(ri) * np.linalg.norm(yj)) + 1e-12
            c = float(ri @ yj) / denom
            if abs(c) > abs(best_c):
                best_c, best_j = c, j
        perm[i] = best_j
        sign[i] = 1.0 if best_c >= 0 else -1.0
        cosines[i] = best_c
        used.add(best_j)
    return perm, sign, R, cosines


def measure_gains(link, tcfg, t200, perm, sign, amp=300):
    """Per-axis input->realized-wrench, returns k (input units / Newton) and the
    full 4x4 input->wrench matrix G (for the diagonality check)."""
    def fw_to_yaml(cmd_fw):
        out = np.zeros(8)
        out[perm] = sign * cmd_fw
        return out

    inputs = {
        "Fx": dict(x=amp, y=0, z=Z_NEUTRAL, r=0),
        "Fy": dict(x=0, y=amp, z=Z_NEUTRAL, r=0),
        "Fz": dict(x=0, y=0, z=Z_NEUTRAL + amp, r=0),
        "Tz": dict(x=0, y=0, z=Z_NEUTRAL, r=amp),
    }
    G = np.zeros((4, 4))        # G[in_axis, wrench_axis]
    for a, ax in enumerate(AXES):
        cmd_fw = link.drive_read(hold=1.0, **inputs[ax])
        W = realized_wrench(tcfg, t200, fw_to_yaml(cmd_fw))
        G[a] = W[list(WRENCH_IDX)]
        link.drive_read(0, 0, Z_NEUTRAL, 0, hold=0.3)
    # gain: input amplitude per realized Newton on the intended axis (signed).
    diag = np.array([G[a, a] for a in range(4)])
    k = amp / diag
    return k, G


def compare_inscribed(link, tcfg, t200, alloc, perm, sign, k):
    """Sweep each axis over a range of commanded wrench and compare the wrench
    ArduSub actually realizes vs the analytic inscribed-box allocator."""
    from ardusub_bridge import PWM_NEUTRAL  # noqa
    caps = {"Fx": 65.0, "Fy": 37.5, "Fz": 90.0, "Tz": 17.0}
    print("\n=== ArduSub-realized vs inscribed-box (per-axis sweep) ===")
    print("axis  cmd_wrench   inscribed_realized   ardusub_realized   (intended axis)")
    for a, ax in enumerate(AXES):
        wi = WRENCH_IDX[a]
        for frac in (0.5, 1.0, 1.5):       # 1.5x intentionally pushes past box
            mag = caps[ax] * frac
            w = np.zeros(6); w[wi] = mag
            # inscribed-box path
            cmd_box = np.clip(alloc.allocate(w), -1, 1)
            W_box = realized_wrench(tcfg, t200, cmd_box)
            # ardusub path
            x = int(np.clip(k[0] * w[0], -1000, 1000))
            y = int(np.clip(k[1] * w[1], -1000, 1000))
            z = int(np.clip(Z_NEUTRAL + k[2] * w[2], 0, 1000))
            r = int(np.clip(k[3] * w[5], -1000, 1000))
            cmd_fw = link.drive_read(x, y, z, r, hold=0.8)
            cmd_yaml = np.zeros(8); cmd_yaml[perm] = sign * cmd_fw
            W_as = realized_wrench(tcfg, t200, cmd_yaml)
            link.drive_read(0, 0, Z_NEUTRAL, 0, hold=0.2)
            print(f"{ax}  {mag:8.2f}   {W_box[wi]:14.2f}     {W_as[wi]:14.2f}")


def main():
    tcfg, t200, alloc = load_geometry()
    print("connecting to ArduSub SITL (spawns if needed)...")
    link = ArduSubLink(spawn=True)
    try:
        perm, sign, R, cosines = solve_mapping(link, alloc)
        print("\n=== firmware motor -> YAML thruster mapping ===")
        for i in range(8):
            print(f"  Motor{i+1} (servo{i+1}) -> thr{perm[i]}  sign={sign[i]:+.0f}"
                  f"  cos={cosines[i]:+.3f}")
        assert len(set(perm.tolist())) == 8, f"perm not a bijection: {perm}"
        assert np.all(np.abs(cosines) > 0.9), \
            f"weak motor match (cos<0.9): {cosines}"

        k, G = measure_gains(link, tcfg, t200, perm, sign)
        print("\n=== input->realized-wrench matrix G (rows=input axis x,y,z,r) ===")
        print("            Fx        Fy        Fz        Tz")
        for a, ax in enumerate(AXES):
            print(f"  {ax}:  " + "  ".join(f"{G[a,b]:8.2f}" for b in range(4)))
        # diagonality: off-diagonal energy should be small vs diagonal.
        offdiag = G - np.diag(np.diag(G))
        ratio = np.linalg.norm(offdiag) / (np.linalg.norm(np.diag(G)) + 1e-9)
        print(f"  off-diagonal/diagonal ratio = {ratio:.3f}  (want << 1)")
        print(f"\n=== per-axis gains k (MANUAL_CONTROL units / Newton) ===")
        for a, ax in enumerate(AXES):
            print(f"  {ax}: k={k[a]:+.3f}")
        assert ratio < 0.2, f"input->wrench not diagonal (ratio={ratio:.3f})"
        assert np.all(np.isfinite(k)), f"non-finite gain: {k}"

        print(f"\n=== safety factor (pilot gain) ===")
        print(f"  JS_GAIN_DEFAULT={link.gain}  JS_THR_GAIN={link.throttle_gain}"
              f"  (full-stick authority scales ~linearly with gain)")

        # Wrench -> per-thruster demand map (firmware motor order), for the
        # no-saturation clamp. R[i,a] is motor i's command at an `amp`-unit push
        # on axis a; k[a] is input-units/Newton, so demand_i = sum_a B[i,a]*W_a
        # with B[i,a] = R[i,a]*k[a]/amp. Saturation iff max_i|demand_i| > 1. The
        # clamp scales the whole wrench by 1/max so NO thruster clips — the
        # combined/diagonal magnitude may exceed a single axis (that's fine; the
        # limit is per-thruster, not a sphere).
        AMP = 300.0                      # push amplitude used in solve/measure
        demand = R * (k / AMP)           # (8, 4)
        print("=== no-saturation clamp: single-axis authority + which limit binds ===")
        for a, ax in enumerate(AXES):
            cap = (500.0 if ax == "Fz" else 1000.0) / abs(k[a])   # stick-range authority
            motor_at_cap = float(np.max(np.abs(demand[:, a]))) * cap
            binds = "stick" if motor_at_cap < 1.0 else "motor"
            print(f"  {ax}: authority={cap:6.2f}  thruster-demand-at-cap={motor_at_cap:.2f} "
                  f"-> {binds}-limited")
        ArduSubCalib(perm=perm, sign=sign, k=k, demand=demand,
                     gain=link.gain, throttle_gain=link.throttle_gain).save()
        print(f"[saved] calibration -> {CALIB_PATH}")

        compare_inscribed(link, tcfg, t200, alloc, perm, sign, k)
        print("\nGATE PASSED: mapping bijective, wrench diagonal, calibration saved.")
    finally:
        link.close()


if __name__ == "__main__":
    main()
