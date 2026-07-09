"""Route controller wrench through a real ArduSub (SITL) MANUAL-mode mixer.

This replaces the analytic `ThrustAllocator` (inscribed-box pseudo-inverse) with
ArduSub's *actual* motor mixing + saturation reallocation, so the simulated
control path matches what the vehicle does on deployment through BlueOS/ArduSub.

Pipeline (per control tick), MANUAL mode (no ArduSub inner loops):

    controller wrench [Fx Fy Fz Tx Ty Tz]   (Tx,Ty are 0 in our 4-DOF setup)
      -> MANUAL_CONTROL(x,y,z,r)             (normalize; drop roll/pitch)
      -> ArduSub AP_Motors6DOF mix + uniform saturation scaling
      -> SERVO_OUTPUT_RAW (8 motor PWM)
      -> per-motor command in [-1,1], reordered to OUR YAML thruster index
      -> (downstream unchanged) T200Group.step -> sum_to_wrench -> fossen

Because MANUAL-mode motor output is a *static, state-independent* function of the
4-axis input, we do NOT feed sensors back to SITL — we send MANUAL_CONTROL and
read SERVO_OUTPUT_RAW. The conversion constants (motor->thruster permutation,
per-motor sign, wrench->input gains) are produced by `ardusub_check.py` and saved
to a calibration file; this module loads them. Run that harness first.

Conventions discovered in bring-up (see ardusub_check.py / commit notes):
  - FRAME_CONFIG=2 (Vectored_6DOF) matches BlueROVHeavy.yaml (8 thrusters).
  - SERVO1..8 -> Motor1..8, but the SITL default leaves SERVO8_FUNCTION unset to a
    motor; we force SERVO8_FUNCTION=40 (k_motor8) on connect.
  - MANUAL_CONTROL: x,y,r in [-1000,1000] (0=neutral); z in [0,1000] (500=neutral).
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np

# --- environment / paths (override via env vars) -------------------------------
AP_ROOT = os.environ.get("ARDUPILOT_ROOT", os.path.expanduser("~/ardupilot"))
ARDUSUB_BIN = os.environ.get(
    "ARDUSUB_BIN", os.path.join(AP_ROOT, "build", "sitl", "bin", "ardusub"))
ARDUSUB_DEFAULTS = os.environ.get(
    "ARDUSUB_DEFAULTS",
    ",".join(os.path.join(AP_ROOT, "Tools", "autotest", "default_params", p)
             for p in ("sub.parm", "sub-6dof.parm")))
ARDUSUB_MODEL = os.environ.get("ARDUSUB_MODEL", "vectored_6dof")
ARDUSUB_CONN = os.environ.get("ARDUSUB_CONN", "tcp:127.0.0.1:5760")
ARDUSUB_RUNDIR = os.environ.get("ARDUSUB_RUNDIR", "/tmp/ardusub_sitl")
ARDUSUB_SPEEDUP = float(os.environ.get("ARDUSUB_SPEEDUP", "1"))
# The pilot input "safety factor": ArduSub scales MANUAL_CONTROL by `gain`
# (JS_GAIN_DEFAULT). At gain=0.5 (BlueOS default) full stick reaches only ~half
# of motor authority; at 1.0 it reaches full. Heave is additionally scaled by
# JS_THR_GAIN. ArduSub latches `gain` at boot (init_joystick), so we set it via a
# launch-time param overlay, NOT at runtime. The calibration is gain-specific.
ARDUSUB_GAIN = float(os.environ.get("ARDUSUB_GAIN", "0.5"))
ARDUSUB_THR_GAIN = float(os.environ.get("ARDUSUB_THR_GAIN", "1.0"))
CALIB_PATH = os.environ.get(
    "ARDUSUB_CALIB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "data", "ardusub_calib.npz"))

PWM_NEUTRAL = 1500.0
PWM_PER_CMD = 400.0          # command in [-1,1] -> PWM 1500 +/- 400 (1100..1900)
Z_NEUTRAL = 500              # MANUAL_CONTROL z neutral (0..1000 convention)
SERVO_OUTPUT_RAW_ID = 36
K_MOTOR8_FUNC = 40           # SRV_Channel k_motor8


# ------------------------------------------------------------------------------
# Calibration: the only vehicle/firmware-specific constants the runtime needs.
# Produced by ardusub_check.py.
# ------------------------------------------------------------------------------
@dataclass
class ArduSubCalib:
    """Maps the firmware's motor outputs to our YAML thruster convention, and
    the controller wrench to MANUAL_CONTROL inputs.

    perm:  (8,) int — perm[i] = YAML thruster index that SITL servo-channel i
           (Motor i+1) corresponds to.
    sign:  (8,) +/-1 — command sign flip (firmware motor +dir vs YAML +thrust_axis).
    k:     (4,) signed MANUAL_CONTROL units per Newton for (Fx, Fy, Fz, Tz).
           x = k0*Fx, y = k1*Fy, z = 500 + k2*Fz, r = k3*Tz (then clipped).
    gain:  pilot "safety factor" (JS_GAIN_DEFAULT) this calibration was taken at.
           k and the saturation ceiling are gain-specific; the bridge verifies the
           live SITL booted with this gain.
    throttle_gain: JS_THR_GAIN at calibration time (extra heave scaling).
    demand: (8,4) wrench->per-thruster demand map (firmware motor order) for the
           no-saturation clamp: demand_i = sum_a demand[i,a]*W_a; saturation iff
           max_i|demand_i| > 1. None if the calibration predates the clamp.
    """
    perm: np.ndarray
    sign: np.ndarray
    k: np.ndarray
    gain: float = ARDUSUB_GAIN
    throttle_gain: float = ARDUSUB_THR_GAIN
    demand: np.ndarray = None

    def save(self, path: str = CALIB_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arrs = dict(perm=self.perm, sign=self.sign, k=self.k,
                    gain=self.gain, throttle_gain=self.throttle_gain)
        if self.demand is not None:
            arrs["demand"] = self.demand
        np.savez(path, **arrs)

    @classmethod
    def load(cls, path: str = CALIB_PATH) -> "ArduSubCalib":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ArduSub calibration not found at {path}. "
                f"Run `python ardusub_check.py` first to generate it.")
        d = np.load(path)
        return cls(perm=d["perm"].astype(int), sign=d["sign"].astype(float),
                   k=d["k"].astype(float),
                   gain=float(d["gain"]) if "gain" in d else ARDUSUB_GAIN,
                   throttle_gain=(float(d["throttle_gain"])
                                  if "throttle_gain" in d else ARDUSUB_THR_GAIN),
                   demand=(d["demand"].astype(float) if "demand" in d else None))


# ------------------------------------------------------------------------------
# SITL process lifecycle
# ------------------------------------------------------------------------------
def _heartbeat_reachable(conn: str = ARDUSUB_CONN, timeout: float = 2.0) -> bool:
    """Quick check: is a SITL already up at `conn`? (TCP connect probe.)"""
    if not conn.startswith("tcp:"):
        return False
    try:
        host, port = conn[len("tcp:"):].split(":")
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _write_overlay(gain: float, throttle_gain: float) -> str:
    """Launch-time param overlay: motor8 output, headless arming, and the
    boot-latched pilot gain ("safety factor"). Returns the overlay path."""
    path = os.path.join(ARDUSUB_RUNDIR, "pool_overlay.parm")
    with open(path, "w") as f:
        f.write(f"SERVO8_FUNCTION {K_MOTOR8_FUNC}\n")
        f.write("ARMING_CHECK 0\n")
        f.write(f"JS_GAIN_DEFAULT {gain}\n")
        f.write(f"JS_THR_GAIN {throttle_gain}\n")
    return path


def ensure_sitl(*, gain: float = ARDUSUB_GAIN,
                throttle_gain: float = ARDUSUB_THR_GAIN,
                wait_heartbeat: bool = True, timeout: float = 40.0):
    """Make sure an ArduSub SITL is running and reachable at ARDUSUB_CONN.

    If one is already listening, reuse it (its gain is whatever it booted with;
    ArduSubLink verifies it matches). Otherwise spawn the ardusub binary as a
    detached background process with a param overlay that sets the pilot gain at
    boot. Returns the Popen handle if we spawned one, else None.
    """
    if _heartbeat_reachable():
        return None
    if not os.path.exists(ARDUSUB_BIN):
        raise FileNotFoundError(
            f"ardusub SITL binary not found at {ARDUSUB_BIN}. Build it with:\n"
            f"  cd {AP_ROOT} && ./waf configure --board sitl && ./waf sub\n"
            f"(or set ARDUSUB_BIN).")
    os.makedirs(ARDUSUB_RUNDIR, exist_ok=True)
    overlay = _write_overlay(gain, throttle_gain)
    cmd = [ARDUSUB_BIN, "-I0", "--model", ARDUSUB_MODEL,
           "--speedup", str(ARDUSUB_SPEEDUP),
           "--defaults", f"{ARDUSUB_DEFAULTS},{overlay}"]
    proc = subprocess.Popen(
        cmd, cwd=ARDUSUB_RUNDIR,
        stdout=open(os.path.join(ARDUSUB_RUNDIR, "sitl.log"), "w"),
        stderr=subprocess.STDOUT, start_new_session=True)
    if wait_heartbeat:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if _heartbeat_reachable(timeout=1.0):
                break
            if proc.poll() is not None:
                raise RuntimeError(
                    f"ardusub SITL exited (code {proc.returncode}); see "
                    f"{ARDUSUB_RUNDIR}/sitl.log")
            time.sleep(0.5)
        else:
            raise TimeoutError(f"SITL did not come up within {timeout}s")
    return proc


# ------------------------------------------------------------------------------
# Low-level MAVLink link (connect / arm / drive / read)
# ------------------------------------------------------------------------------
class ArduSubLink:
    """Thin pymavlink wrapper for MANUAL-mode mixer use.

    Owns the MAVLink connection, forces the params we need (motor8 output,
    arming checks off), holds the vehicle in MANUAL + armed, streams
    SERVO_OUTPUT_RAW, and exposes send-command / read-motors primitives.
    """

    def __init__(self, conn: str = ARDUSUB_CONN, *, spawn: bool = True,
                 servo_rate_hz: float = 50.0, stream_hz: float = 50.0,
                 read_settle: float = 0.05, gain: float = ARDUSUB_GAIN,
                 throttle_gain: float = ARDUSUB_THR_GAIN):
        self.gain = float(gain)
        self.throttle_gain = float(throttle_gain)
        if spawn:
            self._proc = ensure_sitl(gain=self.gain,
                                     throttle_gain=self.throttle_gain)
        else:
            self._proc = None
        from pymavlink import mavutil          # local import: optional dep
        self._mav = mavutil
        self.master = mavutil.mavlink_connection(conn, source_system=255)
        self.master.wait_heartbeat(timeout=30)
        self.tgt = self.master.target_system
        self.tcomp = self.master.target_component
        self._send_lock = threading.Lock()
        self._init_params()
        self._verify_gain()
        self._set_mode("MANUAL")
        self._request_servo_stream(servo_rate_hz)
        self.arm()
        # Background streamer: ArduSub neutralizes input that isn't streamed
        # (pilot-input failsafe, FS_PILOT_TIMEOUT) and has a ~0.3 s startup gate
        # before accepting input. Continuously sending the latest setpoint keeps
        # us past both, so per-tick allocate() can set-then-read with low latency.
        self._setpoint = (0, 0, Z_NEUTRAL, 0)
        self._read_settle = float(read_settle)
        self._stream_dt = 1.0 / float(stream_hz)
        self._running = True
        self._sender = threading.Thread(target=self._stream_loop, daemon=True)
        self._sender.start()
        time.sleep(0.6)              # warm past the startup-neutral gate

    def _stream_loop(self) -> None:
        while self._running:
            x, y, z, r = self._setpoint
            self.send_manual(x, y, z, r)
            time.sleep(self._stream_dt)

    def get_param(self, name: str, timeout: float = 3.0):
        self.master.mav.param_request_read_send(self.tgt, self.tcomp,
                                                name.encode(), -1)
        for _ in range(40):
            m = self.master.recv_match(type="PARAM_VALUE", blocking=True,
                                       timeout=timeout)
            if m and m.param_id.strip("\x00") == name:
                return float(m.param_value)
        return None

    def _verify_gain(self) -> None:
        """The pilot gain is boot-latched; ensure the live SITL matches what this
        link/calibration expects (a reused SITL may have booted differently)."""
        live = self.get_param("JS_GAIN_DEFAULT")
        if live is not None and abs(live - self.gain) > 1e-3:
            raise RuntimeError(
                f"ArduSub SITL booted with JS_GAIN_DEFAULT={live} but this run "
                f"expects gain={self.gain}. The pilot gain is latched at boot; "
                f"stop the running SITL so it can be relaunched with the right "
                f"gain (it is set via the launch param overlay).")

    # --- setup -----------------------------------------------------------------
    def _set_param(self, name: str, val: float) -> None:
        self.master.mav.param_set_send(
            self.tgt, self.tcomp, name.encode(), float(val),
            self._mav.mavlink.MAV_PARAM_TYPE_REAL32)
        time.sleep(0.05)

    def _init_params(self) -> None:
        # SITL default leaves Motor8 unwired; arming checks block headless arm.
        self._set_param("SERVO8_FUNCTION", K_MOTOR8_FUNC)
        self._set_param("ARMING_CHECK", 0)

    def _set_mode(self, name: str) -> None:
        self.master.set_mode(name)
        time.sleep(0.2)

    def _request_servo_stream(self, hz: float) -> None:
        self.master.mav.command_long_send(
            self.tgt, self.tcomp,
            self._mav.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            SERVO_OUTPUT_RAW_ID, int(1e6 / hz), 0, 0, 0, 0, 0)

    # --- arm / drive / read ----------------------------------------------------
    def arm(self) -> None:
        for force in (0, 21196):                # try normal, then force-arm magic
            self.master.mav.command_long_send(
                self.tgt, self.tcomp,
                self._mav.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                1, force, 0, 0, 0, 0, 0)
            ack = self.master.recv_match(type="COMMAND_ACK", blocking=True,
                                         timeout=3)
            if ack and ack.result == 0:
                return
        raise RuntimeError("failed to arm ArduSub SITL")

    def disarm(self) -> None:
        self.master.mav.command_long_send(
            self.tgt, self.tcomp,
            self._mav.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            0, 0, 0, 0, 0, 0, 0)

    def send_manual(self, x: int, y: int, z: int, r: int) -> None:
        """x,y,r in [-1000,1000]; z in [0,1000]. Called by the streamer thread."""
        with self._send_lock:
            self.master.mav.manual_control_send(
                self.tgt, int(x), int(y), int(z), int(r), 0)

    def read_motors(self, drain: float = 0.03) -> np.ndarray:
        """Newest SERVO_OUTPUT_RAW as 8 commands in [-1,1] (servo channels 1..8).

        Drains the queued messages for `drain` s and returns the freshest one;
        callers settle the setpoint first so the value already reflects it."""
        t0 = time.time()
        last = None
        while time.time() - t0 < drain:
            m = self.master.recv_match(type="SERVO_OUTPUT_RAW", blocking=True,
                                       timeout=0.05)
            if m is not None:
                last = m
        if last is None:                         # nothing queued — wait for one
            last = self.master.recv_match(type="SERVO_OUTPUT_RAW", blocking=True,
                                          timeout=1.0)
        if last is None:
            raise RuntimeError("no SERVO_OUTPUT_RAW received")
        pwm = np.array([getattr(last, f"servo{i}_raw") for i in range(1, 9)],
                       dtype=float)
        return (pwm - PWM_NEUTRAL) / PWM_PER_CMD

    def drive_read(self, x, y, z, r, *, hold: float = 0.0) -> np.ndarray:
        """Set the streamed MANUAL_CONTROL setpoint, settle, and return the 8
        realized commands. `hold` extends the settle for calibration (steady
        state); per-tick control uses the default read_settle."""
        self._setpoint = (int(x), int(y), int(z), int(r))
        time.sleep(max(self._read_settle, hold))
        return self.read_motors()

    def close(self) -> None:
        self._running = False
        if getattr(self, "_sender", None) is not None:
            self._sender.join(timeout=1.0)
        try:
            self.disarm()
            self.master.close()
        except Exception:
            pass


# ------------------------------------------------------------------------------
# Drop-in allocator: same interface as ThrustAllocator.allocate(wrench)->cmds[8]
# ------------------------------------------------------------------------------
class ArduSubMixer:
    """Drop-in replacement for `ThrustAllocator` that routes the wrench through
    ArduSub's MANUAL-mode mixer. `allocate(wrench6)` returns 8 per-thruster
    commands in [-1,1] in OUR YAML thruster order, ready for `T200Group.step`.
    """

    def __init__(self, *, link: ArduSubLink | None = None,
                 calib: ArduSubCalib | None = None, spawn: bool = True,
                 clamp_saturation: bool = True):
        self.calib = calib if calib is not None else ArduSubCalib.load()
        # No-saturation clamp: scale any wrench whose realization would clip a
        # thruster (per-thruster limit, NOT a magnitude/sphere cap). Needs the
        # demand map from calibration; off if the calib predates it.
        self.clamp_saturation = bool(clamp_saturation) and self.calib.demand is not None
        self.last_scale = 1.0
        # Spawn/verify the link at the gain ("safety factor") the calibration was
        # taken at, so k and the saturation ceiling are consistent.
        self.link = link if link is not None else ArduSubLink(
            spawn=spawn, gain=self.calib.gain,
            throttle_gain=self.calib.throttle_gain)

    def saturation_scale(self, wrench6: np.ndarray) -> float:
        """Largest s in (0,1] so s*wrench triggers NO saturation, anywhere.

        Two distinct limits, whichever binds first (both scale the whole wrench,
        so direction is preserved either way):
          - stick range: MANUAL_CONTROL inputs clip at +/-1000 (z half-range 500).
            This is what caps single-axis authority at gain 0.5.
          - per-thruster: a motor command would exceed [-1,1]. Binds for
            multi-axis combos even when each axis is within its stick range.
        A surge+sway diagonal can still exceed a single axis (combined ~sqrt(2)x)
        because neither limit is a magnitude/sphere cap. 1.0 = already feasible.
        """
        if not self.clamp_saturation:
            return 1.0
        Fx, Fy, Fz, Tz = wrench6[0], wrench6[1], wrench6[2], wrench6[5]
        k = self.calib.k
        stick = max(abs(k[0] * Fx) / 1000.0, abs(k[1] * Fy) / 1000.0,
                    abs(k[2] * Fz) / 500.0, abs(k[3] * Tz) / 1000.0)
        motor = float(np.max(np.abs(self.calib.demand @ np.array([Fx, Fy, Fz, Tz]))))
        worst = max(stick, motor)
        return 1.0 if worst <= 1.0 else 1.0 / worst

    def wrench_to_manual(self, wrench6: np.ndarray) -> tuple[int, int, int, int]:
        """4-DOF wrench (Fx,Fy,Fz,Tz; Tx,Ty ignored) -> MANUAL_CONTROL ints."""
        k = self.calib.k
        Fx, Fy, Fz, Tz = wrench6[0], wrench6[1], wrench6[2], wrench6[5]
        x = int(np.clip(k[0] * Fx, -1000, 1000))
        y = int(np.clip(k[1] * Fy, -1000, 1000))
        z = int(np.clip(Z_NEUTRAL + k[2] * Fz, 0, 1000))
        r = int(np.clip(k[3] * Tz, -1000, 1000))
        return x, y, z, r

    def allocate(self, wrench6: np.ndarray) -> np.ndarray:
        """wrench [Fx Fy Fz Tx Ty Tz] -> 8 commands in [-1,1], YAML order.

        With clamp_saturation (default), the wrench is first scaled down — if and
        only if it would clip a thruster — so ArduSub never saturates and the
        realized direction is preserved exactly."""
        wrench6 = np.asarray(wrench6, dtype=float)
        if abs(wrench6[3]) > 1e-6 or abs(wrench6[4]) > 1e-6:
            # 4-DOF interface: roll/pitch torque cannot be commanded via
            # MANUAL_CONTROL and is silently dropped. Our controllers don't
            # request it, so warn loudly if that assumption ever breaks.
            import warnings
            warnings.warn(f"ArduSubMixer dropping roll/pitch torque "
                          f"Tx={wrench6[3]:.3g} Ty={wrench6[4]:.3g}")
        self.last_scale = self.saturation_scale(wrench6)
        if self.last_scale < 1.0:
            wrench6 = wrench6 * self.last_scale   # direction-preserving, no clip
        x, y, z, r = self.wrench_to_manual(wrench6)
        cmd_fw = self.link.drive_read(x, y, z, r)        # firmware motor order
        cmd_yaml = np.zeros(8)
        cmd_yaml[self.calib.perm] = self.calib.sign * cmd_fw
        return cmd_yaml

    def close(self) -> None:
        self.link.close()
