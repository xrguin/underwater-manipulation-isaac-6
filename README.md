# Underwater manipulation — Isaac Sim 6

Focused BlueROV Heavy EDMDc + Isaac Sim workspace using an ArduSub-style
4-axis control interface.

## Runtime

The machine-local `rov_isaac6` Conda environment provides short commands:

```bash
conda activate rov_isaac6
python -m EDMDc.teleop_tank
```

Its environment-local `python` and `python3` launchers deliberately dispatch
to Isaac Sim 6's bundled Python 3.12.13 while applying the required Kit
library and extension paths. Anaconda's own CPython 3.12 binary is not used to
host Kit. The direct equivalent remains:

```bash
/home/miaodong/Documents/isaac-sim-6.0/python.sh -m EDMDc.teleop_tank
```

## Control Interface

The normal control input is:

```text
u = [surge, sway, heave, yaw] in [-1, 1]^4
```

This is the ArduSub MANUAL_CONTROL stick convention:

```text
surge -> x
sway  -> y
heave -> z around neutral
yaw   -> r
```

The simulation path is:

```text
4-axis command [-1,1]
  -> EDMDc.ardusub_allocator.StaticArduSubAllocator
  -> 8 YAML-order thruster commands
  -> optional ArduSub STABILIZE roll/pitch motor mixing
  -> T200 motor lag/thrust model
  -> realized 6-DOF body wrench
  -> Fossen hydrodynamics + Isaac rigid-body dynamics
```

The saved calibration file is `data/ardusub_calib.npz`. The copied
`ardusub_bridge.py` and `ardusub_check.py` are kept for calibration/reference;
the default sim loop does not require live ArduSub SITL.

## ArduSub STABILIZE-style attitude control

`EDMDc.teleop_tank`, `EDMDc.backstep_edmdc_lqr`, and
`EDMDc.backstep_edmdc_mpc` now default to the axis-ownership contract used by
the real `EDMDc_bluerov/control_v2/keyboard_stabilize.py` launcher:

- roll and pitch target level using differential vertical-thruster commands;
- heave remains the existing direct normalized command, with no depth hold;
- surge, sway, and yaw remain the existing controller/operator commands; and
- the full six-DOF Fossen/Isaac plant remains active.

The implementation is in `EDMDc/ardusub_stabilize.py`. Motor patterns are
derived from the YAML thruster geometry, and correction is inserted before
the existing T200 lag/thrust model. The real-robot repository does not save
the onboard ArduSub attitude parameters, so the simulator defaults are
explicit configurable P/P gains (`angle_p=(4.5,4.5)`,
`rate_p=(0.2,0.2)`) with a per-motor correction limit of `0.35`; they are not
presented as an exact copy of an unknown flight-controller parameter file.

Use the unstabilized plant explicitly when reproducing an old control run:

```bash
python -m EDMDc.teleop_tank --attitude-mode manual
python -m EDMDc.backstep_edmdc_lqr --attitude-mode manual
python -m EDMDc.backstep_edmdc_mpc --attitude-mode manual
```

`GripperScene` itself keeps `ardusub_stabilize=False` by default. Existing
dataset collectors therefore do not change behavior or schema silently. New
callers can opt in with `GripperScene(..., ardusub_stabilize=True)` and pass
an `ArduSubStabilizeConfig` to tune the attitude loop.

## Main Commands

Collect Isaac data:

```bash
conda activate rov_isaac6
python -m EDMDc.collect_isaac --n 100 --headless \
  --out EDMDc/data/with_gripper/ardusub_train.npz
```

Run GUI teleoperation with RTX Real-Time:

```bash
python -m EDMDc.teleop_tank
```

This uses STABILIZE-style roll/pitch leveling by default. It still has no
depth hold: releasing the heave keys returns collective vertical command to
zero.

Run the focused Isaac 6 migration validation:

```bash
python -m EDMDc.validate_isaac6
```

Add `--camera` to require a rendered 640×480 RGBA frame, or run
`EDMDc.teleop_tank --self-test` to verify all command signs and tank
collisions headlessly, including recovery from a deliberate roll/pitch upset.
See [ISAAC6_MIGRATION.md](ISAAC6_MIGRATION.md) for the API mapping, validation
record, and known non-physics warnings.

Train EDMDc on the 4-axis command input:

```bash
python -m EDMDc.edmdc EDMDc/data/with_gripper/ardusub_train.npz \
  --lam 1e-3 --out EDMDc/model/ardusub_edmdc.npz
```

Run backstepping + LQR:

```bash
python -m EDMDc.backstep_edmdc_lqr --model EDMDc/model/ardusub_edmdc.npz
```

Run backstepping + MPC:

```bash
python -m EDMDc.backstep_edmdc_mpc --model EDMDc/model/ardusub_edmdc.npz
```

## Notes

`U` in new datasets is the 4-axis normalized ArduSub command. `U_realized`
stores the physical 6-DOF body wrench produced after ArduSub allocation and
T200 dynamics when available.
