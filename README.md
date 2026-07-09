# underwater-manipulation-ardusub-sim

Focused BlueROV Heavy EDMDc + Isaac Sim workspace using an ArduSub-style
4-axis control interface.

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
  -> T200 motor lag/thrust model
  -> realized 6-DOF body wrench
  -> Fossen hydrodynamics + Isaac rigid-body dynamics
```

The saved calibration file is `data/ardusub_calib.npz`. The copied
`ardusub_bridge.py` and `ardusub_check.py` are kept for calibration/reference;
the default sim loop does not require live ArduSub SITL.

## Main Commands

Collect Isaac data:

```bash
python -m EDMDc.collect_isaac --n 100 --headless \
  --out EDMDc/data/with_gripper/ardusub_train.npz
```

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
