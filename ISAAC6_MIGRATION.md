# Isaac Sim 6 migration

This copy targets Isaac Sim `6.0.0-rc.59+release` and its bundled Python
3.12.13 runtime:

```bash
ISAAC_PY=/home/miaodong/Documents/isaac-sim-6.0/python.sh
```

For interactive use on this machine, activate the isolated launcher
environment:

```bash
conda activate rov_isaac6
python -m EDMDc.teleop_tank
```

The environment's `python`/`python3` wrappers execute Isaac's exact bundled
interpreter and initialize its native Kit paths. They do not use Anaconda's
CPython to host Kit: although it has the same Python version, its modified
`sys.version` string is incompatible with this Isaac 6 release candidate.

Do not run simulator entry points in the old `marinegym42` environment. The
original Isaac Sim 4.2 project and its assets are not modified by this
migration.

## API strategy

Physics and stage management migrate directly to Isaac 6 Core Experimental.
`EDMDc/isaac6_compat.py` is the narrow NumPy-facing boundary used by the
existing plant and collector:

| Isaac 4.2 usage | Isaac 6 implementation |
|---|---|
| `omni.isaac.core.World` | `SimulationManager`, `RenderingManager`, and Core Experimental app/stage utilities |
| `omni.isaac.core.articulations.Articulation` | `isaacsim.core.experimental.prims.Articulation` |
| `omni.isaac.core.prims.RigidPrimView` | `isaacsim.core.experimental.prims.RigidPrim` with NumPy-compatible shapes |
| `omni.isaac.core.utils.stage` | `isaacsim.core.experimental.utils.stage` |
| `omni.isaac.debug_draw` | `isaacsim.util.debug_draw` |
| `omni.isaac.sensor.Camera` | renamed Isaac 6 `isaacsim.sensors.camera.Camera`, isolated in the compatibility module |
| `from isaacsim import SimulationApp` | unchanged documented standalone bootstrap |

The camera is the one temporary non-Experimental component. In this Isaac 6
release candidate, constructing
`isaacsim.sensors.experimental.rtx.CameraSensor` after the standalone timeline
has started blocks during render-product attachment. The renamed high-level
Isaac 6 camera package works correctly and retains the existing
`initialize()/get_rgba()` contract. No removed `omni.isaac` compatibility
imports remain in the Python source.

Hydrodynamics, six-DOF state propagation, Fossen parameters, ArduSub command
allocation, the T200 model, controller interfaces, USD/YAML assets, and EDMDc
state definitions were not changed.

## Timeline and keyboard safety

Core Experimental physics tensor views are invalid while the timeline is
paused or stopped. `GripperScene` now:

- never reads, steps, or applies forces through a stopped physics view;
- returns a copied last-finite state while paused;
- applies zero thrust and does not advance T200 motor state while paused;
- pumps render/UI events without advancing simulation time; and
- resumes physics only after the timeline reports playing.

The collector suspends its settle/collection loops on the same boundary.
Teleoperation clears held keys on pause and resume. It also clears commands
when the application loses focus or no press/repeat event arrives within the
configurable `--command-timeout` (default 0.75 s), covering renderer menus that
consume a key-release event. Space remains the immediate manual panic clear.

## Validation

Run the full headless physics check:

```bash
$ISAAC_PY -m EDMDc.validate_isaac6
```

Run it on a renderer-capable host and optionally save the inspected frame:

```bash
$ISAAC_PY -m EDMDc.validate_isaac6 --camera \
  --camera-out /tmp/isaac6_rtx_realtime.png
```

Run the bidirectional command-sign and collision check:

```bash
$ISAAC_PY -m EDMDc.teleop_tank --self-test
```

The controlled entry points enable an optional simulator-side
ArduSub-STABILIZE-style roll/pitch loop. It adds geometry-derived differential
commands only to the four vertical motors before the unchanged T200 model. It
does not add a depth controller or change the four-command interface.

Validated on an NVIDIA GeForce RTX 5080 with driver 580.173.02:

- clean headless startup and shutdown;
- tank, BlueROV Heavy, and Newton Gripper asset roots load;
- vehicle articulation and runtime fixed joint initialize;
- the physical gripper offset remains `[0.148, 0.0, -0.1]` m;
- pose, velocity, applied wrench, thruster state, and resumed state are finite;
- zero command moves from z=-0.216 m to z=-0.449 m in eight simulated seconds,
  matching the 4.2 reference and settling against the tank floor;
- pause applies no force, does not advance the T200 state, and resumes with a
  valid physics interface;
- W/S, A/D, E/Q, and F/R have the expected positive/negative surge, sway,
  heave, and yaw signs;
- T200 throttle, RPM, and allocator outputs remain within their original
  saturation limits;
- the east wall stops a sustained surge at x=2.004 m without tunneling;
- with STABILIZE enabled, a `(+15 deg roll, -12 deg pitch)` upset recovers to
  approximately `(+0.10 deg, +1.88 deg)` in four simulated seconds, with the
  differential motor correction bounded at `0.35`;
- the Isaac 6 debug-draw interface is available;
- the RTX camera returns finite `(480, 640, 4)` RGBA output;
- RTX Real-Time (`RayTracedLighting`) starts on the RTX 5080 and the inspected
  frame is clean without the severe 4.2 speckle/noise;
- a short visible teleoperation run initializes both cameras, processes the
  GUI event loop, advances physics, and shuts down cleanly; and
- all 15 focused STABILIZE and Gaussian/EDMDc tests pass under the bundled
  Python, including the OSQP MPC solve.

A collector smoke run retained the existing schema and units:

```text
X            (N, 6) float64
X_next       (N, 6) float64
U            (N, 4) float64, ardusub_normalized_[-1,1]
U_realized   (N, 6) float64
traj_idx     (N,)   int32
step_idx     (N,)   int32
input_names  [surge, sway, heave, yaw]
```

## Warning classification

Material-binding/USD diagnostics from the authored BlueROV asset are cosmetic
unless they accompany an invalid required prim or a failed material/texture
load. The validator checks the required prim roots, articulation tensor view,
fixed-joint targets, finite dynamics, collision behavior, and rendered pixels
separately, so a cosmetic binding warning cannot be mistaken for a physics or
asset-composition pass.

The first camera frames can also report an annotator warm-up warning before
valid pixels arrive. A low-resolution DLSS advisory can occur because the
internal render input is 320×240 before producing the requested 640×480
output. Both are renderer advisories, not physics failures. GPU/NVML and
OmniHub errors observed only inside a restricted no-GPU sandbox are
environment diagnostics; host-GPU validation enumerated and used the RTX
5080 successfully.
