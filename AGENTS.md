# Project progress

## Gaussian dictionary comparison

- Design: keep the Fossen/Isaac plant at full six-DOF; do not force `p`, `q`,
  `phi`, or `theta` to zero.
- Reduced learned state: `[u, v, w, r]`, selected from the existing six-state
  collector snapshots.
- Lift: four physical velocities plus two configurable Gaussian RBF
  observables by default.
- Dynamics: regular additive EDMDc, `z[k+1] = A z[k] + B u[k]`.
- Inputs: the existing normalized `[surge, sway, heave, yaw]` command by
  default; four task-axis realized wrench is optional when available.
- Integration: consume the existing `collect_fossen.py` and `collect_isaac.py`
  dataset schema; expose a full-six-state MPC interface returning the same
  four normalized commands as the current controller.
- Validation: focused unit tests plus held-out one-step/rollout comparison with
  the current 34-D model and a hold-last baseline.

## Status

- [x] Existing EDMDc dictionary, data schema, evaluator, and MPC inspected.
- [x] Gaussian dictionary/training implementation added under
  `Gaussian_dictionary/`.
- [x] Fair held-out comparison evaluator added.
- [x] Simulation-facing MPC adapter added.
- [x] Usage and model assumptions documented.
- [x] Run tests in the available local Python environment: eight applicable
  tests pass; the optional MPC solve test is skipped because the local `rov`
  environment does not contain SciPy/OSQP.
- [x] Run an end-to-end synthetic train/evaluate smoke check, including the
  current 34-D model comparison interface and CSV output.
