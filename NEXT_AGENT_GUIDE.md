# Next-agent guide: Gaussian dictionary vs 34D EDMDc

## Start here

The next objective is to collect real simulator trajectories in the free-drift
(`--infinite-env`) configuration, train the Gaussian and 34D models on exactly
the same training trajectories, tune them on exactly the same validation
trajectories, and compare them once on an untouched test set.

Read these files before changing anything:

- `AGENTS.md`
- `CLAUDE.md`
- `Gaussian_dictionary/README.md`
- `Gaussian_dictionary/gaussian_edmdc.py`
- `Gaussian_dictionary/evaluate.py`
- `EDMDc/edmdc.py`
- `EDMDc/collect_fossen.py`
- `EDMDc/collect_isaac.py`

Do not rewrite the existing Fossen dynamics or force roll/pitch states to zero.
The existing `EDMDc/` implementation should remain unchanged unless the user
explicitly approves a change.

## Confirmed model design

The simulator remains full six-DOF:

```text
nu = [u, v, w, p, q, r]
```

The reduced Gaussian model selects only the controlled velocities:

```text
x = [u, v, w, r]
z = [u, v, w, r, gaussian_1(x), gaussian_2(x)]
z[k+1] = A z[k] + B command[k]
command = [surge, sway, heave, yaw]
```

Each Gaussian is one scalar function of all four selected velocities. The
default model therefore has `A` shape `6 x 6` and `B` shape `6 x 4`. The number
of RBFs, center method, seed, width scale, and ridge penalty are configurable.

The 34D comparison model uses the existing dictionary:

```text
1 constant + 6 velocities + 21 quadratic products + 6 |nu_i|nu_i terms
```

It produces `A` shape `34 x 34` and `B` shape `34 x 4` when trained on the
normalized four-axis command.

Positions are not part of either velocity predictor. Pose integration and the
outer position controller remain outside the learned velocity model. Roll and
pitch must still be recorded/monitored because restoring forces can make them
hidden variables for the reduced model if they become large.

## What is already implemented

- `Gaussian_dictionary/gaussian_edmdc.py`: dictionary fitting, additive EDMDc
  regression, model serialization, recursive rollout, and existing-dataset
  loading.
- `Gaussian_dictionary/evaluate.py`: common one-step and recursive-rollout
  metrics for Gaussian, 34D, and hold-last models on `[u,v,w,r]`; writes CSV.
- `Gaussian_dictionary/mpc.py`: simulation-facing MPC class that accepts a
  full six-velocity measurement/reference and returns four normalized ArduSub
  commands.
- `Gaussian_dictionary/test_gaussian_edmdc.py`: focused mathematical,
  serialization, schema, and interface tests.
- `Gaussian_dictionary/README.md`: current commands and assumptions.

No existing source under `EDMDc/` was modified for this experiment.

## What has and has not been verified

Verified:

- Eight applicable unit tests pass in the local `rov` environment.
- The lift uses all four selected velocities and does not set `p,q` to zero.
- Synthetic training, saving/loading, recursive rollout, and CSV comparison
  work end-to-end.
- Both model formats can be evaluated on the same dataset.

Not yet verified:

- No new Fossen or Isaac training dataset has been collected for this model.
- No held-out simulator comparison has been completed.
- No Gaussian MPC closed-loop simulation has been run.
- The optional MPC solve test was skipped locally because the `rov`
  environment does not contain SciPy/OSQP.
- The local `rov` environment also lacks PyYAML, so it cannot currently launch
  `EDMDc.collect_fossen`. Run collection in the configured simulator
  environment (`marinegym` per `CLAUDE.md`) or repair dependencies first.

At handoff time, the repository contained `data/ardusub_calib.npz`, but no
collected EDMDc train/validation/test trajectory files were found.

## Synthetic comparison: diagnostic only

Both models were trained with `lambda=1e-6` on the same 480 synthetic samples
(eight trajectories) and evaluated on those same samples. This is an in-sample
software diagnostic, not validation.

| Horizon | Gaussian total RMSE | 34D total RMSE |
|---:|---:|---:|
| One step | 0.00003798 | 0.000000081 |
| 5 steps | 0.0001947 | 0.0000860 |
| 10 steps | 0.0002106 | 0.0001243 |
| 25 steps | 0.0003592 | 0.0003681 |
| 50 steps | 0.0002720 | 0.0002869 |

Interpretation:

- The synthetic generator contained linear and `|x|x` damping terms, which
  directly match the 34D dictionary. Its nearly exact one-step fit is expected.
- The Gaussian model used 60 fitted `A,B` coefficients versus 1,292 for 34D,
  yet produced similar 25- and 50-step error.
- These results do not establish real simulator accuracy or generalization.

Synthetic stability diagnostics:

| Diagnostic | Gaussian | 34D |
|---|---:|---:|
| Spectral radius | 0.976689 | 0.999999997 |
| Eigenvalues outside unit circle | 0 | 0 |
| Largest tested `||A^k||_2` | about 1.29 | about 9.41 |
| Zero-input velocity norm after 500 steps | `4.81e-9` | `7.30e-5` |

The near-unit 34D eigenvalue is primarily the structural constant observable
`1[k+1]=1[k]`; it is not by itself evidence of unstable physical velocity. The
power-norm comparison is also affected by different feature scaling. Closed-
loop stability cannot be inferred from open-loop `A` alone.

## Required data design

Use three disjoint sets of complete trajectories. Never randomly split rows
from the same trajectory because adjacent time samples would leak across sets.

Recommended full collection target:

```text
training:    200 trajectories, seed 101
validation:   40 trajectories, seed 202
test:         40 trajectories, seed 303
```

If Isaac collection is expensive, first run a pilot with `20 / 5 / 5`
trajectories. Confirm data quality and the complete workflow before launching
the full collection.

Keep these identical across all three sets:

- Simulator and vehicle configuration.
- Gripper/payload configuration.
- `dt`, episode duration, mass, buoyancy setting, allocator, and thruster model.
- Four-axis command definition and excitation policy.
- Initial-condition policy.

Use different random seeds. Keep complete trajectory IDs in one set only.

The APRBS excitation should cover positive, negative, combined-axis, and
zero/drain behavior for surge, sway, heave, and yaw. Check input histograms,
state coverage, saturation, truncation, NaNs, and whether roll/pitch remain in
the intended operating range. Save `U_realized` when available for diagnostics,
but use normalized commanded `U` for the primary controller-model comparison.

## Collection sequence

### 1. Fast Fossen pilot

Fossen has no wall-contact dynamics, so it is the fastest way to verify the
experiment and tune data volume. The local `rov` environment currently lacks
PyYAML; use the simulator environment with the repository dependencies.

Example pilot commands:

```bash
conda activate marinegym
python -m EDMDc.collect_fossen --n 20 --episode-s 5 --dt 0.0166666667 \
  --seed 101 --out EDMDc/data/gaussian/fossen_train_pilot.npz
python -m EDMDc.collect_fossen --n 5 --episode-s 5 --dt 0.0166666667 \
  --seed 202 --out EDMDc/data/gaussian/fossen_validation_pilot.npz
python -m EDMDc.collect_fossen --n 5 --episode-s 5 --dt 0.0166666667 \
  --seed 303 --out EDMDc/data/gaussian/fossen_test_pilot.npz
```

Do not use `--fx-only`; the comparison requires all four controlled axes.

### 2. Isaac free-drift collection

`EDMDc.collect_isaac` has `--infinite-env` enabled by default. It expands the
Python truncation envelope to approximately 500 m radially and +/-100 m
vertically. The decorative pool geometry does not enforce collision in this
collector; see `CLAUDE.md` for the environment caveat.

Use `--target-kept` so each split contains the requested number of complete,
non-truncated APRBS trajectories. Example pilot commands:

```bash
conda activate marinegym
python -m EDMDc.collect_isaac --target-kept 20 --episode-s 5 --seed 101 \
  --headless --infinite-env \
  --out EDMDc/data/gaussian/isaac_train_pilot.npz
python -m EDMDc.collect_isaac --target-kept 5 --episode-s 5 --seed 202 \
  --headless --infinite-env \
  --out EDMDc/data/gaussian/isaac_validation_pilot.npz
python -m EDMDc.collect_isaac --target-kept 5 --episode-s 5 --seed 303 \
  --headless --infinite-env \
  --out EDMDc/data/gaussian/isaac_test_pilot.npz
```

The Isaac collector enables the gripper by default. Use `--no-gripper` only if
the intended comparison is specifically the bare ROV, and then use it for all
splits. Keep payload-cube settings consistent as well.

Step-response trajectories are valuable as a separate stress test. Do not mix
them silently into only one of the train/validation files. If collected, label
and report their results separately from APRBS validation.

## Data-quality gate before training

Do not train until all of the following are reported for every split:

- Array shapes for `X`, `X_next`, `U`, and `U_realized` when present.
- Number of complete trajectories and samples per trajectory.
- No NaN or infinite values.
- Command minimum, maximum, mean, standard deviation, and saturation fraction
  for every axis.
- Velocity range/standard deviation for all six axes.
- Roll/pitch rate distribution and, if available, attitude range.
- Unique trajectory IDs with no overlap between train, validation, and test.
- Identical `dt`, input definitions, gripper/payload configuration, and
  buoyancy configuration across splits.

If roll/pitch or other omitted variables vary enough to make `[u,v,w,r]`
non-Markovian, report that evidence before expanding the Gaussian dictionary.

## Training and model selection

Train both candidates on the exact same training file. Start with a common
ridge value only as a baseline; then tune each model fairly on validation data.

Baseline commands:

```bash
python -m Gaussian_dictionary.gaussian_edmdc \
  EDMDc/data/gaussian/isaac_train_pilot.npz \
  --n-rbfs 2 --center-method kmeans --seed 7 --lam 1e-3 \
  --out Gaussian_dictionary/model/gaussian_2rbf_pilot.npz

python -m EDMDc.edmdc EDMDc/data/gaussian/isaac_train_pilot.npz \
  --lam 1e-3 --out EDMDc/model/edmdc_34d_pilot.npz
```

Suggested validation candidates:

- Gaussian RBF count: `2, 4, 8, 16`.
- Gaussian center method: start with `kmeans`; compare reproducible `random`
  centers only after the baseline works.
- Gaussian width scale: `0.5, 1.0, 2.0`.
- Ridge penalty for both families: `1e-6, 1e-5, 1e-4, 1e-3, 1e-2`.

Do not select a model using training RMSE. Choose hyperparameters using
validation rollout performance and stability diagnostics. Once selected,
freeze them before opening the test result.

## Fair validation and final comparison

Run the same validation file, trajectory subset, state axes, inputs, and
horizons for both models:

```bash
python -m Gaussian_dictionary.evaluate \
  Gaussian_dictionary/model/gaussian_2rbf_pilot.npz \
  EDMDc/data/gaussian/isaac_validation_pilot.npz \
  --compare-34d EDMDc/model/edmdc_34d_pilot.npz \
  --horizons 1 5 10 25 50 100 \
  --out Gaussian_dictionary/results/validation_pilot.csv
```

After hyperparameters are frozen, repeat once on the untouched test file and
save a separate `test_*.csv`.

Compare at minimum:

- One-step RMSE per axis and total on `[u,v,w,r]`.
- Recursive rollout RMSE at common horizons and in seconds.
- Normalized RMSE so velocity axes with different scales are comparable.
- Hold-last baseline.
- Zero-input equilibrium drift.
- Spectral radius, eigenvalues outside the unit circle, and transient growth.
- Diverged or non-finite rollout count.
- Prediction time and fitted parameter count.

The 34D model predicts `p,q` and the Gaussian model does not. The main fair
accuracy table must compare `[u,v,w,r]`; report 34D `p,q` accuracy separately.

## Closed-loop phase comes after modeling validation

Open-loop `rho(A)<1` does not prove a better controller. After selecting
models from held-out prediction results, run both inside controllers using the
same reference trajectories, command/rate limits, disturbances, payload, and
initial conditions. Report:

- Velocity and position/yaw tracking RMSE.
- Command effort and saturation fraction.
- Solver failures and computation time.
- Overshoot and settling time.
- Robustness to payload and parameter mismatch.

For fixed linear feedback `u=-Kz`, the closed-loop matrix is `A-BK`; for MPC,
closed-loop stability must be assessed from actual plant/controller rollouts,
not from open-loop `A` alone.

## Definition of done for the next agent

- Data collection command and environment are recorded.
- Train, validation, and test trajectory sets are disjoint and pass QA.
- Both model families are trained on the same training snapshots.
- Hyperparameters are selected only from the common validation set.
- A final untouched test CSV is saved and summarized without overstating the
  synthetic results.
- Stability findings distinguish constant-observable modes from physical
  velocity instability.
- No closed-loop superiority claim is made until the controller is run against
  the simulator plant.
