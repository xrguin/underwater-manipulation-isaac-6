# Next-agent guide: Gaussian dictionary vs 34D EDMDc

## Start here

The next objective is to collect real simulator trajectories in the free-drift
configuration at **20 Hz control / 60 Hz physics** (via
`EDMDc.collect_tank --infinite --control-hz 20`, NOT `collect_isaac`, whose
`--dt` changes the physics step itself), train the Gaussian and 34D models on
exactly the same training trajectories, tune them on exactly the same
validation trajectories, and compare them once on an untouched test set.

Decisions locked 2026-07-21 (session with user):

- Sampling rate 20 Hz (`dt = 0.05 s`), physics stays at 60 Hz (decimation 3,
  zero-order-held commands, states logged at control instants).
- All four command axes excited (`--yaw-amp > 0`); free drift, no
  station-keeping, no yaw hold.
- Excitation box **±0.25** on every axis — see "Input convention" below.
- Comparison is modeling-only for now (open-loop prediction). No MPC yet.
- Primary metric: sliding-window K-step RMSE (`Gaussian_dictionary.evaluate_kstep`)
  in addition to the from-start rollout in `Gaussian_dictionary.evaluate`.

Read these files before changing anything:

- `AGENTS.md`
- `CLAUDE.md`
- `Gaussian_dictionary/README.md`
- `Gaussian_dictionary/gaussian_edmdc.py`
- `Gaussian_dictionary/evaluate.py`
- `EDMDc/edmdc.py`
- `EDMDc/collect_fossen.py`
- `EDMDc/collect_isaac.py`
- `EDMDc/collect_tank.py` — the actual 20 Hz collector (`--infinite` mode)

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

### Input convention (decided 2026-07-21)

At deployment the pilot chain applies safety factor 0.5 × pilot gain 0.5, so
the vehicle only ever receives normalized commands in **[-0.25, 0.25]** (on
the [-1, 1] full-throttle scale). Convention:

- **Record `U` in real post-gain units ([-0.25, 0.25])**, never rescaled to
  [-1, 1]. For a model linear in `u`, a constant rescale is absorbed into `B`
  (predictions identical), so rescaling buys nothing and creates a unit trap
  the moment the gain chain changes. This matches standard sysID practice
  (record the input the plant actually receives — cf. ArduPilot logging
  post-gain RCOU, not stick inputs).
- **Excite inside the deployment box (±0.25)**. The throttle→thrust map is
  nonlinear (ESC deadband + T200 polynomial, `EDMDc/thrusters.py`), so the
  fitted `B` is a local linearization averaged over the training amplitude
  distribution — training at full ±1 would learn the wrong gain for ±0.25
  operation. Same lesson as the earlier inscribed-box wrench caps.
- Heave keeps the collector's deadband floor: `|cmd_z|` is drawn from
  `[Z_FLOOR=0.20, 0.25]`, a narrow band. Check the heave input histogram in
  QA and note it in the report.
- Store the gain factors with the dataset so the scale is recoverable
  (`aprbs_amp*` fields already cover this; safety factor and pilot gain are
  0.5 × 0.5 for all splits).
- Later controller configs must use `u_min/u_max = ±0.25`.

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

Episodes are 60 s of free drift at 20 Hz (~1,200 pairs each when the guard
does not trip). Pilot first, full collection only after QA passes:

```text
pilot:  training 20 episodes seed 101 | validation 5 episodes seed 202
        | test 5 episodes seed 303
full:   revisit the volume after the pilot learning curve; the guide's
        original 200/40/40 target is an upper bound, not a commitment.
```

`collect_tank` seeds each episode with `SeedSequence((seed, ep))`, so
different `--seed` values give fully disjoint excitation streams per split.

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

Collection runs through `EDMDc.collect_tank --infinite`: deep-pool free
drift, spawn (0, 0, -5), fully exogenous 4-axis APRBS (no station-keeping, no
yaw hold), episode ends only on tipover (|roll| or |pitch| > 60°) or the
generous pool-interior guard (±13 m walls, z ∈ (-9.3, -0.6)). Do NOT use
`collect_isaac --dt 0.05` for the 20 Hz runs — there `--dt` sets
`physics_dt`, which changes the physics itself instead of decimating control.

The collector writes `<stem>_train_<TS>.npz` / `<stem>_test_<TS>.npz` and
crashes on `--test-episodes 0`, so each split runs with one extra dummy test
episode whose output file is discarded; the split content is the run's
"train" file. Pilot commands:

```bash
conda activate marinegym
# training split (20 kept episodes)
python -m EDMDc.collect_tank --infinite --control-hz 20 \
  --amp 0.25 --amp-z 0.25 --yaw-amp 0.25 --hold 1.0 --hold-min 0.4 \
  --episodes 21 --test-episodes 1 --seconds 60 --seed 101
# validation split (5 kept episodes)
python -m EDMDc.collect_tank --infinite --control-hz 20 \
  --amp 0.25 --amp-z 0.25 --yaw-amp 0.25 --hold 1.0 --hold-min 0.4 \
  --episodes 6 --test-episodes 1 --seconds 60 --seed 202
# test split (5 kept episodes)
python -m EDMDc.collect_tank --infinite --control-hz 20 \
  --amp 0.25 --amp-z 0.25 --yaw-amp 0.25 --hold 1.0 --hold-min 0.4 \
  --episodes 6 --test-episodes 1 --seconds 60 --seed 303
```

Copy the three kept files to canonical names before training:

```text
EDMDc/data/gaussian/free20_train_pilot.npz
EDMDc/data/gaussian/free20_validation_pilot.npz
EDMDc/data/gaussian/free20_test_pilot.npz
```

Hold times 0.4–1.0 s match the validated Jul-18 free-drift run. The vehicle
configuration is whatever `GripperScene` defaults to — keep it identical
across splits (it is, since all three runs use the same code path).

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
  EDMDc/data/gaussian/free20_train_pilot.npz \
  --n-rbfs 2 --center-method kmeans --seed 7 --lam 1e-3 \
  --out Gaussian_dictionary/model/gaussian_2rbf_pilot.npz

python -m EDMDc.edmdc EDMDc/data/gaussian/free20_train_pilot.npz \
  --lam 1e-3 --out EDMDc/model/edmdc_34d_free20_pilot.npz
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
horizons for both models. Two complementary metrics:

1. `Gaussian_dictionary.evaluate` — from-start recursive rollout (one error
   sample per trajectory per horizon; dominated by the initial transient).
2. `Gaussian_dictionary.evaluate_kstep` — **primary metric**: sliding-window
   K-step RMSE over every valid start index in every trajectory (the 20 Hz
   analog of the old `validate_heldout_sweep` convention; K=10 at 20 Hz is
   the old K=30 at 60 Hz, both 0.5 s).

```bash
python -m Gaussian_dictionary.evaluate \
  Gaussian_dictionary/model/gaussian_2rbf_pilot.npz \
  EDMDc/data/gaussian/free20_validation_pilot.npz \
  --compare-34d EDMDc/model/edmdc_34d_free20_pilot.npz \
  --horizons 1 5 10 25 50 100 \
  --out Gaussian_dictionary/results/validation_pilot.csv

python -m Gaussian_dictionary.evaluate_kstep \
  Gaussian_dictionary/model/gaussian_2rbf_pilot.npz \
  EDMDc/data/gaussian/free20_validation_pilot.npz \
  --compare-34d EDMDc/model/edmdc_34d_free20_pilot.npz \
  --horizons 1 5 10 20 40 100 \
  --out Gaussian_dictionary/results/validation_pilot_kstep.csv
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

Deferred by user decision (2026-07-21): the current campaign is
dictionary-design comparison only — no MPC runs yet. When the closed-loop
phase does start, remember the controller command caps are ±0.25 (see "Input
convention"), not the ±1.0 defaults in `GaussianMPCConfig`.

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
