# Working notes for AI assistants on this codebase

Concise reference for navigating, running, and modifying the underwater
manipulation 2 / EDMDc project. See `README.md` for project overview.

## Environment

**Python invocation**: `conda activate marinegym && python -m EDMDc.<module>`.
The marinegym conda env has an activation hook that sources Isaac Sim 4.2's
`setup_conda_env.sh`, prepending Isaac's bundled PYTHONPATH (numpy 1.26,
scipy 1.10, cvxpy 1.6.7). Calling `~/miniconda3/envs/marinegym/bin/python`
**directly** (no `conda activate`) bypasses the hook and `import isaacsim`
fails.

Alternative (from a non-conda shell):
`PYTHONUNBUFFERED=1 /home/xzha/Documents/isaac-sim-4.2/python.sh -m EDMDc.<module>`.

User-site deps for Isaac:
```
/home/xzha/Documents/isaac-sim-4.2/python.sh -m pip install --user \
    "cvxpy<1.7" osqp matplotlib opencv-python PyQt5
```
`cvxpy<1.7` is mandatory because 1.7+ requires `scipy.sparse.eye_array`
(needs scipy ≥1.11; Isaac bundles 1.10).

## Module discipline

- **Argparse FIRST**, then `SimulationApp(...)`, then `omni.isaac.*` imports.
  `--help` must work without booting Isaac. Pattern is followed in
  `collect_isaac{,_38}.py`, `open_loop_inputs_gripper{,_38}.py`, etc.
- **ffmpeg resolution BEFORE `SimulationApp(...)`** — Isaac's Kit boot
  sanitizes `PATH`. Standard preamble in collection / open-loop scripts.
- **Project-root on `sys.path` after Isaac boot** — Isaac resets `PYTHONPATH`
  during startup, so the per-module preamble re-adds project root for
  relative imports (`from .foo import ...`).

## Key data conventions

- Body velocity ν = [u, v, w, p, q, r] — surge, sway, heave, roll-rate, pitch-rate, yaw-rate. Body frame, NED.
- World pose η = [x, y, z, roll (φ), pitch (θ), yaw (ψ)]. Euler-ZYX intrinsic.
- Wrench U = [Fx, Fy, Fz, Tx, Ty, Tz] — body frame.
- 4-DOF task mask (deployment + default training): Fx, Fy, Fz, Tz active; Tx, Ty pinned to 0.
- Quaternions stored wxyz everywhere (Isaac's convention). Conversion helpers in `EDMDc/fossen.py`.

## Dictionary dimensions

**The default state dimension is 34** (`EDMDc.edmdc`). The 38-D variant is
the optional richer alternative.

| trainer | dict dim | features | when to use |
|---|---|---|---|
| `EDMDc.edmdc` | **34 (default)** | 1 + ν(6) + ν⊗ν pairs(21) + \|ν\|·ν(6) | LQR/MPC navigation; passive attitude — what we ship |
| `EDMDc.edmdc_38` | 38 | 34-D + sin(φ), cos(φ), sin(θ), cos(θ) | When attitude dynamics matter (high-attitude excursions, attitude-aware MPC) |

The 38-D extension captures the gripper-induced attitude coupling (the
ROV+gripper has a passive equilibrium at φ=0°, θ≈17°). See
`EDMDc/debug_journal_gripper.md` Bug #2 for the motivating evidence.

A single 38-D collection (`collect_isaac_38.py`) produces a dataset that
the 34-D trainer can also consume — `EDMDc.edmdc` reads only `X, U, X_next`
and ignores the extra `Eta, Eta_next` columns. So if you might want both,
collect with `collect_isaac_38.py` and train the 34-D path on it.

## Wrench limits & APRBS schemes

Three excitation modes coexist in the codebase. Pick via CLI flags:

| flag | mode | per-axis caps (4-DOF) |
|---|---|---|
| **(default)** | **Continuous-uniform APRBS in inscribed box** | **Fx ±22.57, Fy ±23.47, Fz +88.17/−69.15, Tz ±6.59** |
| `--no-continuous-aprbs` | Legacy discrete-amplitude APRBS at full polytope-vertex caps | Fx ±65, Fy ±37.5, Fz +102/−80, Tz ±17 |
| `--fx-only` | APRBS on Fx only (legacy single-axis test) | only Fx active |

Per-rotor cap is `MAX_THRUST_PER_MOTOR_N = 25` N (half the T200 spec; the
ThrustAllocator + T200Group enforce this at runtime).

**Why the default changed**: under the legacy discrete-APRBS at full caps,
~80 % of joint commands require per-rotor thrusts > 25 N. The allocator +
T200 saturate to keep each rotor at ≤25 N, but the dataset stores the
*commanded* wrench, not the realized one — biasing the LS fit by a factor
of ~5× on the worst-case corner samples. The current default
(continuous-APRBS in inscribed box) puts every sample inside the
allocator's feasible polytope, so `U_recorded = U_realized` and the LS
fit is clean.

## Pool / envelope settings

- `WATER_SURFACE_Z = -0.2` (m) — Fossen's buoyancy boundary.
- Default trajectory truncation: 6 m radius × ±1.8 m vertical, |pitch| / |roll| < PITCH_LIMIT.
- `PITCH_LIMIT` default 86°; relaxed to 60° when `--gripper` is on (BlueROV+gripper has passive equilibrium at ~17° pitch; 60° gives 15° headroom over worst observed APRBS excursion of ~45°).
- `--infinite-env` (default ON in collection scripts) bumps truncation to 500 m × ±100 m — used for free-drift APRBS where the vehicle's accumulated motion isn't bounded by a pool. The Isaac `environment.usd` pool geometry doesn't enforce collision; truncation is purely Python-side.

## Reproducibility

- All RNG goes through `trajectory_rng(seed, traj_idx)` (`collect_fossen.py`)
  which uses `np.random.SeedSequence` to give bit-identical APRBS sequences
  per (seed, traj_idx) pair.
- `axis_mask_for(task_dof, fx_only=...)` is the canonical 4-DOF mask
  producer; both collection paths use it.

## Output paths (convention)

| artifact | location |
|---|---|
| Training datasets (34-D) | `EDMDc/data/with_gripper/isaac_gripper_<TS>.npz` |
| Training datasets (38-D) | `EDMDc/data/with_gripper_38/isaac_gripper_38_<TS>.npz` |
| Trained models | `EDMDc/model/edmdc_gripper{,_38}_<TS>.npz` |
| Open-loop eval | `EDMDc/data/with_gripper{,_38}/open_loop/<TS>_<axis>.{npz,mp4}` |
| LQR trajectories | `EDMDc/data/with_gripper{,_38}/backstep_lqr/<TS>.npz` |
| All plots | `EDMDc/data/plots/...` (mirrors data subdir) |
| Chase-cam recordings | `EDMDc/recording/<script>_<TS>.mp4` |

When adding a new flag that changes excitation/training (e.g.,
`--continuous-aprbs`), the caps used should be saved into the npz via the
existing `wrench_cmd_max_pos`/`wrench_cmd_max_neg` fields so downstream
analysis can recover the run config without parsing CLI args.

## Common gotchas

1. **Isaac stage warnings about missing materials** are noise; the USD CAD
   references mat paths outside its own scope. Doesn't affect physics.
2. **`omni.physx.tensors Duplicate link name 'base_link'`** appears with
   `--gripper` because the runtime FixedJoint creates a composite
   articulation. Cosmetic — physics is correct.
3. **Live matplotlib in same process as Isaac Kit crashes the Qt event loop.**
   Use the out-of-process `EDMDc.live_plot_client` instead. See memory
   `mpc_live_plot_qt_conflict.md`.
4. **`numpy 2.x` and Isaac don't mix.** Isaac calls `np._no_nep50_warning()`
   which doesn't exist in numpy 2.x. Stay on numpy 1.26 (Isaac's bundle).
5. **`collect_isaac_38.py` produces a superset of `collect_isaac.py`'s
   output.** The 34-D trainer (which is the default — see "Dictionary
   dimensions") reads `X, U, X_next` and ignores the extra `Eta, Eta_next`
   columns. A single 38-D collection can train both dictionaries; no need
   to re-collect for the 34-D path.

## When investigating model accuracy

Use the matched-distribution metric, not the sustained-step metric:

```bash
# Standard sysID metric: K-step prediction RMSE on held-out APRBS
python -m EDMDc.validate_heldout_sweep --K 30 --skip-collect
```

The sustained single-axis step test (`EDMDc.open_loop_inputs_gripper`) is
useful as a *diagnostic* but conflates model fit with training-distribution
coverage. Avoid using it as the primary metric without controlling for
amplitude distribution.

## Tx, Ty axes — DON'T excite them

The BlueROV2 Heavy with Newton Gripper is passively roll/pitch-stable via
the buoyancy-COB / COG offset + quadratic angular damping. Excitation on
Tx and Ty drives the vehicle into tipover within 1–2 s (we measured 37 %
truncation under APRBS at cap 7.5 N·m). All training and control runs
should keep Tx, Ty pinned to 0 (4-DOF mask). See the session history in
README's "Reverted earlier 6-DOF Tx/Ty excitation experiments".

## Open follow-ups (known compromises, not bugs)

These are intentional or accepted gaps that a future session may want to
revisit. None of them blocks the current default pipeline.

1. **LQR `u_min` / `u_max` in `EDMDc/lqr.py:35-36` still point at the
   *original* full-cap values** (Fx ±65, Fy ±37.5, Fz +102/−80, Tz ±17),
   not the inscribed caps the model was trained on. At deployment the
   LQR can technically command wrenches outside the model's training
   distribution. In practice the LQR uses < 50 % of training std width, so
   this never triggers, but if you switch to MPC with aggressive cost
   weights it could matter. Easy fix: pass a custom `LQRConfig` with
   inscribed caps from `EDMDc.collect_fossen.INSCRIBED_BOX_CAPS_POS/NEG`.

2. **Sustained single-axis step tests at inscribed-cap amplitudes drive the
   vehicle into the training-distribution tail.** Most severe on Tz: at
   sustained Tz = 6.59 N·m, the terminal yaw rate is 2.06 rad/s because
   yaw has zero linear damping in `BlueROVHeavy.yaml`. The 8-second test
   spends most of its duration at r values that continuous-APRBS holds
   (0.1–0.5 s) rarely reach.

   **The mechanism is LS multicollinearity**, not training-distribution
   tail per se — see `EDMDc/debug_journal_gripper.md` Bug #3 for the
   full diagnosis. The linear and quadratic damping columns are nearly
   collinear in the narrow training velocity window; LS picks any pair
   satisfying the local fit, so the model's effective damping curve
   crosses Isaac's at only one velocity. The failure manifests on Fz/Tz
   (where APRBS holds don't reach the cap-step terminal velocity) but
   not on Fx/Fy (where they do).

   Options: (a) use deployment-realistic amplitudes (`Fx +15, Fy ±6,
   Fz +13/−8, Tz ±2`, derived from canonical LQR command range — see
   README "Key findings"), (b) lengthen `APRBS_HOLD_MAX_STEPS` for
   Tz specifically, (c) accept that this test stresses the boundary
   and rely on K-step held-out RMSE as the primary metric.

3. **The 38-D dictionary lacks product features** (e.g., `sin(φ)·cos(θ)`,
   `cos(φ)·q`). It has the four trig terms as standalone features and
   represents pitch restoring torque via `sin(θ)` alone, but the buoyancy
   roll-restoring torque `sin(φ)·cos(θ)` and Euler-rate coupling
   `cos(φ)·q` are not directly representable. This is why high-attitude
   regimes (φ or θ > ~45°) degrade. Fix would be a 42-D extension adding
   the four trig-pair products. Not done because LQR-deployment
   trajectories stay near the (0°, 17°) equilibrium and don't visit
   that regime.

4. **APRBS hold times are axis-agnostic** (`APRBS_HOLD_MIN/MAX_STEPS` in
   `collect_fossen.py`). Per-axis time constants differ (surge τ ≈ 0.1 s,
   sway similar, heave ~0.3 s, yaw effectively ~∞ at low r because no
   linear damping). Longer holds on Tz specifically would let r reach
   steady state during training and improve the model's high-r
   prediction without growing the data quantity.

5. **The earlier sweep used `--infinite-env` (500 m × ±100 m truncation)**,
   while the physical Isaac pool USD is roughly 25×25×2 m with no
   collision geometry. For the inscribed-APRBS sweep the vehicle's
   accumulated drift over 5 s is ≲ 1 m, so the truncation envelope is
   irrelevant. But if a future experiment uses larger amplitudes or
   longer episodes, the truncation envelope vs physical pool mismatch
   may matter.

6. **Deployment caps could be halved again** — the LQR closed-loop uses
   only 11–47 % of training-std width on every axis (measured during
   the 3-waypoint nav). The inscribed caps already match what the LQR
   typically commands; tightening further would risk under-coverage on
   recovery maneuvers but might give a slightly cleaner fit at
   deployment-typical amplitudes. Not pursued.

## Memory pointers

Project-specific memories that have come up repeatedly this session:

- `env_isaac_sim_python.md` — marinegym activation hook details.
- `mpc_live_plot_qt_conflict.md` — Qt-vs-Kit GUI conflict resolution.
- `debug_journal_gripper.md` — chronological diagnosis log; Bug #2 has the
  34-D dictionary's failure to represent attitude-dependent gripper torques.
