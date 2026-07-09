# norm_inputs — EDMDc / ARX with normalized [-1,1] control inputs

Self-contained experiment folder (gripper-only, **no payload cube**).
**Question:** if the control input is the 6-DOF body wrench **normalized per axis
to `[-1,1]`** instead of absolute Newtons, can we still construct EDMDc and ARX,
and does it change accuracy?

**Answer (verified on Isaac data): yes / no change.** EDMDc and ARX both fit
cleanly, EDMDc is spectrally stable, and 30-step held-out accuracy is identical
(marginally better) to the baseline Newton-input pipeline. Normalizing inputs
only rescales `B` (`B_newton = B / symmetric_caps`), nothing else.

## Design (locked)

| decision | choice |
|---|---|
| Input `U` | 6-DOF wrench **normalized per axis to [-1,1]** (4-DOF mask: Fx, Fy, Fz, Tz; Tx, Ty = 0) |
| Excitation | APRBS in the **inscribed box** of the achievable wrench polytope (the realizable region the shipping pipeline uses) |
| Per-axis scale | symmetric `sᵢ = min(cap_pos, cap_neg)` = `[22.57, 23.47, 69.15, –, –, 6.59]` N → `u_newton = sᵢ·u_norm` |
| Thruster model | linear unit-thrust (`thrusters_norm.py`, `linear=True`, `thrust = 25·c`) → `U_recorded = U_realized` |
| Code layout | self-contained copy; existing `EDMDc/*` untouched |

> Note: an earlier iteration used a *thruster-cube* (zonotope) excitation where
> the per-axis range went to `2√2`. It was dropped because "cube" collided with
> the payload-cube pickup task and added concept overhead for no accuracy gain.
> The current design uses the standard inscribed box, just normalized to [-1,1].

### Why a single symmetric scale per axis
EDMDc/ARX are linear in `U`, so for `u_norm → force` to be linear (and `B` to be
the Newton-`B` rescaled), each axis needs **one** scale. The inscribed caps are
asymmetric on heave (Fz +88.17/−69.15 N); using them as separate ± scales would
make the map piecewise-linear. The symmetric box is inside the achievable
polytope, so with the linear allocator every sample is realizable (no
saturation). Cost: slight heave-up coverage loss (Fz capped at 69.15 N).

## Files

| file | role |
|---|---|
| `norm_aprbs.py` | `SYMMETRIC_CAPS`, `denorm_to_newton`/`norm_from_newton`, `aprbs_sequence_normalized` (inscribed-box, [-1,1]) |
| `thrusters_norm.py` | vendored linear-mode T200/allocator (`pool_test_simulation/thrusters.py`) |
| `collect_norm.py` | Isaac collection; records `U` normalized to [-1,1] and `C` (8 thruster cmds) → `data/norm_*.npz` |
| `edmdc_norm.py` | EDMDc trainer (reuses `EDMDc.edmdc`); saves `B` and `B_newton = B/sᵢ` |
| `edmdc_delay_norm.py` | HODMDc+ARX trainer (reuses `EDMDc.edmdc_delay`) |
| `validate_norm.py` | K-step held-out accuracy (train/test split, EDMDc vs ARX, plot) |
| `compare_inputs.py` | normalized-input vs baseline Newton comparison (raw + NRMSE, bar chart) |
| `data/`, `model/` | outputs |

## Run

```bash
conda activate marinegym
# 1. Collect normalized-input APRBS data (4-DOF, gripper, inscribed box)
python -m EDMDc.norm_inputs.collect_norm --n 120 --episode-s 5 --headless \
    --out EDMDc/norm_inputs/data/norm_inscribed_train.npz
# 2. Train EDMDc and ARX
python -m EDMDc.norm_inputs.edmdc_norm        EDMDc/norm_inputs/data/norm_inscribed_train.npz
python -m EDMDc.norm_inputs.edmdc_delay_norm  EDMDc/norm_inputs/data/norm_inscribed_train.npz --state-delay 3 --input-delay 1
# 3. Accuracy + comparison
python -m EDMDc.norm_inputs.validate_norm     EDMDc/norm_inputs/data/norm_inscribed_train.npz --K 30
python -m EDMDc.norm_inputs.compare_inputs \
    --norm EDMDc/norm_inputs/data/norm_inscribed_train.npz \
    --baseline EDMDc/norm_inputs/data/baseline_wrench_train.npz --K 30
```

`U_newton = symmetric_caps · U`. The baseline (`baseline_wrench_train.npz`) is a
matched `collect_isaac` run (Newton wrench, same inscribed box, n=120, ep=5,
seed=0, gripper) — so the only difference is the input representation.

## Results (120 traj × 5 s; norm 106,086 / baseline 104,199 snapshots; 96 train / 24 test)

Both EDMDc stable (ρ=1.0). **30-step held-out RMSE** — normalized [-1,1] vs
baseline Newton, both inscribed box ([compare_inputs_nrmse.png](data/plots/compare_inputs_nrmse.png)):

| metric | norm EDMDc | base EDMDc | norm ARX | base ARX |
|---|---|---|---|---|
| raw RMSE (agg) | 0.1455 | 0.1501 | 0.1038 | 0.1040 |
| **NRMSE (agg)** | **0.701** | 0.731 | **0.499** | 0.510 |

ARX gain over EDMDc: **+28.8%** (norm), +30.2% (baseline).

**Conclusions:**
1. Normalizing inputs to [-1,1] does **not** change modeling accuracy — norm and
   Newton are within ~2–4% (norm marginally better), confirming the units only
   rescale `B`.
2. ARX(ds=3,di=1) improves EDMDc by ~29% on held-out K-step in both, biggest on
   the attitude rates p/q — matching `edmdc_time_delay_embedding`.
