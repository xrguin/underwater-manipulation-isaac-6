# Gaussian dictionary EDMDc

This folder is an isolated comparison implementation. It does not replace or
modify the current 34-D EDMDc code.

## Model definition

The Fossen and Isaac simulators remain full six-DOF and continue to save

```text
nu = [u, v, w, p, q, r].
```

At the model boundary, this package selects

```text
x = [u, v, w, r]
```

and constructs the default six-dimensional lifted state

```text
z = [u, v, w, r, gaussian_1(x), gaussian_2(x)].
```

Each Gaussian is **one scalar function of all four velocities**:

```text
gaussian_j(x) = exp(-0.5 * sum_i ((x_i - center[j,i]) / width[j])^2).
```

The learned controlled model is the regular additive EDMDc form

```text
z[k+1] = A z[k] + B command[k]
command = [surge, sway, heave, yaw].
```

With two RBFs, `A` is `6 x 6` and `B` is `6 x 4`. The simulator's `p` and `q`
are neither set to zero nor altered. They are simply outside this reduced
predictor. This is the main limitation to keep in mind if roll/pitch coupling
becomes important.

The published approach describes randomly distributed Gaussian functions.
This implementation defaults to reproducible k-means centers because two RBFs
must cover the data efficiently. Use `--center-method random` for the closer
random-center reproduction. Centers are fit after standardizing each velocity,
so different units and magnitudes do not make one axis dominate the distance.

## Use the existing simulation data

No Fossen rewrite or new collector is required. Create training and held-out
datasets using the existing commands, for example:

```bash
conda activate marinegym
python -m EDMDc.collect_fossen --n 100 --out EDMDc/data/gaussian_train.npz
python -m EDMDc.collect_fossen --n 20 --out EDMDc/data/gaussian_test.npz
```

Isaac data from `python -m EDMDc.collect_isaac ...` uses the same `X`, `X_next`,
and `U` interface and can be used in exactly the same way.

## Train

The paper-style default is two Gaussian observables:

```bash
python -m Gaussian_dictionary.gaussian_edmdc \
  EDMDc/data/gaussian_train.npz \
  --n-rbfs 2 --lam 1e-3 \
  --out Gaussian_dictionary/model/gaussian_edmdc.npz
```

Useful controlled comparisons include:

```bash
# Closer reproduction of random Gaussian centers
python -m Gaussian_dictionary.gaussian_edmdc EDMDc/data/gaussian_train.npz \
  --center-method random --seed 7

# Test whether more Gaussian coverage helps
python -m Gaussian_dictionary.gaussian_edmdc EDMDc/data/gaussian_train.npz \
  --n-rbfs 8
```

Fit the four task-axis realized wrench instead of the normalized command only
when the dataset includes `U_realized`:

```bash
python -m Gaussian_dictionary.gaussian_edmdc EDMDc/data/gaussian_train.npz \
  --wrench realized
```

## Compare against the current 34-D dictionary

Always compare on a held-out file, with the same initial states, commands, and
horizons:

```bash
python -m Gaussian_dictionary.evaluate \
  Gaussian_dictionary/model/gaussian_edmdc.npz \
  EDMDc/data/gaussian_test.npz \
  --compare-34d EDMDc/model/ardusub_edmdc.npz
```

The evaluator reports one-step and recursive rollout RMSE for `[u,v,w,r]`,
adds a hold-last baseline, and saves a CSV under
`Gaussian_dictionary/results/`. Do not judge the dictionaries from training
error alone.

## MPC/simulation adapter

`GaussianEDMDcMPC` accepts the same full velocity measurement/reference shape
used by the current controller and returns the same four normalized commands:

```python
from Gaussian_dictionary.mpc import GaussianEDMDcMPC

controller = GaussianEDMDcMPC.from_npz(
    "Gaussian_dictionary/model/gaussian_edmdc.npz"
)
command = controller.step(nu_measured_6d, nu_reference_6d)
```

Internally it selects `[u,v,w,r]`. This lets a new simulation runner swap the
controller class without changing Fossen, allocation, thruster, or vehicle
dynamics. Because this reduced model does not predict `p,q`, compare it first
in tasks where roll and pitch remain naturally small; the 34-D model remains
the safer candidate when those couplings matter.

## Test

```bash
python -m unittest Gaussian_dictionary.test_gaussian_edmdc -v
```
