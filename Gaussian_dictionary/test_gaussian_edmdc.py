"""Focused tests for the Gaussian dictionary and simulator-facing adapter."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Gaussian_dictionary.gaussian_edmdc import (
    CONTROL_DIM,
    STATE_INDICES,
    GaussianDictionary,
    GaussianEDMDcModel,
    fit_dictionary,
    fit_edmdc,
    load_dataset_arrays,
    select_controlled_state,
)


class GaussianDictionaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dictionary = GaussianDictionary(
            centers=np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
            widths=np.array([1.0, 1.0]),
            state_mean=np.zeros(4),
            state_scale=np.ones(4),
        )

    def test_lift_has_four_states_plus_two_scalar_gaussians(self) -> None:
        state = np.array([0.1, -0.2, 0.3, -0.4])
        lifted = self.dictionary.lift(state)
        self.assertEqual(lifted.shape, (6,))
        np.testing.assert_allclose(lifted[:4], state)

    def test_each_gaussian_uses_every_state_coordinate(self) -> None:
        baseline = self.dictionary.rbf(np.zeros(4))
        for coordinate in range(4):
            perturbed = np.zeros(4)
            perturbed[coordinate] = 0.2
            values = self.dictionary.rbf(perturbed)
            self.assertFalse(np.isclose(values[0], baseline[0]))
            self.assertFalse(np.isclose(values[1], baseline[1]))

    def test_full_state_selection_does_not_zero_p_or_q(self) -> None:
        full = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        selected = select_controlled_state(full)
        np.testing.assert_array_equal(selected, np.array([1.0, 2.0, 3.0, 6.0]))
        np.testing.assert_array_equal(full, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    def test_existing_dataset_schema_and_realized_wrench_selection(self) -> None:
        dataset = {
            "X": np.arange(18.0).reshape(3, 6),
            "X_next": np.arange(18.0, 36.0).reshape(3, 6),
            "U": np.arange(12.0).reshape(3, 4),
            "U_realized": np.arange(18.0).reshape(3, 6),
        }
        state, state_next, commanded = load_dataset_arrays(dataset, "commanded")
        np.testing.assert_array_equal(state, dataset["X"][:, list(STATE_INDICES)])
        np.testing.assert_array_equal(state_next, dataset["X_next"][:, list(STATE_INDICES)])
        np.testing.assert_array_equal(commanded, dataset["U"])
        _, _, realized = load_dataset_arrays(dataset, "realized")
        np.testing.assert_array_equal(realized, dataset["U_realized"][:, list(STATE_INDICES)])


class GaussianModelTests(unittest.TestCase):
    def _fit_synthetic_model(self) -> tuple[GaussianEDMDcModel, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        state = rng.uniform(-1.0, 1.0, size=(2000, 4))
        control = rng.uniform(-0.8, 0.8, size=(2000, CONTROL_DIM))
        physical_A = np.array([
            [0.92, 0.02, 0.00, 0.01],
            [-0.01, 0.90, 0.02, 0.00],
            [0.00, -0.01, 0.89, 0.01],
            [0.01, 0.00, -0.01, 0.94],
        ])
        physical_B = np.diag([0.08, 0.07, 0.06, 0.05])
        state_next = state @ physical_A.T + control @ physical_B.T
        dictionary = fit_dictionary(state, n_rbfs=2, center_method="kmeans", seed=3)
        A, B = fit_edmdc(state, state_next, control, dictionary, lam=1e-12)
        return GaussianEDMDcModel(A, B, dictionary, dt=0.05), state, state_next, control

    def test_fit_recovers_linear_physical_dynamics(self) -> None:
        model, state, state_next, control = self._fit_synthetic_model()
        rmse = float(np.sqrt(np.mean((model.predict(state, control) - state_next) ** 2)))
        self.assertLess(rmse, 1e-9)

    def test_rollout_shape(self) -> None:
        model, state, _, control = self._fit_synthetic_model()
        predicted = model.rollout(state[0], control[:12])
        self.assertEqual(predicted.shape, (13, 4))
        np.testing.assert_allclose(predicted[0], state[0])

    def test_full_velocity_adapter_carries_p_and_q(self) -> None:
        model, state, _, control = self._fit_synthetic_model()
        full = np.array([state[0, 0], state[0, 1], state[0, 2], 0.7, -0.8, state[0, 3]])
        predicted = model.predict_full_nu(full, control[0])
        self.assertEqual(predicted.shape, (6,))
        self.assertEqual(predicted[3], full[3])
        self.assertEqual(predicted[4], full[4])

    def test_model_save_load_round_trip(self) -> None:
        model, state, _, control = self._fit_synthetic_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = GaussianEDMDcModel.load(path)
            np.testing.assert_allclose(
                loaded.predict(state[:10], control[:10]),
                model.predict(state[:10], control[:10]),
            )

    @unittest.skipUnless(
        importlib.util.find_spec("osqp") is not None and importlib.util.find_spec("scipy") is not None,
        "osqp/scipy are not installed",
    )
    def test_mpc_accepts_full_six_state_interface(self) -> None:
        from Gaussian_dictionary.mpc import GaussianEDMDcMPC, GaussianMPCConfig

        model, state, _, _ = self._fit_synthetic_model()
        config = GaussianMPCConfig(N=4)
        controller = GaussianEDMDcMPC(model, config)
        full = np.array([state[0, 0], state[0, 1], state[0, 2], 0.2, -0.3, state[0, 3]])
        command = controller.step(full, np.zeros(6))
        self.assertEqual(command.shape, (4,))
        self.assertTrue(np.all(command >= np.asarray(config.u_min)))
        self.assertTrue(np.all(command <= np.asarray(config.u_max)))
        self.assertTrue(np.all(np.abs(command) <= np.asarray(config.du_max) + 1e-5))


if __name__ == "__main__":
    unittest.main()
