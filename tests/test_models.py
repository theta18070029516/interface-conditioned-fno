from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from advection_fno.config import ExperimentConfig
from advection_fno.data import generate_dataset
from advection_fno.models import (
    design_matrix,
    exact_shift_multiplier,
    fit_optax_check,
    fit_ridge,
    fit_shared_learnable_rho,
    predict,
    spectral_convolution,
)


class ModelIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(n_x=32)
        self.rng = np.random.default_rng(10)

    def test_shared_rho_one_equals_plain(self) -> None:
        batch = 5
        u0 = self.rng.normal(size=(batch, self.config.n_x))
        gamma0 = self.rng.choice([-1.0, 1.0], size=u0.shape)
        gamma1 = self.rng.choice([-1.0, 1.0], size=u0.shape)
        weights = self.rng.normal(size=6) + 1j * self.rng.normal(size=6)
        weights[0] = weights[0].real
        parameters = {"weights": weights}
        plain = predict("plain", parameters, u0, gamma0, gamma1)
        for model in ("shared", "shared_learnable_rho"):
            shared = predict(model, parameters, u0, gamma0, gamma1, rho=1.0)
            np.testing.assert_allclose(shared, plain, rtol=1.0e-6, atol=1.0e-6)

    def test_shared_constant_gamma_equals_plain_for_every_rho(self) -> None:
        u0 = self.rng.normal(size=(4, self.config.n_x))
        gamma = np.ones_like(u0)
        weights = self.rng.normal(size=8) + 1j * self.rng.normal(size=8)
        weights[0] = weights[0].real
        parameters = {"weights": weights}
        plain = predict("plain", parameters, u0, gamma, gamma)
        for model in ("shared", "shared_learnable_rho"):
            for rho in (0.0, 0.25, 0.5, 1.0):
                shared = predict(model, parameters, u0, gamma, gamma, rho=rho)
                np.testing.assert_allclose(shared, plain, rtol=1.0e-6, atol=1.0e-6)

    def test_full_shift_multiplier_matches_integer_grid_translation(self) -> None:
        n_x = self.config.n_x
        pad = self.config.pad_points
        signal = np.zeros(n_x)
        signal[8:16] = 1.0
        padded = np.pad(signal, (pad, pad), mode="edge")
        shift_cells = round(self.config.displacement / self.config.dx)
        expected = np.roll(padded, shift_cells)[pad : pad + n_x]
        weights = exact_shift_multiplier(
            n_x,
            self.config.dx,
            self.config.displacement,
            pad_fraction=self.config.pad_fraction,
        )
        actual = spectral_convolution(
            weights, signal, pad_fraction=self.config.pad_fraction
        )
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)

    def test_ridge_fit_returns_finite_models(self) -> None:
        dataset = generate_dataset(32, "B", seed=11, config=self.config)
        results = {}
        for model in (
            "plain",
            "hidden_width_2",
            "two_channel_fno",
            "shared",
            "dual",
        ):
            result = fit_ridge(model, dataset, modes=4, batch_size=8)
            results[model] = result
            self.assertTrue(np.isfinite(result.condition_number))
            self.assertTrue(np.isfinite(result.train_mse))
            estimate = predict(
                model,
                result.parameters,
                dataset.u0,
                dataset.gamma0,
                dataset.gamma1,
                rho=result.rho,
            )
            self.assertEqual(estimate.shape, dataset.u1.shape)
            self.assertTrue(np.all(np.isfinite(estimate)))
        self.assertEqual(
            results["two_channel_fno"].parameter_count,
            results["dual"].parameter_count,
        )
        self.assertEqual(
            results["hidden_width_2"].parameter_count,
            results["dual"].parameter_count,
        )

    def test_design_matrix_matches_forward_evaluation(self) -> None:
        dataset = generate_dataset(5, "B", seed=12, config=self.config)
        modes = 4
        per_kernel = 2 * modes - 1
        for model in (
            "plain",
            "hidden_width_2",
            "two_channel_fno",
            "shared",
            "dual",
        ):
            parameter_count = per_kernel if model in ("plain", "shared") else 2 * per_kernel
            theta = self.rng.normal(size=parameter_count)
            if model in ("plain", "shared"):
                real = theta[:modes]
                imag = np.concatenate([[0.0], theta[modes:]])
                parameters = {"weights": real + 1j * imag}
            else:
                parameters = {}
                for branch, key in enumerate(("weights0", "weights1")):
                    part = theta[branch * per_kernel : (branch + 1) * per_kernel]
                    real = part[:modes]
                    imag = np.concatenate([[0.0], part[modes:]])
                    parameters[key] = real + 1j * imag
            rho = 0.25 if model == "shared" else 0.0
            matrix = np.asarray(
                design_matrix(
                    model,
                    dataset.u0,
                    dataset.gamma0,
                    dataset.gamma1,
                    modes,
                    rho=rho,
                )
            )
            from_matrix = np.einsum("sxp,p->sx", matrix, theta)
            from_forward = predict(
                model,
                parameters,
                dataset.u0,
                dataset.gamma0,
                dataset.gamma1,
                rho=rho,
            )
            np.testing.assert_allclose(from_matrix, from_forward, rtol=2.0e-6, atol=2.0e-6)

    def test_hidden_width_two_collapses_to_plain(self) -> None:
        dataset = generate_dataset(32, "B", seed=15, config=self.config)
        ridge = 1.0e-3
        plain = fit_ridge("plain", dataset, modes=4, ridge=ridge, batch_size=8)
        hidden = fit_ridge(
            "hidden_width_2", dataset, modes=4, ridge=ridge, batch_size=8
        )
        plain_prediction = predict(
            "plain",
            plain.parameters,
            dataset.u0,
            dataset.gamma0,
            dataset.gamma1,
        )
        hidden_prediction = predict(
            "hidden_width_2",
            hidden.parameters,
            dataset.u0,
            dataset.gamma0,
            dataset.gamma1,
        )
        np.testing.assert_allclose(
            hidden.parameters["weights0"], hidden.parameters["weights1"]
        )
        np.testing.assert_allclose(
            hidden_prediction, plain_prediction, rtol=1.0e-6, atol=1.0e-6
        )
        self.assertEqual(hidden.parameter_ridge, 2.0 * ridge)

    def test_two_channel_fno_is_standard_linear_channel_mixing(self) -> None:
        u0 = self.rng.normal(size=(4, self.config.n_x))
        gamma0 = self.rng.choice([-1.0, 1.0], size=u0.shape)
        gamma1_a = self.rng.choice([-1.0, 1.0], size=u0.shape)
        gamma1_b = -gamma1_a
        weights0 = self.rng.normal(size=6) + 1j * self.rng.normal(size=6)
        weights1 = self.rng.normal(size=6) + 1j * self.rng.normal(size=6)
        weights0[0] = weights0[0].real
        weights1[0] = weights1[0].real
        parameters = {"weights0": weights0, "weights1": weights1}
        expected = spectral_convolution(weights0, u0) + spectral_convolution(
            weights1, gamma0
        )
        actual_a = predict(
            "two_channel_fno", parameters, u0, gamma0, gamma1_a
        )
        actual_b = predict(
            "two_channel_fno", parameters, u0, gamma0, gamma1_b
        )
        np.testing.assert_allclose(actual_a, expected, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(actual_b, expected, rtol=1.0e-6, atol=1.0e-6)

    def test_optax_cross_check_approaches_ridge_solution(self) -> None:
        dataset = generate_dataset(20, "A", seed=13, config=self.config)
        ridge = fit_ridge("shared", dataset, modes=2, batch_size=10)
        optax_result = fit_optax_check(
            "shared",
            dataset,
            modes=2,
            steps=2_000,
            learning_rate=0.05,
        )
        relative_gap = abs(optax_result.train_mse - ridge.train_mse) / max(
            ridge.train_mse, 1.0e-12
        )
        self.assertLess(relative_gap, 1.0e-3)

    def test_learnable_rho_recovers_synthetic_coupling(self) -> None:
        dataset = generate_dataset(64, "B", seed=14, config=self.config)
        modes = 4
        target_rho = 0.8
        weights = self.rng.normal(size=modes) + 1j * self.rng.normal(size=modes)
        weights[0] = weights[0].real
        target = predict(
            "shared",
            {"weights": weights},
            dataset.u0,
            dataset.gamma0,
            dataset.gamma1,
            rho=target_rho,
        )
        synthetic = replace(dataset, u1=target)
        result = fit_shared_learnable_rho(
            synthetic,
            modes,
            rho_initial=0.2,
            ridge=1.0e-10,
            batch_size=16,
            steps=2_000,
            learning_rate=2.0e-2,
        )
        self.assertEqual(result.model, "shared_learnable_rho")
        self.assertGreater(result.rho, 0.0)
        self.assertLess(result.rho, 1.0)
        self.assertAlmostEqual(result.rho, target_rho, delta=5.0e-2)
        self.assertEqual(result.parameter_count, 2 * modes)
        self.assertTrue(np.isfinite(result.train_mse))
        self.assertLess(result.train_mse, 1.0e-7)


if __name__ == "__main__":
    unittest.main()
