from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from deep_fno_shared_benchmark.checkpoint import load_checkpoint, save_checkpoint
from deep_fno_shared_benchmark.config import ModelConfig
from deep_fno_shared_benchmark.models import (
    apply_fno,
    apply_shared,
    init_fno,
    init_shared,
    parameter_count,
    shared_spectral_term,
    spectral_convolution,
)


class DeepModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig(
            n_layers=2,
            hidden_channels=4,
            n_modes=3,
            lifting_channels=8,
            projection_channels=8,
        )
        rng = np.random.default_rng(10)
        self.inputs = rng.normal(size=(3, 3, 32)).astype(np.float32)
        self.inputs[:, 0] = np.linspace(-1.0, 1.0, 32)
        self.inputs[:, 2] = rng.choice([-1.0, 1.0], size=(3, 32))

    def test_forward_shapes_and_gamma_bounds(self) -> None:
        fno = init_fno(jax.random.PRNGKey(1), self.config)
        shared = init_shared(jax.random.PRNGKey(2), self.config)
        fno_output = apply_fno(fno, jnp.asarray(self.inputs), self.config)
        shared_output = apply_shared(shared, jnp.asarray(self.inputs), self.config)
        self.assertEqual(fno_output["u"].shape, (3, 1, 32))
        self.assertEqual(shared_output["u"].shape, (3, 1, 32))
        self.assertEqual(shared_output["gamma"].shape, (3, 1, 32))
        self.assertTrue(np.all(np.isfinite(np.asarray(shared_output["u"]))))
        self.assertTrue(np.all(np.abs(np.asarray(shared_output["gamma"])) <= 1.0))

    def test_models_accept_a_finer_grid_without_new_parameters(self) -> None:
        params = init_shared(jax.random.PRNGKey(3), self.config)
        fine = np.resize(self.inputs, (3, 3, 64)).astype(np.float32)
        fine[:, 0] = np.linspace(-1.0, 1.0, 64)
        output = apply_shared(params, jnp.asarray(fine), self.config)
        self.assertEqual(output["u"].shape, (3, 1, 64))

    def test_rho_one_matches_plain_spectral_term(self) -> None:
        key = jax.random.PRNGKey(4)
        value = jax.random.normal(key, (2, 4, 32))
        real = jax.random.normal(key, (4, 4, 3))
        weight = (real + 1j * real).astype(jnp.complex64)
        gamma_in = jnp.where(value[:, 0] > 0, 1.0, -1.0)
        gamma_out = -gamma_in
        plain = spectral_convolution(value, weight, self.config.pad_fraction)
        shared = shared_spectral_term(
            value,
            gamma_in,
            gamma_out,
            weight,
            jnp.asarray(1.0),
            self.config.pad_fraction,
        )
        np.testing.assert_allclose(shared, plain, rtol=1.0e-5, atol=1.0e-5)

    def test_constant_gamma_matches_plain_for_any_rho(self) -> None:
        relative_l2_limit = 5.0e-4
        scaled_max_limit = 5.0e-4
        key = jax.random.PRNGKey(5)
        value = jax.random.normal(key, (2, 4, 32))
        weight = jax.random.normal(key, (4, 4, 3)).astype(jnp.complex64)
        gamma = jnp.ones((2, 32))
        plain = spectral_convolution(value, weight, self.config.pad_fraction)
        plain_np = np.asarray(plain, dtype=np.float64)
        plain_l2 = max(float(np.linalg.norm(plain_np.ravel())), 1.0e-12)
        plain_scale = max(1.0, float(np.max(np.abs(plain_np))))

        for rho in (0.0, 0.25, 0.8, 1.0):
            shared = shared_spectral_term(
                value,
                gamma,
                gamma,
                weight,
                jnp.asarray(rho, dtype=jnp.float32),
                self.config.pad_fraction,
            )
            residual = np.asarray(shared, dtype=np.float64) - plain_np
            max_abs = float(np.max(np.abs(residual)))
            relative_l2 = float(np.linalg.norm(residual.ravel()) / plain_l2)
            scaled_max = max_abs / plain_scale
            details = (
                f"rho={rho}: max_abs={max_abs:.8e}, "
                f"relative_l2={relative_l2:.8e}, "
                f"scaled_max={scaled_max:.8e}"
            )
            self.assertLess(relative_l2, relative_l2_limit, msg=details)
            self.assertLess(scaled_max, scaled_max_limit, msg=details)

    def test_gradients_are_finite(self) -> None:
        params = init_shared(jax.random.PRNGKey(6), self.config)

        def objective(current):
            output = apply_shared(current, jnp.asarray(self.inputs), self.config)
            return jnp.mean(output["u"] ** 2) + jnp.mean(output["gamma"] ** 2)

        gradients = jax.grad(objective)(params)
        for leaf in jax.tree_util.tree_leaves(gradients):
            self.assertTrue(np.all(np.isfinite(np.asarray(leaf))))

    def test_shared_reports_extra_parameters(self) -> None:
        fno = init_fno(jax.random.PRNGKey(7), self.config)
        shared = init_shared(jax.random.PRNGKey(7), self.config)
        self.assertGreater(parameter_count(shared), parameter_count(fno))

    def test_npz_checkpoint_round_trip(self) -> None:
        params = init_shared(jax.random.PRNGKey(8), self.config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.npz"
            save_checkpoint(path, params, {"kind": "shared", "seed": 8})
            restored, metadata = load_checkpoint(path, params)
        self.assertEqual(metadata["kind"], "shared")
        expected = apply_shared(params, jnp.asarray(self.inputs), self.config)["u"]
        actual = apply_shared(restored, jnp.asarray(self.inputs), self.config)["u"]
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
