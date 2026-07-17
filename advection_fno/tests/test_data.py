from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from advection_fno.config import ExperimentConfig
from advection_fno.data import generate_dataset, load_dataset, save_dataset


class DataGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(n_x=32)

    def test_endpoint_samples_are_continuous_and_have_constant_gamma(self) -> None:
        dataset = generate_dataset(
            20,
            "A",
            seed=1,
            config=self.config,
            s_mode="endpoints",
        )
        self.assertTrue(np.all(dataset.metadata["is_endpoint"]))
        for index, s in enumerate(dataset.metadata["s"]):
            expected = 1.0 if s == self.config.x_min else -1.0
            np.testing.assert_array_equal(dataset.gamma0[index], expected)
            np.testing.assert_array_equal(dataset.gamma1[index], expected)
            self.assertAlmostEqual(float(np.ptp(dataset.u0[index])), 0.0, places=12)
            self.assertAlmostEqual(float(np.ptp(dataset.u1[index])), 0.0, places=12)

    def test_internal_interface_moves_by_c_times_T(self) -> None:
        dataset = generate_dataset(
            128,
            "A",
            seed=2,
            config=self.config,
            s_mode="interior_grid",
        )
        locations = dataset.metadata["output_jump_location"]
        np.testing.assert_allclose(
            locations,
            dataset.metadata["s"] + self.config.velocity * self.config.final_time,
        )
        for index in np.flatnonzero(dataset.metadata["has_output_jump"]):
            changes = np.count_nonzero(np.diff(dataset.gamma1[index]))
            self.assertEqual(changes, 1)

    def test_false_interface_dataset_has_smooth_target(self) -> None:
        dataset = generate_dataset(
            32,
            "A",
            seed=3,
            config=self.config,
            s_mode="interior_grid",
            force_zero_jump=True,
        )
        np.testing.assert_allclose(dataset.metadata["J"], 0.0)
        self.assertTrue(np.all(np.ptp(dataset.u0, axis=1) < 1.0e-12))
        self.assertTrue(np.all(np.ptp(dataset.u1, axis=1) < 1.0e-12))

    def test_npz_round_trip_uses_no_pickle(self) -> None:
        dataset = generate_dataset(8, "B", seed=4, config=self.config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.npz"
            save_dataset(path, dataset)
            restored = load_dataset(path)
        np.testing.assert_allclose(restored.u0, dataset.u0)
        np.testing.assert_allclose(restored.gamma1, dataset.gamma1)
        np.testing.assert_allclose(restored.metadata["smooth_coefficients"], dataset.metadata["smooth_coefficients"])


if __name__ == "__main__":
    unittest.main()
