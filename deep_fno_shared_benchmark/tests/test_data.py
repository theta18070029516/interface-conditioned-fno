from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from deep_fno_shared_benchmark.config import DataConfig
from deep_fno_shared_benchmark.data import (
    ENDPOINT,
    EXITING,
    INTERNAL,
    ZERO_JUMP,
    DataNormalizer,
    generate_cases,
    load_cases,
    sample_cases,
    save_cases,
)


class ContinuousDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DataConfig()

    def test_category_counts_follow_registered_mixture(self) -> None:
        cases = generate_cases(100, seed=1, config=self.config)
        counts = np.bincount(cases.category, minlength=4)
        np.testing.assert_array_equal(counts, np.array([70, 10, 10, 10]))

    def test_internal_interface_transports_exact_distance(self) -> None:
        cases = generate_cases(128, seed=2, config=self.config)
        dataset = sample_cases(cases, 256)
        for index in np.flatnonzero(cases.category == INTERNAL):
            changes0 = np.flatnonzero(np.diff(dataset.gamma0[index]))
            changes1 = np.flatnonzero(np.diff(dataset.gamma_target[index]))
            self.assertEqual(changes0.size, 1)
            self.assertEqual(changes1.size, 1)
            location0 = 0.5 * (
                dataset.x[changes0[0]] + dataset.x[changes0[0] + 1]
            )
            location1 = 0.5 * (
                dataset.x[changes1[0]] + dataset.x[changes1[0] + 1]
            )
            self.assertAlmostEqual(
                location1 - location0,
                self.config.displacement,
                delta=2.0 * self.config.domain_length / dataset.n_x,
            )

    def test_continuous_cases_use_canonical_gamma(self) -> None:
        cases = generate_cases(100, seed=3, config=self.config)
        dataset = sample_cases(cases, 64)
        continuous = np.isin(cases.category, [ZERO_JUMP, ENDPOINT])
        np.testing.assert_array_equal(dataset.gamma0[continuous], 1.0)
        np.testing.assert_array_equal(dataset.gamma_target[continuous], 1.0)
        np.testing.assert_array_equal(cases.jump[cases.category == ZERO_JUMP], 0.0)

    def test_exiting_interfaces_leave_no_target_jump(self) -> None:
        cases = generate_cases(100, seed=4, config=self.config)
        dataset = sample_cases(cases, 128)
        exiting = cases.category == EXITING
        self.assertTrue(np.all(np.ptp(dataset.gamma0[exiting], axis=1) > 0))
        self.assertTrue(np.all(np.ptp(dataset.gamma_target[exiting], axis=1) == 0))

    def test_same_cases_resample_without_parameter_drift(self) -> None:
        cases = generate_cases(32, seed=5, config=self.config)
        coarse = sample_cases(cases, 64)
        fine = sample_cases(cases, 128)
        np.testing.assert_array_equal(coarse.cases.sample_id, fine.cases.sample_id)
        np.testing.assert_allclose(
            coarse.cases.interface_location,
            fine.cases.interface_location,
        )
        self.assertEqual(fine.n_x, 2 * coarse.n_x)

    def test_case_npz_round_trip_uses_no_pickle(self) -> None:
        cases = generate_cases(17, seed=6, config=self.config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.npz"
            save_cases(path, cases)
            restored = load_cases(path)
        np.testing.assert_array_equal(restored.category, cases.category)
        np.testing.assert_allclose(restored.sin_coefficients, cases.sin_coefficients)
        self.assertEqual(restored.config, cases.config)

    def test_global_normalizer_round_trip(self) -> None:
        dataset = sample_cases(generate_cases(32, seed=7, config=self.config), 64)
        normalizer = DataNormalizer.fit(dataset)
        restored = normalizer.denormalize(normalizer.normalize(dataset.u_target))
        np.testing.assert_allclose(restored, dataset.u_target, rtol=1.0e-6, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
