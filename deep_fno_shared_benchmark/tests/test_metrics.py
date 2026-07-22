from __future__ import annotations

import unittest

import numpy as np

from deep_fno_shared_benchmark.metrics import _high_frequency_errors


class HighFrequencyMetricTests(unittest.TestCase):
    def test_zero_target_tail_uses_absolute_error_only(self) -> None:
        n_x = 64
        target = np.ones((2, n_x), dtype=np.float64)
        prediction = target.copy()
        prediction[0] += 0.1 * (-1.0) ** np.arange(n_x)

        absolute, relative = _high_frequency_errors(
            prediction, target, cutoff=4
        )

        self.assertGreater(absolute[0], 0.0)
        self.assertEqual(absolute[1], 0.0)
        self.assertTrue(np.all(np.isnan(relative)))

    def test_relative_error_is_reported_for_nonzero_target_tail(self) -> None:
        n_x = 64
        grid = np.arange(n_x, dtype=np.float64) / n_x
        target = np.sin(2.0 * np.pi * 8.0 * grid)[None, :]
        prediction = 1.25 * target

        absolute, relative = _high_frequency_errors(
            prediction, target, cutoff=4
        )

        self.assertGreater(absolute[0], 0.0)
        np.testing.assert_allclose(relative, 0.25, rtol=1.0e-12, atol=1.0e-12)

    def test_absolute_error_uses_resolution_consistent_fft_scaling(self) -> None:
        errors = []
        for n_x in (64, 128):
            grid = np.arange(n_x, dtype=np.float64) / n_x
            target = np.zeros((1, n_x), dtype=np.float64)
            prediction = np.sin(2.0 * np.pi * 8.0 * grid)[None, :]
            absolute, _ = _high_frequency_errors(
                prediction, target, cutoff=4
            )
            errors.append(absolute[0])

        np.testing.assert_allclose(errors[0], errors[1], rtol=1.0e-12, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
