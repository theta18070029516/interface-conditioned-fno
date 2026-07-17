from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from advection_fno.experiment import main


class EndToEndSmokeTest(unittest.TestCase):
    def test_smoke_experiment_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result"
            status = main(
                [
                    "--preset",
                    "smoke",
                    "--stage",
                    "A",
                    "--output-dir",
                    str(output),
                    "--skip-ablations",
                ]
            )
            self.assertEqual(status, 0)
            for filename in ("metrics.csv", "summary.csv", "comparisons.csv", "manifest.json"):
                path = output / filename
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
            model_files = list((output / "models").glob("*shared_learnable_rho*.npz"))
            self.assertTrue(model_files)
            with np.load(model_files[0]) as saved:
                rho = float(saved["rho"])
                self.assertGreater(rho, 0.0)
                self.assertLess(rho, 1.0)
                self.assertIn("rho_eta", saved.files)
            two_channel_files = list((output / "models").glob("*two_channel_fno*.npz"))
            self.assertTrue(two_channel_files)
            with np.load(two_channel_files[0]) as saved:
                self.assertEqual(str(saved["model"]), "two_channel_fno")
                self.assertIn("weights0", saved.files)
                self.assertIn("weights1", saved.files)
            hidden_files = list((output / "models").glob("*hidden_width_2*.npz"))
            self.assertTrue(hidden_files)
            with np.load(hidden_files[0]) as saved:
                self.assertEqual(str(saved["model"]), "hidden_width_2")
                self.assertAlmostEqual(
                    float(saved["parameter_ridge"]), 2.0 * float(saved["ridge"])
                )


if __name__ == "__main__":
    unittest.main()
