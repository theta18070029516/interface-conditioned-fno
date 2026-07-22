from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from deep_fno_shared_benchmark.experiment import main


class EndToEndSmokeTest(unittest.TestCase):
    def test_smoke_experiment_writes_reproducible_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "smoke"
            captured_output = io.StringIO()
            with redirect_stdout(captured_output), redirect_stderr(captured_output):
                status = main(
                    [
                        "--preset",
                        "smoke",
                        "--models",
                        "fno",
                        "shared",
                        "--epochs",
                        "1",
                        "--checkpoint-every",
                        "1",
                        "--log-every",
                        "1",
                        "--progress",
                        "--bootstrap-repetitions",
                        "100",
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(status, 0)
            text = captured_output.getvalue()
            self.assertIn("[fno seed=0] epoch 1/1", text)
            self.assertIn("[shared seed=0] epoch 1/1", text)
            self.assertIn("validation_relative_l2=", text)
            for filename in ("summary.csv", "manifest.json", "paired_bootstrap.json"):
                path = output / filename
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
            with (output / "summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            for row in rows:
                self.assertIn("high_frequency_absolute_error_mean", row)
                self.assertIn("high_frequency_relative_error_mean", row)
                self.assertIn(
                    "high_frequency_relative_error_valid_fraction", row
                )
            for model in ("fno", "shared"):
                run = output / "seed_0" / model
                self.assertTrue((run / "checkpoint.npz").exists())
                self.assertTrue(
                    (run / "checkpoints" / "epoch_0001.npz").exists()
                )
                with np.load(
                    run / "checkpoints" / "epoch_0001.npz",
                    allow_pickle=False,
                ) as saved:
                    checkpoint_metadata = json.loads(str(saved["metadata_json"]))
                self.assertEqual(checkpoint_metadata["epoch"], 1)
                self.assertEqual(checkpoint_metadata["kind"], model)
                self.assertFalse(checkpoint_metadata["resume_capable"])
                self.assertIn("validation_relative_l2", checkpoint_metadata["metrics"])
                self.assertTrue((run / "history.csv").exists())
                self.assertTrue((run / "predictions_n64.npz").exists())
            with (output / "manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(
                manifest["runtime_controls"],
                {
                    "checkpoint_every": 1,
                    "log_every": 1,
                    "progress": True,
                },
            )


if __name__ == "__main__":
    unittest.main()
