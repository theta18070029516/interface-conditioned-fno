from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from deep_fno_shared_benchmark.checkpoint import load_checkpoint, save_checkpoint
from deep_fno_shared_benchmark.config import DataConfig, ModelConfig, TrainingConfig
from deep_fno_shared_benchmark.data import (
    INTERNAL,
    DataNormalizer,
    generate_cases,
    load_cases,
    sample_cases,
    save_cases,
)
from deep_fno_shared_benchmark.experiment import main
from deep_fno_shared_benchmark.gpu_smoke_v2 import run as run_gpu_smoke
from deep_fno_shared_benchmark.formal_v2 import (
    FIXED_HYPERPARAMETERS,
    ensure_fixed_hyperparameter_manifest,
)
from deep_fno_shared_benchmark.models import (
    apply_model,
    init_model,
    parameter_count,
    rho_values,
)


class ProtocolV2DataTests(unittest.TestCase):
    def test_all_jumps_are_nonzero_and_remain_in_registered_intervals(self) -> None:
        config = DataConfig.v2()
        cases = generate_cases(512, seed=21, config=config)
        self.assertTrue(np.all(cases.category == INTERNAL))
        self.assertGreaterEqual(float(np.min(np.abs(cases.jump))), 0.5)
        self.assertLessEqual(float(np.max(np.abs(cases.jump))), 1.5)
        self.assertGreaterEqual(float(np.min(cases.interface_location)), -0.75)
        self.assertLessEqual(float(np.max(cases.interface_location)), 0.25)
        final_location = cases.interface_location + config.displacement
        self.assertGreaterEqual(float(np.min(final_location)), -0.25)
        self.assertLessEqual(float(np.max(final_location)), 0.75)

    def test_analytic_gamma_transport_is_resolution_independent(self) -> None:
        config = DataConfig.v2()
        cases = generate_cases(48, seed=22, config=config)
        for resolution in (64, 128, 256):
            dataset = sample_cases(cases, resolution)
            expected0 = np.where(
                dataset.x[None, :] < cases.interface_location[:, None], -1.0, 1.0
            )
            expected_target = np.where(
                dataset.x[None, :]
                < (cases.interface_location + config.displacement)[:, None],
                -1.0,
                1.0,
            )
            np.testing.assert_array_equal(dataset.gamma0, expected0)
            np.testing.assert_array_equal(dataset.gamma_target, expected_target)

    def test_v2_cases_round_trip_with_protocol_metadata(self) -> None:
        cases = generate_cases(31, seed=24, config=DataConfig.v2())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v2_cases.npz"
            save_cases(path, cases)
            restored = load_cases(path)
        self.assertEqual(restored.config.protocol_version, 2)
        self.assertEqual(restored.config, cases.config)
        np.testing.assert_array_equal(restored.interface_location, cases.interface_location)

    def test_v2_config_rejects_gamma_loss_and_early_stopping(self) -> None:
        with self.assertRaises(ValueError):
            TrainingConfig(protocol_version=2, gamma_loss_weight=0.0)
        with self.assertRaises(ValueError):
            TrainingConfig(
                protocol_version=2,
                gamma_loss_weight=1.0,
                early_stopping=False,
            )


class ProtocolV2ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig(
            n_layers=4,
            hidden_channels=4,
            n_modes=3,
            lifting_channels=8,
            projection_channels=8,
        )
        rng = np.random.default_rng(23)
        self.inputs3 = rng.normal(size=(2, 3, 32)).astype(np.float32)
        self.inputs4 = rng.normal(size=(2, 4, 32)).astype(np.float32)
        self.inputs4[:, 2] = rng.choice([-1.0, 1.0], size=(2, 32))
        self.inputs4[:, 3] = rng.choice([-1.0, 1.0], size=(2, 32))

    def test_three_and_four_channel_contracts(self) -> None:
        fno = init_model("fno", jax.random.PRNGKey(1), self.config)
        fno_gamma = init_model("fno_gamma", jax.random.PRNGKey(2), self.config)
        oracle = init_model("shared_oracle", jax.random.PRNGKey(3), self.config)
        oracle_rho0 = init_model(
            "shared_oracle_rho0", jax.random.PRNGKey(4), self.config
        )
        self.assertEqual(fno["lifting"][0]["weight"].shape[1], 3)
        self.assertEqual(fno_gamma["lifting"][0]["weight"].shape[1], 4)
        self.assertEqual(oracle["lifting"][0]["weight"].shape[1], 4)
        self.assertEqual(oracle_rho0["lifting"][0]["weight"].shape[1], 4)
        self.assertNotIn("gamma_blocks", oracle)
        self.assertNotIn("gamma_blocks", oracle_rho0)

    def test_fixed_rho0_oracle_has_no_trainable_rho_and_equal_split(self) -> None:
        learnable = init_model("shared_oracle", jax.random.PRNGKey(31), self.config)
        fixed = init_model("shared_oracle_rho0", jax.random.PRNGKey(31), self.config)
        self.assertTrue(any("rho_eta" in block for block in learnable["blocks"]))
        self.assertFalse(any("rho_eta" in block for block in fixed["blocks"]))
        self.assertLess(parameter_count(fixed), parameter_count(learnable))

        diagnostics = apply_model(
            "shared_oracle_rho0",
            fixed,
            jnp.asarray(self.inputs4),
            self.config,
            return_diagnostics=True,
        )
        for rho in diagnostics["rho_layers"]:
            self.assertEqual(float(np.asarray(rho)), 0.0)
        fixed_config = replace(self.config, rho_mode="fixed_zero")
        expected = apply_model(
            "shared_oracle",
            fixed,
            jnp.asarray(self.inputs4),
            fixed_config,
        )["u"]
        np.testing.assert_array_equal(diagnostics["u"], expected)
        for rho in rho_values(fixed, fixed_config):
            self.assertEqual(rho.shape, (4, 4))
            np.testing.assert_array_equal(rho, np.zeros((4, 4), dtype=np.float32))

    def test_oracle_gate_sequence_and_superresolution(self) -> None:
        params = init_model("shared_oracle", jax.random.PRNGKey(4), self.config)
        diagnostics = apply_model(
            "shared_oracle",
            params,
            jnp.asarray(self.inputs4),
            self.config,
            return_diagnostics=True,
        )
        self.assertEqual(len(diagnostics["gamma_layers"]), 5)
        np.testing.assert_array_equal(
            diagnostics["gamma_layers"][0], self.inputs4[:, 2]
        )
        for gamma in diagnostics["gamma_layers"][1:]:
            np.testing.assert_array_equal(gamma, self.inputs4[:, 3])
        fine = np.resize(self.inputs4, (2, 4, 64)).astype(np.float32)
        output = apply_model(
            "shared_oracle", params, jnp.asarray(fine), self.config
        )
        self.assertEqual(output["u"].shape, (2, 1, 64))

    def test_v2_models_have_finite_gradients_and_round_trip(self) -> None:
        for index, (kind, inputs) in enumerate(
            (
                ("fno", self.inputs3),
                ("fno_gamma", self.inputs4),
                ("shared_oracle", self.inputs4),
                ("shared_oracle_rho0", self.inputs4),
            ),
            start=5,
        ):
            params = init_model(kind, jax.random.PRNGKey(index), self.config)

            def objective(current):
                return jnp.mean(
                    apply_model(kind, current, jnp.asarray(inputs), self.config)["u"]
                    ** 2
                )

            gradients = jax.grad(objective)(params)
            for leaf in jax.tree_util.tree_leaves(gradients):
                self.assertTrue(np.all(np.isfinite(np.asarray(leaf))))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"{kind}.npz"
                save_checkpoint(path, params, {"kind": kind})
                restored, metadata = load_checkpoint(path, params)
            self.assertEqual(metadata["kind"], kind)
            expected = apply_model(kind, params, jnp.asarray(inputs), self.config)["u"]
            actual = apply_model(kind, restored, jnp.asarray(inputs), self.config)["u"]
            np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


class ProtocolV2SmokeTests(unittest.TestCase):
    def test_formal_hyperparameters_are_fixed_and_immutable(self) -> None:
        self.assertEqual(
            FIXED_HYPERPARAMETERS,
            {"learning_rate": 1.0e-3, "weight_decay": 1.0e-4},
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest, path = ensure_fixed_hyperparameter_manifest(tmp)
            self.assertEqual(
                manifest["selected_hyperparameters"], FIXED_HYPERPARAMETERS
            )
            self.assertTrue(path.is_file())
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved["selected_hyperparameters"]["learning_rate"] = 3.0e-4
            path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "manifest"):
                ensure_fixed_hyperparameter_manifest(tmp)

    def test_physical_gpu_six_is_rejected_before_device_query(self) -> None:
        with self.assertRaisesRegex(ValueError, "permanently excluded"):
            run_gpu_smoke(6)

    def test_fixed_rho0_oracle_end_to_end_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "shared_oracle_rho0_smoke"
            status = main(
                [
                    "--preset",
                    "smoke_v2",
                    "--models",
                    "shared_oracle_rho0",
                    "--rho-mode",
                    "fixed_zero",
                    "--epochs",
                    "2",
                    "--checkpoint-every",
                    "1",
                    "--log-every",
                    "1",
                    "--run-tag",
                    "main_v2",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(status, 0)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["models"], ["shared_oracle_rho0"])
            self.assertEqual(manifest["model_config"]["rho_mode"], "fixed_zero")
            self.assertEqual(
                manifest["rho_policy"]["shared_oracle_rho0"],
                "fixed_zero_non_trainable",
            )
            run = output / "seed_0" / "shared_oracle_rho0"
            with np.load(run / "checkpoint_best.npz", allow_pickle=False) as saved:
                self.assertFalse(any("rho_eta" in str(path) for path in saved["paths"]))
            with np.load(run / "rho_layers.npz", allow_pickle=False) as saved:
                self.assertTrue(saved.files)
                for name in saved.files:
                    np.testing.assert_array_equal(
                        saved[name], np.zeros((8, 8), dtype=np.float32)
                    )
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["64"]["oracle_gate_max_abs_error"], 0.0)
            self.assertEqual(metrics["64"]["rho_fixed_zero_max_abs_error"], 0.0)

    def test_v2_end_to_end_outputs_and_checkpoint_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "formal_main_v2_smoke"
            status = main(
                [
                    "--preset",
                    "smoke_v2",
                    "--models",
                    "fno",
                    "fno_gamma",
                    "shared_oracle",
                    "--epochs",
                    "2",
                    "--checkpoint-every",
                    "1",
                    "--log-every",
                    "1",
                    "--bootstrap-repetitions",
                    "100",
                    "--run-tag",
                    "main_v2",
                    "--output-dir",
                    str(output),
                ]
            )
            with (output / "manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["protocol_version"], 2)
            self.assertFalse(manifest["training_config"]["early_stopping"])
            self.assertEqual(manifest["training_config"]["gamma_loss_weight"], 0.0)
            self.assertTrue((output / "paired_bootstrap.json").exists())

            for model in ("fno", "fno_gamma", "shared_oracle"):
                run = output / "seed_0" / model
                for filename in (
                    "checkpoint.npz",
                    "checkpoint_best.npz",
                    "checkpoint_final.npz",
                    "history.csv",
                ):
                    self.assertTrue((run / filename).exists(), filename)
                with (run / "history.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    history = list(csv.DictReader(handle))
                self.assertEqual(len(history), 2)
                self.assertNotIn("train_gamma_mse", history[0])
                self.assertNotIn("validation_gamma_mse", history[0])
                best_row = min(
                    history, key=lambda row: float(row["validation_relative_l2"])
                )
                with np.load(run / "checkpoint_best.npz", allow_pickle=False) as saved:
                    best_metadata = json.loads(str(saved["metadata_json"]))
                with np.load(run / "checkpoint_final.npz", allow_pickle=False) as saved:
                    final_metadata = json.loads(str(saved["metadata_json"]))
                self.assertEqual(best_metadata["best_epoch"], int(float(best_row["epoch"])))
                self.assertEqual(final_metadata["epoch"], 2)
                with np.load(
                    run / "checkpoints" / "epoch_0002.npz", allow_pickle=False
                ) as periodic, np.load(
                    run / "checkpoint_final.npz", allow_pickle=False
                ) as final:
                    parameter_keys = sorted(
                        key for key in final.files if key.startswith("array_")
                    )
                    for key in parameter_keys:
                        np.testing.assert_array_equal(final[key], periodic[key])
                with np.load(run / "predictions_n64.npz", allow_pickle=False) as saved:
                    self.assertNotIn("gamma_mse", saved.files)
                    self.assertNotIn("false_jump", saved.files)
                    if model in {"fno_gamma", "shared_oracle"}:
                        np.testing.assert_array_equal(
                            saved["gamma_input"], saved["gamma_target"]
                        )
            oracle_run = output / "seed_0" / "shared_oracle"
            self.assertTrue((oracle_run / "rho_layers.npz").exists())
            with (oracle_run / "metrics.json").open(encoding="utf-8") as handle:
                oracle_metrics = json.load(handle)
            self.assertEqual(oracle_metrics["64"]["oracle_gate_max_abs_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
