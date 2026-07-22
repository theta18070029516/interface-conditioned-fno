"""Command-line runner for paired FNO/Shared experiments."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint import save_checkpoint
from .config import DataConfig, ModelConfig, TrainingConfig
from .data import (
    DataNormalizer,
    DeepAdvectionDataset,
    generate_cases,
    sample_cases,
    save_cases,
)
from .metrics import hierarchical_paired_bootstrap, prediction_metrics
from .models import ModelKind, apply_model, parameter_count, rho_values
from .training import EpochReport, predict_dataset, prepare_data, train_model


V1_MODELS = ("fno", "shared")
V2_MODELS = ("fno", "fno_gamma", "shared_oracle")
V2_SUPPLEMENTAL_MODELS = ("shared_oracle_rho0",)
V2_ALLOWED_MODELS = V2_MODELS + V2_SUPPLEMENTAL_MODELS
GAMMA_TARGET_MODELS = {"fno_gamma", "shared_oracle", "shared_oracle_rho0"}
ORACLE_SHARED_MODELS = {"shared_oracle", "shared_oracle_rho0"}


PRESETS = {
    "smoke": {
        "n_x": 64,
        "train_size": 128,
        "validation_size": 64,
        "test_size": 128,
        "seeds": (0,),
        "resolutions": (64, 128),
        "model": ModelConfig(
            n_layers=2,
            hidden_channels=8,
            n_modes=4,
            lifting_channels=16,
            projection_channels=16,
        ),
        "training": TrainingConfig(
            epochs=3,
            batch_size=32,
            patience=3,
            validation_batch_size=64,
        ),
    },
    "dev": {
        "n_x": 256,
        "train_size": 4_000,
        "validation_size": 1_000,
        "test_size": 1_000,
        "seeds": (0,),
        "resolutions": (256, 512),
        "model": ModelConfig(),
        "training": TrainingConfig(epochs=80, patience=20),
    },
    "main": {
        "n_x": 256,
        "train_size": 16_000,
        "validation_size": 2_000,
        "test_size": 4_000,
        "seeds": (0, 1, 2, 3, 4),
        "resolutions": (256, 512, 1024),
        "model": ModelConfig(),
        "training": TrainingConfig(),
    },
    "smoke_v2": {
        "n_x": 64,
        "train_size": 64,
        "validation_size": 32,
        "test_size": 64,
        "seeds": (0,),
        "resolutions": (64, 128),
        "model": ModelConfig(
            n_layers=2,
            hidden_channels=8,
            n_modes=4,
            lifting_channels=16,
            projection_channels=16,
        ),
        "training": TrainingConfig(
            epochs=2,
            batch_size=16,
            patience=2,
            gamma_loss_weight=0.0,
            validation_batch_size=32,
            protocol_version=2,
            early_stopping=False,
        ),
    },
    "dev_v2": {
        "n_x": 256,
        "train_size": 4_000,
        "validation_size": 1_000,
        "test_size": 1_000,
        "seeds": (0, 1),
        "resolutions": (256, 512),
        "model": ModelConfig(),
        "training": TrainingConfig(
            epochs=500,
            gamma_loss_weight=0.0,
            protocol_version=2,
            early_stopping=False,
        ),
    },
    "main_v2": {
        "n_x": 256,
        "train_size": 16_000,
        "validation_size": 2_000,
        "test_size": 4_000,
        "seeds": (0, 1, 2, 3, 4),
        "resolutions": (256, 512, 1024),
        "model": ModelConfig(),
        "training": TrainingConfig(
            epochs=500,
            gamma_loss_weight=0.0,
            protocol_version=2,
            early_stopping=False,
        ),
    },
}


def _parse_seeds(value: str | None, default: Iterable[int]) -> tuple[int, ...]:
    if value is None:
        return tuple(default)
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=True)
        + "\n",
        encoding="utf-8",
    )


def _rho_summary(values: list[np.ndarray]) -> list[dict[str, float | int]]:
    result = []
    for layer, value in enumerate(values, start=1):
        flat = np.asarray(value, dtype=np.float64).ravel()
        result.append(
            {
                "layer": layer,
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat)),
                "q05": float(np.quantile(flat, 0.05)),
                "median": float(np.median(flat)),
                "q95": float(np.quantile(flat, 0.95)),
                "near_zero_fraction": float(np.mean(flat < 1.0e-3)),
                "near_one_fraction": float(np.mean(flat > 1.0 - 1.0e-3)),
            }
        )
    return result


def _environment_manifest() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }


def _is_v2_preset(name: str) -> bool:
    return name.endswith("_v2")


def _gamma_integrity(
    kind: ModelKind,
    params: dict[str, object],
    dataset: DeepAdvectionDataset,
    normalizer: DataNormalizer,
    model_config: ModelConfig,
) -> dict[str, float]:
    if kind not in GAMMA_TARGET_MODELS:
        return {}
    prepared = prepare_data(dataset, normalizer, kind)
    input_error = float(
        np.max(np.abs(prepared.inputs[:, 3, :] - dataset.gamma_target))
    )
    result = {"gamma_target_input_max_abs_error": input_error}
    if kind in ORACLE_SHARED_MODELS:
        batch = prepared.inputs[: min(2, len(prepared))]
        diagnostics = apply_model(
            kind,
            params,
            jnp.asarray(batch),
            model_config,
            return_diagnostics=True,
        )
        expected = [batch[:, 2, :]] + [
            batch[:, 3, :] for _ in range(model_config.n_layers)
        ]
        actual = diagnostics["gamma_layers"]
        if len(actual) != len(expected):
            raise RuntimeError("oracle gamma diagnostic sequence has an invalid length")
        gate_error = max(
            float(np.max(np.abs(np.asarray(observed) - target)))
            for observed, target in zip(actual, expected, strict=True)
        )
        result["oracle_gate_max_abs_error"] = gate_error
        if kind == "shared_oracle_rho0":
            if any("rho_eta" in block for block in params["blocks"]):
                raise RuntimeError(
                    "fixed-rho Oracle-Shared must not contain trainable rho_eta"
                )
            rho_error = max(
                float(np.max(np.abs(np.asarray(rho))))
                for rho in diagnostics["rho_layers"]
            )
            result["rho_fixed_zero_max_abs_error"] = rho_error
    return result


def _save_rho_layers(
    path: Path,
    values: list[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            f"layer_{index}": np.asarray(value, dtype=np.float32)
            for index, value in enumerate(values, start=1)
        },
    )


def run_experiment(args: argparse.Namespace) -> int:
    is_v2 = _is_v2_preset(args.preset)
    checkpoint_every = (
        args.checkpoint_every
        if args.checkpoint_every is not None
        else (10 if is_v2 else 0)
    )
    log_every = args.log_every if args.log_every is not None else (10 if is_v2 else 0)
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if log_every < 0:
        raise ValueError("log_every must be non-negative")
    preset = PRESETS[args.preset]
    n_x = args.n_x or preset["n_x"]
    train_size = args.train_size or preset["train_size"]
    validation_size = args.validation_size or preset["validation_size"]
    test_size = args.test_size or preset["test_size"]
    seeds = _parse_seeds(args.seeds, preset["seeds"])
    model_config: ModelConfig = preset["model"]
    training_config: TrainingConfig = preset["training"]
    if args.rho_mode is not None:
        model_config = replace(model_config, rho_mode=args.rho_mode)
    if args.epochs is not None:
        training_config = replace(training_config, epochs=args.epochs)
    if args.learning_rate is not None:
        training_config = replace(
            training_config, learning_rate=args.learning_rate
        )
    if args.weight_decay is not None:
        training_config = replace(training_config, weight_decay=args.weight_decay)
    if args.gamma_loss_weight is not None:
        if is_v2:
            raise ValueError("protocol v2 forbids --gamma-loss-weight")
        training_config = replace(
            training_config, gamma_loss_weight=args.gamma_loss_weight
        )
    data_config = DataConfig.v2() if is_v2 else DataConfig()
    output_dir = Path(args.output_dir)
    output_text = str(output_dir.resolve()).lower().replace("\\", "/")
    if is_v2 and "formal_main_v1" in output_text:
        raise ValueError("protocol-v2 output must not target formal_main_v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    default_models = V2_MODELS if is_v2 else V1_MODELS
    models: tuple[ModelKind, ...] = tuple(args.models or default_models)
    if not models:
        raise ValueError("at least one model must be requested")
    allowed_models = set(V2_ALLOWED_MODELS if is_v2 else V1_MODELS)
    invalid_models = sorted(set(models) - allowed_models)
    if invalid_models:
        raise ValueError(
            f"preset {args.preset} does not allow models {invalid_models}; "
            f"allowed={sorted(allowed_models)}"
        )
    if "shared_oracle_rho0" in models:
        if len(models) != 1:
            raise ValueError(
                "shared_oracle_rho0 must run separately so its fixed rho config "
                "cannot alter another model"
            )
        if args.rho_mode not in {None, "fixed_zero"}:
            raise ValueError("shared_oracle_rho0 requires --rho-mode fixed_zero")
        model_config = replace(model_config, rho_mode="fixed_zero")

    resolutions = tuple(preset["resolutions"])
    if n_x != preset["n_x"]:
        resolutions = (n_x, 2 * n_x, 4 * n_x)
        if args.preset in {"smoke", "smoke_v2"}:
            resolutions = resolutions[:2]

    test_cases = generate_cases(
        test_size, seed=90_000, config=data_config, sample_id_offset=9_000_000
    )
    save_cases(output_dir / "datasets" / "test_cases.npz", test_cases)
    summary_rows: list[dict[str, object]] = []
    paired_relative_l2: dict[str, list[np.ndarray]] = {kind: [] for kind in models}

    for seed in seeds:
        train_cases = generate_cases(
            train_size,
            seed=10_000 + seed,
            config=data_config,
            sample_id_offset=seed * 1_000_000,
        )
        validation_cases = generate_cases(
            validation_size,
            seed=20_000 + seed,
            config=data_config,
            sample_id_offset=5_000_000 + seed * 100_000,
        )
        save_cases(
            output_dir / "datasets" / f"seed_{seed}_train_cases.npz",
            train_cases,
        )
        save_cases(
            output_dir / "datasets" / f"seed_{seed}_validation_cases.npz",
            validation_cases,
        )
        train_dataset = sample_cases(train_cases, n_x)
        validation_dataset = sample_cases(validation_cases, n_x)
        normalizer = DataNormalizer.fit(train_dataset)

        for kind in models:
            run_dir = output_dir / f"seed_{seed}" / kind

            def save_periodic_checkpoint(report: EpochReport) -> None:
                if (
                    checkpoint_every == 0
                    or report.epoch % checkpoint_every != 0
                ):
                    return
                save_checkpoint(
                    run_dir
                    / "checkpoints"
                    / f"epoch_{report.epoch:04d}.npz",
                    report.params,
                    {
                        "checkpoint_type": "periodic_current_params",
                        "resume_capable": False,
                        "kind": kind,
                        "seed": seed,
                        "epoch": report.epoch,
                        "metrics": report.metrics,
                        "best_epoch": report.best_epoch,
                        "best_validation_relative_l2": (
                            report.best_validation_relative_l2
                        ),
                        "stale_epochs": report.stale_epochs,
                        "normalizer": {
                            "mean": normalizer.mean,
                            "std": normalizer.std,
                        },
                    },
                )
                _write_csv(run_dir / "history.csv", list(report.history))

            start_train = time.perf_counter()
            result = train_model(
                kind,
                train_dataset,
                validation_dataset,
                normalizer,
                model_config,
                training_config,
                seed=seed,
                epoch_callback=save_periodic_checkpoint,
                log_every=log_every,
                show_progress=args.progress,
                progress_description=f"{kind} seed={seed}",
            )
            train_seconds = time.perf_counter() - start_train
            if is_v2 and len(result.history) != training_config.epochs:
                raise RuntimeError(
                    "protocol-v2 training did not complete every configured epoch"
                )
            history_best = min(
                result.history, key=lambda row: row["validation_relative_l2"]
            )
            if int(history_best["epoch"]) != result.best_epoch:
                raise RuntimeError("best checkpoint is inconsistent with history")
            common_checkpoint_metadata = {
                "protocol_version": data_config.protocol_version,
                "kind": kind,
                "seed": seed,
                "rho_policy": (
                    "fixed_zero_non_trainable"
                    if kind == "shared_oracle_rho0"
                    else model_config.rho_mode
                    if kind in {"shared", *ORACLE_SHARED_MODELS}
                    else "not_applicable"
                ),
                "normalizer": {
                    "mean": normalizer.mean,
                    "std": normalizer.std,
                },
            }
            save_checkpoint(
                run_dir / "checkpoint_best.npz",
                result.params,
                {
                    **common_checkpoint_metadata,
                    "checkpoint_type": "best_validation_relative_l2",
                    "best_epoch": result.best_epoch,
                    "metrics": history_best,
                },
            )
            save_checkpoint(
                run_dir / "checkpoint_final.npz",
                result.final_params,
                {
                    **common_checkpoint_metadata,
                    "checkpoint_type": "final_epoch",
                    "epoch": result.completed_epochs,
                    "metrics": result.history[-1],
                },
            )
            # Backward-compatible alias: formal inference always uses best params.
            save_checkpoint(
                run_dir / "checkpoint.npz",
                result.params,
                {
                    **common_checkpoint_metadata,
                    "checkpoint_type": "best_validation_relative_l2_alias",
                    "best_epoch": result.best_epoch,
                    "metrics": history_best,
                },
            )
            _write_csv(run_dir / "history.csv", list(result.history))
            resolution_metrics: dict[str, dict[str, float]] = {}

            for resolution in resolutions:
                test_dataset = sample_cases(test_cases, resolution)
                start_prediction = time.perf_counter()
                prediction, gamma_prediction = predict_dataset(
                    kind,
                    result.params,
                    test_dataset,
                    normalizer,
                    model_config,
                    batch_size=training_config.validation_batch_size,
                )
                prediction_seconds = time.perf_counter() - start_prediction
                metrics, per_sample = prediction_metrics(
                    prediction,
                    test_dataset,
                    gamma_prediction=gamma_prediction,
                    spectral_cutoff=model_config.n_modes,
                    include_continuous_metrics=not is_v2,
                )
                metrics["prediction_seconds"] = prediction_seconds
                metrics.update(
                    _gamma_integrity(
                        kind,
                        result.params,
                        test_dataset,
                        normalizer,
                        model_config,
                    )
                )
                resolution_metrics[str(resolution)] = metrics
                gamma_input = (
                    test_dataset.gamma_target
                    if kind in GAMMA_TARGET_MODELS
                    else np.empty((0,), dtype=np.float32)
                )
                np.savez_compressed(
                    run_dir / f"predictions_n{resolution}.npz",
                    prediction=prediction.astype(np.float32),
                    target=test_dataset.u_target,
                    gamma_prediction=(
                        np.asarray(gamma_prediction, dtype=np.float32)
                        if gamma_prediction is not None
                        else np.empty((0,), dtype=np.float32)
                    ),
                    gamma_target=test_dataset.gamma_target,
                    gamma0=test_dataset.gamma0,
                    gamma_input=np.asarray(gamma_input, dtype=np.float32),
                    sample_id=test_cases.sample_id,
                    jump=test_cases.jump,
                    interface_location=test_cases.interface_location,
                    plateau=(
                        np.all(test_cases.sin_coefficients == 0.0, axis=1)
                        & np.all(test_cases.cos_coefficients == 0.0, axis=1)
                    ),
                    **{key: np.asarray(value) for key, value in per_sample.items()},
                )
                row: dict[str, object] = {
                    "seed": seed,
                    "model": kind,
                    "resolution": resolution,
                    "parameter_count": parameter_count(result.params),
                    "best_epoch": result.best_epoch,
                    "train_seconds": train_seconds,
                }
                row.update(metrics)
                summary_rows.append(row)
                if resolution == n_x:
                    paired_relative_l2[kind].append(per_sample["relative_l2"])

            _write_json(run_dir / "metrics.json", resolution_metrics)
            if kind in {"shared", *ORACLE_SHARED_MODELS}:
                rho = rho_values(result.params, model_config)
                _write_json(
                    run_dir / "rho_summary.json",
                    _rho_summary(rho),
                )
                _save_rho_layers(run_dir / "rho_layers.npz", rho)

    _write_csv(output_dir / "summary.csv", summary_rows)
    if not is_v2 and {"fno", "shared"}.issubset(paired_relative_l2):
        fno = np.stack(paired_relative_l2["fno"], axis=0)
        shared = np.stack(paired_relative_l2["shared"], axis=0)
        if fno.shape == shared.shape:
            interval = hierarchical_paired_bootstrap(
                shared - fno,
                repetitions=args.bootstrap_repetitions,
                seed=12345,
            )
            _write_json(
                output_dir / "paired_bootstrap.json",
                {
                    "metric": "shared_minus_fno_relative_l2",
                    "mean": interval.mean,
                    "lower_95": interval.lower,
                    "upper_95": interval.upper,
                    "repetitions": args.bootstrap_repetitions,
                },
            )
    if is_v2 and set(V2_MODELS).issubset(paired_relative_l2):
        stacked = {
            kind: np.stack(paired_relative_l2[kind], axis=0)
            for kind in V2_MODELS
        }
        contrasts = {
            "shared_oracle_minus_fno_gamma": (
                stacked["shared_oracle"] - stacked["fno_gamma"]
            ),
            "fno_gamma_minus_fno": stacked["fno_gamma"] - stacked["fno"],
            "shared_oracle_minus_fno": (
                stacked["shared_oracle"] - stacked["fno"]
            ),
        }
        bootstrap_payload: dict[str, object] = {
            "metric": "paired_per_sample_relative_l2_difference",
            "repetitions": args.bootstrap_repetitions,
            "seed": 12345,
            "contrasts": {},
        }
        for name, differences in contrasts.items():
            interval = hierarchical_paired_bootstrap(
                differences,
                repetitions=args.bootstrap_repetitions,
                seed=12345,
            )
            bootstrap_payload["contrasts"][name] = {
                "mean": interval.mean,
                "lower_95": interval.lower,
                "upper_95": interval.upper,
            }
        _write_json(output_dir / "paired_bootstrap.json", bootstrap_payload)

    manifest = {
        "protocol_version": data_config.protocol_version,
        "run_tag": args.run_tag or ("main_v2" if is_v2 else "main_v1"),
        "preset": args.preset,
        "models": models,
        "seeds": seeds,
        "n_x": n_x,
        "resolutions": resolutions,
        "sizes": {
            "train": train_size,
            "validation": validation_size,
            "test": test_size,
        },
        "data_config": data_config.to_dict(),
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "runtime_controls": {
            "checkpoint_every": checkpoint_every,
            "log_every": log_every,
            "progress": args.progress,
        },
        "checkpoint_policy": {
            "periodic": "current parameters",
            "best": "minimum validation relative L2",
            "final": "last configured epoch",
            "formal_inference": "checkpoint_best.npz",
        },
        "rho_policy": {
            kind: (
                "fixed_zero_non_trainable"
                if kind == "shared_oracle_rho0"
                else model_config.rho_mode
                if kind in {"shared", *ORACLE_SHARED_MODELS}
                else "not_applicable"
            )
            for kind in models
        },
        "gamma_policy": (
            {
                "loss": "disabled",
                "oracle_sequence": [
                    "gamma0",
                    *["gammaT" for _ in range(model_config.n_layers)],
                ],
            }
            if is_v2
            else {"loss": "recursive Shared only"}
        ),
        "environment": _environment_manifest(),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(sorted(set(V1_MODELS + V2_ALLOWED_MODELS))),
        default=None,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", help="comma-separated training seeds")
    parser.add_argument("--n-x", type=int)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--gamma-loss-weight", type=float)
    parser.add_argument("--run-tag")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="save current model parameters every N epochs; v2 defaults to 10",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="print aggregated training and validation metrics every N epochs; v2 defaults to 10",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="show an epoch-level tqdm progress bar",
    )
    parser.add_argument(
        "--rho-mode",
        choices=("matrix", "layer_scalar", "fixed_zero", "fixed_one"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_experiment(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
