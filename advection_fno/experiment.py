"""Command-line runner for the complete feature-conditioned advection study."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PRESETS, ExperimentConfig
from .data import AdvectionDataset, apply_gamma_ablation, generate_dataset, save_dataset
from .metrics import build_comparisons, evaluate_predictions, summarize_rows, write_csv
from .models import FitResult, fit_ridge, fit_shared_learnable_rho, predict, save_fit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--stage", choices=("A", "B", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("results/advection_feature_fno"))
    parser.add_argument("--ridge", type=float, default=1.0e-8)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--x64", action="store_true", help="Enable JAX float64 before computation")
    parser.add_argument(
        "--skip-ablations", action="store_true", help="Skip perturbed-gamma diagnostic suites"
    )
    return parser.parse_args(argv)


def _derived_seed(seed: int, stage: str, split: int) -> int:
    return seed * 100_000 + (0 if stage == "A" else 50_000) + split


def _fit_and_evaluate(
    *,
    model_label: str,
    fit_model: str,
    train: AdvectionDataset,
    suites: dict[str, AdvectionDataset],
    modes: int,
    rho: float,
    ridge: float,
    batch_size: int,
    stage: str,
    seed: int,
    output_dir: Path,
    rows: list[dict[str, object]],
    budget_modes: int | None = None,
) -> FitResult:
    result = fit_ridge(
        fit_model,
        train,
        modes,
        rho=rho,
        ridge=ridge,
        batch_size=batch_size,
    )
    reported_modes = modes if budget_modes is None else budget_modes
    save_fit(
        output_dir
        / "models"
        / f"{stage}_seed{seed}_{model_label}_m{reported_modes}_retained{modes}.npz",
        result,
    )
    for suite_name, dataset in suites.items():
        estimate = predict(
            fit_model,
            result.parameters,
            dataset.u0,
            dataset.gamma0,
            dataset.gamma1,
            rho=result.rho,
            pad_fraction=dataset.config.pad_fraction,
        )
        rows.extend(
            evaluate_predictions(
                estimate,
                dataset,
                context={
                    "stage": stage,
                    "seed": seed,
                    "suite": suite_name,
                    "model": model_label,
                    "fit_model": fit_model,
                    "modes": reported_modes,
                    "retained_modes": modes,
                    "rho": result.rho,
                    "ridge": result.ridge,
                    "parameter_ridge": result.parameter_ridge,
                    "parameter_count": result.parameter_count,
                    "condition_number": result.condition_number,
                    "train_mse": result.train_mse,
                },
            )
        )
    return result


def _fit_learnable_rho_and_evaluate(
    *,
    train: AdvectionDataset,
    suites: dict[str, AdvectionDataset],
    modes: int,
    rho_initial: float,
    ridge: float,
    batch_size: int,
    steps: int,
    learning_rate: float,
    stage: str,
    seed: int,
    output_dir: Path,
    rows: list[dict[str, object]],
) -> FitResult:
    result = fit_shared_learnable_rho(
        train,
        modes,
        rho_initial=rho_initial,
        ridge=ridge,
        batch_size=batch_size,
        steps=steps,
        learning_rate=learning_rate,
    )
    save_fit(
        output_dir
        / "models"
        / f"{stage}_seed{seed}_shared_learnable_rho_m{modes}_retained{modes}.npz",
        result,
    )
    for suite_name, dataset in suites.items():
        estimate = predict(
            result.model,
            result.parameters,
            dataset.u0,
            dataset.gamma0,
            dataset.gamma1,
            rho=result.rho,
            pad_fraction=dataset.config.pad_fraction,
        )
        rows.extend(
            evaluate_predictions(
                estimate,
                dataset,
                context={
                    "stage": stage,
                    "seed": seed,
                    "suite": suite_name,
                    "model": "shared_learnable_rho",
                    "fit_model": result.model,
                    "modes": modes,
                    "retained_modes": modes,
                    "rho": result.rho,
                    "rho_initial": result.rho_initial,
                    "optimization_steps": result.optimization_steps,
                    "ridge": result.ridge,
                    "parameter_ridge": result.parameter_ridge,
                    "parameter_count": result.parameter_count,
                    "condition_number": result.condition_number,
                    "train_mse": result.train_mse,
                },
            )
        )
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.x64:
        jax.config.update("jax_enable_x64", True)
    preset = PRESETS[args.preset]
    config = ExperimentConfig(n_x=int(preset["n_x"]))
    stages = ("A", "B") if args.stage == "all" else (args.stage,)
    modes_values = [int(value) for value in preset["modes"]]
    scan_mode = int(preset["rho_scan_mode"])
    seeds = [int(value) for value in preset["seeds"]]
    batch_size = int(preset["fit_batch_size"])
    diagnostic_size = int(preset["diagnostic_size"])
    learnable_rho_initial = float(preset["learnable_rho_initial"])
    learnable_rho_steps = int(preset["learnable_rho_steps"])
    learnable_rho_learning_rate = float(preset["learnable_rho_learning_rate"])
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    metric_rows: list[dict[str, object]] = []

    for stage in stages:
        sizes = preset["sizes"][stage]
        for seed in seeds:
            train = generate_dataset(
                int(sizes["train"]), stage, _derived_seed(seed, stage, 1), config
            )
            validation = generate_dataset(
                int(sizes["validation"]), stage, _derived_seed(seed, stage, 2), config
            )
            test = generate_dataset(
                int(sizes["test"]), stage, _derived_seed(seed, stage, 3), config
            )
            endpoint = generate_dataset(
                diagnostic_size,
                stage,
                _derived_seed(seed, stage, 4),
                config,
                s_mode="endpoints",
            )
            offgrid = generate_dataset(
                diagnostic_size,
                stage,
                _derived_seed(seed, stage, 5),
                config,
                s_mode="offgrid",
            )
            false_interface = generate_dataset(
                diagnostic_size,
                stage,
                _derived_seed(seed, stage, 6),
                config,
                s_mode="interior_grid",
                force_zero_jump=True,
            )
            if args.save_data:
                data_dir = output_dir / "datasets"
                save_dataset(data_dir / f"{stage}_seed{seed}_train.npz", train)
                save_dataset(data_dir / f"{stage}_seed{seed}_validation.npz", validation)
                save_dataset(data_dir / f"{stage}_seed{seed}_test.npz", test)

            fitted_at_scan: dict[str, FitResult] = {}
            predictions_at_scan: dict[str, np.ndarray] = {}
            for modes in modes_values:
                suites = {"iid": test}
                if modes == scan_mode:
                    suites.update(
                        {
                            "endpoint_continuous": endpoint,
                            "offgrid": offgrid,
                            "false_interface": false_interface,
                        }
                    )
                plain = _fit_and_evaluate(
                    model_label="plain",
                    fit_model="plain",
                    train=train,
                    suites=suites,
                    modes=modes,
                    rho=0.0,
                    ridge=args.ridge,
                    batch_size=batch_size,
                    stage=stage,
                    seed=seed,
                    output_dir=output_dir,
                    rows=metric_rows,
                )
                hidden_width_2 = _fit_and_evaluate(
                    model_label="hidden_width_2",
                    fit_model="hidden_width_2",
                    train=train,
                    suites=suites,
                    modes=modes,
                    rho=0.0,
                    ridge=args.ridge,
                    batch_size=batch_size,
                    stage=stage,
                    seed=seed,
                    output_dir=output_dir,
                    rows=metric_rows,
                )
                two_channel_fno = _fit_and_evaluate(
                    model_label="two_channel_fno",
                    fit_model="two_channel_fno",
                    train=train,
                    suites=suites,
                    modes=modes,
                    rho=0.0,
                    ridge=args.ridge,
                    batch_size=batch_size,
                    stage=stage,
                    seed=seed,
                    output_dir=output_dir,
                    rows=metric_rows,
                )
                shared = _fit_and_evaluate(
                    model_label="shared",
                    fit_model="shared",
                    train=train,
                    suites=suites,
                    modes=modes,
                    rho=0.0,
                    ridge=args.ridge,
                    batch_size=batch_size,
                    stage=stage,
                    seed=seed,
                    output_dir=output_dir,
                    rows=metric_rows,
                )
                shared_learnable_rho = _fit_learnable_rho_and_evaluate(
                    train=train,
                    suites=suites,
                    modes=modes,
                    rho_initial=learnable_rho_initial,
                    ridge=args.ridge,
                    batch_size=batch_size,
                    steps=learnable_rho_steps,
                    learning_rate=learnable_rho_learning_rate,
                    stage=stage,
                    seed=seed,
                    output_dir=output_dir,
                    rows=metric_rows,
                )
                dual = _fit_and_evaluate(
                    model_label="dual",
                    fit_model="dual",
                    train=train,
                    suites=suites,
                    modes=modes,
                    rho=0.0,
                    ridge=args.ridge,
                    batch_size=batch_size,
                    stage=stage,
                    seed=seed,
                    output_dir=output_dir,
                    rows=metric_rows,
                )
                available_modes = (config.n_x + 2 * config.pad_points) // 2 + 1
                wide_modes = min(2 * modes, available_modes)
                wide = _fit_and_evaluate(
                    model_label="wide_plain",
                    fit_model="plain",
                    train=train,
                    suites=suites,
                    modes=wide_modes,
                    rho=0.0,
                    ridge=args.ridge,
                    batch_size=batch_size,
                    stage=stage,
                    seed=seed,
                    output_dir=output_dir,
                    rows=metric_rows,
                    budget_modes=modes,
                )
                if modes == scan_mode:
                    fitted_at_scan = {
                        "plain": plain,
                        "hidden_width_2": hidden_width_2,
                        "two_channel_fno": two_channel_fno,
                        "shared": shared,
                        "shared_learnable_rho": shared_learnable_rho,
                        "dual": dual,
                        "wide_plain": wide,
                    }
                    for label, result in fitted_at_scan.items():
                        predictions_at_scan[label] = predict(
                            result.model,
                            result.parameters,
                            test.u0,
                            test.gamma0,
                            test.gamma1,
                            rho=result.rho,
                            pad_fraction=config.pad_fraction,
                        )

            if scan_mode in modes_values:
                metric_rows.extend(
                    evaluate_predictions(
                        test.u1,
                        test,
                        context={
                            "stage": stage,
                            "seed": seed,
                            "suite": "iid",
                            "model": "exact_shift",
                            "fit_model": "analytic",
                            "modes": 0,
                            "retained_modes": 0,
                            "rho": 0.0,
                            "parameter_count": 0,
                            "condition_number": 1.0,
                            "train_mse": 0.0,
                        },
                    )
                )
                for rho in (0.25, 0.5, 0.75, 1.0):
                    _fit_and_evaluate(
                        model_label=f"shared_rho_{rho:g}",
                        fit_model="shared",
                        train=train,
                        suites={"iid": test},
                        modes=scan_mode,
                        rho=rho,
                        ridge=args.ridge,
                        batch_size=batch_size,
                        stage=stage,
                        seed=seed,
                        output_dir=output_dir,
                        rows=metric_rows,
                    )
                if not args.skip_ablations:
                    ablations: list[tuple[str, AdvectionDataset]] = [
                        ("unshifted", apply_gamma_ablation(test, "unshifted")),
                        ("constant", apply_gamma_ablation(test, "constant")),
                    ]
                    ablations.extend(
                        (f"shift_{cells:+d}", apply_gamma_ablation(test, "shift_output", cells=cells))
                        for cells in (-4, -2, -1, 1, 2, 4)
                    )
                    ablations.extend(
                        (f"smooth_{cells}", apply_gamma_ablation(test, "smooth", cells=cells))
                        for cells in (1, 2, 4)
                    )
                    for ablation_name, ablated in ablations:
                        for label in (
                            "two_channel_fno",
                            "shared",
                            "shared_learnable_rho",
                            "dual",
                        ):
                            result = fitted_at_scan[label]
                            estimate = predict(
                                result.model,
                                result.parameters,
                                ablated.u0,
                                ablated.gamma0,
                                ablated.gamma1,
                                rho=result.rho,
                                pad_fraction=config.pad_fraction,
                            )
                            metric_rows.extend(
                                evaluate_predictions(
                                    estimate,
                                    ablated,
                                    context={
                                        "stage": stage,
                                        "seed": seed,
                                        "suite": f"ablation_{ablation_name}",
                                        "model": label,
                                        "fit_model": result.model,
                                        "modes": scan_mode,
                                        "rho": result.rho,
                                        "parameter_count": result.parameter_count,
                                        "condition_number": result.condition_number,
                                        "train_mse": result.train_mse,
                                    },
                                )
                            )
                if seed == seeds[0]:
                    _plot_predictions(
                        test,
                        predictions_at_scan,
                        output_dir / "figures" / f"stage_{stage}_predictions.png",
                    )

    summaries = summarize_rows(metric_rows)
    comparisons = build_comparisons(
        metric_rows, repetitions=int(preset["bootstrap_reps"])
    )
    write_csv(output_dir / "metrics.csv", metric_rows)
    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "comparisons.csv", comparisons)
    _plot_metric_curves(summaries, output_dir / "figures" / "relative_l2_vs_modes.png")
    duration = time.perf_counter() - start_time
    manifest = {
        "schema_version": 1,
        "preset": args.preset,
        "stages": list(stages),
        "config": config.to_dict(),
        "modes": modes_values,
        "models": [
            "plain",
            "wide_plain",
            "hidden_width_2",
            "two_channel_fno",
            "shared",
            "shared_learnable_rho",
            "dual",
        ],
        "rho_scan": [0.0, 0.25, 0.5, 0.75, 1.0],
        "learnable_rho": {
            "initial": learnable_rho_initial,
            "steps": learnable_rho_steps,
            "learning_rate": learnable_rho_learning_rate,
            "parameterization": "rho = sigmoid(eta)",
        },
        "seeds": seeds,
        "ridge": args.ridge,
        "x64": bool(jax.config.read("jax_enable_x64")),
        "devices": [str(device) for device in jax.devices()],
        "python": sys.version,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "duration_seconds": duration,
        "output_files": ["metrics.csv", "summary.csv", "comparisons.csv"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _plot_predictions(
    dataset: AdvectionDataset,
    predictions: dict[str, np.ndarray],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates = np.flatnonzero(dataset.metadata["has_output_jump"])
    index = int(candidates[0]) if candidates.size else 0
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(dataset.x, dataset.u1[index], "k-", linewidth=2.0, label="Exact")
    for label, values in predictions.items():
        axes[0].plot(dataset.x, values[index], linewidth=1.2, label=label)
        axes[1].plot(dataset.x, values[index] - dataset.u1[index], label=label)
    axes[0].axvline(dataset.metadata["output_jump_location"][index], color="0.5", linestyle="--")
    axes[0].set_ylabel("u(x, 1)")
    axes[0].legend(ncol=2)
    axes[1].axhline(0.0, color="k", linewidth=0.8)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("prediction - exact")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_metric_curves(summaries: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for axis, stage in zip(axes, ("A", "B"), strict=True):
        for model in (
            "plain",
            "hidden_width_2",
            "two_channel_fno",
            "shared",
            "shared_learnable_rho",
            "dual",
            "wide_plain",
        ):
            points = [
                row
                for row in summaries
                if row["stage"] == stage and row["suite"] == "iid" and row["model"] == model
            ]
            grouped: dict[int, list[float]] = {}
            for row in points:
                grouped.setdefault(int(row["modes"]), []).append(float(row["relative_l2_mean"]))
            if not grouped:
                continue
            x = sorted(grouped)
            y = [float(np.mean(grouped[value])) for value in x]
            axis.plot(x, y, marker="o", label=model)
        axis.set_title(f"Stage {stage}")
        axis.set_xlabel("retained modes")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("mean relative L2")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
