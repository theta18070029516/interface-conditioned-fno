"""Interface-aware metrics and paired bootstrap comparisons."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .data import AdvectionDataset

METRIC_NAMES = (
    "relative_l2",
    "mae",
    "linf",
    "interface_mae",
    "far_field_mae",
    "jump_amplitude_error",
    "overshoot",
    "spurious_jump_error",
    "total_variation_error",
)


def evaluate_predictions(
    prediction: np.ndarray,
    dataset: AdvectionDataset,
    *,
    context: dict[str, object] | None = None,
    interface_cells: int = 4,
    overshoot_cells: int = 8,
    trace_cells: int = 2,
) -> list[dict[str, object]]:
    """Compute one row of global and interface metrics per sample."""

    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != dataset.u1.shape:
        raise ValueError(f"Prediction shape {prediction.shape} != target shape {dataset.u1.shape}")
    context = dict(context or {})
    rows: list[dict[str, object]] = []
    dx = dataset.config.dx
    for index in range(len(dataset)):
        target = dataset.u1[index]
        estimate = prediction[index]
        error = estimate - target
        target_rms = float(np.sqrt(np.mean(target**2)))
        relative_l2 = float(np.sqrt(np.mean(error**2)) / max(target_rms, 1.0e-10))
        has_jump = bool(dataset.metadata["has_output_jump"][index])
        location = float(dataset.metadata["output_jump_location"][index])
        interface_mae = np.nan
        far_field_mae = float(np.mean(np.abs(error)))
        jump_error = np.nan
        overshoot = np.nan
        if has_jump:
            interface_mask = np.abs(dataset.x - location) <= interface_cells * dx
            far_mask = ~interface_mask
            if np.any(interface_mask):
                interface_mae = float(np.mean(np.abs(error[interface_mask])))
            if np.any(far_mask):
                far_field_mae = float(np.mean(np.abs(error[far_mask])))
            target_jump = _trace_jump(target, dataset.x, location, trace_cells)
            estimate_jump = _trace_jump(estimate, dataset.x, location, trace_cells)
            if np.isfinite(target_jump) and np.isfinite(estimate_jump):
                jump_error = float(abs(estimate_jump - target_jump))
            overshoot_mask = np.abs(dataset.x - location) <= overshoot_cells * dx
            if np.any(overshoot_mask):
                local_target = target[overshoot_mask]
                local_estimate = estimate[overshoot_mask]
                upper = float(np.max(local_target))
                lower = float(np.min(local_target))
                overshoot = float(
                    max(
                        np.max(local_estimate - upper),
                        np.max(lower - local_estimate),
                        0.0,
                    )
                )
        difference_error = np.diff(estimate) - np.diff(target)
        signal_scale = max(float(np.ptp(target)), target_rms, 1.0e-10)
        spurious_jump = float(np.max(np.abs(difference_error)) / signal_scale)
        tv_error = float(abs(np.sum(np.abs(np.diff(estimate))) - np.sum(np.abs(np.diff(target)))))
        row: dict[str, object] = {
            **context,
            "sample_id": int(dataset.metadata["sample_id"][index]),
            "s": float(dataset.metadata["s"][index]),
            "J": float(dataset.metadata["J"][index]),
            "is_endpoint": bool(dataset.metadata["is_endpoint"][index]),
            "has_output_jump": has_jump,
            "output_jump_location": location,
            "relative_l2": relative_l2,
            "mae": float(np.mean(np.abs(error))),
            "linf": float(np.max(np.abs(error))),
            "interface_mae": interface_mae,
            "far_field_mae": far_field_mae,
            "jump_amplitude_error": jump_error,
            "overshoot": overshoot,
            "spurious_jump_error": spurious_jump,
            "total_variation_error": tv_error,
        }
        rows.append(row)
    return rows


def _trace_jump(values: np.ndarray, x: np.ndarray, location: float, cells: int) -> float:
    left = np.flatnonzero(x < location)
    right = np.flatnonzero(x > location)
    if left.size < cells or right.size < cells:
        return float("nan")
    return float(np.mean(values[right[:cells]]) - np.mean(values[left[-cells:]]))


def summarize_rows(
    rows: Iterable[dict[str, object]],
    *,
    group_keys: tuple[str, ...] = ("stage", "seed", "suite", "model", "modes", "rho"),
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in group_keys)].append(row)
    summaries: list[dict[str, object]] = []
    for key, group in groups.items():
        summary = {name: value for name, value in zip(group_keys, key, strict=True)}
        summary["n_samples"] = len(group)
        for metric in METRIC_NAMES:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else np.nan
            summary[f"{metric}_median"] = float(np.median(finite)) if finite.size else np.nan
        summaries.append(summary)
    return summaries


def paired_bootstrap_reduction(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap reduction of candidate mean error relative to baseline mean error."""

    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    finite = np.isfinite(baseline) & np.isfinite(candidate)
    baseline = baseline[finite]
    candidate = candidate[finite]
    if baseline.size == 0 or float(np.mean(baseline)) <= 1.0e-15:
        return {"reduction": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    point = 1.0 - float(np.mean(candidate)) / float(np.mean(baseline))
    rng = np.random.default_rng(seed)
    boot = np.empty(repetitions, dtype=np.float64)
    written = 0
    chunk = 256
    while written < repetitions:
        count = min(chunk, repetitions - written)
        indices = rng.integers(0, baseline.size, size=(count, baseline.size))
        base_means = np.mean(baseline[indices], axis=1)
        candidate_means = np.mean(candidate[indices], axis=1)
        boot[written : written + count] = 1.0 - candidate_means / np.maximum(
            base_means, 1.0e-15
        )
        written += count
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "reduction": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(baseline.size),
    }


def build_comparisons(
    rows: list[dict[str, object]],
    *,
    repetitions: int,
) -> list[dict[str, object]]:
    """Create paired comparisons against plain and parameter-matched baselines."""

    comparisons: list[dict[str, object]] = []
    stages = sorted({str(row["stage"]) for row in rows})
    modes_values = sorted({int(row["modes"]) for row in rows if int(row["modes"]) > 0})
    for stage in stages:
        for modes in modes_values:
            baseline_rows = [
                row
                for row in rows
                if row["stage"] == stage
                and row["suite"] == "iid"
                and row["model"] == "plain"
                and int(row["modes"]) == modes
            ]
            baseline_map = {
                (int(row["seed"]), int(row["sample_id"])): row for row in baseline_rows
            }
            if not baseline_map:
                continue
            for model, reference_model in (
                ("hidden_width_2", "plain"),
                ("two_channel_fno", "plain"),
                ("shared", "plain"),
                ("shared_learnable_rho", "plain"),
                ("dual", "plain"),
                ("dual", "hidden_width_2"),
                ("dual", "two_channel_fno"),
                ("dual", "wide_plain"),
            ):
                reference_rows = [
                    row
                    for row in rows
                    if row["stage"] == stage
                    and row["suite"] == "iid"
                    and row["model"] == reference_model
                    and int(row["modes"]) == modes
                ]
                reference_map = {
                    (int(row["seed"]), int(row["sample_id"])): row
                    for row in reference_rows
                }
                candidate_map = {
                    (int(row["seed"]), int(row["sample_id"])): row
                    for row in rows
                    if row["stage"] == stage
                    and row["suite"] == "iid"
                    and row["model"] == model
                    and int(row["modes"]) == modes
                    and (model != "shared" or float(row["rho"]) == 0.0)
                }
                keys = sorted(set(reference_map) & set(candidate_map))
                for metric in ("interface_mae", "overshoot", "relative_l2"):
                    baseline = np.asarray([reference_map[key][metric] for key in keys], dtype=float)
                    candidate = np.asarray([candidate_map[key][metric] for key in keys], dtype=float)
                    result = paired_bootstrap_reduction(
                        baseline,
                        candidate,
                        repetitions=repetitions,
                        seed=(
                            17_000
                            + modes
                            + {
                                "hidden_width_2": 125,
                                "two_channel_fno": 250,
                                "shared": 0,
                                "shared_learnable_rho": 500,
                                "dual": 1_000,
                            }[model]
                            + {
                                "plain": 0,
                                "hidden_width_2": 1_000,
                                "two_channel_fno": 2_000,
                                "wide_plain": 4_000,
                            }[reference_model]
                        ),
                    )
                    comparisons.append(
                        {
                            "stage": stage,
                            "modes": modes,
                            "model": model,
                            "reference_model": reference_model,
                            "metric": metric,
                            **result,
                            "passes_20_percent": bool(
                                np.isfinite(result["reduction"])
                                and result["reduction"] >= 0.2
                                and result["ci_low"] > 0.0
                            ),
                        }
                    )
            for model in (
                "hidden_width_2",
                "two_channel_fno",
                "shared",
                "shared_learnable_rho",
                "dual",
            ):
                endpoint_reference = {
                    (int(row["seed"]), int(row["sample_id"])): row
                    for row in rows
                    if row["stage"] == stage
                    and row["suite"] == "endpoint_continuous"
                    and row["model"] == "plain"
                    and int(row["modes"]) == modes
                }
                endpoint_candidate = {
                    (int(row["seed"]), int(row["sample_id"])): row
                    for row in rows
                    if row["stage"] == stage
                    and row["suite"] == "endpoint_continuous"
                    and row["model"] == model
                    and int(row["modes"]) == modes
                }
                endpoint_keys = sorted(set(endpoint_reference) & set(endpoint_candidate))
                if endpoint_keys:
                    baseline_values = np.asarray(
                        [endpoint_reference[key]["relative_l2"] for key in endpoint_keys], dtype=float
                    )
                    candidate_values = np.asarray(
                        [endpoint_candidate[key]["relative_l2"] for key in endpoint_keys], dtype=float
                    )
                    reduction = paired_bootstrap_reduction(
                        baseline_values,
                        candidate_values,
                        repetitions=repetitions,
                        seed=31_000
                        + modes
                        + {
                            "hidden_width_2": 125,
                            "two_channel_fno": 250,
                            "shared": 0,
                            "shared_learnable_rho": 500,
                            "dual": 1_000,
                        }[model],
                    )
                    ratio = 1.0 - reduction["reduction"]
                    ratio_low = 1.0 - reduction["ci_high"]
                    ratio_high = 1.0 - reduction["ci_low"]
                    spurious = np.asarray(
                        [endpoint_candidate[key]["spurious_jump_error"] for key in endpoint_keys],
                        dtype=float,
                    )
                    comparisons.append(
                        {
                            "stage": stage,
                            "modes": modes,
                            "model": model,
                            "reference_model": "plain",
                            "metric": "endpoint_relative_l2_ratio",
                            "ratio": float(ratio),
                            "ci_low": float(ratio_low),
                            "ci_high": float(ratio_high),
                            "n": int(len(endpoint_keys)),
                            "mean_spurious_jump_error": float(np.mean(spurious)),
                            "passes_endpoint_1_05": bool(
                                np.isfinite(ratio)
                                and ratio <= 1.05
                                and float(np.mean(spurious)) <= 0.01
                            ),
                        }
                    )
    return comparisons


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
