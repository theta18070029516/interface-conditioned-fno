"""Solution, interface, spectral, and uncertainty metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import INTERNAL, DeepAdvectionDataset


def per_sample_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean((np.asarray(prediction) - np.asarray(target)) ** 2, axis=-1)


def per_sample_relative_l2(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    residual = np.linalg.norm(np.asarray(prediction) - np.asarray(target), axis=-1)
    denominator = np.maximum(np.linalg.norm(np.asarray(target), axis=-1), epsilon)
    return residual / denominator


def total_variation(value: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.diff(np.asarray(value), axis=-1)), axis=-1)


def _zero_crossing(gamma: np.ndarray, x: np.ndarray) -> np.ndarray:
    gamma = np.asarray(gamma)
    result = np.full(gamma.shape[0], np.nan, dtype=np.float64)
    for sample in range(gamma.shape[0]):
        changes = np.flatnonzero(gamma[sample, :-1] * gamma[sample, 1:] <= 0)
        if changes.size == 0:
            continue
        index = int(changes[np.argmin(np.abs(gamma[sample, changes]))])
        left, right = gamma[sample, index], gamma[sample, index + 1]
        fraction = 0.5 if right == left else float(-left / (right - left))
        result[sample] = x[index] + fraction * (x[index + 1] - x[index])
    return result


def _safe_nanmean(value: np.ndarray) -> float:
    finite = np.asarray(value)[np.isfinite(value)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _interface_window_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    x: np.ndarray,
    locations: np.ndarray,
    mask: np.ndarray,
    width: float,
) -> np.ndarray:
    output = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    for sample in np.flatnonzero(mask):
        local = np.abs(x - locations[sample]) <= width
        if np.any(local):
            output[sample] = np.mean(
                np.abs(prediction[sample, local] - target[sample, local])
            )
    return output


def _jump_amplitude(value: np.ndarray, x: np.ndarray, locations: np.ndarray) -> np.ndarray:
    output = np.full(value.shape[0], np.nan, dtype=np.float64)
    dx = float(x[1] - x[0])
    inner = max(2.0 * dx, 0.02)
    outer = max(8.0 * dx, 0.08)
    for sample, location in enumerate(locations):
        left = (x >= location - outer) & (x <= location - inner)
        right = (x >= location + inner) & (x <= location + outer)
        if np.any(left) and np.any(right):
            output[sample] = np.mean(value[sample, right]) - np.mean(
                value[sample, left]
            )
    return output


def _high_frequency_errors(
    prediction: np.ndarray,
    target: np.ndarray,
    cutoff: int,
    *,
    relative_tolerance: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute and conditionally defined relative spectral-tail errors.

    ``norm="forward"`` makes the Fourier coefficients comparable across grids on
    the same physical domain.  Relative error is undefined when the target tail
    is negligible compared with the full target spectrum; those entries are
    represented by NaN and excluded from aggregate relative-error statistics.
    """

    if cutoff < 0:
        raise ValueError("cutoff must be non-negative")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")
    pred_spectrum = np.fft.rfft(prediction, axis=-1, norm="forward")
    target_spectrum = np.fft.rfft(target, axis=-1, norm="forward")
    if cutoff >= pred_spectrum.shape[-1]:
        absolute = np.zeros(prediction.shape[0], dtype=np.float64)
        relative = np.full(prediction.shape[0], np.nan, dtype=np.float64)
        return absolute, relative
    absolute = np.linalg.norm(
        pred_spectrum[:, cutoff:] - target_spectrum[:, cutoff:], axis=-1
    )
    target_tail = np.linalg.norm(target_spectrum[:, cutoff:], axis=-1)
    target_scale = np.linalg.norm(target_spectrum, axis=-1)
    threshold = relative_tolerance * np.maximum(
        target_scale, np.finfo(np.float64).tiny
    )
    valid = target_tail > threshold
    relative = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    np.divide(absolute, target_tail, out=relative, where=valid)
    return absolute, relative


def prediction_metrics(
    prediction: np.ndarray,
    dataset: DeepAdvectionDataset,
    *,
    gamma_prediction: np.ndarray | None = None,
    spectral_cutoff: int = 16,
    interface_width: float = 0.05,
    include_continuous_metrics: bool = True,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(dataset.u_target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes do not match")
    locations = dataset.cases.interface_location + dataset.cases.config.displacement
    internal = dataset.cases.category == INTERNAL
    continuous = ~internal
    mse = per_sample_mse(prediction, target)
    relative_l2 = per_sample_relative_l2(prediction, target)
    interface_mae = _interface_window_mae(
        prediction,
        target,
        dataset.x,
        locations,
        internal,
        interface_width,
    )
    predicted_jump = _jump_amplitude(prediction, dataset.x, locations)
    target_jump = _jump_amplitude(target, dataset.x, locations)
    jump_error = np.abs(predicted_jump - target_jump)
    tv_error = np.abs(total_variation(prediction) - total_variation(target))
    overshoot = np.maximum(
        np.max(prediction, axis=-1) - np.max(target, axis=-1), 0.0
    )
    undershoot = np.maximum(
        np.min(target, axis=-1) - np.min(prediction, axis=-1), 0.0
    )
    high_frequency_absolute, high_frequency_relative = _high_frequency_errors(
        prediction, target, spectral_cutoff
    )

    per_sample = {
        "mse": mse,
        "relative_l2": relative_l2,
        "interface_mae": interface_mae,
        "predicted_jump_amplitude": predicted_jump,
        "target_jump_amplitude": target_jump,
        "jump_amplitude_error": jump_error,
        "total_variation_error": tv_error,
        "overshoot": overshoot,
        "undershoot": undershoot,
        "high_frequency_absolute_error": high_frequency_absolute,
        "high_frequency_relative_error": high_frequency_relative,
    }
    summary = {
        "mse_mean": float(np.mean(mse)),
        "relative_l2_mean": float(np.mean(relative_l2)),
        "relative_l2_median": float(np.median(relative_l2)),
        "interface_mae_mean": _safe_nanmean(interface_mae[internal]),
        "jump_amplitude_error_mean": _safe_nanmean(jump_error[internal]),
        "total_variation_error_mean": float(np.mean(tv_error)),
        "overshoot_mean": float(np.mean(overshoot)),
        "undershoot_mean": float(np.mean(undershoot)),
        "high_frequency_absolute_error_mean": float(
            np.mean(high_frequency_absolute)
        ),
        "high_frequency_relative_error_mean": _safe_nanmean(
            high_frequency_relative
        ),
        "high_frequency_relative_error_valid_fraction": float(
            np.mean(np.isfinite(high_frequency_relative))
        ),
    }
    if include_continuous_metrics:
        false_jump = np.max(np.abs(np.diff(prediction, axis=-1)), axis=-1)
        per_sample["false_jump"] = false_jump
        summary["continuous_false_jump_mean"] = _safe_nanmean(
            false_jump[continuous]
        )

    if gamma_prediction is not None:
        gamma_prediction = np.asarray(gamma_prediction, dtype=np.float64)
        if gamma_prediction.shape != dataset.gamma_target.shape:
            raise ValueError("gamma prediction has an invalid shape")
        gamma_mse = per_sample_mse(gamma_prediction, dataset.gamma_target)
        predicted_location = _zero_crossing(gamma_prediction, dataset.x)
        target_location = _zero_crossing(dataset.gamma_target, dataset.x)
        location_error = np.abs(predicted_location - target_location)
        false_interface = np.isfinite(predicted_location) & continuous
        per_sample.update(
            {
                "gamma_mse": gamma_mse,
                "gamma_location_error": location_error,
                "gamma_false_interface": false_interface.astype(np.float64),
            }
        )
        summary.update(
            {
                "gamma_mse_mean": float(np.mean(gamma_mse)),
                "gamma_location_error_mean": _safe_nanmean(location_error[internal]),
                "gamma_false_interface_rate": float(np.mean(false_interface[continuous]))
                if np.any(continuous)
                else float("nan"),
            }
        )
    return summary, per_sample


@dataclass(frozen=True)
class BootstrapInterval:
    mean: float
    lower: float
    upper: float


def hierarchical_paired_bootstrap(
    differences: np.ndarray,
    *,
    repetitions: int = 10_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Resample training seeds, then shared test samples within each seed."""

    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 2 or min(differences.shape) < 1:
        raise ValueError("differences must have shape [seeds, samples]")
    rng = np.random.default_rng(seed)
    n_seeds, n_samples = differences.shape
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        seed_indices = rng.integers(0, n_seeds, size=n_seeds)
        sample_indices = rng.integers(0, n_samples, size=(n_seeds, n_samples))
        selected = differences[seed_indices]
        selected = np.take_along_axis(selected, sample_indices, axis=1)
        estimates[index] = np.mean(selected)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return BootstrapInterval(
        mean=float(np.mean(differences)),
        lower=float(lower),
        upper=float(upper),
    )
