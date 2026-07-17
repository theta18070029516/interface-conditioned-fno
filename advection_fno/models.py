"""Linear Fourier multipliers and feature-conditioned operator factorizations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import AdvectionDataset

ModelName = Literal[
    "plain",
    "hidden_width_2",
    "two_channel_fno",
    "shared",
    "shared_learnable_rho",
    "dual",
]


@dataclass(frozen=True)
class FitResult:
    model: ModelName
    modes: int
    rho: float
    ridge: float
    parameters: dict[str, np.ndarray]
    condition_number: float
    train_mse: float
    parameter_count: int
    rho_initial: float | None = None
    optimization_steps: int = 0
    parameter_ridge: float | None = None


def _as_batch(signal: jax.Array | np.ndarray) -> tuple[jax.Array, bool]:
    value = jnp.asarray(signal)
    if value.ndim == 1:
        return value[None, :], True
    if value.ndim != 2:
        raise ValueError("Signals must have shape [n_x] or [batch, n_x]")
    return value, False


def operator_basis(
    signal: jax.Array | np.ndarray,
    modes: int,
    *,
    pad_fraction: float = 0.5,
) -> jax.Array:
    """Basis response for every identifiable real/imaginary multiplier parameter."""

    signal, _ = _as_batch(signal)
    n_x = int(signal.shape[-1])
    pad = int(round(n_x * pad_fraction))
    padded = jnp.pad(signal, ((0, 0), (pad, pad)), mode="edge")
    spectrum = jnp.fft.rfft(padded, axis=-1)
    n_frequency = int(spectrum.shape[-1])
    if modes < 1 or modes > n_frequency:
        raise ValueError(f"modes must be in [1, {n_frequency}], got {modes}")
    selectors = jnp.eye(n_frequency, dtype=spectrum.dtype)[:modes]
    real_spectra = spectrum[:, None, :] * selectors[None, :, :]
    real_outputs = jnp.fft.irfft(real_spectra, n=padded.shape[-1], axis=-1)
    real_outputs = jnp.transpose(real_outputs[..., pad : pad + n_x], (0, 2, 1))
    if modes == 1:
        return real_outputs
    imag_spectra = spectrum[:, None, :] * (1j * selectors[None, 1:, :])
    imag_outputs = jnp.fft.irfft(imag_spectra, n=padded.shape[-1], axis=-1)
    imag_outputs = jnp.transpose(imag_outputs[..., pad : pad + n_x], (0, 2, 1))
    return jnp.concatenate([real_outputs, imag_outputs], axis=-1)


@partial(jax.jit, static_argnames=("model", "modes", "rho", "pad_fraction"))
def design_matrix(
    model: ModelName,
    u0: jax.Array | np.ndarray,
    gamma0: jax.Array | np.ndarray,
    gamma1: jax.Array | np.ndarray,
    modes: int,
    *,
    rho: float = 0.0,
    pad_fraction: float = 0.5,
) -> jax.Array:
    """Return the spatial design tensor [batch, n_x, parameters]."""

    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    u0, _ = _as_batch(u0)
    gamma0, _ = _as_batch(gamma0)
    gamma1, _ = _as_batch(gamma1)
    base = operator_basis(u0, modes, pad_fraction=pad_fraction)
    if model == "plain":
        return base
    if model == "hidden_width_2":
        return jnp.concatenate([base, base], axis=-1)
    if model == "two_channel_fno":
        interface_channel = operator_basis(
            gamma0, modes, pad_fraction=pad_fraction
        )
        return jnp.concatenate([base, interface_channel], axis=-1)
    feature = operator_basis(gamma0 * u0, modes, pad_fraction=pad_fraction)
    feature = gamma1[..., None] * feature
    if model in ("shared", "shared_learnable_rho"):
        a = 0.5 * (1.0 + rho)
        b = 0.5 * (1.0 - rho)
        return a * base + b * feature
    if model == "dual":
        return jnp.concatenate([base, feature], axis=-1)
    raise ValueError(f"Unknown model: {model}")


def _theta_to_weights(theta: np.ndarray, modes: int) -> np.ndarray:
    expected = 2 * modes - 1
    if theta.size != expected:
        raise ValueError(f"Expected {expected} parameters, got {theta.size}")
    real = theta[:modes]
    imag = np.concatenate([np.zeros(1, dtype=theta.dtype), theta[modes:]])
    return real + 1j * imag


def _parameters_from_theta(model: ModelName, theta: np.ndarray, modes: int) -> dict[str, np.ndarray]:
    per_kernel = 2 * modes - 1
    if model in ("plain", "shared", "shared_learnable_rho"):
        return {"weights": _theta_to_weights(theta, modes)}
    return {
        "weights0": _theta_to_weights(theta[:per_kernel], modes),
        "weights1": _theta_to_weights(theta[per_kernel:], modes),
    }


def fit_ridge(
    model: ModelName,
    dataset: AdvectionDataset,
    modes: int,
    *,
    rho: float = 0.0,
    ridge: float = 1.0e-8,
    batch_size: int = 64,
) -> FitResult:
    """Fit the globally optimal quadratic-loss solution for the linear model."""

    if model == "shared_learnable_rho":
        raise ValueError(
            "fit_ridge cannot optimize rho; use fit_shared_learnable_rho instead"
        )
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    per_kernel = 2 * modes - 1
    if model == "hidden_width_2":
        effective = fit_ridge(
            "plain",
            dataset,
            modes,
            rho=rho,
            ridge=ridge,
            batch_size=batch_size,
        )
        parameters = {
            "weights0": 0.5 * effective.parameters["weights"],
            "weights1": 0.5 * effective.parameters["weights"],
        }
        prediction = predict(
            model,
            parameters,
            dataset.u0,
            dataset.gamma0,
            dataset.gamma1,
            rho=rho,
            pad_fraction=dataset.config.pad_fraction,
        )
        return FitResult(
            model=model,
            modes=modes,
            rho=rho,
            ridge=ridge,
            parameters=parameters,
            condition_number=effective.condition_number,
            train_mse=float(np.mean((prediction - dataset.u1) ** 2)),
            parameter_count=2 * per_kernel,
            parameter_ridge=2.0 * ridge,
        )
    parameter_count = per_kernel if model in ("plain", "shared") else 2 * per_kernel
    gram = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    rhs = np.zeros(parameter_count, dtype=np.float64)
    observations = 0

    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        design = design_matrix(
            model,
            dataset.u0[start:stop],
            dataset.gamma0[start:stop],
            dataset.gamma1[start:stop],
            modes,
            rho=rho,
            pad_fraction=dataset.config.pad_fraction,
        )
        flat_design = design.reshape((-1, parameter_count))
        target = jnp.asarray(dataset.u1[start:stop]).reshape((-1,))
        batch_gram = flat_design.T @ flat_design
        batch_rhs = flat_design.T @ target
        gram += np.asarray(batch_gram, dtype=np.float64)
        rhs += np.asarray(batch_rhs, dtype=np.float64)
        observations += int(target.size)

    gram /= observations
    rhs /= observations
    regularized = gram + ridge * np.eye(parameter_count, dtype=np.float64)
    condition_number = float(np.linalg.cond(regularized))
    try:
        theta = np.linalg.solve(regularized, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(regularized, rhs, rcond=None)[0]
    parameters = _parameters_from_theta(model, theta, modes)
    prediction = predict(
        model,
        parameters,
        dataset.u0,
        dataset.gamma0,
        dataset.gamma1,
        rho=rho,
        pad_fraction=dataset.config.pad_fraction,
    )
    train_mse = float(np.mean((prediction - dataset.u1) ** 2))
    return FitResult(
        model=model,
        modes=modes,
        rho=rho,
        ridge=ridge,
        parameters=parameters,
        condition_number=condition_number,
        train_mse=train_mse,
        parameter_count=parameter_count,
        parameter_ridge=ridge,
    )


def fit_shared_learnable_rho(
    dataset: AdvectionDataset,
    modes: int,
    *,
    rho_initial: float = 0.5,
    ridge: float = 1.0e-8,
    batch_size: int = 64,
    steps: int = 2_000,
    learning_rate: float = 2.0e-2,
) -> FitResult:
    """Jointly fit one shared Fourier multiplier and a constrained scalar rho.

    The trainable unconstrained scalar ``eta`` is mapped through
    ``rho = sigmoid(eta)``.  Sufficient statistics are accumulated in batches,
    so optimization does not materialize a full ``[samples, x, modes]`` tensor.
    """

    if not 0.0 < rho_initial < 1.0:
        raise ValueError("rho_initial must lie strictly inside (0, 1)")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if steps <= 0 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")

    parameter_count = 2 * modes - 1
    gram_cc = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    gram_ce = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    gram_ee = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    rhs_c = np.zeros(parameter_count, dtype=np.float64)
    rhs_e = np.zeros(parameter_count, dtype=np.float64)
    target_square = 0.0
    observations = 0

    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        base = operator_basis(
            dataset.u0[start:stop],
            modes,
            pad_fraction=dataset.config.pad_fraction,
        )
        feature = operator_basis(
            dataset.gamma0[start:stop] * dataset.u0[start:stop],
            modes,
            pad_fraction=dataset.config.pad_fraction,
        )
        feature = jnp.asarray(dataset.gamma1[start:stop])[..., None] * feature
        constant = 0.5 * (base + feature)
        rho_coefficient = 0.5 * (base - feature)
        flat_constant = constant.reshape((-1, parameter_count))
        flat_coefficient = rho_coefficient.reshape((-1, parameter_count))
        target = jnp.asarray(dataset.u1[start:stop]).reshape((-1,))
        gram_cc += np.asarray(flat_constant.T @ flat_constant, dtype=np.float64)
        gram_ce += np.asarray(flat_constant.T @ flat_coefficient, dtype=np.float64)
        gram_ee += np.asarray(flat_coefficient.T @ flat_coefficient, dtype=np.float64)
        rhs_c += np.asarray(flat_constant.T @ target, dtype=np.float64)
        rhs_e += np.asarray(flat_coefficient.T @ target, dtype=np.float64)
        target_square += float(np.asarray(target @ target, dtype=np.float64))
        observations += int(target.size)

    gram_cc /= observations
    gram_ce /= observations
    gram_ee /= observations
    rhs_c /= observations
    rhs_e /= observations
    target_square /= observations
    identity = np.eye(parameter_count, dtype=np.float64)

    initial_gram = (
        gram_cc
        + rho_initial * (gram_ce + gram_ce.T)
        + rho_initial**2 * gram_ee
        + ridge * identity
    )
    initial_rhs = rhs_c + rho_initial * rhs_e
    try:
        theta_initial = np.linalg.solve(initial_gram, initial_rhs)
    except np.linalg.LinAlgError:
        theta_initial = np.linalg.lstsq(initial_gram, initial_rhs, rcond=None)[0]

    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    statistics = tuple(
        jnp.asarray(value, dtype=dtype)
        for value in (gram_cc, gram_ce, gram_ee, rhs_c, rhs_e, target_square)
    )
    parameters = {
        "theta": jnp.asarray(theta_initial, dtype=dtype),
        "eta": jnp.asarray(np.log(rho_initial / (1.0 - rho_initial)), dtype=dtype),
    }
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(parameters)

    def objective(current: dict[str, jax.Array]) -> jax.Array:
        current_theta = current["theta"]
        current_rho = jax.nn.sigmoid(current["eta"])
        current_cc, current_ce, current_ee, current_c, current_e, current_y2 = statistics
        current_gram = (
            current_cc
            + current_rho * (current_ce + current_ce.T)
            + current_rho**2 * current_ee
        )
        current_rhs = current_c + current_rho * current_e
        return (
            current_theta @ current_gram @ current_theta
            - 2.0 * current_theta @ current_rhs
            + current_y2
            + ridge * jnp.sum(current_theta**2)
        )

    @jax.jit
    def optimize(
        initial_parameters: dict[str, jax.Array], initial_state: optax.OptState
    ) -> tuple[dict[str, jax.Array], optax.OptState]:
        def body(
            _: int,
            carry: tuple[dict[str, jax.Array], optax.OptState],
        ) -> tuple[dict[str, jax.Array], optax.OptState]:
            current, current_state = carry
            gradient = jax.grad(objective)(current)
            updates, current_state = optimizer.update(gradient, current_state, current)
            return optax.apply_updates(current, updates), current_state

        return jax.lax.fori_loop(
            0, steps, body, (initial_parameters, initial_state)
        )

    parameters, _ = optimize(parameters, state)
    theta = np.asarray(parameters["theta"], dtype=np.float64)
    eta = float(np.asarray(parameters["eta"], dtype=np.float64))
    rho = float(jax.nn.sigmoid(parameters["eta"]))
    final_gram = (
        gram_cc
        + rho * (gram_ce + gram_ce.T)
        + rho**2 * gram_ee
        + ridge * identity
    )
    fitted_parameters = {
        "weights": _theta_to_weights(theta, modes),
        "rho_eta": np.asarray(eta, dtype=np.float64),
    }
    prediction = predict(
        "shared_learnable_rho",
        fitted_parameters,
        dataset.u0,
        dataset.gamma0,
        dataset.gamma1,
        rho=rho,
        pad_fraction=dataset.config.pad_fraction,
    )
    return FitResult(
        model="shared_learnable_rho",
        modes=modes,
        rho=rho,
        ridge=ridge,
        parameters=fitted_parameters,
        condition_number=float(np.linalg.cond(final_gram)),
        train_mse=float(np.mean((prediction - dataset.u1) ** 2)),
        parameter_count=parameter_count + 1,
        rho_initial=rho_initial,
        optimization_steps=steps,
        parameter_ridge=ridge,
    )


def fit_optax_check(
    model: ModelName,
    dataset: AdvectionDataset,
    modes: int,
    *,
    rho: float = 0.0,
    ridge: float = 1.0e-8,
    steps: int = 2_000,
    learning_rate: float = 5.0e-2,
) -> FitResult:
    """Fit the same quadratic objective with Optax as an implementation cross-check.

    Ridge fitting remains the registered estimator. This function exists to verify
    that automatic differentiation and the explicit normal equations optimize the
    same forward model on small diagnostic datasets.
    """

    if model == "shared_learnable_rho":
        raise ValueError("Use fit_shared_learnable_rho for the learnable-rho model")
    if steps <= 0 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    design = design_matrix(
        model,
        dataset.u0,
        dataset.gamma0,
        dataset.gamma1,
        modes,
        rho=rho,
        pad_fraction=dataset.config.pad_fraction,
    )
    parameter_count = int(design.shape[-1])
    flat_design = design.reshape((-1, parameter_count))
    target = jnp.asarray(dataset.u1).reshape((-1,))
    theta = jnp.zeros(parameter_count, dtype=flat_design.dtype)
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(theta)

    @jax.jit
    def step(current: jax.Array, optimizer_state: optax.OptState):
        def objective(parameters: jax.Array) -> jax.Array:
            residual = flat_design @ parameters - target
            parameter_ridge = 2.0 * ridge if model == "hidden_width_2" else ridge
            return jnp.mean(residual**2) + parameter_ridge * jnp.sum(parameters**2)

        loss, gradient = jax.value_and_grad(objective)(current)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, current)
        return optax.apply_updates(current, updates), optimizer_state, loss

    loss = jnp.asarray(np.inf)
    for _ in range(steps):
        theta, state, loss = step(theta, state)
    theta_host = np.asarray(theta, dtype=np.float64)
    parameters = _parameters_from_theta(model, theta_host, modes)
    estimate = predict(
        model,
        parameters,
        dataset.u0,
        dataset.gamma0,
        dataset.gamma1,
        rho=rho,
        pad_fraction=dataset.config.pad_fraction,
    )
    return FitResult(
        model=model,
        modes=modes,
        rho=rho,
        ridge=ridge,
        parameters=parameters,
        condition_number=float("nan"),
        train_mse=float(np.mean((estimate - dataset.u1) ** 2)),
        parameter_count=parameter_count,
        parameter_ridge=2.0 * ridge if model == "hidden_width_2" else ridge,
    )


@partial(jax.jit, static_argnames=("pad_fraction",))
def spectral_convolution(
    weights: jax.Array | np.ndarray,
    signal: jax.Array | np.ndarray,
    *,
    pad_fraction: float = 0.5,
) -> jax.Array:
    """Apply a retained-mode Fourier multiplier on the padded interval."""

    signal, squeezed = _as_batch(signal)
    weights = jnp.asarray(weights)
    n_x = int(signal.shape[-1])
    pad = int(round(n_x * pad_fraction))
    padded = jnp.pad(signal, ((0, 0), (pad, pad)), mode="edge")
    spectrum = jnp.fft.rfft(padded, axis=-1)
    if weights.shape != (weights.size,):
        raise ValueError("weights must be one-dimensional")
    if weights.size > spectrum.shape[-1]:
        raise ValueError("There are more multiplier modes than available FFT modes")
    multiplier = jnp.zeros(spectrum.shape[-1], dtype=spectrum.dtype)
    multiplier = multiplier.at[: weights.size].set(weights.astype(spectrum.dtype))
    output = jnp.fft.irfft(spectrum * multiplier[None, :], n=padded.shape[-1], axis=-1)
    output = output[:, pad : pad + n_x]
    return output[0] if squeezed else output


def predict(
    model: ModelName,
    parameters: dict[str, np.ndarray],
    u0: np.ndarray,
    gamma0: np.ndarray,
    gamma1: np.ndarray,
    *,
    rho: float = 0.0,
    pad_fraction: float = 0.5,
) -> np.ndarray:
    """Evaluate a fitted operator and return a host NumPy array."""

    if model == "plain":
        output = spectral_convolution(parameters["weights"], u0, pad_fraction=pad_fraction)
    elif model == "hidden_width_2":
        hidden0 = spectral_convolution(
            parameters["weights0"], u0, pad_fraction=pad_fraction
        )
        hidden1 = spectral_convolution(
            parameters["weights1"], u0, pad_fraction=pad_fraction
        )
        output = hidden0 + hidden1
    elif model == "two_channel_fno":
        solution_channel = spectral_convolution(
            parameters["weights0"], u0, pad_fraction=pad_fraction
        )
        interface_channel = spectral_convolution(
            parameters["weights1"], gamma0, pad_fraction=pad_fraction
        )
        output = solution_channel + interface_channel
    elif model in ("shared", "shared_learnable_rho"):
        a = 0.5 * (1.0 + rho)
        b = 0.5 * (1.0 - rho)
        base = spectral_convolution(parameters["weights"], u0, pad_fraction=pad_fraction)
        feature = spectral_convolution(
            parameters["weights"], gamma0 * u0, pad_fraction=pad_fraction
        )
        output = a * base + b * jnp.asarray(gamma1) * feature
    elif model == "dual":
        base = spectral_convolution(parameters["weights0"], u0, pad_fraction=pad_fraction)
        feature = spectral_convolution(
            parameters["weights1"], gamma0 * u0, pad_fraction=pad_fraction
        )
        output = base + jnp.asarray(gamma1) * feature
    else:
        raise ValueError(f"Unknown model: {model}")
    return np.asarray(output)


def exact_shift_multiplier(
    n_x: int,
    dx: float,
    displacement: float,
    *,
    pad_fraction: float = 0.5,
) -> np.ndarray:
    """Full circular-shift multiplier on the padded computational interval."""

    pad = int(round(n_x * pad_fraction))
    n_extended = n_x + 2 * pad
    n_frequency = n_extended // 2 + 1
    length_extended = n_extended * dx
    wave_numbers = 2.0 * np.pi * np.arange(n_frequency) / length_extended
    return np.exp(-1j * wave_numbers * displacement)


def save_fit(path: str | Path, result: FitResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "model": np.asarray(result.model),
        "modes": np.asarray(result.modes),
        "rho": np.asarray(result.rho),
        "ridge": np.asarray(result.ridge),
        "condition_number": np.asarray(result.condition_number),
        "train_mse": np.asarray(result.train_mse),
        "parameter_count": np.asarray(result.parameter_count),
        "rho_initial": np.asarray(
            np.nan if result.rho_initial is None else result.rho_initial
        ),
        "optimization_steps": np.asarray(result.optimization_steps),
        "parameter_ridge": np.asarray(
            result.ridge if result.parameter_ridge is None else result.parameter_ridge
        ),
    }
    payload.update(result.parameters)
    np.savez_compressed(path, **payload)
