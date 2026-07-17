"""Analytic data for fixed-speed 1D advection with one physical interface."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from .config import ExperimentConfig

Stage = Literal["A", "B"]
SMode = Literal["grid_faces", "interior_grid", "offgrid", "endpoints"]


@dataclass(frozen=True)
class AdvectionDataset:
    """Arrays used by the models plus non-input metadata used for analysis."""

    u0: np.ndarray
    gamma0: np.ndarray
    gamma1: np.ndarray
    u1: np.ndarray
    x: np.ndarray
    metadata: dict[str, np.ndarray]
    stage: Stage
    config: ExperimentConfig

    def __len__(self) -> int:
        return int(self.u0.shape[0])

    def validate(self) -> None:
        expected = (len(self), self.config.n_x)
        for name in ("u0", "gamma0", "gamma1", "u1"):
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f"{name} has shape {value.shape}, expected {expected}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        if self.x.shape != (self.config.n_x,):
            raise ValueError("x has an incompatible shape")
        for key, value in self.metadata.items():
            if value.shape[0] != len(self):
                raise ValueError(f"metadata[{key!r}] does not have one entry per sample")


def _smooth_basis(z: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    """Three C-infinity compactly supported functions, constant near boundaries."""

    q = z / config.smooth_support
    bump = np.zeros_like(q, dtype=np.float64)
    inside = np.abs(q) < 1.0
    bump[inside] = np.exp(1.0 - 1.0 / (1.0 - q[inside] ** 2))
    polynomials = (
        q,
        0.5 * (3.0 * q**2 - 1.0),
        0.5 * (5.0 * q**3 - 3.0 * q),
    )
    basis = np.stack([bump * p for p in polynomials], axis=-1)
    return basis[..., : config.smooth_terms]


def evaluate_background(
    z: np.ndarray,
    means: np.ndarray,
    coefficients: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    """Evaluate the smooth background and its constant exterior extension."""

    basis = _smooth_basis(np.asarray(z, dtype=np.float64), config)
    return means[:, None] + np.einsum("sk,xk->sx", coefficients, basis, optimize=True)


def gamma_values(z: np.ndarray, s: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    """Evaluate binary interface features, including the endpoint convention."""

    z = np.asarray(z, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    gamma = np.where(z[None, :] < s[:, None], -1.0, 1.0)
    tolerance = 8.0 * np.finfo(np.float64).eps
    at_left = s <= config.x_min + tolerance
    at_right = s >= config.x_max - tolerance
    gamma[at_left] = 1.0
    gamma[at_right] = -1.0
    return gamma


def _sample_locations(
    rng: np.random.Generator,
    n_samples: int,
    config: ExperimentConfig,
    mode: SMode,
) -> np.ndarray:
    if mode == "grid_faces":
        indices = rng.integers(0, config.n_x + 1, size=n_samples)
        return config.x_min + indices * config.dx
    if mode == "interior_grid":
        indices = rng.integers(1, config.n_x, size=n_samples)
        return config.x_min + indices * config.dx
    if mode == "offgrid":
        return rng.uniform(config.x_min + config.dx, config.x_max - config.dx, size=n_samples)
    if mode == "endpoints":
        values = np.resize(np.array([config.x_min, config.x_max]), n_samples)
        rng.shuffle(values)
        return values
    raise ValueError(f"Unknown s sampling mode: {mode}")


def generate_dataset(
    n_samples: int,
    stage: Stage,
    seed: int,
    config: ExperimentConfig | None = None,
    *,
    s_mode: SMode = "grid_faces",
    force_zero_jump: bool = False,
) -> AdvectionDataset:
    """Generate exact input-output pairs without a numerical time integrator."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if stage not in ("A", "B"):
        raise ValueError("stage must be 'A' or 'B'")
    config = config or ExperimentConfig()
    rng = np.random.default_rng(seed)
    x = config.grid
    s = _sample_locations(rng, n_samples, config, s_mode)
    means = rng.uniform(config.mean_low, config.mean_high, size=n_samples)
    jump_magnitude = rng.uniform(config.jump_abs_low, config.jump_abs_high, size=n_samples)
    jump_sign = rng.choice(np.array([-1.0, 1.0]), size=n_samples)
    jumps = jump_sign * jump_magnitude
    if force_zero_jump:
        jumps.fill(0.0)

    if stage == "A":
        coefficients = np.zeros((n_samples, config.smooth_terms), dtype=np.float64)
    else:
        coefficients = rng.uniform(
            -config.smooth_amplitude,
            config.smooth_amplitude,
            size=(n_samples, config.smooth_terms),
        )

    gamma0 = gamma_values(x, s, config)
    transported_x = x - config.displacement
    gamma1 = gamma_values(transported_x, s, config)
    background0 = evaluate_background(x, means, coefficients, config)
    background1 = evaluate_background(transported_x, means, coefficients, config)
    u0 = background0 + 0.5 * jumps[:, None] * gamma0
    u1 = background1 + 0.5 * jumps[:, None] * gamma1

    tolerance = 8.0 * np.finfo(np.float64).eps
    is_endpoint = (s <= config.x_min + tolerance) | (s >= config.x_max - tolerance)
    jump_location = s + config.displacement
    has_output_jump = (~is_endpoint) & (jump_location > config.x_min) & (jump_location < config.x_max)
    metadata = {
        "sample_id": np.arange(n_samples, dtype=np.int64),
        "seed": np.full(n_samples, seed, dtype=np.int64),
        "s": s.astype(np.float64),
        "c": np.full(n_samples, config.velocity, dtype=np.float64),
        "T": np.full(n_samples, config.final_time, dtype=np.float64),
        "m": means.astype(np.float64),
        "J": jumps.astype(np.float64),
        "smooth_coefficients": coefficients.astype(np.float64),
        "is_endpoint": is_endpoint.astype(bool),
        "has_output_jump": has_output_jump.astype(bool),
        "output_jump_location": jump_location.astype(np.float64),
        "grid_resolution": np.full(n_samples, config.n_x, dtype=np.int64),
    }
    dataset = AdvectionDataset(
        u0=u0.astype(np.float64),
        gamma0=gamma0.astype(np.float64),
        gamma1=gamma1.astype(np.float64),
        u1=u1.astype(np.float64),
        x=x.astype(np.float64),
        metadata=metadata,
        stage=stage,
        config=config,
    )
    dataset.validate()
    return dataset

def apply_gamma_ablation(
    dataset: AdvectionDataset,
    kind: Literal["unshifted", "constant", "shift_output", "smooth"],
    *,
    cells: float = 1.0,
) -> AdvectionDataset:
    """Return the same physical samples with intentionally perturbed features."""

    gamma0 = np.array(dataset.gamma0, copy=True)
    gamma1 = np.array(dataset.gamma1, copy=True)
    if kind == "unshifted":
        gamma1 = np.array(gamma0, copy=True)
    elif kind == "constant":
        gamma0.fill(1.0)
        gamma1.fill(1.0)
    elif kind == "shift_output":
        locations = dataset.metadata["output_jump_location"] + cells * dataset.config.dx
        gamma1 = _gamma_from_jump_locations(dataset.x, locations, dataset)
    elif kind == "smooth":
        width = max(abs(cells) * dataset.config.dx, np.finfo(np.float64).eps)
        gamma0 = _smooth_gamma(dataset.x, dataset.metadata["s"], width, dataset)
        gamma1 = _smooth_gamma(
            dataset.x,
            dataset.metadata["output_jump_location"],
            width,
            dataset,
        )
    else:
        raise ValueError(f"Unknown gamma ablation: {kind}")
    return replace(dataset, gamma0=gamma0, gamma1=gamma1)


def _gamma_from_jump_locations(
    x: np.ndarray,
    locations: np.ndarray,
    dataset: AdvectionDataset,
) -> np.ndarray:
    gamma = np.where(x[None, :] < locations[:, None], -1.0, 1.0)
    endpoint = dataset.metadata["is_endpoint"]
    left = endpoint & (dataset.metadata["s"] <= dataset.config.x_min)
    right = endpoint & (dataset.metadata["s"] >= dataset.config.x_max)
    gamma[left] = 1.0
    gamma[right] = -1.0
    return gamma


def _smooth_gamma(
    x: np.ndarray,
    locations: np.ndarray,
    width: float,
    dataset: AdvectionDataset,
) -> np.ndarray:
    gamma = np.tanh((x[None, :] - locations[:, None]) / width)
    endpoint = dataset.metadata["is_endpoint"]
    left = endpoint & (dataset.metadata["s"] <= dataset.config.x_min)
    right = endpoint & (dataset.metadata["s"] >= dataset.config.x_max)
    gamma[left] = 1.0
    gamma[right] = -1.0
    return gamma


def save_dataset(path: str | Path, dataset: AdvectionDataset) -> None:
    """Save a dataset without pickle so it is safe and portable."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "u0": dataset.u0,
        "gamma0": dataset.gamma0,
        "gamma1": dataset.gamma1,
        "u1": dataset.u1,
        "x": dataset.x,
        "stage": np.asarray(dataset.stage),
    }
    payload.update({f"meta__{key}": value for key, value in dataset.metadata.items()})
    for key, value in dataset.config.to_dict().items():
        payload[f"config__{key}"] = np.asarray(value)
    np.savez_compressed(path, **payload)


def load_dataset(path: str | Path) -> AdvectionDataset:
    """Load data written by :func:`save_dataset`."""

    with np.load(Path(path), allow_pickle=False) as payload:
        config_values = {
            key.removeprefix("config__"): payload[key].item()
            for key in payload.files
            if key.startswith("config__")
        }
        metadata = {
            key.removeprefix("meta__"): np.asarray(payload[key])
            for key in payload.files
            if key.startswith("meta__")
        }
        dataset = AdvectionDataset(
            u0=np.asarray(payload["u0"]),
            gamma0=np.asarray(payload["gamma0"]),
            gamma1=np.asarray(payload["gamma1"]),
            u1=np.asarray(payload["u1"]),
            x=np.asarray(payload["x"]),
            metadata=metadata,
            stage=str(payload["stage"].item()),
            config=ExperimentConfig(**config_values),
        )
    dataset.validate()
    return dataset
