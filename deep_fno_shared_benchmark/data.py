"""Analytic, resolution-independent data for one-jump linear advection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DataConfig


INTERNAL = 0
EXITING = 1
ZERO_JUMP = 2
ENDPOINT = 3
CATEGORY_NAMES = {
    INTERNAL: "internal_jump",
    EXITING: "exiting_jump",
    ZERO_JUMP: "zero_jump_continuous",
    ENDPOINT: "endpoint_continuous",
}


@dataclass(frozen=True)
class AdvectionCases:
    """Continuous sample parameters that can be evaluated on any regular grid."""

    sample_id: np.ndarray
    seed: np.ndarray
    category: np.ndarray
    mean: np.ndarray
    sin_coefficients: np.ndarray
    cos_coefficients: np.ndarray
    jump: np.ndarray
    interface_location: np.ndarray
    config: DataConfig

    def __len__(self) -> int:
        return int(self.sample_id.size)

    def validate(self) -> None:
        n_samples = len(self)
        vectors = (
            self.sample_id,
            self.seed,
            self.category,
            self.mean,
            self.jump,
            self.interface_location,
        )
        if any(value.shape != (n_samples,) for value in vectors):
            raise ValueError("case vectors must have one entry per sample")
        expected_coefficients = (n_samples, self.config.smooth_terms)
        if self.sin_coefficients.shape != expected_coefficients:
            raise ValueError("sin_coefficients has an invalid shape")
        if self.cos_coefficients.shape != expected_coefficients:
            raise ValueError("cos_coefficients has an invalid shape")
        if not np.all(np.isin(self.category, tuple(CATEGORY_NAMES))):
            raise ValueError("unknown category code")
        numeric = (
            self.mean,
            self.sin_coefficients,
            self.cos_coefficients,
            self.jump,
            self.interface_location,
        )
        if not all(np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("case parameters contain non-finite values")
        if np.any(self.jump[self.category == ZERO_JUMP] != 0):
            raise ValueError("zero-jump cases must have zero jump magnitude")
        if self.config.protocol_version == 2:
            if not np.all(self.category == INTERNAL):
                raise ValueError("protocol-v2 cases must all be internal jumps")
            if np.any(np.abs(self.jump) < self.config.jump_abs_low):
                raise ValueError("protocol-v2 jumps must be non-zero")
            if np.any(np.abs(self.jump) > self.config.jump_abs_high):
                raise ValueError("protocol-v2 jumps exceed the registered range")
            if np.any(self.interface_location < self.config.v2_interface_low) or np.any(
                self.interface_location > self.config.v2_interface_high
            ):
                raise ValueError("protocol-v2 initial interface lies outside its interval")
            transported = self.interface_location + self.config.displacement
            if np.any(transported < -0.25) or np.any(transported > 0.75):
                raise ValueError("protocol-v2 final interface lies outside [-0.25, 0.75]")

    def subset(self, indices: np.ndarray | slice) -> "AdvectionCases":
        result = AdvectionCases(
            sample_id=self.sample_id[indices],
            seed=self.seed[indices],
            category=self.category[indices],
            mean=self.mean[indices],
            sin_coefficients=self.sin_coefficients[indices],
            cos_coefficients=self.cos_coefficients[indices],
            jump=self.jump[indices],
            interface_location=self.interface_location[indices],
            config=self.config,
        )
        result.validate()
        return result


@dataclass(frozen=True)
class DeepAdvectionDataset:
    """A discretization of continuous advection cases."""

    x: np.ndarray
    u0: np.ndarray
    gamma0: np.ndarray
    u_target: np.ndarray
    gamma_target: np.ndarray
    cases: AdvectionCases

    def __len__(self) -> int:
        return len(self.cases)

    @property
    def n_x(self) -> int:
        return int(self.x.size)

    def validate(self) -> None:
        expected = (len(self), self.n_x)
        for name in ("u0", "gamma0", "u_target", "gamma_target"):
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f"{name} has shape {value.shape}, expected {expected}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        if self.x.ndim != 1 or not np.all(np.isfinite(self.x)):
            raise ValueError("x must be a finite one-dimensional grid")
        for gamma in (self.gamma0, self.gamma_target):
            if np.any(np.abs(gamma) > 1.0):
                raise ValueError("gamma labels must lie in [-1, 1]")

    def model_inputs(
        self,
        normalizer: "DataNormalizer",
        *,
        include_gamma_target: bool = False,
    ) -> np.ndarray:
        x = np.broadcast_to(self.x[None, :], self.u0.shape)
        channels = [x, normalizer.normalize(self.u0), self.gamma0]
        if include_gamma_target:
            channels.append(self.gamma_target)
        return np.stack(channels, axis=1).astype(np.float32)

    def normalized_target(self, normalizer: "DataNormalizer") -> np.ndarray:
        return normalizer.normalize(self.u_target)[:, None, :].astype(np.float32)


@dataclass(frozen=True)
class DataNormalizer:
    """One global affine normalization shared by input and target solution fields."""

    mean: float
    std: float

    @classmethod
    def fit(cls, dataset: DeepAdvectionDataset) -> "DataNormalizer":
        values = np.concatenate([dataset.u0.ravel(), dataset.u_target.ravel()])
        std = float(np.std(values))
        if not np.isfinite(std) or std < 1.0e-8:
            raise ValueError("training solution fields have near-zero or invalid variance")
        return cls(mean=float(np.mean(values)), std=std)

    def normalize(self, value: np.ndarray) -> np.ndarray:
        return (np.asarray(value) - self.mean) / self.std

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value) * self.std + self.mean


def _category_counts(n_samples: int, config: DataConfig) -> np.ndarray:
    fractions = np.array(
        [
            config.internal_fraction,
            config.exiting_fraction,
            config.zero_jump_fraction,
            config.endpoint_fraction,
        ],
        dtype=np.float64,
    )
    raw = fractions * n_samples
    counts = np.floor(raw).astype(np.int64)
    remainder = n_samples - int(np.sum(counts))
    if remainder:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts


def generate_cases(
    n_samples: int,
    seed: int,
    config: DataConfig | None = None,
    *,
    sample_id_offset: int = 0,
) -> AdvectionCases:
    """Draw continuous problem parameters with exact pre-registered strata."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    config = config or DataConfig()
    rng = np.random.default_rng(seed)
    counts = _category_counts(n_samples, config)
    category = np.concatenate(
        [np.full(count, code, dtype=np.int8) for code, count in enumerate(counts)]
    )
    rng.shuffle(category)

    means = rng.uniform(config.mean_low, config.mean_high, size=n_samples)
    sin_coefficients = rng.uniform(
        -config.smooth_amplitude,
        config.smooth_amplitude,
        size=(n_samples, config.smooth_terms),
    )
    cos_coefficients = rng.uniform(
        -config.smooth_amplitude,
        config.smooth_amplitude,
        size=(n_samples, config.smooth_terms),
    )
    plateau = rng.random(n_samples) < config.plateau_fraction
    sin_coefficients[plateau] = 0.0
    cos_coefficients[plateau] = 0.0

    jump_abs = rng.uniform(config.jump_abs_low, config.jump_abs_high, size=n_samples)
    jump = jump_abs * rng.choice(np.array([-1.0, 1.0]), size=n_samples)
    jump[category == ZERO_JUMP] = 0.0

    location = np.empty(n_samples, dtype=np.float64)
    internal = category == INTERNAL
    exiting = category == EXITING
    zero_jump = category == ZERO_JUMP
    endpoint = category == ENDPOINT
    if config.protocol_version == 2:
        location[internal] = rng.uniform(
            config.v2_interface_low,
            config.v2_interface_high,
            size=int(np.count_nonzero(internal)),
        )
    else:
        location[internal] = rng.uniform(
            config.x_min + config.interface_margin,
            config.x_max - config.displacement - config.interface_margin,
            size=int(np.count_nonzero(internal)),
        )
    location[exiting] = rng.uniform(
        config.x_max - config.displacement,
        config.x_max - config.interface_margin,
        size=int(np.count_nonzero(exiting)),
    )
    location[zero_jump] = config.x_min
    location[endpoint] = rng.choice(
        np.array([config.x_min, config.x_max]),
        size=int(np.count_nonzero(endpoint)),
    )

    result = AdvectionCases(
        sample_id=np.arange(
            sample_id_offset, sample_id_offset + n_samples, dtype=np.int64
        ),
        seed=np.full(n_samples, seed, dtype=np.int64),
        category=category,
        mean=means.astype(np.float64),
        sin_coefficients=sin_coefficients.astype(np.float64),
        cos_coefficients=cos_coefficients.astype(np.float64),
        jump=jump.astype(np.float64),
        interface_location=location,
        config=config,
    )
    result.validate()
    return result


def _smooth_envelope(z: np.ndarray, support: float) -> tuple[np.ndarray, np.ndarray]:
    q = z / support
    envelope = np.zeros_like(q, dtype=np.float64)
    inside = np.abs(q) < 1.0
    envelope[inside] = np.exp(1.0 - 1.0 / (1.0 - q[inside] ** 2))
    return q, envelope


def _evaluate_background(
    z: np.ndarray,
    cases: AdvectionCases,
) -> np.ndarray:
    config = cases.config
    q, envelope = _smooth_envelope(np.asarray(z, dtype=np.float64), config.smooth_support)
    frequency = np.arange(1, config.smooth_terms + 1, dtype=np.float64)
    decay = frequency**config.spectral_decay
    sin_basis = envelope[:, None] * np.sin(np.pi * q[:, None] * frequency[None, :])
    cos_basis = envelope[:, None] * np.cos(np.pi * q[:, None] * frequency[None, :])
    smooth = np.einsum(
        "sk,xk->sx", cases.sin_coefficients / decay[None, :], sin_basis, optimize=True
    )
    smooth += np.einsum(
        "sk,xk->sx", cases.cos_coefficients / decay[None, :], cos_basis, optimize=True
    )
    return cases.mean[:, None] + smooth


def _gamma_field(
    x: np.ndarray,
    locations: np.ndarray,
    categories: np.ndarray,
) -> np.ndarray:
    gamma = np.where(x[None, :] < locations[:, None], -1.0, 1.0)
    continuous = (categories == ZERO_JUMP) | (categories == ENDPOINT)
    gamma[continuous] = 1.0
    return gamma


def sample_cases(
    cases: AdvectionCases,
    n_x: int,
    *,
    dtype: np.dtype = np.float32,
) -> DeepAdvectionDataset:
    """Evaluate the same continuous cases on an arbitrary cell-centered grid."""

    cases.validate()
    x = cases.config.grid(n_x)
    transported_x = x - cases.config.displacement
    gamma0 = _gamma_field(x, cases.interface_location, cases.category)
    gamma_target = _gamma_field(
        x,
        cases.interface_location + cases.config.displacement,
        cases.category,
    )
    background0 = _evaluate_background(x, cases)
    background_target = _evaluate_background(transported_x, cases)
    u0 = background0 + 0.5 * cases.jump[:, None] * gamma0
    u_target = background_target + 0.5 * cases.jump[:, None] * gamma_target
    dataset = DeepAdvectionDataset(
        x=x.astype(dtype),
        u0=u0.astype(dtype),
        gamma0=gamma0.astype(dtype),
        u_target=u_target.astype(dtype),
        gamma_target=gamma_target.astype(dtype),
        cases=cases,
    )
    dataset.validate()
    return dataset


def save_cases(path: str | Path, cases: AdvectionCases) -> None:
    """Store continuous parameters without pickle-backed object arrays."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_id=cases.sample_id,
        seed=cases.seed,
        category=cases.category,
        mean=cases.mean,
        sin_coefficients=cases.sin_coefficients,
        cos_coefficients=cases.cos_coefficients,
        jump=cases.jump,
        interface_location=cases.interface_location,
        config_keys=np.asarray(list(cases.config.to_dict().keys())),
        config_values=np.asarray(list(cases.config.to_dict().values()), dtype=np.float64),
    )


def load_cases(path: str | Path) -> AdvectionCases:
    with np.load(path, allow_pickle=False) as saved:
        keys = [str(value) for value in saved["config_keys"]]
        values = [float(value) for value in saved["config_values"]]
        raw = dict(zip(keys, values, strict=True))
        integer_keys = {"smooth_terms", "protocol_version"}
        config = DataConfig(
            **{
                key: int(value) if key in integer_keys else value
                for key, value in raw.items()
            }
        )
        cases = AdvectionCases(
            sample_id=np.asarray(saved["sample_id"]),
            seed=np.asarray(saved["seed"]),
            category=np.asarray(saved["category"]),
            mean=np.asarray(saved["mean"]),
            sin_coefficients=np.asarray(saved["sin_coefficients"]),
            cos_coefficients=np.asarray(saved["cos_coefficients"]),
            jump=np.asarray(saved["jump"]),
            interface_location=np.asarray(saved["interface_location"]),
            config=config,
        )
    cases.validate()
    return cases
