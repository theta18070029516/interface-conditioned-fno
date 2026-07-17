"""Configuration objects shared by data generation and experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    """Physical and sampling configuration for the advection benchmark."""

    x_min: float = -1.0
    x_max: float = 1.0
    final_time: float = 1.0
    velocity: float = 0.5
    n_x: int = 256
    smooth_terms: int = 3
    smooth_support: float = 0.75
    smooth_amplitude: float = 0.2
    mean_low: float = -0.5
    mean_high: float = 0.5
    jump_abs_low: float = 0.5
    jump_abs_high: float = 1.5
    pad_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.final_time <= 0:
            raise ValueError("final_time must be positive")
        if self.velocity <= 0:
            raise ValueError("This implementation assumes positive advection velocity")
        if self.n_x < 8 or self.n_x % 2:
            raise ValueError("n_x must be an even integer >= 8")
        if not 0 < self.smooth_support < max(abs(self.x_min), abs(self.x_max)):
            raise ValueError("smooth_support must lie strictly inside the physical domain")
        if not 0 < self.pad_fraction:
            raise ValueError("pad_fraction must be positive")

    @property
    def domain_length(self) -> float:
        return self.x_max - self.x_min

    @property
    def dx(self) -> float:
        return self.domain_length / self.n_x

    @property
    def displacement(self) -> float:
        return self.velocity * self.final_time

    @property
    def grid(self) -> np.ndarray:
        return self.x_min + (np.arange(self.n_x, dtype=np.float64) + 0.5) * self.dx

    @property
    def faces(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.n_x + 1, dtype=np.float64)

    @property
    def pad_points(self) -> int:
        return int(round(self.n_x * self.pad_fraction))

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


PRESETS: dict[str, dict[str, object]] = {
    "smoke": {
        "n_x": 32,
        "sizes": {
            "A": {"train": 64, "validation": 16, "test": 32},
            "B": {"train": 96, "validation": 16, "test": 32},
        },
        "diagnostic_size": 24,
        "modes": [4, 8],
        "rho_scan_mode": 8,
        "seeds": [0],
        "bootstrap_reps": 200,
        "fit_batch_size": 16,
        "learnable_rho_initial": 0.5,
        "learnable_rho_steps": 1_000,
        "learnable_rho_learning_rate": 2.0e-2,
    },
    "full": {
        "n_x": 256,
        "sizes": {
            "A": {"train": 4096, "validation": 512, "test": 1024},
            "B": {"train": 8192, "validation": 1024, "test": 2048},
        },
        "diagnostic_size": 1024,
        "modes": [8, 16, 32, 64],
        "rho_scan_mode": 32,
        "seeds": [0, 1, 2, 3, 4],
        "bootstrap_reps": 10_000,
        "fit_batch_size": 64,
        "learnable_rho_initial": 0.5,
        "learnable_rho_steps": 2_000,
        "learnable_rho_learning_rate": 2.0e-2,
    },
}
