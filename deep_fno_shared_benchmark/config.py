"""Configuration for the deep FNO/Shared advection benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


RhoMode = Literal["matrix", "layer_scalar", "fixed_zero", "fixed_one"]


@dataclass(frozen=True)
class DataConfig:
    """Resolution-independent distribution of analytic advection cases."""

    x_min: float = -1.0
    x_max: float = 1.0
    velocity: float = 0.5
    final_time: float = 1.0
    smooth_terms: int = 8
    smooth_support: float = 0.85
    smooth_amplitude: float = 0.2
    spectral_decay: float = 1.5
    plateau_fraction: float = 0.2
    mean_low: float = -0.5
    mean_high: float = 0.5
    jump_abs_low: float = 0.5
    jump_abs_high: float = 1.5
    internal_fraction: float = 0.7
    exiting_fraction: float = 0.1
    zero_jump_fraction: float = 0.1
    endpoint_fraction: float = 0.1
    interface_margin: float = 0.05
    protocol_version: int = 1
    v2_interface_low: float = -0.75
    v2_interface_high: float = 0.25

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.velocity <= 0 or self.final_time <= 0:
            raise ValueError("velocity and final_time must be positive")
        if self.smooth_terms < 1:
            raise ValueError("smooth_terms must be positive")
        if not 0 < self.smooth_support < max(abs(self.x_min), abs(self.x_max)):
            raise ValueError("smooth_support must lie strictly inside the domain")
        if self.smooth_amplitude < 0 or self.spectral_decay <= 0:
            raise ValueError("invalid smooth-background parameters")
        if not 0 <= self.plateau_fraction <= 1:
            raise ValueError("plateau_fraction must lie in [0, 1]")
        if not 0 < self.jump_abs_low <= self.jump_abs_high:
            raise ValueError("invalid jump range")
        fractions = (
            self.internal_fraction,
            self.exiting_fraction,
            self.zero_jump_fraction,
            self.endpoint_fraction,
        )
        if any(value < 0 for value in fractions):
            raise ValueError("category fractions must be non-negative")
        if abs(sum(fractions) - 1.0) > 1.0e-12:
            raise ValueError("category fractions must sum to one")
        if not 0 < self.interface_margin < 0.5 * self.domain_length:
            raise ValueError("interface_margin is incompatible with the domain")
        if self.x_max - self.displacement - self.interface_margin <= self.x_min:
            raise ValueError("displacement leaves no room for retained interfaces")
        if self.protocol_version not in {1, 2}:
            raise ValueError("protocol_version must be 1 or 2")
        if self.protocol_version == 2:
            if fractions != (1.0, 0.0, 0.0, 0.0):
                raise ValueError("protocol v2 requires internal-only category fractions")
            if not self.x_min < self.v2_interface_low < self.v2_interface_high < self.x_max:
                raise ValueError("invalid protocol-v2 initial interface interval")
            if self.v2_interface_low < -0.75 or self.v2_interface_high > 0.25:
                raise ValueError("protocol-v2 initial interfaces must lie in [-0.75, 0.25]")
            final_low = self.v2_interface_low + self.displacement
            final_high = self.v2_interface_high + self.displacement
            if final_low < -0.25 or final_high > 0.75:
                raise ValueError("protocol-v2 final interfaces must lie in [-0.25, 0.75]")
            if not (self.x_min < final_low <= final_high < self.x_max):
                raise ValueError("protocol-v2 transported interfaces must remain internal")

    @classmethod
    def v2(cls, **overrides: Any) -> "DataConfig":
        """Build the registered v2 internal-discontinuity distribution."""

        values: dict[str, Any] = {
            "protocol_version": 2,
            "internal_fraction": 1.0,
            "exiting_fraction": 0.0,
            "zero_jump_fraction": 0.0,
            "endpoint_fraction": 0.0,
            "v2_interface_low": -0.75,
            "v2_interface_high": 0.25,
        }
        values.update(overrides)
        return cls(**values)

    @property
    def domain_length(self) -> float:
        return self.x_max - self.x_min

    @property
    def displacement(self) -> float:
        return self.velocity * self.final_time

    def grid(self, n_x: int) -> "object":
        import numpy as np

        if n_x < 8 or n_x % 2:
            raise ValueError("n_x must be an even integer >= 8")
        dx = self.domain_length / n_x
        return self.x_min + (np.arange(n_x, dtype=np.float64) + 0.5) * dx

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    """Common architecture plus Shared-specific rho/gamma controls."""

    n_layers: int = 4
    hidden_channels: int = 64
    n_modes: int = 16
    lifting_channels: int = 128
    projection_channels: int = 128
    pad_fraction: float = 0.5
    rho_mode: RhoMode = "matrix"
    rho_initial: float = 0.05
    gamma_initial_magnitude: float = 0.95

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")
        if min(
            self.hidden_channels,
            self.n_modes,
            self.lifting_channels,
            self.projection_channels,
        ) < 1:
            raise ValueError("all channel and mode counts must be positive")
        if self.pad_fraction < 0:
            raise ValueError("pad_fraction must be non-negative")
        if self.rho_mode not in {
            "matrix",
            "layer_scalar",
            "fixed_zero",
            "fixed_one",
        }:
            raise ValueError(f"unknown rho_mode: {self.rho_mode}")
        if not 0 < self.rho_initial < 1:
            raise ValueError("rho_initial must lie strictly inside (0, 1)")
        if not 0 < self.gamma_initial_magnitude < 1:
            raise ValueError("gamma_initial_magnitude must lie inside (0, 1)")

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and model-selection settings."""

    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    warmup_fraction: float = 0.05
    end_learning_rate_fraction: float = 0.01
    gradient_clip: float = 1.0
    patience: int = 40
    gamma_loss_weight: float = 1.0
    validation_batch_size: int = 256
    protocol_version: int = 1
    early_stopping: bool = True

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("epochs, batch_size, and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer parameters")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must lie in [0, 1)")
        if not 0 < self.end_learning_rate_fraction <= 1:
            raise ValueError("end_learning_rate_fraction must lie in (0, 1]")
        if self.gradient_clip <= 0 or self.gamma_loss_weight < 0:
            raise ValueError("invalid loss or clipping parameter")
        if self.validation_batch_size < 1:
            raise ValueError("validation_batch_size must be positive")
        if self.protocol_version not in {1, 2}:
            raise ValueError("protocol_version must be 1 or 2")
        if self.protocol_version == 2:
            if self.early_stopping:
                raise ValueError("protocol v2 forbids early stopping")
            if self.gamma_loss_weight != 0.0:
                raise ValueError("protocol v2 forbids a gamma loss")

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)
