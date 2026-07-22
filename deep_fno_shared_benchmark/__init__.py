"""Deep FNO versus interface-conditioned Shared benchmark."""

from .config import DataConfig, ModelConfig, TrainingConfig
from .data import (
    AdvectionCases,
    DataNormalizer,
    DeepAdvectionDataset,
    generate_cases,
    sample_cases,
)

__all__ = [
    "AdvectionCases",
    "DataConfig",
    "DataNormalizer",
    "DeepAdvectionDataset",
    "ModelConfig",
    "TrainingConfig",
    "generate_cases",
    "sample_cases",
]
