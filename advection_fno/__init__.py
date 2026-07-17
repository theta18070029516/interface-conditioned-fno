"""Feature-conditioned spectral operators for 1D linear advection."""

from .config import ExperimentConfig
from .data import AdvectionDataset, generate_dataset
from .models import FitResult, fit_ridge, fit_shared_learnable_rho, predict

__all__ = [
    "AdvectionDataset",
    "ExperimentConfig",
    "FitResult",
    "fit_ridge",
    "fit_shared_learnable_rho",
    "generate_dataset",
    "predict",
]
