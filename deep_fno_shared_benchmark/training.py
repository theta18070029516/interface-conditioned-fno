"""Deterministic Optax training and batched inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import ModelConfig, TrainingConfig
from .data import DataNormalizer, DeepAdvectionDataset
from .models import ModelKind, Params, apply_model, init_model


@dataclass(frozen=True)
class PreparedData:
    inputs: np.ndarray
    u_target: np.ndarray
    gamma_target: np.ndarray

    def __len__(self) -> int:
        return int(self.inputs.shape[0])


@dataclass(frozen=True)
class TrainResult:
    kind: ModelKind
    params: Params
    final_params: Params
    history: tuple[dict[str, float], ...]
    best_epoch: int
    best_validation_relative_l2: float
    stopped_early: bool

    @property
    def completed_epochs(self) -> int:
        return len(self.history)


@dataclass(frozen=True)
class EpochReport:
    """State exposed after one complete training and validation epoch."""

    epoch: int
    params: Params
    metrics: dict[str, float]
    history: tuple[dict[str, float], ...]
    best_epoch: int
    best_validation_relative_l2: float
    stale_epochs: int


def prepare_data(
    dataset: DeepAdvectionDataset,
    normalizer: DataNormalizer,
    kind: ModelKind = "fno",
) -> PreparedData:
    include_gamma_target = kind in {
        "fno_gamma",
        "shared_oracle",
        "shared_oracle_rho0",
    }
    return PreparedData(
        inputs=dataset.model_inputs(
            normalizer, include_gamma_target=include_gamma_target
        ),
        u_target=dataset.normalized_target(normalizer),
        gamma_target=dataset.gamma_target[:, None, :].astype(np.float32),
    )


def _make_forward(
    kind: ModelKind,
    model_config: ModelConfig,
) -> Callable[[Params, jax.Array], dict[str, jax.Array]]:
    def forward(params: Params, inputs: jax.Array) -> dict[str, jax.Array]:
        return apply_model(kind, params, inputs, model_config)

    return forward


def _validation_metrics(
    forward: Callable[[Params, jax.Array], dict[str, jax.Array]],
    params: Params,
    data: PreparedData,
    batch_size: int,
) -> tuple[float, float, float]:
    squared_error = 0.0
    observations = 0
    relative_l2: list[np.ndarray] = []
    gamma_squared_error = 0.0
    gamma_observations = 0
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        output = forward(params, jnp.asarray(data.inputs[start:stop]))
        prediction = np.asarray(output["u"])
        target = data.u_target[start:stop]
        residual = prediction - target
        squared_error += float(np.sum(residual**2))
        observations += int(residual.size)
        relative_l2.append(
            np.linalg.norm(residual, axis=-1)
            / np.maximum(np.linalg.norm(target, axis=-1), 1.0e-12)
        )
        if "gamma" in output:
            gamma_residual = np.asarray(output["gamma"]) - data.gamma_target[start:stop]
            gamma_squared_error += float(np.sum(gamma_residual**2))
            gamma_observations += int(gamma_residual.size)
    gamma_mse = (
        gamma_squared_error / gamma_observations
        if gamma_observations
        else float("nan")
    )
    return (
        squared_error / observations,
        float(np.mean(np.concatenate(relative_l2, axis=0))),
        gamma_mse,
    )


def train_model(
    kind: ModelKind,
    train_dataset: DeepAdvectionDataset,
    validation_dataset: DeepAdvectionDataset,
    normalizer: DataNormalizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    seed: int,
    epoch_callback: Callable[[EpochReport], None] | None = None,
    log_every: int = 0,
    show_progress: bool = False,
    progress_description: str | None = None,
) -> TrainResult:
    if log_every < 0:
        raise ValueError("log_every must be non-negative")
    if training_config.protocol_version == 2 and kind == "shared":
        raise ValueError("protocol v2 uses shared_oracle, not recursive shared")
    if kind in {"fno_gamma", "shared_oracle"} and training_config.protocol_version != 2:
        raise ValueError(f"{kind} requires protocol_version=2")
    train = prepare_data(train_dataset, normalizer, kind)
    validation = prepare_data(validation_dataset, normalizer, kind)
    steps_per_epoch = max(1, int(np.ceil(len(train) / training_config.batch_size)))
    total_steps = training_config.epochs * steps_per_epoch
    warmup_steps = max(
        1, int(round(total_steps * training_config.warmup_fraction))
    )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=training_config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=(
            training_config.learning_rate
            * training_config.end_learning_rate_fraction
        ),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(training_config.gradient_clip),
        optax.adamw(schedule, weight_decay=training_config.weight_decay),
    )
    params = init_model(kind, jax.random.PRNGKey(seed), model_config)
    optimizer_state = optimizer.init(params)
    forward = _make_forward(kind, model_config)
    forward_jit = jax.jit(forward)

    def loss_function(
        current_params: Params,
        inputs: jax.Array,
        u_target: jax.Array,
        gamma_target: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        output = forward(current_params, inputs)
        u_mse = jnp.mean((output["u"] - u_target) ** 2)
        gamma_mse = jnp.asarray(0.0, dtype=u_mse.dtype)
        if kind == "shared":
            gamma_mse = jnp.mean((output["gamma"] - gamma_target) ** 2)
        total = u_mse + training_config.gamma_loss_weight * gamma_mse
        return total, (u_mse, gamma_mse)

    @jax.jit
    def train_step(
        current_params: Params,
        current_state: optax.OptState,
        inputs: jax.Array,
        u_target: jax.Array,
        gamma_target: jax.Array,
    ) -> tuple[Params, optax.OptState, jax.Array, jax.Array, jax.Array]:
        (loss, (u_mse, gamma_mse)), gradients = jax.value_and_grad(
            loss_function, has_aux=True
        )(current_params, inputs, u_target, gamma_target)
        updates, next_state = optimizer.update(
            gradients, current_state, current_params
        )
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_state, loss, u_mse, gamma_mse

    rng = np.random.default_rng(seed)
    best_params = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), params)
    best_metric = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    stopped_early = False

    progress_bar = None
    epoch_iterator = range(training_config.epochs)
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as error:
            raise RuntimeError(
                "show_progress=True requires tqdm; install it in the active "
                "environment with `python -m pip install tqdm`"
            ) from error
        progress_bar = tqdm(
            epoch_iterator,
            total=training_config.epochs,
            desc=progress_description or f"{kind} seed={seed}",
            unit="epoch",
            dynamic_ncols=True,
        )
        epoch_iterator = progress_bar

    try:
        for epoch in epoch_iterator:
            order = rng.permutation(len(train))
            train_total = 0.0
            train_u = 0.0
            train_gamma = 0.0
            batches = 0
            for start in range(0, len(train), training_config.batch_size):
                indices = order[start : start + training_config.batch_size]
                params, optimizer_state, loss, u_mse, gamma_mse = train_step(
                    params,
                    optimizer_state,
                    jnp.asarray(train.inputs[indices]),
                    jnp.asarray(train.u_target[indices]),
                    jnp.asarray(train.gamma_target[indices]),
                )
                train_total += float(loss)
                train_u += float(u_mse)
                train_gamma += float(gamma_mse)
                batches += 1

            validation_mse, validation_relative_l2, validation_gamma_mse = (
                _validation_metrics(
                    forward_jit,
                    params,
                    validation,
                    training_config.validation_batch_size,
                )
            )
            epoch_number = epoch + 1
            metrics: dict[str, float] = {
                "epoch": float(epoch + 1),
                "train_loss": train_total / batches,
                "train_u_mse": train_u / batches,
                "validation_u_mse": validation_mse,
                "validation_relative_l2": validation_relative_l2,
            }
            if kind == "shared":
                metrics["train_gamma_mse"] = train_gamma / batches
                metrics["validation_gamma_mse"] = validation_gamma_mse
            history.append(metrics)
            if validation_relative_l2 < best_metric:
                best_metric = validation_relative_l2
                best_epoch = epoch_number
                best_params = jax.tree_util.tree_map(
                    lambda value: np.asarray(value).copy(), params
                )
                stale_epochs = 0
            else:
                stale_epochs += 1

            if progress_bar is not None:
                progress_bar.set_postfix(
                    train_loss=f"{metrics['train_loss']:.3e}",
                    validation_relative_l2=(
                        f"{metrics['validation_relative_l2']:.3e}"
                    ),
                    best=f"{best_metric:.3e}@{best_epoch}",
                    refresh=True,
                )

            stopped_early = (
                training_config.early_stopping
                and stale_epochs >= training_config.patience
            )
            is_last_epoch = epoch_number == training_config.epochs
            should_log = log_every > 0 and (
                epoch_number % log_every == 0
                or stopped_early
                or is_last_epoch
            )
            if should_log:
                message = (
                    f"[{kind} seed={seed}] epoch "
                    f"{epoch_number}/{training_config.epochs} | "
                    f"train_loss={metrics['train_loss']:.8e} | "
                    f"train_u_mse={metrics['train_u_mse']:.8e} | "
                    f"validation_u_mse={metrics['validation_u_mse']:.8e} | "
                    "validation_relative_l2="
                    f"{metrics['validation_relative_l2']:.8e} | "
                    f"best={best_metric:.8e}@{best_epoch}"
                )
                if kind == "shared":
                    message += (
                        f" | train_gamma_mse={metrics['train_gamma_mse']:.8e}"
                        " | validation_gamma_mse="
                        f"{metrics['validation_gamma_mse']:.8e}"
                    )
                if stopped_early:
                    message += (
                        f" | early_stop_after={stale_epochs}_stale_epochs"
                    )
                if progress_bar is not None:
                    progress_bar.write(message)
                else:
                    print(message, flush=True)

            if epoch_callback is not None:
                epoch_callback(
                    EpochReport(
                        epoch=epoch_number,
                        params=params,
                        metrics=dict(metrics),
                        history=tuple(dict(row) for row in history),
                        best_epoch=best_epoch,
                        best_validation_relative_l2=best_metric,
                        stale_epochs=stale_epochs,
                    )
                )

            if stopped_early:
                break
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return TrainResult(
        kind=kind,
        params=best_params,
        final_params=jax.tree_util.tree_map(
            lambda value: np.asarray(value).copy(), params
        ),
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_relative_l2=best_metric,
        stopped_early=stopped_early,
    )


def predict_dataset(
    kind: ModelKind,
    params: Params,
    dataset: DeepAdvectionDataset,
    normalizer: DataNormalizer,
    model_config: ModelConfig,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray | None]:
    prepared = prepare_data(dataset, normalizer, kind)
    forward = jax.jit(_make_forward(kind, model_config))
    predictions: list[np.ndarray] = []
    gamma_predictions: list[np.ndarray] = []
    for start in range(0, len(prepared), batch_size):
        stop = min(start + batch_size, len(prepared))
        output = forward(params, jnp.asarray(prepared.inputs[start:stop]))
        predictions.append(np.asarray(output["u"])[:, 0, :])
        if "gamma" in output:
            gamma_predictions.append(np.asarray(output["gamma"])[:, 0, :])
    prediction = normalizer.denormalize(np.concatenate(predictions, axis=0))
    gamma_prediction = (
        np.concatenate(gamma_predictions, axis=0) if gamma_predictions else None
    )
    return prediction, gamma_prediction
