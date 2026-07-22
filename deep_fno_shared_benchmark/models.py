"""Pure-JAX classic FNO and interface-conditioned Shared networks."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .config import ModelConfig


ModelKind = Literal[
    "fno",
    "shared",
    "fno_gamma",
    "shared_oracle",
    "shared_oracle_rho0",
]
Params = dict[str, object]


def _xavier_linear(
    key: jax.Array,
    out_channels: int,
    in_channels: int,
    *,
    bias: bool = True,
) -> dict[str, jax.Array]:
    limit = np.sqrt(6.0 / (in_channels + out_channels))
    weight = jax.random.uniform(
        key,
        (out_channels, in_channels),
        minval=-limit,
        maxval=limit,
        dtype=jnp.float32,
    )
    result = {"weight": weight}
    if bias:
        result["bias"] = jnp.zeros((out_channels,), dtype=jnp.float32)
    return result


def _complex_spectral_weight(
    key: jax.Array,
    out_channels: int,
    in_channels: int,
    n_modes: int,
) -> jax.Array:
    key_real, key_imag = jax.random.split(key)
    scale = 1.0 / np.sqrt(max(1, in_channels * out_channels))
    shape = (out_channels, in_channels, n_modes)
    real = scale * jax.random.normal(key_real, shape, dtype=jnp.float32)
    imag = scale * jax.random.normal(key_imag, shape, dtype=jnp.float32)
    return (real + 1j * imag).astype(jnp.complex64)


def _pointwise(value: jax.Array, linear: dict[str, jax.Array]) -> jax.Array:
    output = jnp.einsum("bix,oi->box", value, linear["weight"], optimize=True)
    if "bias" in linear:
        output = output + linear["bias"][None, :, None]
    return output


def spectral_convolution(
    value: jax.Array,
    weight: jax.Array,
    pad_fraction: float,
) -> jax.Array:
    """Apply a channel-mixing retained-mode rFFT convolution."""

    if value.ndim != 3:
        raise ValueError("value must have shape [batch, channels, n_x]")
    if weight.ndim != 3 or weight.shape[1] != value.shape[1]:
        raise ValueError("spectral weight has incompatible channel dimensions")
    n_x = int(value.shape[-1])
    pad = int(round(n_x * pad_fraction))
    if pad:
        padded = jnp.pad(value, ((0, 0), (0, 0), (pad, pad)), mode="edge")
    else:
        padded = value
    spectrum = jnp.fft.rfft(padded, axis=-1)
    n_modes = int(weight.shape[-1])
    if n_modes > spectrum.shape[-1]:
        raise ValueError(
            f"n_modes={n_modes} exceeds the {spectrum.shape[-1]} available rFFT modes"
        )
    retained = jnp.einsum(
        "bim,oim->bom", spectrum[..., :n_modes], weight, optimize=True
    )
    output_spectrum = jnp.zeros(
        (value.shape[0], weight.shape[0], spectrum.shape[-1]),
        dtype=spectrum.dtype,
    )
    output_spectrum = output_spectrum.at[..., :n_modes].set(retained)
    output = jnp.fft.irfft(output_spectrum, n=padded.shape[-1], axis=-1)
    if pad:
        output = output[..., pad : pad + n_x]
    return output


def shared_spectral_term(
    value: jax.Array,
    gamma_in: jax.Array,
    gamma_out: jax.Array,
    weight: jax.Array,
    rho: jax.Array,
    pad_fraction: float,
) -> jax.Array:
    """Structured term using one shared multiplier for both branches."""

    if gamma_in.ndim == 2:
        gamma_in = gamma_in[:, None, :]
    if gamma_out.ndim == 2:
        gamma_out = gamma_out[:, None, :]
    a = 0.5 * (1.0 + rho)
    b = 0.5 * (1.0 - rho)
    base = spectral_convolution(value, weight * a[..., None], pad_fraction)
    feature = spectral_convolution(
        gamma_in * value,
        weight * b[..., None],
        pad_fraction,
    )
    return base + gamma_out * feature


def _init_backbone(
    key: jax.Array,
    config: ModelConfig,
    *,
    input_channels: int = 3,
) -> Params:
    key_count = 4 + 2 * config.n_layers
    keys = iter(jax.random.split(key, key_count))
    lifting = [
        _xavier_linear(next(keys), config.lifting_channels, input_channels),
        _xavier_linear(
            next(keys), config.hidden_channels, config.lifting_channels
        ),
    ]
    blocks = []
    for _ in range(config.n_layers):
        blocks.append(
            {
                "spectral": _complex_spectral_weight(
                    next(keys),
                    config.hidden_channels,
                    config.hidden_channels,
                    config.n_modes,
                ),
                "pointwise": _xavier_linear(
                    next(keys),
                    config.hidden_channels,
                    config.hidden_channels,
                    bias=False,
                ),
            }
        )
    projection = [
        _xavier_linear(
            next(keys), config.projection_channels, config.hidden_channels
        ),
        _xavier_linear(next(keys), 1, config.projection_channels),
    ]
    return {"lifting": lifting, "blocks": blocks, "projection": projection}


def init_fno(key: jax.Array, config: ModelConfig) -> Params:
    return _init_backbone(key, config)


def init_fno_gamma(key: jax.Array, config: ModelConfig) -> Params:
    """Initialize the four-channel FNO+$\\gamma_T$ baseline."""

    return _init_backbone(key, config, input_channels=4)


def _attach_rho_parameters(params: Params, config: ModelConfig) -> Params:
    eta_initial = float(np.log(config.rho_initial / (1.0 - config.rho_initial)))
    for index, block in enumerate(params["blocks"]):
        block = dict(block)
        if config.rho_mode == "matrix":
            block["rho_eta"] = jnp.full(
                (config.hidden_channels, config.hidden_channels),
                eta_initial,
                dtype=jnp.float32,
            )
        elif config.rho_mode == "layer_scalar":
            block["rho_eta"] = jnp.asarray(eta_initial, dtype=jnp.float32)
        params["blocks"][index] = block
    return params


def init_shared(key: jax.Array, config: ModelConfig) -> Params:
    key_backbone, _ = jax.random.split(key)
    params = _attach_rho_parameters(_init_backbone(key_backbone, config), config)
    gamma_blocks: list[dict[str, jax.Array]] = []
    for _ in params["blocks"]:
        gamma_blocks.append(
            {
                "spectral": jnp.zeros(
                    (1, 1, config.n_modes), dtype=jnp.complex64
                ),
                "pointwise": {
                    "weight": jnp.zeros((1, 1), dtype=jnp.float32),
                },
                "bias": jnp.zeros((1,), dtype=jnp.float32),
            }
        )
    params["gamma_blocks"] = gamma_blocks
    return params


def init_shared_oracle(key: jax.Array, config: ModelConfig) -> Params:
    """Initialize v2 Oracle-Shared without a learned gamma recurrence."""

    params = _init_backbone(key, config, input_channels=4)
    return _attach_rho_parameters(params, config)


def init_shared_oracle_rho0(key: jax.Array, config: ModelConfig) -> Params:
    """Initialize Oracle-Shared with every channel-pair rho fixed to zero."""

    fixed_config = replace(config, rho_mode="fixed_zero")
    return init_shared_oracle(key, fixed_config)


def _apply_lifting(params: Params, inputs: jax.Array) -> jax.Array:
    hidden = jax.nn.gelu(_pointwise(inputs, params["lifting"][0]))
    return _pointwise(hidden, params["lifting"][1])


def _apply_projection(params: Params, hidden: jax.Array) -> jax.Array:
    projected = jax.nn.gelu(_pointwise(hidden, params["projection"][0]))
    return _pointwise(projected, params["projection"][1])


def apply_fno(params: Params, inputs: jax.Array, config: ModelConfig) -> dict[str, jax.Array]:
    hidden = _apply_lifting(params, inputs)
    for block in params["blocks"]:
        global_term = spectral_convolution(
            hidden, block["spectral"], config.pad_fraction
        )
        local_term = _pointwise(hidden, block["pointwise"])
        hidden = jax.nn.gelu(global_term + local_term)
    return {"u": _apply_projection(params, hidden)}


def _block_rho(block: dict[str, jax.Array], config: ModelConfig) -> jax.Array:
    if config.rho_mode == "fixed_zero":
        return jnp.asarray(0.0, dtype=jnp.float32)
    if config.rho_mode == "fixed_one":
        return jnp.asarray(1.0, dtype=jnp.float32)
    return jax.nn.sigmoid(block["rho_eta"])


def apply_shared(
    params: Params,
    inputs: jax.Array,
    config: ModelConfig,
    *,
    return_diagnostics: bool = False,
) -> dict[str, jax.Array | list[jax.Array]]:
    hidden = _apply_lifting(params, inputs)
    gamma_current = inputs[:, 2, :]
    magnitude = jnp.asarray(config.gamma_initial_magnitude, dtype=inputs.dtype)
    q = jnp.arctanh(jnp.clip(magnitude * gamma_current, -0.999999, 0.999999))
    gamma_layers: list[jax.Array] = [gamma_current]
    rho_layers: list[jax.Array] = []

    for block, gamma_block in zip(
        params["blocks"], params["gamma_blocks"], strict=True
    ):
        q_channel = q[:, None, :]
        gamma_residual = _pointwise(q_channel, gamma_block["pointwise"])
        gamma_residual += spectral_convolution(
            q_channel, gamma_block["spectral"], config.pad_fraction
        )
        gamma_residual += gamma_block["bias"][None, :, None]
        q = q + gamma_residual[:, 0, :]
        gamma_next = jnp.tanh(q)
        rho = _block_rho(block, config)
        structured = shared_spectral_term(
            hidden,
            gamma_current,
            gamma_next,
            block["spectral"],
            rho,
            config.pad_fraction,
        )
        hidden = jax.nn.gelu(structured + _pointwise(hidden, block["pointwise"]))
        gamma_current = gamma_next
        gamma_layers.append(gamma_current)
        rho_layers.append(rho)

    result: dict[str, jax.Array | list[jax.Array]] = {
        "u": _apply_projection(params, hidden),
        "gamma": gamma_current[:, None, :],
    }
    if return_diagnostics:
        result["gamma_layers"] = gamma_layers
        result["rho_layers"] = rho_layers
    return result


def apply_shared_oracle(
    params: Params,
    inputs: jax.Array,
    config: ModelConfig,
    *,
    return_diagnostics: bool = False,
) -> dict[str, jax.Array | list[jax.Array]]:
    """Apply fixed gates ``[gamma0, gammaT, ..., gammaT]`` to all blocks."""

    if inputs.ndim != 3 or inputs.shape[1] != 4:
        raise ValueError("oracle-shared models expect [batch, 4, n_x] inputs")
    if "gamma_blocks" in params:
        raise ValueError("oracle-shared parameter trees must not contain gamma_blocks")
    hidden = _apply_lifting(params, inputs)
    gamma0 = inputs[:, 2, :]
    gamma_target = inputs[:, 3, :]
    gamma_layers: list[jax.Array] = [gamma0]
    rho_layers: list[jax.Array] = []

    gamma_current = gamma0
    for block in params["blocks"]:
        gamma_next = gamma_target
        rho = _block_rho(block, config)
        structured = shared_spectral_term(
            hidden,
            gamma_current,
            gamma_next,
            block["spectral"],
            rho,
            config.pad_fraction,
        )
        hidden = jax.nn.gelu(structured + _pointwise(hidden, block["pointwise"]))
        gamma_current = gamma_next
        gamma_layers.append(gamma_current)
        rho_layers.append(rho)

    result: dict[str, jax.Array | list[jax.Array]] = {
        "u": _apply_projection(params, hidden)
    }
    if return_diagnostics:
        result["gamma_layers"] = gamma_layers
        result["rho_layers"] = rho_layers
    return result


def apply_model(
    kind: ModelKind,
    params: Params,
    inputs: jax.Array,
    config: ModelConfig,
    *,
    return_diagnostics: bool = False,
) -> dict[str, jax.Array | list[jax.Array]]:
    if kind == "fno":
        if return_diagnostics:
            return apply_fno(params, inputs, config)
        return apply_fno(params, inputs, config)
    if kind == "fno_gamma":
        return apply_fno(params, inputs, config)
    if kind == "shared":
        return apply_shared(
            params, inputs, config, return_diagnostics=return_diagnostics
        )
    if kind == "shared_oracle":
        return apply_shared_oracle(
            params, inputs, config, return_diagnostics=return_diagnostics
        )
    if kind == "shared_oracle_rho0":
        fixed_config = replace(config, rho_mode="fixed_zero")
        return apply_shared_oracle(
            params,
            inputs,
            fixed_config,
            return_diagnostics=return_diagnostics,
        )
    raise ValueError(f"unknown model kind: {kind}")


def init_model(kind: ModelKind, key: jax.Array, config: ModelConfig) -> Params:
    if kind == "fno":
        return init_fno(key, config)
    if kind == "fno_gamma":
        return init_fno_gamma(key, config)
    if kind == "shared":
        return init_shared(key, config)
    if kind == "shared_oracle":
        return init_shared_oracle(key, config)
    if kind == "shared_oracle_rho0":
        return init_shared_oracle_rho0(key, config)
    raise ValueError(f"unknown model kind: {kind}")


def parameter_count(params: Params) -> int:
    """Count real scalar degrees of freedom, treating complex values as two."""

    total = 0
    for leaf in jax.tree_util.tree_leaves(params):
        size = int(np.prod(leaf.shape))
        total += 2 * size if jnp.issubdtype(leaf.dtype, jnp.complexfloating) else size
    return total


def rho_values(params: Params, config: ModelConfig) -> list[np.ndarray]:
    if "blocks" not in params:
        return []
    values: list[np.ndarray] = []
    for block in params["blocks"]:
        value = np.asarray(_block_rho(block, config), dtype=np.float32)
        if value.ndim == 0:
            value = np.full(
                (config.hidden_channels, config.hidden_channels),
                float(value),
                dtype=np.float32,
            )
        values.append(value)
    return values
