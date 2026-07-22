"""GPU-only protocol-v2 smoke test; select a physical GPU before importing JAX."""

from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np


EXCLUDED_PHYSICAL_GPUS = {6}


def _available_physical_gpus() -> set[int]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    result: set[int] = set()
    for raw_line in completed.stdout.splitlines():
        value = raw_line.strip()
        if value:
            result.add(int(value))
    return result


def run(physical_gpu: int) -> None:
    if physical_gpu in EXCLUDED_PHYSICAL_GPUS:
        raise ValueError("physical GPU 6 is permanently excluded")
    available = _available_physical_gpus()
    if physical_gpu not in available:
        raise ValueError(
            f"physical GPU {physical_gpu} is unavailable; detected={sorted(available)}"
        )
    if "jax" in globals():
        raise RuntimeError("JAX was imported before CUDA_VISIBLE_DEVICES was set")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp

    from deep_fno_shared_benchmark.config import ModelConfig
    from deep_fno_shared_benchmark.models import (
        apply_model,
        init_model,
        shared_spectral_term,
        spectral_convolution,
    )

    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"GPU backend required, got {jax.default_backend()}; devices={jax.devices()}"
        )
    if len(jax.devices()) != 1:
        raise RuntimeError(
            f"expected exactly one visible logical GPU, got {jax.devices()}"
        )

    config = ModelConfig(
        n_layers=2,
        hidden_channels=4,
        n_modes=3,
        lifting_channels=8,
        projection_channels=8,
    )
    key = jax.random.PRNGKey(20260721)
    value = jax.random.normal(key, (2, 4, 32))
    weight = jax.random.normal(key, (4, 4, 3)).astype(jnp.complex64)
    gamma = jnp.ones((2, 32), dtype=jnp.float32)
    plain = spectral_convolution(value, weight, config.pad_fraction)
    plain_np = np.asarray(plain, dtype=np.float64)
    plain_norm = max(float(np.linalg.norm(plain_np.ravel())), 1.0e-12)
    plain_scale = max(float(np.max(np.abs(plain_np))), 1.0)
    for rho in (0.0, 0.25, 0.8, 1.0):
        shared = shared_spectral_term(
            value,
            gamma,
            gamma,
            weight,
            jnp.asarray(rho, dtype=jnp.float32),
            config.pad_fraction,
        )
        residual = np.asarray(shared, dtype=np.float64) - plain_np
        relative_l2 = float(np.linalg.norm(residual.ravel()) / plain_norm)
        scaled_max = float(np.max(np.abs(residual)) / plain_scale)
        if relative_l2 >= 5.0e-4 or scaled_max >= 5.0e-4:
            raise AssertionError(
                f"constant-gamma identity failed at rho={rho}: "
                f"relative_l2={relative_l2:.8e}, scaled_max={scaled_max:.8e}"
            )

    rng = np.random.default_rng(20260721)
    inputs3 = rng.normal(size=(2, 3, 64)).astype(np.float32)
    inputs4 = rng.normal(size=(2, 4, 64)).astype(np.float32)
    inputs4[:, 2:] = rng.choice([-1.0, 1.0], size=(2, 2, 64))
    for model_index, (kind, inputs) in enumerate(
        (
            ("fno", inputs3),
            ("fno_gamma", inputs4),
            ("shared_oracle", inputs4),
            ("shared_oracle_rho0", inputs4),
        )
    ):
        params = init_model(kind, jax.random.fold_in(key, model_index), config)
        if kind == "shared_oracle_rho0":
            if any("rho_eta" in block for block in params["blocks"]):
                raise AssertionError("fixed-rho model contains trainable rho parameters")
            diagnostics = apply_model(
                kind,
                params,
                jnp.asarray(inputs),
                config,
                return_diagnostics=True,
            )
            if any(
                float(np.max(np.abs(np.asarray(rho)))) != 0.0
                for rho in diagnostics["rho_layers"]
            ):
                raise AssertionError("fixed-rho model did not use rho=0")

        def objective(current):
            output = apply_model(kind, current, jnp.asarray(inputs), config)["u"]
            return jnp.mean(output**2)

        loss, gradients = jax.value_and_grad(objective)(params)
        if not np.isfinite(float(loss)):
            raise AssertionError(f"{kind} loss is non-finite")
        for leaf in jax.tree_util.tree_leaves(gradients):
            if not np.all(np.isfinite(np.asarray(leaf))):
                raise AssertionError(f"{kind} has non-finite gradients")
        fine = np.resize(inputs, (2, inputs.shape[1], 128)).astype(np.float32)
        prediction = apply_model(kind, params, jnp.asarray(fine), config)["u"]
        if prediction.shape != (2, 1, 128):
            raise AssertionError(f"{kind} super-resolution shape is invalid")

    print(f"PASS: protocol-v2 GPU smoke test on physical GPU {physical_gpu}")
    print(f"backend={jax.default_backend()} devices={jax.devices()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args(argv)
    run(args.physical_gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
