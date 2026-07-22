"""Portable NPZ checkpoints for nested JAX parameter dictionaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np


PathPart = str | int


def _flatten(
    value: Any,
    prefix: tuple[PathPart, ...] = (),
) -> list[tuple[tuple[PathPart, ...], np.ndarray]]:
    if isinstance(value, dict):
        result: list[tuple[tuple[PathPart, ...], np.ndarray]] = []
        for key in sorted(value):
            result.extend(_flatten(value[key], prefix + (key,)))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            result.extend(_flatten(item, prefix + (index,)))
        return result
    return [(prefix, np.asarray(value))]


def _path_text(path: tuple[PathPart, ...]) -> str:
    return json.dumps(path, ensure_ascii=False, separators=(",", ":"))


def _replace_from_mapping(
    template: Any,
    mapping: dict[str, np.ndarray],
    prefix: tuple[PathPart, ...] = (),
) -> Any:
    if isinstance(template, dict):
        return {
            key: _replace_from_mapping(template[key], mapping, prefix + (key,))
            for key in sorted(template)
        }
    if isinstance(template, list):
        return [
            _replace_from_mapping(item, mapping, prefix + (index,))
            for index, item in enumerate(template)
        ]
    if isinstance(template, tuple):
        return tuple(
            _replace_from_mapping(item, mapping, prefix + (index,))
            for index, item in enumerate(template)
        )
    key = _path_text(prefix)
    if key not in mapping:
        raise ValueError(f"checkpoint is missing parameter {key}")
    return jnp.asarray(mapping[key])


def save_checkpoint(
    path: str | Path,
    params: dict[str, object],
    metadata: dict[str, object] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = _flatten(params)
    payload: dict[str, np.ndarray] = {
        f"array_{index:05d}": array
        for index, (_, array) in enumerate(flattened)
    }
    payload["paths"] = np.asarray([_path_text(item[0]) for item in flattened])
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
    )
    np.savez_compressed(path, **payload)


def load_checkpoint(
    path: str | Path,
    template: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    with np.load(path, allow_pickle=False) as saved:
        paths = [str(item) for item in saved["paths"]]
        mapping = {
            path_text: np.asarray(saved[f"array_{index:05d}"])
            for index, path_text in enumerate(paths)
        }
        metadata = json.loads(str(saved["metadata_json"]))
    expected = {_path_text(path) for path, _ in _flatten(template)}
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise ValueError(f"checkpoint structure mismatch; missing={missing}, extra={extra}")
    return _replace_from_mapping(template, mapping), metadata
