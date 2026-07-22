"""Shared formal-training entry points used by the v2 model notebooks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .experiment import V2_ALLOWED_MODELS, main as experiment_main


FORMAL_SEEDS = (0, 1, 2, 3, 4)
FORMAL_LEARNING_RATE = 1.0e-3
FORMAL_WEIGHT_DECAY = 1.0e-4
FIXED_HYPERPARAMETERS = {
    "learning_rate": FORMAL_LEARNING_RATE,
    "weight_decay": FORMAL_WEIGHT_DECAY,
}


def find_repository_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "deep_fno_shared_benchmark" / "experiment.py").is_file():
            return candidate
    raise FileNotFoundError("无法定位包含 deep_fno_shared_benchmark 的仓库根目录")


def load_locked_hyperparameters(repository_root: str | Path) -> dict[str, object]:
    """Load the optional development-screen result for sensitivity analysis."""

    path = (
        Path(repository_root)
        / "deep_fno_shared_benchmark"
        / "results"
        / "hyperparameter_screen_v2"
        / "hyperparameters_manifest_v2.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "正式训练前必须先完成公共超参数筛选；缺少 " + str(path)
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != 2:
        raise ValueError("超参数 manifest 的 protocol_version 不是 2")
    if manifest.get("selection_status") != "locked":
        raise ValueError("超参数 manifest 尚未锁定")
    selected = manifest.get("selected_hyperparameters", {})
    if set(selected) != {"learning_rate", "weight_decay"}:
        raise ValueError("超参数 manifest 缺少学习率或权重衰减")
    return manifest


def ensure_fixed_hyperparameter_manifest(
    repository_root: str | Path,
) -> tuple[dict[str, object], Path]:
    """Create or validate the immutable user-fixed formal-v2 settings."""

    path = (
        Path(repository_root)
        / "deep_fno_shared_benchmark"
        / "results"
        / "formal_main_v2"
        / "fixed_hyperparameters_manifest_v2.json"
    )
    manifest: dict[str, object] = {
        "protocol_version": 2,
        "run_tag": "main_v2",
        "selection_status": "locked",
        "selection_source": "user_fixed_before_formal_training",
        "selected_hyperparameters": dict(FIXED_HYPERPARAMETERS),
        "hyperparameter_screen_required": False,
    }
    serialized = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                "既有正式 v2 固定超参数 manifest 与当前协议不一致：" + str(path)
            )
    return manifest, path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _is_complete(
    run_output: Path,
    model: str,
    seed: int,
    epochs: int,
    expected_hyperparameters: dict[str, float],
) -> bool:
    run = run_output / f"seed_{seed}" / model
    required = (
        run_output / "manifest.json",
        run / "history.csv",
        run / "checkpoint_best.npz",
        run / "checkpoint_final.npz",
        run / "predictions_n256.npz",
        run / "predictions_n512.npz",
        run / "predictions_n1024.npz",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    manifest = json.loads((run_output / "manifest.json").read_text(encoding="utf-8"))
    if model == "shared_oracle_rho0":
        if manifest.get("model_config", {}).get("rho_mode") != "fixed_zero":
            return False
        if (
            manifest.get("rho_policy", {}).get(model)
            != "fixed_zero_non_trainable"
        ):
            return False
    training = manifest.get("training_config", {})
    if any(
        float(training.get(key, float("nan"))) != value
        for key, value in expected_hyperparameters.items()
    ):
        return False
    return len(_read_rows(run / "history.csv")) == epochs


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_formal_model(
    model: str,
    *,
    repository_root: str | Path | None = None,
    seeds: tuple[int, ...] = FORMAL_SEEDS,
    epochs: int = 500,
    progress: bool = True,
) -> Path:
    if model not in V2_ALLOWED_MODELS:
        raise ValueError(f"v2 formal model must be one of {V2_ALLOWED_MODELS}")
    if epochs != 500:
        raise ValueError("registered v2 formal training requires exactly 500 epochs")
    root = find_repository_root(repository_root)
    project = root / "deep_fno_shared_benchmark"
    hyperparameter_manifest, hyperparameter_path = (
        ensure_fixed_hyperparameter_manifest(root)
    )
    selected = dict(hyperparameter_manifest["selected_hyperparameters"])
    result_root = project / "results" / "formal_main_v2" / model
    result_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        seed_output = result_root / f"seed_{seed}_run"
        if _is_complete(seed_output, model, seed, epochs, selected):
            print(f"跳过已完整运行：{model} seed={seed}")
            continue
        if seed_output.exists() and any(seed_output.iterdir()):
            raise RuntimeError(
                f"检测到不完整的既有目录，拒绝覆盖：{seed_output}"
            )
        arguments = [
            "--preset",
            "main_v2",
            "--models",
            model,
            "--seeds",
            str(seed),
            "--epochs",
            "500",
            "--learning-rate",
            str(selected["learning_rate"]),
            "--weight-decay",
            str(selected["weight_decay"]),
            "--checkpoint-every",
            "10",
            "--log-every",
            "10",
            "--bootstrap-repetitions",
            "10000",
            "--run-tag",
            "main_v2",
            "--output-dir",
            str(seed_output),
        ]
        if model == "shared_oracle_rho0":
            arguments.extend(["--rho-mode", "fixed_zero"])
        if progress:
            arguments.append("--progress")
        status = experiment_main(arguments)
        if status != 0 or not _is_complete(
            seed_output, model, seed, epochs, selected
        ):
            raise RuntimeError(f"正式训练或产物审计失败：{model} seed={seed}")

    summary_rows: list[dict[str, str]] = []
    for seed in seeds:
        summary_rows.extend(
            _read_rows(result_root / f"seed_{seed}_run" / "summary.csv")
        )
    _write_csv(result_root / "summary_all_seeds.csv", summary_rows)

    aggregate_manifest = {
        "protocol_version": 2,
        "run_tag": "main_v2",
        "model": model,
        "seeds": list(seeds),
        "epochs": epochs,
        "training_resolution": 256,
        "inference_resolutions": [256, 512, 1024],
        "sizes": {"train": 16000, "validation": 2000, "test": 4000},
        "selected_hyperparameters": selected,
        "hyperparameter_selection_source": hyperparameter_manifest[
            "selection_source"
        ],
        "hyperparameter_manifest_sha256": hashlib.sha256(
            hyperparameter_path.read_bytes()
        ).hexdigest(),
        "checkpoint_for_inference": "checkpoint_best.npz",
        "periodic_checkpoint_every": 10,
        "physical_gpu_6_excluded": True,
        "rho_policy": (
            "fixed_zero_non_trainable"
            if model == "shared_oracle_rho0"
            else "registered_model_default"
        ),
    }
    (result_root / "aggregate_manifest.json").write_text(
        json.dumps(aggregate_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return result_root
