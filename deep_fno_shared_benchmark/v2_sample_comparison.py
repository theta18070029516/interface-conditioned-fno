"""Sample-wise v2 comparison utilities used by the companion notebook.

All inference is performed from the registered continuous cases and each run's
``checkpoint_best.npz``.  No interpolation of the N=256 arrays is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Mapping
import hashlib
import json
from typing import Any

import numpy as np

from .checkpoint import load_checkpoint
from .config import ModelConfig
from .data import DataNormalizer, DeepAdvectionDataset, load_cases, sample_cases
from .models import ModelKind, init_model
from .training import predict_dataset


MODEL_KINDS: tuple[ModelKind, ...] = (
    "fno",
    "fno_gamma",
    "shared_oracle",
    "shared_oracle_rho0",
)
MODEL_LABELS: dict[ModelKind, str] = {
    "fno": "FNO",
    "fno_gamma": r"FNO+$\gamma_T$",
    "shared_oracle": "Oracle-Shared",
    "shared_oracle_rho0": r"Oracle-Shared ($\rho=0$)",
}
MODEL_COLORS: dict[ModelKind, str] = {
    "fno": "#0072B2",
    "fno_gamma": "#E69F00",
    "shared_oracle": "#009E73",
    "shared_oracle_rho0": "#CC79A7",
}


@dataclass(frozen=True)
class SampleFields:
    """One continuous case evaluated on one grid."""

    x: np.ndarray
    u0: np.ndarray
    uT: np.ndarray
    gamma0: np.ndarray
    gammaT: np.ndarray
    sample_id: int
    xi0: float
    xiT: float
    jump: float


@dataclass(frozen=True)
class ModelRun:
    """Validated best-checkpoint bundle for one model and one seed."""

    kind: ModelKind
    seed: int
    run_dir: Path
    manifest_path: Path
    train_cases_path: Path
    test_cases_path: Path
    checkpoint_path: Path
    params: dict[str, object]
    model_config: ModelConfig
    normalizer: DataNormalizer
    best_epoch: int


@dataclass(frozen=True)
class ResolutionComparison:
    """Truth, four predictions, and metrics at one resolution."""

    split: str
    sample_index: int
    resolution: int
    sample: SampleFields
    predictions: dict[ModelKind, np.ndarray]
    metrics: tuple[dict[str, float | int | str], ...]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON 顶层必须是对象：{path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_model_config(manifest: Mapping[str, Any]) -> ModelConfig:
    raw = manifest.get("model_config")
    if not isinstance(raw, Mapping):
        raise KeyError("manifest 缺少 model_config")
    return ModelConfig(**dict(raw))


def load_model_run(results_root: str | Path, kind: ModelKind, seed: int) -> ModelRun:
    """Load exactly the formal v2 best checkpoint for ``kind`` and ``seed``."""

    if kind not in MODEL_KINDS:
        raise ValueError(f"v2 不支持模型类型：{kind}")
    root = Path(results_root).expanduser().resolve()
    outer_run_dir = root / kind / f"seed_{seed}_run"
    run_dir = outer_run_dir / f"seed_{seed}" / kind
    manifest_path = outer_run_dir / "manifest.json"
    checkpoint_path = run_dir / "checkpoint_best.npz"
    train_cases_path = outer_run_dir / "datasets" / f"seed_{seed}_train_cases.npz"
    test_cases_path = outer_run_dir / "datasets" / "test_cases.npz"
    required = (
        manifest_path,
        checkpoint_path,
        train_cases_path,
        test_cases_path,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"正式 v2 产物不完整：{missing}")

    manifest = _read_json(manifest_path)
    if int(manifest.get("protocol_version", -1)) != 2:
        raise AssertionError(f"不是 protocol v2：{manifest_path}")
    if manifest.get("preset") != "main_v2":
        raise AssertionError(f"不是 main_v2 正式运行：{manifest_path}")
    if list(manifest.get("models", [])) != [kind]:
        raise AssertionError(f"manifest 模型与目录不一致：{manifest_path}")
    if list(manifest.get("seeds", [])) != [seed]:
        raise AssertionError(f"manifest seed 与目录不一致：{manifest_path}")
    if manifest.get("checkpoint_policy", {}).get("formal_inference") != "checkpoint_best.npz":
        raise AssertionError("正式推理策略不是 checkpoint_best.npz")

    import jax

    model_config = _build_model_config(manifest)
    template = init_model(kind, jax.random.PRNGKey(seed), model_config)
    params, metadata = load_checkpoint(checkpoint_path, template)
    if metadata.get("checkpoint_type") != "best_validation_relative_l2":
        raise AssertionError(f"checkpoint 类型不正确：{checkpoint_path}")
    if metadata.get("kind") != kind or int(metadata.get("seed", -1)) != seed:
        raise AssertionError(f"checkpoint 元数据与请求不一致：{checkpoint_path}")
    normalizer_raw = metadata.get("normalizer")
    if not isinstance(normalizer_raw, Mapping):
        raise KeyError(f"checkpoint 缺少 normalizer：{checkpoint_path}")
    normalizer = DataNormalizer(
        mean=float(normalizer_raw["mean"]),
        std=float(normalizer_raw["std"]),
    )
    if not np.isfinite(normalizer.mean) or not np.isfinite(normalizer.std) or normalizer.std <= 0:
        raise ValueError(f"normalizer 非法：{normalizer}")
    return ModelRun(
        kind=kind,
        seed=seed,
        run_dir=run_dir,
        manifest_path=manifest_path,
        train_cases_path=train_cases_path,
        test_cases_path=test_cases_path,
        checkpoint_path=checkpoint_path,
        params=params,
        model_config=model_config,
        normalizer=normalizer,
        best_epoch=int(metadata["best_epoch"]),
    )


def _cases_path(run: ModelRun, split: str) -> Path:
    if split == "train":
        return run.train_cases_path
    if split == "test":
        return run.test_cases_path
    raise ValueError("split 必须是 'train' 或 'test'")


def _validate_matched_runs(runs: Mapping[ModelKind, ModelRun], split: str) -> None:
    if set(runs) != set(MODEL_KINDS):
        raise AssertionError("必须同时提供四种 v2 模型")
    seeds = {run.seed for run in runs.values()}
    if len(seeds) != 1:
        raise AssertionError(f"四模型 seed 不一致：{seeds}")
    normalizers = {
        (round(run.normalizer.mean, 12), round(run.normalizer.std, 12))
        for run in runs.values()
    }
    if len(normalizers) != 1:
        raise AssertionError(f"同一种子的四模型 normalizer 不一致：{normalizers}")
    configs = set()
    for run in runs.values():
        config = run.model_config.to_dict()
        config.pop("rho_mode")
        configs.add(json.dumps(config, sort_keys=True))
    if len(configs) != 1:
        raise AssertionError("四模型除 rho_mode 外的 model_config 不一致")
    if runs["shared_oracle_rho0"].model_config.rho_mode != "fixed_zero":
        raise AssertionError("固定 rho=0 模型的 rho_mode 不是 fixed_zero")
    case_hashes = {_file_sha256(_cases_path(run, split)) for run in runs.values()}
    if len(case_hashes) != 1:
        raise AssertionError(f"四模型的 {split} 连续样本文件不一致")


def _one_sample_dataset(cases_path: Path, sample_index: int, resolution: int) -> DeepAdvectionDataset:
    cases = load_cases(cases_path)
    if not 0 <= sample_index < len(cases):
        raise IndexError(f"样本索引 {sample_index} 超出 [0, {len(cases) - 1}]")
    selected_cases = cases.subset(np.asarray([sample_index], dtype=np.int64))
    return sample_cases(selected_cases, int(resolution))


def _sample_fields(dataset: DeepAdvectionDataset) -> SampleFields:
    cases = dataset.cases
    xi0 = float(cases.interface_location[0])
    return SampleFields(
        x=np.asarray(dataset.x, dtype=np.float64),
        u0=np.asarray(dataset.u0[0], dtype=np.float64),
        uT=np.asarray(dataset.u_target[0], dtype=np.float64),
        gamma0=np.asarray(dataset.gamma0[0], dtype=np.float64),
        gammaT=np.asarray(dataset.gamma_target[0], dtype=np.float64),
        sample_id=int(cases.sample_id[0]),
        xi0=xi0,
        xiT=xi0 + cases.config.displacement,
        jump=float(cases.jump[0]),
    )


def _metric_rows(
    split: str,
    sample_index: int,
    resolution: int,
    sample: SampleFields,
    predictions: Mapping[ModelKind, np.ndarray],
    interface_half_width: float,
) -> tuple[dict[str, Any], ...]:
    target = sample.uT
    target_norm = max(float(np.linalg.norm(target)), np.finfo(np.float64).eps)
    interface_mask = np.abs(sample.x - sample.xiT) <= interface_half_width
    if not np.any(interface_mask):
        interface_mask[np.argmin(np.abs(sample.x - sample.xiT))] = True
    rows: list[dict[str, Any]] = []
    for kind in MODEL_KINDS:
        residual = np.asarray(predictions[kind], dtype=np.float64) - target
        rows.append(
            {
                "split": split,
                "sample_index": sample_index,
                "sample_id": sample.sample_id,
                "resolution": resolution,
                "model_kind": kind,
                "model": MODEL_LABELS[kind],
                "mse": float(np.mean(residual**2)),
                "relative_l2": float(np.linalg.norm(residual) / target_norm),
                "mae": float(np.mean(np.abs(residual))),
                "interface_mae": float(np.mean(np.abs(residual[interface_mask]))),
                "max_abs_error": float(np.max(np.abs(residual))),
            }
        )
    return tuple(rows)


def build_sample_comparison(
    *,
    results_root: str | Path,
    seed: int,
    split: str,
    sample_index: int,
    resolutions: Iterable[int] = (256, 512, 1024),
    interface_half_width: float = 0.04,
) -> tuple[dict[ModelKind, ModelRun], list[ResolutionComparison]]:
    """Run matched best-checkpoint inference for one sample at all resolutions."""

    runs = {kind: load_model_run(results_root, kind, seed) for kind in MODEL_KINDS}
    _validate_matched_runs(runs, split)
    reference_cases_path = _cases_path(runs["fno"], split)
    comparisons: list[ResolutionComparison] = []
    expected_sample_id: int | None = None
    for raw_resolution in resolutions:
        resolution = int(raw_resolution)
        dataset = _one_sample_dataset(reference_cases_path, sample_index, resolution)
        sample = _sample_fields(dataset)
        if expected_sample_id is None:
            expected_sample_id = sample.sample_id
        elif sample.sample_id != expected_sample_id:
            raise AssertionError("跨分辨率 sample_id 不一致")
        predictions: dict[ModelKind, np.ndarray] = {}
        for kind in MODEL_KINDS:
            prediction, gamma_prediction = predict_dataset(
                kind,
                runs[kind].params,
                dataset,
                runs[kind].normalizer,
                runs[kind].model_config,
                batch_size=1,
            )
            if gamma_prediction is not None:
                raise AssertionError(f"v2 模型 {kind} 不应输出预测 gamma")
            prediction = np.asarray(prediction[0], dtype=np.float64)
            if prediction.shape != (resolution,) or not np.all(np.isfinite(prediction)):
                raise ValueError(f"{kind} 在 N={resolution} 的预测无效")
            predictions[kind] = prediction
        comparisons.append(
            ResolutionComparison(
                split=split,
                sample_index=sample_index,
                resolution=resolution,
                sample=sample,
                predictions=predictions,
                metrics=_metric_rows(
                    split,
                    sample_index,
                    resolution,
                    sample,
                    predictions,
                    interface_half_width,
                ),
            )
        )
    return runs, comparisons


def metrics_frame(comparisons: Iterable[ResolutionComparison]):
    import pandas as pd

    rows = [row for comparison in comparisons for row in comparison.metrics]
    return pd.DataFrame(rows).sort_values(["resolution", "model_kind"]).reset_index(drop=True)


def source_data_frame(comparisons: Iterable[ResolutionComparison]):
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        sample = comparison.sample
        for grid_index, x_value in enumerate(sample.x):
            row: dict[str, Any] = {
                "split": comparison.split,
                "sample_index": comparison.sample_index,
                "sample_id": sample.sample_id,
                "resolution": comparison.resolution,
                "grid_index": grid_index,
                "x": float(x_value),
                "u0": float(sample.u0[grid_index]),
                "uT_exact": float(sample.uT[grid_index]),
                "gamma0": float(sample.gamma0[grid_index]),
                "gammaT": float(sample.gammaT[grid_index]),
                "xi0": sample.xi0,
                "xiT": sample.xiT,
                "jump": sample.jump,
            }
            for kind in MODEL_KINDS:
                prediction = float(comparison.predictions[kind][grid_index])
                row[f"{kind}_prediction"] = prediction
                row[f"{kind}_absolute_error"] = abs(prediction - float(sample.uT[grid_index]))
            rows.append(row)
    return pd.DataFrame(rows)


def configure_research_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "lines.linewidth": 1.1,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def plot_sample_comparison(
    comparisons: Iterable[ResolutionComparison],
    *,
    zoom_half_width: float = 0.12,
):
    """Create a double-column quantitative grid with full-domain and interface views."""

    import matplotlib.pyplot as plt

    configure_research_style()
    comparisons = list(comparisons)
    if not comparisons:
        raise ValueError("comparisons 不能为空")
    figure, axes = plt.subplots(
        len(comparisons),
        2,
        figsize=(7.2, 2.05 * len(comparisons)),
        gridspec_kw={"width_ratios": (1.55, 1.0)},
        squeeze=False,
    )
    plotted_values = [
        values
        for comparison in comparisons
        for values in (
            comparison.sample.u0,
            comparison.sample.uT,
            *(comparison.predictions[kind] for kind in MODEL_KINDS),
        )
    ]
    global_min = min(float(np.min(values)) for values in plotted_values)
    global_max = max(float(np.max(values)) for values in plotted_values)
    y_margin = 0.04 * max(global_max - global_min, np.finfo(np.float64).eps)
    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    for row_index, comparison in enumerate(comparisons):
        sample = comparison.sample
        for column_index, axis in enumerate(axes[row_index]):
            axis.plot(sample.x, sample.uT, color="black", linewidth=1.6, label="Exact")
            axis.plot(
                sample.x,
                sample.u0,
                color="#8A8A8A",
                linewidth=0.8,
                linestyle=":",
                label=r"Initial $u_0$",
            )
            for kind in MODEL_KINDS:
                axis.plot(
                    sample.x,
                    comparison.predictions[kind],
                    color=MODEL_COLORS[kind],
                    linewidth=1.0,
                    label=MODEL_LABELS[kind],
                )
            axis.axvline(
                sample.xiT,
                color="#666666",
                linewidth=0.8,
                linestyle="--",
                zorder=0,
            )
            axis.text(
                0.015,
                0.98,
                panel_letters[2 * row_index + column_index],
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                fontweight="bold",
            )
            axis.tick_params(direction="in", length=3)
            axis.set_ylim(global_min - y_margin, global_max + y_margin)
            axis.set_ylabel(r"$u(x,T)$")
            if row_index == len(comparisons) - 1:
                axis.set_xlabel(r"$x$")
        axes[row_index, 0].set_title(f"N={comparison.resolution}: full domain")
        axes[row_index, 1].set_title(f"N={comparison.resolution}: interface zoom")
        axes[row_index, 1].set_xlim(
            sample.xiT - zoom_half_width,
            sample.xiT + zoom_half_width,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=6,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.955), h_pad=1.0, w_pad=1.0)
    return figure


def export_comparison(
    comparisons: Iterable[ResolutionComparison],
    output_dir: str | Path,
    *,
    stem: str,
    figure: Any | None = None,
) -> dict[str, Path]:
    """Export metrics, source data, editable vectors, rasters, and QA metadata."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = list(comparisons)
    paths: dict[str, Path] = {
        "metrics_csv": output_dir / f"{stem}_metrics.csv",
        "source_data_csv": output_dir / f"{stem}_source_data.csv",
    }
    metrics_frame(comparisons).to_csv(paths["metrics_csv"], index=False)
    source_data_frame(comparisons).to_csv(paths["source_data_csv"], index=False)
    if figure is not None:
        paths.update(
            {
                "figure_svg": output_dir / f"{stem}.svg",
                "figure_pdf": output_dir / f"{stem}.pdf",
                "figure_tiff": output_dir / f"{stem}.tiff",
                "figure_png": output_dir / f"{stem}.png",
            }
        )
        figure.savefig(paths["figure_svg"])
        figure.savefig(paths["figure_pdf"])
        figure.savefig(paths["figure_tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        figure.savefig(paths["figure_png"], dpi=400)
        svg_text = paths["figure_svg"].read_text(encoding="utf-8")
        qa = {
            "backend": "python_matplotlib",
            "figure_width_inches": float(figure.get_size_inches()[0]),
            "figure_height_inches": float(figure.get_size_inches()[1]),
            "svg_contains_editable_text": "<text" in svg_text,
            "exports": {
                key: {"path": str(path), "bytes": path.stat().st_size}
                for key, path in paths.items()
                if key.startswith("figure_")
            },
        }
        if not qa["svg_contains_editable_text"]:
            raise AssertionError("SVG 文字未保持可编辑状态")
        paths["figure_qa_json"] = output_dir / f"{stem}_figure_qa.json"
        paths["figure_qa_json"].write_text(
            json.dumps(qa, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return paths
