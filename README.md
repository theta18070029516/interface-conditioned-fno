# Interface-Conditioned Fourier Neural Operators

Structured linear Fourier/Green operators for learning PDE solution maps with known interfaces or discontinuity features.

> **Project status:** this repository is an early, controlled validation of the method. The current 1D advection study provides mechanism-level evidence, not a final claim of performance across PDE classes.

## Core idea

Classical Fourier neural operators use a translation-invariant spectral kernel. A known discontinuity feature can instead condition the kernel at both its source and target points:

$$
G_\gamma(x,y)=\kappa_0(x-y)+\gamma_{\mathrm{out}}(x)\kappa_1(x-y)\gamma_{\mathrm{in}}(y).
$$

The corresponding single-layer linear operator is

$$
\mathcal T_\gamma f
=\mathcal K_0f+\gamma_{\mathrm{out}}\mathcal K_1\bigl(\gamma_{\mathrm{in}}f\bigr),
$$

where $\mathcal K_0$ and $\mathcal K_1$ are learned Fourier multipliers. The feature is therefore not merely concatenated as another input channel: it selects interactions at the input side and gates the result at the output side. A discontinuous pullback kernel can thus be obtained from smooth translation-invariant spectral kernels.

For a binary feature, the shared-multiplier special case is

$$
\mathcal T_{\mathrm{shared}}f
=a\mathcal Kf+b\gamma_{\mathrm{out}}\mathcal K\bigl(\gamma_{\mathrm{in}}f\bigr),
\qquad
a=\frac{1+\rho}{2},\quad b=\frac{1-\rho}{2}.
$$

Both forms require only two spectral convolutions. The feature determines the possible jump location; the learned multipliers determine the jump amplitude and smooth structure on each side.

## Current testbed: 1D linear advection

The first testbed is the nonperiodic, constant-velocity advection equation

$$
u_t+c u_x=0,\qquad x\in[-1,1],\quad t\in[0,1],\quad c=0.5,
$$

whose solution is known exactly:

$$
u(x,T)=u_0(x-cT).
$$

Each initial condition contains at most one jump. The binary interface features satisfy

$$
\gamma_1(x)=\gamma_0(x-cT).
$$

The inflow--outflow boundary condition makes samples with $s=\pm1$ genuine continuous controls on the open physical interval. Two data stages are evaluated:

- **Stage A:** a two-plateau jump, isolating discontinuity recovery.
- **Stage B:** a compact smooth background plus a jump, testing simultaneous smooth and discontinuous reconstruction.

## Implemented models

| Model | Operator | Purpose |
|---|---|---|
| `plain` | $\mathcal Ku_0$ | Standard one-channel linear Fourier baseline. |
| `hidden_width_2` | $\mathcal K_1u_0+\mathcal K_2u_0$ | Negative control: it has duplicated parameters but the same linear function class as `plain`. |
| `two_channel_fno` | $\mathcal K_u u_0+\mathcal K_\gamma\gamma_0$ | Parameter-matched generic two-input-channel baseline; it has no product feature or output gate. |
| `wide_plain` | $\mathcal K_{2M}u_0$ | Ordinary baseline with twice the retained spectral modes. |
| `shared` | $a\mathcal Ku_0+b\gamma_1\mathcal K(\gamma_0u_0)$ | Structured interface operator with a shared multiplier. |
| `shared_learnable_rho` | `shared` with learned $\rho$ | Tests whether the cross-interface coupling should be learned. |
| `dual` | $\mathcal K_0u_0+\gamma_1\mathcal K_1(\gamma_0u_0)$ | Main two-multiplier interface-conditioned operator. |

All fixed models are fitted by ridge-regression normal equations. `shared_learnable_rho` jointly optimizes a global scalar $\rho$ and its shared multiplier.

## Deep FNO / Shared Benchmark v2

除上述线性算子验证外，仓库还包含一个 JAX 实现的神经算子基准：在一维常速平流中学习含**区域内部非零间断**的终态解。v2 在 $N=256$ 训练，并在 $N=256/512/1024$ 上测试同分辨率误差与零样本超分辨率。

该基准以五个独立训练 seed、每个 seed 为 16,000/2,000/4,000 个训练/验证/测试样本，统一比较四个模型：

- **FNO**：不提供终态界面先验的基线；
- **FNO+$\\gamma_{T}$**：将解析终态界面作为额外输入；
- **Oracle-Shared**：采用固定解析门控序列的结构化谱算子；
- **Oracle-Shared ($\\rho=0$)**：固定等权双分支的后注册消融模型。

全部模型使用相同的优化设置（学习率 $10^{-3}$、权重衰减 $10^{-4}$、500 epoch），并按验证集相对 $L_{2}$ 选择最佳 checkpoint。比较采用共享测试样本的分层配对 bootstrap，报告相对 $L_{2}$、MSE、界面 MAE、跳跃幅值、总变差、过冲/欠冲及高频误差等指标。

- 基准代码和复现实验说明：[deep_fno_shared_benchmark/README.md](deep_fno_shared_benchmark/README.md)
- 完整实验结果与图表：[四模型 v2 对比测试报告](deep_fno_shared_benchmark/results/formal_main_v2/comparison_four_models/four_model_main_v2_comparison_report.md)

## Initial evidence

The committed results are a single-seed smoke experiment at $N=32$ and retained mode budget $M=8$. Lower is better.

| Model | Stage A relative $L^2$ | Stage B relative $L^2$ | Interpretation |
|---|---:|---:|---|
| `plain` | 0.2179 | 0.1718 | Standard spectral baseline. |
| `wide_plain` | 0.1379 | 0.1101 | More modes help, but do not close the gap. |
| `shared` | 0.1035 | 0.0942 | The structured shared kernel is substantially better. |
| `dual` | **0.0988** | **0.0721** | Best current model in both stages. |

At $M=8$, `dual` reduces relative $L^2$ error versus `plain` by 54.7% in Stage A and 58.0% in Stage B. It also improves over `wide_plain`, while `hidden_width_2` is numerically identical to `plain` and the generic `two_channel_fno` remains close to `plain`. These controls support the intended explanation: the gain comes from the structured feature interaction, not simply from more parameters or an extra channel.

The same experiment also establishes the current limitation: performance is sensitive to a wrong, shifted, or smoothed output interface, and independently trained feature models do not yet preserve `plain`-level error on all continuous endpoint controls. The result is promising but not yet publication-grade evidence.

## Tests and result artefacts

| Item | What it checks | Link |
|---|---|---|
| Data tests | Exact advection transport, internal/endpoint interfaces, and dataset serialization. | [`test_data.py`](advection_fno/tests/test_data.py) |
| Model-identity tests | Fourier design matrices, `shared` identities, learnable $\rho$, two-channel mixing, and the width-two equivalence. | [`test_models.py`](advection_fno/tests/test_models.py) |
| End-to-end smoke test | Generation, fitting, evaluation, and saved experiment outputs. | [`test_smoke.py`](advection_fno/tests/test_smoke.py) |
| Full 1D test report | Method, protocol, figures, ablations, endpoint controls, and limitations. | [1D advection equation test report](advection_fno/1D%20advection%20equation%20test%20report.md) |
| Aggregate metrics | Per-model mean/median metrics for every suite. | [`summary.csv`](advection_fno/results/smoke_hidden_width_2/summary.csv) |
| Paired comparisons | Bootstrap comparisons against `plain` and parameter-matched controls. | [`comparisons.csv`](advection_fno/results/smoke_hidden_width_2/comparisons.csv) |
| Run configuration | Dataset, model, precision, environment, and runtime metadata. | [`manifest.json`](advection_fno/results/smoke_hidden_width_2/manifest.json) |
| Visual results | Error-versus-mode curve and representative Stage A/B predictions. | [figures](advection_fno/results/smoke_hidden_width_2/figures) |

Only the compact, citable smoke bundle is versioned. Large sample arrays, fitted weights, logs, and other exploratory runs are deliberately ignored by Git.

## Repository layout

```text
advection_fno/
├── config.py                         # Experiment presets and configuration
├── data.py                           # Analytic advection data generation
├── models.py                         # Linear spectral operators and ridge fits
├── metrics.py                        # Errors, diagnostics, and comparisons
├── experiment.py                     # Command-line experiment entry point
├── tests/                            # Unit and smoke tests
├── results/                          # Local outputs; only one compact bundle is tracked
└── 1D advection equation test report.md
```

## Reproduce the smoke experiment

Install the package and its dependencies in a JAX-capable Python environment:

```bash
pip install -e .
```

Run the checks and a fresh smoke experiment from the repository root:

```bash
python -m unittest discover -s advection_fno/tests -v

python -m advection_fno.experiment \
  --preset smoke \
  --stage all \
  --output-dir advection_fno/results/smoke_local \
  --save-data \
  --x64
```

For the planned multi-seed GPU study, replace `smoke` with `full` and use a separate ignored output directory such as `advection_fno/results/full`.

## Next steps

- Repeat the benchmark at higher resolution with multiple training seeds and seed-level uncertainty estimates.
- Include continuous controls during model selection and test constraints or reliability gating for absent interfaces.
- Evaluate imperfect or predicted interfaces, variable transport, and nonlinear conservation laws.
- Extend the feature-conditioned kernel view beyond the current single-layer linear operator.

## License

This project is released under the [MIT License](LICENSE).
