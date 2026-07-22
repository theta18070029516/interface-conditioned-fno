# Deep FNO / Shared Benchmark

一维含内部间断线性平流问题上的 Fourier Neural Operator（FNO）与结构化 Shared 谱算子对比基准。当前正式实验为协议 v2：所有样本均含区域内部非零间断，并在相同数据、相同优化配置和严格配对测试下比较四个模型。

## 本仓库包含的内容

- 可复现实验源码、单元测试、GPU smoke test 与远程验证脚本；
- 四个 v2 正式模型的训练 notebook；
- 四模型统计比较 notebook，以及保留嵌入输出的已执行版本；
- 四模型正式测试报告、报告中的 PNG 图和可审计的 source-data CSV。

为保持仓库轻量，训练 case、预测数组、checkpoint、周期模型快照和其他原始结果目录均不纳入版本控制。运行正式训练后，这些文件会在本地或服务器的 `results/` 下重新生成。

## 问题与模型

空间区域为 $[-1,1]$，线性平流速度 $c=0.5$、终止时间 $T=1$。每个 v2 样本具有一个内部非零跳跃：

$$
\xi_{0}\in[-0.75,0.25],
\qquad
\xi_{T}=\xi_{0}+0.5\in[-0.25,0.75].
$$

| 模型 | 输入 | 作用 |
| --- | --- | --- |
| FNO | $(x,u_{0},\gamma_{0})$ | 不提供终态界面先验的基线 |
| FNO+$\gamma_{T}$ | $(x,u_{0},\gamma_{0},\gamma_{T})$ | 量化解析终态界面先验的信息收益 |
| Oracle-Shared | 同上 | 使用固定门控序列 $[\gamma_{0},\gamma_{T},\gamma_{T},\gamma_{T},\gamma_{T}]$ |
| Oracle-Shared（$\rho=0$） | 同上 | 后注册消融：所有 $\rho_{ij}^{\ell}=0$，即等权混合两条谱分支 |

Oracle-Shared 不含 v1 的递推 $q_{\ell}$ 或 `gamma_blocks`；四个模型都仅优化终态解场 MSE。正式 v2 固定学习率为 $10^{-3}$、权重衰减为 $10^{-4}$，每个模型使用 5 个训练 seed、每个 seed 使用 16,000/2,000/4,000 个训练/验证/测试样本，在 $N=256$ 训练并直接测试 $N=256/512/1024$。

详细协议见 [implementation_plan_v2.md](implementation_plan_v2.md)；已完成的四模型结果见 [正式对比报告](results/formal_main_v2/comparison_four_models/four_model_main_v2_comparison_report.md)。

## 安装与环境

代码已在 Python 3.11、JAX、Optax、NumPy、Matplotlib、Pandas、tqdm、Jupyter 环境中验证。GPU 用户应先按自己的 CUDA 驱动与硬件配置安装相应 JAX 后端，再安装本项目其余依赖：

```bash
conda create -n sno python=3.11
conda activate sno
pip install -r deep_fno_shared_benchmark/requirements.txt
```

从仓库根目录运行模块，例如：

```bash
python -m deep_fno_shared_benchmark.experiment --help
```

## 验证

物理 GPU 6 被永久排除。服务器上选择一块其他空闲 GPU 后，运行完整单元测试和 v2 GPU smoke test：

```bash
conda activate sno
bash deep_fno_shared_benchmark/remote_validate_v2.sh <physical_gpu_index>
```

该脚本会先运行全部测试，再验证 JAX 的 GPU 后端、Oracle 门控、固定 $\rho=0$ 策略，以及跨分辨率前向。

## 正式训练与比较

1. 在不同 GPU 上分别运行 `train_fno_main_v2.ipynb`、`train_fno_gamma_main_v2.ipynb`、`train_shared_oracle_main_v2.ipynb` 与 `train_shared_oracle_rho0_main_v2.ipynb`。
2. 每个 notebook 会训练 5 个 seed，各 500 个完整 epoch；每 10 个 epoch 记录指标并保存周期 checkpoint。
3. 使用验证相对 $L_{2}$ 最小的 `checkpoint_best.npz` 在 $N=256/512/1024$ 推理。
4. 运行 `compare_three_models_main_v2.ipynb`。文件名因兼容性保留，但内容已比较四个模型；它执行 10,000 次分层配对 bootstrap，输出比较图、source data 与中文报告。

`build_v2_notebooks.py` 可重新生成上述 notebook。`v2_same_seed_sample_comparison.ipynb` 可在相同 seed、相同样本和三种分辨率下并排检查四个模型的训练或测试预测。

## 结果与可追溯性

[four_model_main_v2_comparison_report.md](results/formal_main_v2/comparison_four_models/four_model_main_v2_comparison_report.md) 是面向读者的正式报告；报告所需的 PNG 图位于同级 `figures/`，数值来源位于同级 `source_data/`。比较采用 5 个训练 seed 与共享测试样本的分层配对 bootstrap；详细的协议、产物和检查结果保存在：

- [comparison_manifest.json](results/formal_main_v2/comparison_four_models/comparison_manifest.json)
- [artifact_audit.csv](results/formal_main_v2/comparison_four_models/source_data/artifact_audit.csv)
- [paired_hierarchical_bootstrap.csv](results/formal_main_v2/comparison_four_models/source_data/paired_hierarchical_bootstrap.csv)

## 目录结构

```text
deep_fno_shared_benchmark/
├── config.py, data.py, models.py, training.py, metrics.py
├── experiment.py, formal_v2.py, build_v2_notebooks.py
├── train_*_main_v2.ipynb
├── compare_three_models_main_v2.ipynb
├── tests/
└── results/formal_main_v2/comparison_four_models/
    ├── four_model_main_v2_comparison_report.md
    ├── figures/*.png
    └── source_data/*.csv
```
