# Deep FNO / Shared Benchmark v2 实施方案

## 1. 目标与版本隔离

v2 的目标是在全部样本均含区域内部间断的条件下，公平区分三类贡献：基础 FNO 的拟合能力、显式提供终态界面先验 $\gamma_T$ 的信息增益，以及将该先验用于 Shared 结构门控的额外架构增益。

- v2 固定使用 `protocol_version=2`、正式运行标签 `main_v2` 和目录 `results/formal_main_v2`。
- v1 的 `implementation_plan.md`、递推式 `shared`、训练 notebook、checkpoint 与 `results/formal_main_v1` 保持可用。
- v2 正式推理只读取验证集相对 $L_2$ 最优 checkpoint；最终轮 checkpoint 仅用于审计训练是否完整结束。

## 2. 数据协议

线性平流速度与终止时间沿用 v1：$c=0.5$、$T=1$，因此位移为 $cT=0.5$。v2 的每个样本均为非零跳跃的内部间断：

$$
\xi_0\sim\mathcal{U}[-0.75,0.25],\qquad
\xi_T=\xi_0+0.5\in[-0.25,0.75].
$$

其余分布保持 v1 定义：

- 跳跃绝对幅值在 $[0.5,1.5]$ 上采样，符号等概率为正或负；
- 平滑背景的均值、谱衰减、支撑区间与系数幅值不变；
- 平台背景比例保持为 $20\%$；
- 删除离开区域、零跳跃与端点连续样本。

连续参数先生成一次，再在 $N=256$、$512$、$1024$ 的胞心网格上解析采样，以保证跨分辨率样本身份、界面位置和输运距离完全一致。

## 3. 模型与损失

### 3.1 三个正式模型

| 模型名 | 输入通道 | 界面信息的用途 |
|---|---:|---|
| `fno` | 3 | $(x,u_0,\gamma_0)$，不提供 $\gamma_T$ |
| `fno_gamma` | 4 | $(x,u_0,\gamma_0,\gamma_T)$，仅作为普通输入通道 |
| `shared_oracle` | 4 | 同上，并将解析 $\gamma$ 用作 Shared 谱层门控 |

### 3.2 Oracle-Shared 门控

v1 的 `shared` 保留递推变量 $q_\ell$、soft-$\tanh$ 与 `gamma_blocks`。v2 的 `shared_oracle` 不含这些参数和计算，其四个谱块使用固定门控对：

$$
(\gamma_0,\gamma_T),\quad
(\gamma_T,\gamma_T),\quad
(\gamma_T,\gamma_T),\quad
(\gamma_T,\gamma_T).
$$

用于诊断的层序列为：

$$
[\gamma_0,\gamma_T,\gamma_T,\gamma_T,\gamma_T].
$$

三种 v2 模型均只最小化解场均方误差：

$$
\mathcal{L}=\mathrm{MSE}(\hat u_T,u_T).
$$

v2 配置禁止启用 $\gamma$ 损失。

## 4. 训练、模型选择与 checkpoint

- 正式数据规模：训练集 16,000、验证集 2,000、测试集 4,000。
- 正式训练分辨率：$N=256$；测试分辨率：$N=256$、$512$、$1024$。
- 正式训练 seed：0、1、2、3、4。
- 每个 seed 完整训练 500 epoch，不使用早停。
- 每 10 epoch 打印训练损失、验证 MSE、验证相对 $L_2$ 并保存当前参数。
- 每次训练保存 `checkpoint_best.npz`、`checkpoint_final.npz`；`checkpoint.npz` 作为最优模型的向后兼容别名。
- `history.csv` 必须恰好含 500 个 epoch，最佳轮必须对应其中最小的验证相对 $L_2$，最终 checkpoint 必须对应第 500 轮。

## 5. 固定正式超参数

根据正式训练前的用户决策，三种模型固定使用同一组优化超参数：

$$
\mathrm{learning\ rate}=10^{-3},\qquad
\mathrm{weight\ decay}=10^{-4}.
$$

- 正式训练不再以开发集筛选结果作为前置条件。
- 三种模型和五个 seed 必须使用完全相同的固定值，保证比较公平。
- 首次正式训练会在 `results/formal_main_v2` 创建不可变的 `fixed_hyperparameters_manifest_v2.json`，后续训练必须校验其内容一致。
- 原公共筛选入口保留为可选敏感性分析，不影响正式 v2 结果，也不能覆盖固定正式 manifest。

## 6. 评价与统计

### 6.1 预注册比较

主比较是在 $N=256$ 上逐样本计算：

$$
\Delta_{\mathrm{primary}}
=L_{2,\mathrm{rel}}(\mathrm{shared\_oracle})
-L_{2,\mathrm{rel}}(\mathrm{fno\_gamma}).
$$

次要比较包括：

- `fno_gamma - fno`：终态界面先验的信息增益；
- `shared_oracle - fno`：先验与结构的联合增益；
- 三模型在 $N=512$、$1024$ 的零样本超分辨率表现。

每个差值均保持训练 seed 与测试样本配对，并执行 10,000 次分层配对 bootstrap：先重采样 seed，再在每个被抽中的 seed 内重采样共享测试样本，报告均值及 $95\%$ 置信区间。

### 6.2 指标与子组

保留以下逐样本或汇总指标：MSE、相对 $L_2$、界面窗口 MAE、跳跃幅值及其绝对误差、总变差误差、过冲、欠冲、高频绝对误差与高频相对误差。

删除连续样本指标和预测 $\gamma$ 指标，改为审计解析 $\gamma_T$ 是否与第四输入通道、Oracle 门控逐点完全一致。

预注册子组为：

- 平台背景与平滑背景；
- 正跳跃与负跳跃；
- 按 $|\Delta u|$ 三等分的跳跃幅值组；
- 按 $\xi_T$ 三等分的终态界面位置组。

对 `shared_oracle` 额外保存每层 $\rho_{ij}^{\ell}$ 原始矩阵，并报告均值、标准差、5%/50%/95% 分位数、接近 0/1 的饱和比例和逐层热图。

## 7. 产物

- `hyperparameter_screen_v2.ipynb`：可选超参数敏感性分析，不是正式训练前置步骤。
- `train_fno_main_v2.ipynb`、`train_fno_gamma_main_v2.ipynb`、`train_shared_oracle_main_v2.ipynb`：三张卡可独立并行执行的正式训练入口，物理 GPU 6 永久排除。
- `compare_three_models_main_v2.ipynb`：统计比较、科研图、source-data CSV、比较 manifest 和中文 Markdown 报告入口。
- `results/formal_main_v2`：仅存放 v2 正式结果，不覆盖 v1。

## 8. 验收门槛

交付正式训练入口前必须同时满足：

1. v2 数据全部为内部非零跳跃，初态和终态界面满足注册边界；跨分辨率解析输运一致。
2. 三/四通道输入正确；Oracle 门控序列正确且参数树不存在 `gamma_blocks`。
3. 三模型梯度有限并支持更细网格前向。
4. 禁用早停后运行完整指定 epoch；最佳与最终 checkpoint 语义正确。
5. 三模型 checkpoint round-trip 和 v2 端到端 smoke test 通过。
6. 原有 18 项 v1 测试继续通过。
7. notebook JSON、中文 Markdown、路径、GPU 选择、协议 manifest 和输出审计全部通过。
8. 服务器正式训练前运行完整测试与 GPU smoke test，并拒绝物理 GPU 6。
