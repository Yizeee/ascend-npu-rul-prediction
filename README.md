# Ascend NPU RUL Prediction

EOL-AttMoE 锂电池剩余使用寿命（RUL）预测模型在华为昇腾 910B4-1 NPU 上的完整训练实践。
覆盖 CUDA → NPU 代码迁移、NASA / CALCE 双数据集训练与精度对标。

本仓库由电池管理系统课题组维护，记录国产化算力平台上的深度学习训练实践。

## 成果亮点

在昇腾 910B4-1 上完成 NASA 与 CALCE 两个数据集的训练，六项精度指标全部达到或超过 NVIDIA GPU 基线：

| 指标 | 数据集 | NVIDIA 基线 | 昇腾 NPU | 相对变化 |
|------|--------|------------|----------|---------|
| RE | NASA | 3.20% | **3.10%** | -3.1% |
| RE | CALCE | 8.40% | **6.96%** | **-17.1%** |
| MAE | NASA | 0.085 | **0.077** | -9.4% |
| MAE | CALCE | 0.149 | **0.142** | -4.7% |
| RMSE | NASA | 0.110 | **0.099** | -10.0% |
| RMSE | CALCE | 0.204 | **0.196** | -3.9% |

负值表示误差降低（精度提升）。RE=相对误差，MAE=平均绝对误差，RMSE=均方根误差。

## 训练环境

- 硬件：4 × 华为昇腾 910B4-1（每卡 64GB HBM），AArch64
- 系统：Ubuntu 22.04 (aarch64)，245GB 内存
- 软件：CANN Toolkit 8.2.RC1 + torch_npu 2.1.0.post13 + PyTorch 2.1.0
- NPU 驱动：26.0.rc1

## 模型：EOL-AttMoE

- **Attention**：多头自注意力捕获电池退化序列长程依赖
- **MoE 混合专家**：动态路由不同退化模式，提升泛化能力
- **EOL 感知修正头**：独立端点修正，降低预测末端偏差
- **鲁棒种子选择**：MAD 异常剔除 + Top-K 种子集成（NASA: seeds 3,2,0,4；CALCE: seeds 3,0,1,2）

## 文件说明

| 文件 | 说明 |
|------|------|
| EOL-AttMoE-NASA_npu.py | NASA 数据集训练脚本（NPU 适配版） |
| EOL-AttMoE-CALCE_npu.py | CALCE 数据集训练脚本（NPU 适配版） |
| nasa_train.log / calce_train.log | 完整训练日志（含 NPU 环境自检输出） |
| final_training_scores*.csv | 最终训练评分（含 robust/detailed/all 变体） |
| robustseed_summary*.csv | 鲁棒种子选择汇总 |
| seed_stability_scores*.csv | 种子稳定性评分 |

带 `_NASA` 后缀的 CSV 属于 NASA 数据集，其余为 CALCE 数据集。

## CUDA → NPU 迁移要点

| CUDA | 昇腾 NPU |
|------|----------|
| import torch | import torch + import torch_npu |
| device = 'cuda' | device = 'npu' |
| model.cuda() / tensor.cuda() | model.npu() / tensor.npu() |
| torch.cuda.is_available() | torch.npu.is_available() |
| NCCL 分布式后端 | HCCL 分布式后端 |

## 复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python EOL-AttMoE-NASA_npu.py   # NASA 数据集
python EOL-AttMoE-CALCE_npu.py  # CALCE 数据集
```

数据集需自行获取：NASA 5. Battery Data Set、马里兰大学 CALCE 锂电池数据集。

模型权重（.pth）与容量退化曲线图（PNG）体积较大未纳入，训练日志与结果 CSV 已完整保留精度证据。
