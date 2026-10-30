# 正式版本选择

## 默认复现组合

| 用途 | 正式入口 | 配置 | 模型或数据 |
|---|---|---|---|
| 部署场推理与ONNX一致性验证 | `outputs/pinn_sofc/`根训练器；`tools/run.py check` | `outputs/pinn_sofc/config_v20.yaml` | `runs/stage3_pde_v20/final.pt`与`pinn_sofc_v20.onnx` |
| 重训历史v20流程 | 根训练器的`train.py` | `config_v20.yaml` | 从`runs/stage3_pde_v19/best.pt`热启动，输出写入新目录 |
| 七天基线与温度调度 | `tools/run.py dispatch`、`tools/run.py sampling` | `outputs/microgrid_sys/configs/` | 冻结的五探针、101×5轴向和15,567点训练网格三份LUT |
| 容量与经济分析 | `tools/run.py capacity`、`tools/run.py sensitivity` | `components.yaml`、`scenario.yaml`、`economics.yaml`、`ems.yaml`、`coupling.yaml` | 合成负荷、部件参数；PINN核查使用v20 LUT |
| 安装后的训练管线检查 | `tools/run.py train-smoke` | `config_release_smoke.yaml` | 包内小样本；一个轮次、PDE关闭 |

以上`runs/`路径相对于 `outputs/pinn_sofc/`。v20部署选择 **final.pt**；v19的 **best.pt**只作为历史热启动起点。`config_v20_nopde.yaml`是v20结构的无PDE配置，不代表三种子配对实验的全部协议。`config_v20_smoke.yaml`保留历史试运行设置；新安装检查统一使用 `config_release_smoke.yaml`。

## 三份温度LUT

文件均位于 `outputs/microgrid_sys/data/`，使用相同工况轴并保留相同的非温度字段。

| 文件 | 温度提取 | 使用角色 |
|---|---|---|
| `pinn_lut_v20.npz` | 每域中线的五个探针 | 历史部署口径及调度对照 |
| `pinn_lut_v20_axial_101.npz` | 101个轴向位置×5个域中线 | 轴向提取调度与共同交叉评价 |
| `pinn_lut_v20_full_training_mesh.npz` | 15,567个训练网格点 | 更密的空间提取诊断 |

三份表均为准稳态代理量；五探针最大值、轴向中线最大值与训练网格最大值具有不同的空间含义。

## 独立研究训练器

路径前缀为 `outputs/pinn_sofc/experiments/`。

| 目录 | 研究用途 | 配置或验证入口 | 与部署模型的关系 |
|---|---|---|---|
| `paired_20261014/trainer/` | 随机初始化、等监督样本及更新次数的配对训练 | 配对训练实现；历史三种子检查点不在本包 | 不作为部署权重 |
| `physics_corrected_20261015/trainer/` | 热学边界、压力语义、训练配点与梯度预算修正 | `configs/corrected_formal_template.yaml`；`research/test_corrected_physics_20261015.py` | 独立物理修正分支 |
| `paired_corrected_20261015_stage12/trainer/` | 修正后PINN与data-only阶段配对 | `configs/seed42_pinn.yaml`、`seed42_data_only.yaml` | 保留对应阶段协议 |
| `electrochem_corrected_20261015/trainer/` | 温度相关电导率、电化学闭合与电荷源修正 | `configs/seed42_pinn.yaml`、`seed42_data_only.yaml`；电化学及热学回归验证 | 不替换历史v20部署模型 |
| `interface_semantic_budget_20261015/trainer/` | 分组与语义界面梯度预算对照 | `configs/group.yaml`、`semantic.yaml`；`research/test_semantic_interface_20261015.py` | 需要高级验证附件和CUDA |

名称包含`pilot`、`continuous`或`resume`的配置保留历史试运行与断点实验用途。正式研究请选择上表中的模板或配对配置，并记录新训练的输出目录、随机种子和选择规则。后续物理修正训练器与历史v20分别构成不同实验，不互换权重或精度指标。

## 发布版本

- **v1.0.0**：冻结研究代码、模型、LUT和数据附件；三份大附件的文件内容与SHA-256继续作为数据身份。
- **v1.0.1**：增加正式版本选择、结果索引和固定环境；科学代码、模型参数和数据不变。
- **v1.0.2**：增加完整英文README及中英文切换，更新项目说明；科学代码、模型参数和数据不变。
