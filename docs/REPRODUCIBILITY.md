# 复现范围与验证条件

正式入口、配置和检查点选择见 [正式版本选择](VERSIONS.md)；逐项输入、命令与输出见 [结果复现索引](RESULTS_INDEX.md)；固定环境见 [环境说明](ENVIRONMENT.md)。

## 可直接运行

- 0D系统组件、SOC滞环EMS、负荷、设备尺寸、放大、经济和排放核算。
- 使用冻结v20 ONNX与三份LUT的场推理、域检查和七天调度。
- 容量网格搜索及经济敏感性代码；`reference_results/capacity/`保留历史结果，新的运行写入 `outputs/microgrid_sys/results/`。
- 真实小样本的一轮训练；它不具备完整训练数据覆盖度。
- 两种候选温度表重建，使用已提供的完整训练网格坐标。

## 需要附件或外部软件

- 完整v20训练/评估：`sofc-training-data-v20.zip`。默认训练配置和发布模型具有明确角色；已有发布检查点目录禁止作为新训练输出覆盖。
- COMSOL二维DOE与导出：`sofc-comsol-reference.zip`及COMSOL 6.3许可。
- 三维模型构建/求解：COMSOL及原生源码，实际求解需时间和内存；编译通过不能代替数值求解验证。
- 物理修正、热学回归和电化学制造解检查：完整训练附件，使用包内初始温度场、源公式对照表与审计记录。
- 语义界面梯度验证：完整训练附件、`sofc-advanced-validation.zip`及CUDA环境。历史第30轮检查点仅用于验证已有模型梯度，不能据此宣称迁移后的精确续训成立。

高级验证命令（仓库根目录）：

```sh
python research/test_corrected_physics_20261015.py
python research/test_electrochem_20261015.py
python research/test_electrochem_thermal_regression_20261015.py
python research/test_semantic_interface_20261015.py
python research/test_ts_permutation_20261016.py
```

## 实验角色

| 模块 | 角色 |
|---|---|
| `pinn_sofc/`根训练器 | 历史v20部署模型，含其原有近似物理闭合 |
| `experiments/paired_20261014/trainer/` | 随机初始化配对使用的联合更新训练器 |
| `experiments/physics_corrected_20261015/trainer/` | 热学算子、压力语义、训练配点与梯度预算修正 |
| `experiments/paired_corrected_20261015_stage12/trainer/` | 热学修正配对阶段 |
| `experiments/electrochem_corrected_20261015/trainer/` | 温度相关电导率、电化学闭合和电荷源修正 |
| `experiments/interface_semantic_budget_20261015/trainer/` | 界面项的语义分区梯度预算 |

这些训练器保留各自实现，未用后续修正替换历史部署模型。历史启动脚本含协议锁、已有结果、逐轮检查点和绝对路径哈希，无法仅复制脚本复现，因此主发布包提供训练器及配置，而没有将这些脚本包装成可直接运行的一键命令。逐种子历史完整训练和全部中间COMSOL模型未收录。

## 路径处理

- Python研究验证脚本由当前源码位置向上寻找 `.sofc-project`，构造仓库内路径。
- YAML的 `data_root`、`run_dir`和初始温度场路径使用仓库相对路径；读取后才解析为本机绝对路径。合并表名相对于 `data_root`。
- CLI中的相对 `--run-dir`按对应训练器位置解析；统一入口负责切换模块工作目录。
- 二维和三维Java中的原项目路径改为 `ProjectPaths.resolve(...)`；COMSOL安装由参数或环境变量指定。
- 部署检查点及高级验证检查点的配置元数据调整为相对路径，网络权重不变；高级验证检查点的协议与数据指纹仍记录原始实验身份，不用于迁移后代码的直接精确续训。
- 来源清单记录原文件指纹，发布清单记录实际发布文件指纹。路径调整后的文件不应与原始文件字节哈希混为一谈。

## 科学口径

- 不将五探针温度称为连续全场最大温度；全训练网格最大值也不保证连续域最大值。
- 1173 K、1135 K及安全裕度是所用情景假设，不是制造商认证限值。
- 新旧调度的终端SOC存在差异；主燃料能量不含引燃油，GHG账本含引燃油。
- 容量和调度代表周不同，历史容量结果与重跑调度结果分别核对。
- 域外运行点不填补温度，不推导全周动态热安全、应力或寿命保证。
- 本次整理不改变模型参数、已冻结LUT或科学结论。数值流程是否通过以 `VALIDATION.md` 中的实际运行记录为准。
