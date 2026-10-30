# 研究数据与模型授权

Copyright © 2026 张晏瑞。

本项目的研究数据和模型采用 **Creative Commons Attribution 4.0 International（CC BY 4.0）**。允许复制、修改、再分发及商业使用；使用时应保留署名、提供许可证链接，并注明所作修改。

- 许可证：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 法律文本：[LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt)
- 署名与引用：[CITATION.cff](CITATION.cff)、[引用说明](docs/CITATION.md)

## 适用范围

| 材料 | 路径或附件 |
|---|---|
| 工况、负荷、采样、训练数据与网格坐标 | `outputs/pinndata/`；`outputs/microgrid_sys/data/`中的CSV和NPZ |
| 发布模型与归一化参数 | `outputs/pinn_sofc/runs/`中的PT、ONNX、归一化及掩码JSON |
| 实验初始温度场和审计数据 | `outputs/pinn_sofc/experiments/*/initial_reference/` |
| 研究结果、对照表和公式验证数据 | `reference_results/`；`outputs/microgrid_sys/results/`中随包分发的CSV |
| 完整训练数据 | `sofc-training-data-v20.zip`中的Parquet文件 |
| COMSOL基准模型 | `sofc-comsol-reference.zip`中的MPH文件 |
| 高级验证检查点 | `sofc-advanced-validation.zip`中的PT文件 |

上述授权同时适用于 `v1.0.0` 发布的相应研究数据和模型。压缩包内的代码与说明文件仍按各自的MIT声明授权。

## 署名示例

> 数据与模型：张晏瑞，SOFC 多尺度建模与海岛微电网研究代码，GitHub，https://github.com/IrisVow/sofc-multiscale-microgrid，CC BY 4.0。使用的数据附件版本：v1.0.0。修改：按实际情况说明。

Python第三方依赖、PyTorch运行时和COMSOL软件本身遵循各自的许可证，不包含在本项目对原创代码、数据和模型的授权中。
