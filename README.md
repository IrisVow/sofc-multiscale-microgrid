# SOFC 多尺度建模与海岛微电网研究代码

本项目连接 COMSOL 单电池建模与数据导出、PINN 多场代理、温度提取与查找表、容量经济分析以及微电网能量管理。发布包包含研究代码、配置、验证脚本和必要的小型数据；完整训练数据与 COMSOL 基准模型作为同一项目的 Release 附件分发。

## 目录

| 路径 | 内容 |
|---|---|
| `outputs/pinn_sofc/` | v20 训练、评估、ONNX导出、网络与物理残差；`experiments/` 保存独立的物理修正与界面预算训练器 |
| `outputs/microgrid_sys/` | 系统部件、负荷、调度、PINN耦合、容量经济和敏感性分析；含YAML配置及三份温度LUT |
| `outputs/pinndata/` | 工况元数据、小样本、完整训练网格坐标；完整训练表导入到各 `merged/` 目录 |
| `work/` | COMSOL二维DOE导出与公共路径解析类 |
| `outputs/verify3d/` | 三维模型构建、求解和诊断源码 |
| `research/java/` | 反应传输、组分、界面通量、网格和参考求解的原生研究类 |
| `research/test_*.py` | 物理方程、制造解、界面梯度和组分置换验证 |
| `reference_results/` | 已完成的容量分析和三种温度提取调度对比汇总 |
| `tools/` | 数据附件导入、统一运行、LUT重建和COMSOL编译入口 |
| `docs/` | 复现范围、路径调整、来源与验证记录 |

保留原模块间的 `outputs/` 层级，以减少对科学实现和导入关系的改动。运行时根据 `.sofc-project` 定位仓库，配置文件中的项目路径均相对于仓库或模块。请保留这个标记文件。

## 安装

建议使用 Python 3.12，在仓库根目录执行：

```sh
python -m venv .venv
# 激活虚拟环境后：
python -m pip install -r requirements.txt
```

CPU可以运行调度、容量分析、推理和基础验证。PINN完整训练建议使用支持CUDA的PyTorch环境；历史语义界面验证脚本使用CUDA。COMSOL步骤另外需要本机安装COMSOL 6.3及相应模块许可。Python依赖不包含COMSOL。

## 无需完整训练数据的入口

以下命令可在仓库根目录运行，使用当前Python解释器：

```sh
python tools/run.py check
python tools/run.py dispatch 7 --pinn
python tools/run.py sampling
python tools/run.py capacity --pinn
python tools/run.py sensitivity
python tools/run.py train-smoke
```

`check`执行组件、EMS、负荷、尺寸、放大、LUT、ONNX/PyTorch、公共推理API、容量经济和四组分置换检查。性能断言依赖运行机器。v18历史模型未收录，对应兼容检查明确跳过。

`sampling`重跑基线及三种温度提取方式的四种控制模式，共13组七天运行，并与 `reference_results/ems_three_sampling/summary.csv` 中的发电量、主燃料、GHG和终端SOC核对。输出写入 `outputs/microgrid_sys/results/sampling_replay/`。再次运行时用 `--output` 指定新目录。

`train-smoke`使用真实数据的确定性小样本执行一个轮次，仅检查训练管线。它关闭PDE项，不能用于复现v20精度。默认输出目录存在时会拒绝覆盖，可用 `--run-dir runs/new_smoke` 指定新目录。

## 完整数据与模型附件

| Release附件 | 用途 |
|---|---|
| `sofc-training-data-v20.zip` | v20所用的完整稳态和瞬态合并Parquet数据 |
| `sofc-comsol-reference.zip` | 二维原始基准模型 `work/sofc1_FINAL2_solved.mph` |
| `sofc-advanced-validation.zip` | 语义界面验证需要的历史第30轮检查点 |

将需要的附件下载到 `downloads/`，然后执行：

```sh
python tools/data_assets.py verify --asset training
python tools/data_assets.py import --asset training
python tools/data_assets.py import --asset comsol
python tools/data_assets.py import --asset advanced
```

也可用 `--directory` 指定附件所在目录。脚本同时检查压缩包和内部文件的大小及SHA-256；若目标文件已存在且内容不同，会停止而不覆盖。GitHub仓库为 `IrisVow/sofc-multiscale-microgrid`，首次发布标签为 `v1.0.0`。仓库为私有，需要具有访问权限的账号下载附件。将附件下载到本地后执行上面的导入命令即可；若使用可访问的Release下载地址，也可通过 `download --base-url` 自动下载、校验并导入。

完整Parquet、`.mph`及下载缓存已加入 `.gitignore`。请把三个ZIP上传到Release附件区，主仓库上传本目录中的代码、小型数据和说明。

## 完整训练和评估

导入训练数据后，在 `outputs/pinn_sofc/` 中运行：

```sh
python train.py --config config_v20.yaml --init runs/stage3_pde_v19/best.pt --run-dir runs/new_v20
python evaluate.py --run_dir runs/stage3_pde_v20 --config config_v20.yaml --ckpt final.pt --pools steady_val steady_test transient_test
python export_onnx.py --config config_v20.yaml --ckpt runs/stage3_pde_v20/final.pt --out runs/stage3_pde_v20/new_export.onnx
```

v20生产部署使用 `final.pt`，ONNX和三份LUT对应冻结部署模型；v19的 `best.pt`仅用于重现v20热启动。发布检查点只调整了配置中的路径元数据，网络参数保持一致。全新随机初始化训练与历史热启动流程应分别报告。

实验训练器位于 `experiments/*/trainer/`。在相应训练器目录运行 `train.py --config ../configs/配置名.yaml --run-dir runs/new_experiment`。实验配置的原始协议和用途见 [复现范围](docs/REPRODUCIBILITY.md)。历史检查点不能直接用于迁移后代码的精确断点续训；可使用 `--init`开展新的权重热启动实验。

## 温度LUT重建

主仓库包含三份已冻结LUT以及15,567点的训练网格几何，可独立重建候选温度表：

```sh
python tools/build_sampling_luts.py --sampling axial_101 --output outputs/microgrid_sys/results/lut_rebuild/axial101.npz
python tools/build_sampling_luts.py --sampling full_training_mesh --output outputs/microgrid_sys/results/lut_rebuild/full_mesh.npz
```

仅替换 `tmax_k`，保留原LUT工况轴、热源和电势字段。不同计算设备可能产生浮点差异；重建脚本不会覆盖部署LUT。

## COMSOL

在仓库根目录运行，`--comsol-home`指向本机的 `Multiphysics` 安装目录；也可设置 `COMSOL_HOME`。不再固定安装盘符。

```sh
python tools/comsol.py --comsol-home "本机COMSOL的Multiphysics目录" --compile-only DoeAll
python tools/comsol.py --comsol-home "本机COMSOL的Multiphysics目录" DoeAll
python tools/comsol.py --comsol-home "本机COMSOL的Multiphysics目录" Build3D
```

执行DOE前导入COMSOL基准附件。执行构建或求解会产生新模型和数据，可能耗时较长。`research/java/` 中的诊断和精化类通常要求指定既有模型、日志和导出路径；其参数以源码 `main` 为准。它们是可组合研究工具，不是无需输入模型的一键复现命令。

## 研究边界

负荷为合成典型海岛负荷。温度阈值、经济和排放参数含工程情景假设。三种温度LUT均为准稳态表，不包含热历史；域外温度保持未知。历史控制器在严格阈值下存在搜索下限可能提高低功率请求的问题，源码及对比保持该行为以复现已有结果。

容量研究与温度调度研究的窗口及设备配置不同；各实验的部署、随机配对与热启动用途不同。不能将其合并为已验证的容量—热约束联合优化。完整边界及验证要求见 [复现范围](docs/REPRODUCIBILITY.md)。

本包按“核心代码、配置与必要数据、验证脚本、项目说明”整理，论文正文、组会材料、Origin工程、编译产物和临时文件未纳入。项目授权许可证由作者另行确定，本次未代选。
