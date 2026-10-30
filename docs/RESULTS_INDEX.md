# 结果与复现索引

命令默认从仓库根目录执行。`M`表示 `outputs/microgrid_sys/`，`P`表示 `outputs/pinn_sofc/`；表中路径均展开相对于仓库。论文对应关系使用完整工作版的LaTeX标签与章节主题，避免随排版变化的图表编号。论文正文不在代码仓库中。

| 结果或论文对应项 | 输入和配置 | 命令或入口 | 输出和已归档证据 | 复现条件 |
|---|---|---|---|---|
| 部署模型十二变量精度，`tab:pervariable` | `P/config_v20.yaml`、v20 `final.pt`、完整合并训练表 | `python tools/run.py evaluate` | `P/runs/stage3_pde_v20/eval_metrics_final.csv`、`eval_transient_by_t_final.csv`、`probe_T_t_final.csv` | 需导入训练附件；已打包已有`eval_metrics_final.csv` |
| 部署验证指标可视化 | 已存v20评估CSV | 在`M`运行 `python scripts/fig_metrics.py --version v20` | `M/results/figures/v20/fig_metrics_v20.png`及指标汇总CSV | 主仓库即可运行；属于验证可视化，不承诺重建论文全部排版 |
| PINN配置、损失与采样，`tab:configuration`、`tab:collocation` | `P/config_v20.yaml`、`models/`、`losses/`、`physics/` | 在`P`运行 `python train.py --config config_v20.yaml --init runs/stage3_pde_v19/best.pt --run-dir runs/new_v20` | 新目录中的配置、训练日志、归一化参数与检查点 | 需完整训练附件；保留历史热启动规则 |
| 三种温度提取的七天电量、燃料、排放与SOC，`tab:results`、`fig:tradeoffs` | `M/configs/`和三份v20 LUT | `python tools/run.py sampling` | `M/results/sampling_replay/summary.csv`、`timeseries/*.csv`、`validation.json`；历史对照`reference_results/ems_three_sampling/summary.csv` | 主仓库即可运行；自动核对5类历史指标 |
| 七天与首日轨迹，`fig:week`、`fig:day` | 上一行产生的13条时序；B0/F2/A2对应baseline、five_probe__limit1135_m2、axial101__limit1135_m2 | 先运行`sampling`，读取对应`timeseries/*.csv` | 功率、SOC、燃料与域内温度列；归档汇总见上述summary | 时序可重算；论文专用排版脚本未纳入 |
| 温度交叉评价，`fig:cross` | 同一调度时序、三份温度LUT | 归档证据：`reference_results/ems_three_sampling/temperature_cross_evaluation.csv` | 有效步数、未知发电步数、最大与平均温度、超阈值统计 | 交叉表已归档；`sampling`重建轨迹，但不自动重算此交叉表 |
| 适用域与控制下限，`tab:audit`、`fig:feasibility` | 三种提取方式和严格阈值调度 | 同一`sampling`入口；查看`pinn_*`时序列及保留控制逻辑 | `reference_results/ems_three_sampling/coverage_and_control_audit.csv`、`issue_steps.csv` | 审计汇总已归档；避免将域外温度当作已验证 |
| 全训练网格与轴向温度LUT | 冻结v20检查点、网格坐标、原五探针LUT | `python tools/build_sampling_luts.py --sampling axial_101 --output outputs/microgrid_sys/results/lut_rebuild/axial101.npz`；全网格将sampling改为`full_training_mesh`并换输出名 | 新NPZ；原LUT不被覆盖 | 主仓库即可运行；全网格计算较重 |
| 初始容量可行域与成本，`fig:capacity`、`tab:designs`、`tab:config` | `M/configs/components.yaml`、`scenario.yaml`、`economics.yaml`、`ems.yaml` | `python tools/run.py capacity --pinn` | `M/results/capacity/capacity_scan.csv`、`capacity_opt.json`、`capacity_scan.png`；历史证据`reference_results/capacity/` | 主仓库即可运行；新结果按当前扫描配置产生 |
| 低备用九点补扫，`tab:designs`下限设计 | 150 kW SOFC、独立的电池/机组边界组合 | 归档证据：`reference_results/capacity/boundary_sweep.json` | 各边界组合的可行性、LCOE及排放 | 历史记录可查；当前容量入口不单独重建此补扫 |
| 投资、尺寸、排放对照，`tab:config` | 容量分析结果和部件/经济参数 | 先运行capacity；在`M`运行 `python scripts/compare_paper_benchmarks.py` | `M/results/capacity/e2_paper_benchmarks.*`；历史对照同名`reference_results/capacity/`文件 | 先产生`capacity_opt.json`；对照按该次容量结果计算 |
| 经济与热参数敏感性，`fig:sensitivity`、`tab:sensitivity` | 推荐容量结果、经济/组件/耦合配置 | 先运行capacity，再运行 `python tools/run.py sensitivity` | `M/results/capacity/e3_sensitivity.csv`、JSON与PNG；历史对照`reference_results/capacity/e3_sensitivity.*` | 主仓库即可运行；重新仿真输出与历史记录分别报告 |
| 热学、电化学与界面机制验证 | 各独立trainer/configs、训练数据、初始参考场 | `research/test_corrected_physics_20261015.py`、`test_electrochem_20261015.py`、`test_electrochem_thermal_regression_20261015.py`、`test_semantic_interface_20261015.py` | 各分支`M/results/`下验证JSON；已验证结果见`docs/VALIDATION.md` | 需训练附件；语义验证另需高级附件与CUDA |
| 二维DOE导出与三维构建，几何与守恒/网格章节 | COMSOL基准附件、`work/*.java`、`outputs/verify3d/`、`research/java/` | `python tools/comsol.py --comsol-home "安装目录" --compile-only DoeAll`；构建或求解时选择相应原生类 | 原生模型、导出场及诊断文件，依具体类参数而定 | 需COMSOL；103源码编译已验证，求解需另执行 |
| 三种子配对、热启动消融、独立瞬态与热记忆，`tab:paired`、`tab:ablation`、`tab:practical`、`tab:thermalstate` | 分支训练器与原始逐种子检查点、协议锁、独立参考场 | `docs/VERSIONS.md`列出训练实现角色 | 本仓库未分发这些实验的全部历史检查点与逐工况导出 | 不提供声称可重建这些表的一键命令；新实验应单独记录协议 |

容量研究与调度研究使用不同代表周与设备组合。索引对应各自输入和结果，不将两者合并为同一热约束容量—调度联合优化。
