# 已验证环境与固定依赖

## 环境身份

| 项目 | 已验证值 |
|---|---|
| 操作系统与架构 | Windows 11，AMD64 |
| Python | 3.12.13 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA运行时 | 12.1 |
| NumPy / pandas / PyArrow | 2.5.2 / 3.0.5 / 25.0.1 |
| ONNX / ONNX Runtime | 1.22.0 / 1.28.0 |
| COMSOL | Multiphysics 6.3，使用其自带Java工具编译 |

完整包版本见根目录 `environment.snapshot.json`，直接与间接Python依赖均固定在 `requirements-lock-windows-py312-cu121.txt`。该文件由已运行验证的环境依赖闭包生成，不包含Origin、PDF处理等与仓库运行无关的工具。

## 安装命令

准备Python 3.12.13后，在仓库根目录运行：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock-windows-py312-cu121.txt
.venv/Scripts/python.exe tools/run.py check
.venv/Scripts/python.exe tools/run.py train-smoke
```

锁定文件使用PyTorch官方CUDA 12.1轮子索引，并固定其他包的实际版本。CUDA训练需要兼容的NVIDIA驱动；CUDA运行时版本不等于系统必须另装相同版本的CUDA Toolkit。COMSOL为独立软件，不通过pip安装。

## 使用其他平台或CPU环境

`requirements.txt`保存依赖范围，适合另行选择CPU版PyTorch或其他系统的环境。上述Windows/CUDA锁定文件不作为Linux、macOS或CPU专用安装的通用锁。迁移到其他组合后，应记录新的版本清单，并重新运行目标流程的验证。

## 验证状态

此处固定的是已完成 `docs/VALIDATION.md` 中验证的实际环境。此次补充没有新建空环境重新安装；固定依赖快照与一次独立的干净环境安装验证属于不同证据。
