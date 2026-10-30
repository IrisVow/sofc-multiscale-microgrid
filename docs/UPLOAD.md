# 上传清单

1. 将主代码压缩包解压到新文件夹，把解压后的内容作为GitHub仓库根目录；保留 `.sofc-project`、`.gitignore` 和 `.gitattributes`。
2. 主仓库包括 `README.md`、`requirements.txt`、`data_manifest.json`、`tools/`、`docs/`、`research/`、`work/*.java`、`outputs/`内的核心代码、配置、发布检查点、ONNX、LUT和小型数据，以及 `reference_results/`。
3. 在同一项目创建Release，将 `sofc-training-data-v20.zip`、`sofc-comsol-reference.zip` 和 `sofc-advanced-validation.zip`作为附件上传。
4. 完整训练数据、COMSOL模型、下载缓存和重新运行结果由 `.gitignore`排除。使用Git上传时先检查 `git status --short`，确认没有将附件、临时产物或个人配置加入提交。
5. 仓库创建后，可在README补入实际Release链接；`data_assets.py download --base-url`也支持该地址，无需修改代码内的本机路径。

目标仓库为 `https://github.com/IrisVow/sofc-multiscale-microgrid`，可见性为Private，首次发布标签为 `v1.0.0`。原工作目录和上游原始项目继续保留。
