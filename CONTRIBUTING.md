# 协作约定

1. 从 `main` 创建功能分支，提交前同步最新代码，用 Pull Request 合并。
2. 先按 README 安装环境、下载固定版本源代码和数据。不要提交 `upstream`、`code/vendor`、`data/raw`、`data/processed`、`output`、PDF、环境和凭据。
3. 修改预处理必须更新 `docs/REPRODUCTION.md` 的协议说明和回归测试，重新生成独立的新实验结果，不沿用旧模型的数值结论。
4. 模型兼容性修改放在 `code/setup_project.py` 的补丁步骤，不手改生成的 vendor 文件。不要把兼容性修复描述为算法创新。
5. PR 写清改动目的、运行命令、测试结果以及影响的实验协议。运行 `python -B -m pytest tests -q -p no:cacheprovider`；涉及训练结果时，还要运行独立验证脚本。
6. 每个实验使用唯一 `--run-tag`，保留 config、history、best.pt、预测和验证记录。完整复现前不声称达到了论文精度。

后续论文级工作：恢复原始 TensorFlow 概率混合逻辑或忠实移植并验证；获取 PLANT；严格对齐数据/缩放；按论文范围调参；五个随机种子；统计/机器学习/深度基线；无测试集参与的变量选择；复现表格与解释分析。
