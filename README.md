# DSC6013 - IMV-LSTM 论文复现

Guo, Lin & Antulov-Fantulin, **Exploring Interpretable LSTM Neural Networks over Multi-Variable Data**, ICML 2019。
[论文](https://proceedings.mlr.press/v97/guo19b.html) · [作者源码](https://github.com/weilai0980/IMV-LSTM) · [作者推荐的 PyTorch 实现](https://github.com/KurochkinAlexey/IMV_LSTM)

本项目提供可校验的数据下载、原始代码下载、Windows/CPU 兼容修补、两种模型训练、预测和注意力导出、独立结果验证。
**当前目标是跑通复现链路，不等于完整复现论文结果。** 原始 TensorFlow 代码保留供对照，运行使用作者推荐的第三方 PyTorch 实现。该实现采用 MSE，未实现论文的概率混合 EM 训练。

## 目录

```text
code/                 下载、预处理、训练、验证代码
  vendor/             自动生成的兼容版模型（不提交）
data/raw/             原始数据（不提交）
data/processed/       处理后数组和数据说明（不提交）
output/               模型、预测、指标、注意力及验证记录（不提交）
upstream/             固定版本原始源码（下载生成，不提交）
tests/                长期保留的回归测试
docs/                 复现范围、方法差异及合作说明
sources.lock.json     数据与源码的固定 URL、commit、SHA256
requirements.txt      直接依赖版本
requirements-lock.txt 本次 Windows/Python 3.12 环境的全部依赖版本
```

## 从零启动（Windows PowerShell，Python 3.12）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -B code/setup_project.py
.\.venv\Scripts\python.exe -B code/prepare_data.py
.\.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B code/train.py
.\.venv\Scripts\python.exe -B code/verify_results.py
```

Linux/macOS 使用 `.venv/bin/python`，安装 `requirements.txt`；完整锁文件来自 Windows，不作为其他平台的通用锁文件。
默认 `train.py` 跑 SML2010、PM2.5、NASDAQ × IMV-Tensor、IMV-Full，共 6 次，每次最多 10 epochs、seed 42、每变量 10 个单元、CPU。验证集 RMSE 选最优模型，测试集只用于最后评价。

只跑一个组合或增加轮数：

```powershell
.\.venv\Scripts\python.exe -B code/train.py --datasets sml --models tensor --epochs 150 --run-tag sml_tensor_150
.\.venv\Scripts\python.exe -B code/verify_results.py --run-tag sml_tensor_150
```

不同实验请使用不同 `--run-tag`。程序拒绝覆盖已有运行目录。训练中每轮落盘历史和最佳权重；当前版本没有优化器断点续训，异常中断可保留诊断证据，需新 tag 重跑该组合。其他已完成组合无需重跑。

`setup_project.py` 对已有文件只校验，不重复下载；校验失败会停止，避免悄悄覆盖修改。网络失败可直接重试。所有下载地址和校验值见 `sources.lock.json`。

## 数据、指标与解释

- 数据快照来自上述 PyTorch 仓库，保留完整原始文件；数据下载不要求账户或付费。
- 用原始行顺序划分训练/验证/测试，下一步预测；仅训练集拟合缩放参数；不回填未来值、不制造末尾目标。
- SML2010 目标为房间温度，对齐论文图中变量列表；与第三方 notebook 的餐厅温度目标不同。
- 保存原始单位、训练集 min-max 单位两套 RMSE/MAE及上一时点预测基线。不同单位的误差不可混用。
- 注意力导出是该 PyTorch 模型的 alpha/beta，**不是论文 EM 后验重要性，也不是因果解释**。
- PLANT 数据、完整基线、五次重复、超参数搜索及变量选择实验尚未完成。

详细划分、窗口、缺失值处理和论文差异见 [复现说明](docs/REPRODUCTION.md)。协作见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 来源与发布范围

原作者源代码及第三方模型所有权归原作者；所下载快照未附明确 LICENSE，本仓库通过下载脚本取得原始源代码，不重新发布这些文件或替它们授予许可。
这里只发布本项目编写的复现工具、测试、文档和来源清单。PDF、数据、训练权重、输出、虚拟环境及缓存均不上传。
