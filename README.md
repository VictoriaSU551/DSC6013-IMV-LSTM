# IMV-LSTM：三套公开数据的完整实验与严格核验

基于 Guo, Lin & Antulov-Fantulin 的ICML2019论文独立实现。此次独立autoDL实验包含概率混合EM、五种子重复、基线、变量选择和解释核验；旧第三方MSE短跑链路保留在下方历史说明，二者输出和实现分开。

- **已验证**：150项HPO、105次全变量训练、90次变量选择重训；195正式深度run全部独立checkpoint核验通过，15组经典基线全部严格通过，科学测试77项通过。
- **实际结论**：Full在PM2.5上RMSE约21.13，persistence约22.13（改善4.53%）；SML/NASDAQ及Tensor存在明显负结果。六个IMV变量选择组合中，Pearson均优于后验排名。未重现论文普遍性能优势，未隐藏失败或挑seed。
- **复现范围**：SML2010、PM2.5、NASDAQ；PLANT原始快照未取得，16组主模型搜索并非论文300组穷举，缩放/PM协议及统计exact-diffuse初始化变体均披露。不能称四数据集数值完全复现。

阅读[详细中文实验报告](results/reproduction_20261018/复现实验报告.md)、[汇总指标](results/reproduction_20261018/summary_metrics.csv)、[数据与来源审计](docs/SOURCE_AUDIT.md)、[独立运行与恢复](docs/RUNNING_AUTODL.md)。完整过程和失败保留记录见[LEDGER](docs/LEDGER.md)，数值恢复细节见[REPAIR_NEXT](docs/REPAIR_NEXT.md)。

`code/repro/`为独立自写概率模型/训练/统计修复/核验/报告；`configs/reproduction.json`为预登记预算与种子，Linux直接依赖和实际环境锁为`requirements-autodl*.txt`。原始数据、无明确许可第三方代码、权重和大日志不发布；小结果与来源指纹在`results/`。本次公开commit不触发额外云端训练，已在独立autoDL环境执行测试与核验。

## 历史：2026-10-14的六次短跑链路

以下保留原说明，仅描述原commit89433d8及此前的短跑；其中“尚未完成”等状态不代表本次`code/repro`实验状态。

# DSC6013 - IMV-LSTM 论文复现

Guo, Lin & Antulov-Fantulin, **Exploring Interpretable LSTM Neural Networks over Multi-Variable Data**, ICML 2019。
[论文](https://proceedings.mlr.press/v97/guo19b.html) · [作者源码](https://github.com/weilai0980/IMV-LSTM) · [作者推荐的 PyTorch 实现](https://github.com/KurochkinAlexey/IMV_LSTM)

本项目提供可校验的数据下载、原始代码下载、Windows/CPU 兼容修补、两种模型训练、预测和注意力导出、独立结果验证。
**当前目标是跑通复现链路，不等于完整复现论文结果。** 原始 TensorFlow 代码保留供对照，运行使用作者推荐的第三方 PyTorch 实现。该实现采用 MSE，未实现论文的概率混合 EM 训练。

## 本次完成情况（2026-10-14）

- 原始源码与三套数据共 29 个文件通过 SHA256 校验。
- 两种模型 × 三个数据集共 6 次 CPU 训练，每次 10 epochs，均已完成。
- 独立重载检查通过：数据时序对齐、缩放、最优验证集轮次、测试预测、RMSE/MAE及注意力。
- 9 项回归测试、独立 Git 检出后的数据逐数组比对以及 GitHub Actions 测试通过。
- 这些证据仅确认运行与数值复核，不确认完整论文数值复现或所有模型优于简单基线。

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
