# IMV-LSTM 复现项目

Guo, Lin & Antulov-Fantulin, *Exploring Interpretable LSTM Neural Networks over Multi-Variable Data*, ICML 2019。
[正式论文与补充材料](https://proceedings.mlr.press/v97/guo19b.html) · [详细中文报告](reports/复现实验报告.md) · [最终审查](docs/FINAL_AUDIT.md) · [运行说明](docs/RUNNING.md)

这是三套公开数据上具有真实训练产物和核验证据的独立方法复现。**不是四数据集完整数值复现**：尚未取得可核实的PLANT快照，主模型搜索为16组而非论文300组；预处理、基线适配和统计初始化差异见报告。

公开数据实验包括150项验证集超参搜索、105次全变量深度训练、90次变量选择重训和15组经典基线。种子固定为11、23、47、71、101，确定性基线只拟合一次。195个正式深度模型的原独立推断证书及其源码、数据、checkpoint绑定均保留，整理时重新核验全部文件和保存预测指标。

Full在本次PM2.5协议下优于persistence约4.53%；其他五个数据集×IMV结构组合未优于persistence。六个IMV变量选择组合Pearson均优于后验重要性排名。负结果和全部种子保留，不能声称普遍重现论文优势。

## 最终目录

```text
code/                  数据取得、预处理、模型、训练与最终验证
configs/               实际实验配置和统计审计规则
tests/                 数学、数据、恢复、完整性和Windows锁测试
data/raw/              三套原始数据（本地）
data/processed/        处理数组及来源/切分/缩放说明（本地）
output/reproduction/   195个最佳模型、预测、解释及15组最终基线（本地）
reports/               唯一最终中文报告、7个CSV、2张图及审计证据
docs/                  来源、运行和最终审查说明
upstream/              固定版本参考源码（本地，可按SHA重新取得）
sources.lock.json      29项资产的固定URL、版本和SHA256
```

最佳模型和科学解释轨迹保留；恢复checkpoint、HPO权重、失败候选、排错日志、旧短跑和重复环境清理。清理后的留存文件以`reports/evidence/final_manifest.json`为准；`original_*`manifest只作历史来源证明。

## Windows复核本地成果

已配好`.venv-final`。重建环境使用Python3.12：

```powershell
python -m venv .venv-final
.\.venv-final\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-final\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONUTF8='1'
$env:PYTHONPATH=(Resolve-Path code).Path
.\.venv-final\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
.\.venv-final\Scripts\python.exe -B code/verify_final.py
```

默认核查全部留存SHA、195项原独立推断证书的当前绑定、全部保存预测指标、HPO和训练集变量选择。额外执行数据逐数组重建、当前设备15项代表checkpoint和六组统计参数重放：

```powershell
.\.venv-final\Scripts\python.exe -B code/verify_final.py --rebuild-data --replay representative --replay-statistics
# 在当前设备重新推断全部195个checkpoint：
.\.venv-final\Scripts\python.exe -B code/verify_final.py --replay all
```

原Linux/CUDA核验与当前Windows/CPU重放分开记录。可计算CSV和图的重建使用`code/rebuild_reports.py --destination output/rebuilt_reports`，不覆盖最终报告。

## 从GitHub冷启动

公开仓库提供自写代码、测试、配置、来源锁、报告、图表、汇总表和紧凑证据。原始数据、第三方未明确许可源码、大权重、环境和论文PDF保留本地，不进入公开Git历史。

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH=(Resolve-Path code).Path
.\.venv-final\Scripts\python.exe -B code/setup_project.py
.\.venv-final\Scripts\python.exe -B code/prepare_data.py
.\.venv-final\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
.\.venv-final\Scripts\python.exe -B code/verify_final.py --public
```

`--public`只核验公开代码/报告，不证明clone中存在未发布权重。NPZ压缩容器的元数据可能影响文件SHA；重建一致性应逐数组比较，不能改既有实验的数据绑定。正式重训需要新目录、匹配GPU和时间，见[运行说明](docs/RUNNING.md)。
