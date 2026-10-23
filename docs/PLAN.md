# IMV-LSTM Complete Public-Data Reproduction Implementation Plan

**Goal:** 在独立autoDL项目中补全论文核心方法、三套公开数据上的比较与解释实验，尽力取得PLANT；最终交付精简结果与详细中文分析。
**Architecture:** 保留原有兼容PyTorch MSE链路；新增独立 `code/repro/` 实现论文式(1)-(13)，分离搜索、重复训练、解释与独立核验。任务状态原子写盘、失败保留、重启跳过通过验证的结果。
**Tech Stack:** Python3.10 / PyTorch2.6 CUDA12.4 / NumPy1.26 / sklearn / statsmodels / XGBoost。所有包独立安装，不复用任何实验环境。
**Spec:** `docs/USER_CONTRACT.md`、论文正文与附录；基线代码起点89433d8965c8957bd66598b78a83fd06375ba6db。

## Global Constraints
训练窗口10；仅训练集拟合缩放；SML原始行3200/3600、NASDAQ35100/37830；PM2.5为已登记60/20/20时间切分（论文未提供）。5种子11/23/47/71/101。Adam batch64。验证RMSE选超参/epoch，不读测试选择；原单位和训练minmax单位同时报告。随机性/环境/源码/数据哈希均落盘。

## Review Focus
EM后验必须含真实训练y且detach；推断预测不能用y；时间注意力排除最后状态并明确lag定义；Tensor变量隐藏流独立；Full门允许跨变量；变量筛选只看训练集；统计标准差与标准误分开；失败/未收敛不能算完成；不可将不同缩放误差直接等同论文数值；不可覆盖已完成目录。

## Task 1: 环境与数据审计
- [ ] 下载29个哈希固定资产，独立venv安装并冻结Linux依赖；原仓库9项测试。
- [ ] 审计SML采样/断点/目标，PM缺失窗口与风向编码，NASDAQ padding来源与边界。
- [ ] 调查PLANT原始发布来源；不可得则保存来源证据并明确留缺，不用其他光伏集冒充。

## Task 2: 原生论文模型与公式验证
Files: `code/repro/models.py`, `tests/test_paper_models.py`。
- [ ] 先测试高斯混合密度/后验、EM与NLL梯度一致性、预测不依赖标签、Tensor隔离、Full参数量与注意力维度、batch1。
- [ ] 实现Full/Tensor、变量独立均值方差头、共享变量打分、T-1时间注意力、EM q与全局I、均值加权预测。
- [ ] LSTM、DUAL encoder-decoder attention、RETAIN回归基线；各自结构/时序检查。

## Task 3: 训练、搜索、恢复与登记
Files: `code/repro/engine.py`, `code/repro/suite.py`, `configs/reproduction.json`。
- [ ] 持久化优化器/RNG/epoch以恢复；保存最优验证权重、训练/验证曲线、时间、训练I/T、变量排名与原/缩放单位指标。
- [ ] 主模型每dataset/model预定16组覆盖论文全部units/lr/dropout/L2值的均衡子集（原文全笛卡尔积300组/组合，明确本次未穷尽），搜索seed11，最多80epochs/patience20；主实验150epochs/patience35，5固定seed。
- [ ] 深度基线每组合6组units/lr候选，5seed；IMV-MSE消融用相同选定架构5seed。
- [ ] 3数据×(IMV2+MSE2+LSTM/DUAL/RETAIN3)×5=105最终深度运行；主方法HPO96、深度baseline HPO54共150搜索单位。
- [ ] RF/XGT/ENET/STRX/ARIMAX+Persistence：验证集选择，随机基线5seed；确定性baseline单次拟合并标注，不伪造独立重复。

## Task 4: 变量选择与可解释性
- [ ] IMV2分别按训练后验I、训练Pearson top50%重训；DUAL/RETAIN按训练attention/contribution top50%重训，3datasets×6methods×5seeds=90运行。每seed使用该seed全变量模型训练集排名。
- [ ] 导出5seed变量I/先验attention差异、时间T、训练I轨迹、排名稳定性；只做预测关联解释，不当作因果。
- [ ] 参数量核对Lemma3.1；同D比较标准LSTM/Full/Tensor更新参数和时间。额外说明网络总参数和recurrent参数不同。

## Task 5: 独立核验、报告与GitHub
- [ ] 独立重载每个最终checkpoint并逐点复算预测/指标，检查target_row、哈希和最佳val epoch；搜索文件不作为最终结果。
- [ ] 汇总均值/SD/SE、相对persistence与基线差异、论文口径差异；纳入失败/缺失/未收敛证据。统计测试若使用时间序列需block bootstrap。
- [ ] 详细中文MD解释全部方法、数据、搜索范围、实验过程、结论、偏差、资源、复现命令；少量概括图/CSV。
- [ ] 完整自写代码与文档推GitHub，精简交付回Mac，验证SHA后关闭本任务heartbeat。

## 进度与资源规则
进度分母为环境数据审计1、方法验证1、150搜索、105全变量深度运行、90变量选择、每数据5类经典baseline15、独立验证报告发布1，共363单位；baseline内部搜索不另计。PLANT作为不可得项显式登记，不把未做内容说成100%论文复现。总体进度是本公开数据计划完成率，不是论文数值匹配率。任何计划调整须在LEDGER说明原由并重算分母。
每个worker限2CPU线程，GPU单任务、最多占60%显存；检测其他计算进程时等待，不抢占。磁盘不足仅清理本任务可重下载pip缓存，不清其他实验。不设付费资源自动扩展。


## 执行结束状态
三公开数据预登记363单位已执行且通过相应验收，15经典与195正式深度完整核验PASS，77测试PASS。论文普遍性能优势未重现，PLANT缺失、16/300搜索和预处理/初始化差异不隐藏。结果和完整实际过程以results/reproduction_20261018/复现实验报告.md、RUNNING_AUTODL.md及LEDGER为准。
