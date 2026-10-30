# 复核与重新运行

最终本地包保留最佳模型、预测、训练解释轨迹和统计物理参数，使用`code/verify_final.py`复核。默认复算全部保存预测指标，并在全部文件SHA相同的条件下重新绑定原195项独立checkpoint推断证书。`--replay representative`重新载入三数据集五种深度结构的seed11，共15项；`--replay all`重放195项。`--replay-statistics`以固定物理参数重新过滤六组状态空间模型，不再优化。

原195项推断包含全部测试点和EM训练后验；当前本机代表重放不替代原全部推断。原统计score/边界/多步长证书来自Linux训练环境，跨平台固定参数预测另行检查，不混称同一认证。保存指标保持经典rtol1e-9/atol1e-10的精度合同；深度预测先恢复scaled float32，再转float64逆变换。

PM2.5共线风向变量的QR主元在不同BLAS平台可能选出不同列。保存的物理参数必须与**原训练列身份**一起载入；最终重放使用`fit_details.json`的`exogenous_original_indices`，不重新选择QR列。该修复位于最终验证适配层，原数值训练源码及结果不变。

`--rebuild-data`在内存从原始文件重建全部数组并逐元素比较，不覆盖数据。HPO按150项完整候选及各自验证历史最小值核对，变量选择只使用训练集。

## 独立重训三公开数据实验

在**新独立目录**clone仓库，使用Linux、Python3.10、PyTorch2.6和可用CUDA。`requirements.txt`为直接依赖；`reports/evidence/original_linux_environment.txt`只记录旧环境，含平台相关包及旧本地wheel路径，不作跨平台安装清单。Windows实际环境见`requirements-windows-lock.txt`。

```bash
python -m venv ../venv
mkdir -p ../logs ../tmp ../cache/matplotlib
export PYTHONPATH="$PWD/code"
export PYTHONUTF8=1
export TMPDIR="$(dirname "$PWD")/tmp"
export XDG_CACHE_HOME="$(dirname "$PWD")/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
../venv/bin/python -m pip install -r requirements.txt
../venv/bin/python code/setup_project.py
../venv/bin/python code/prepare_data.py
../venv/bin/python -B -m pytest tests -q -p no:cacheprovider > ../logs/tests.log
../venv/bin/python -B -m repro.suite
```

原suite执行150项HPO、105全变量、90变量选择和15组基线。GPU单worker、线程2；深度训练保存优化器/RNG以恢复。最终精简包已删除恢复文件，**不能当作中断训练目录续跑**。新统计基线若不收敛，必须对新运行独立诊断并生成新的来源绑定；不能改旧policy或复用本次固定审批冒充通过。

`code/repro/`保存实际训练、求解和原完整目录审查逻辑；这些工具针对清理前的研究运行目录。最终目录用`verify_final.py`。跨平台重训不保证逐bit相同，也不弥补PLANT和未公开预处理缺口。重建CSV和图会写入新目录，最终审阅报告不变。
