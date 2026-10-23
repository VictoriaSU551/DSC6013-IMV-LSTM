# 独立autoDL完整实验运行与验收

本次公开范围为三套公开数据的完整预登记实验：150 HPO、105全变量、90选择、15经典组及195深度完整核验均已通过。完整原始数据/训练权重/逐点预测/日志/失败与审批证据保留在autoDL独立根目录；GitHub只存自写代码、配置、锁、小结果和证据摘要。PLANT、300组合穷举及逐项论文数值一致不在已完成范围。

## 新独立根目录的基础运行

```bash
# 在新的隔离根目录内clone本分支为repo，再进入repo；不要复用其他实验环境/cache。
python -m venv ../venv
export PYTHONPATH="$PWD/code"
export TMPDIR="$(dirname "$PWD")/tmp"
export XDG_CACHE_HOME="$(dirname "$PWD")/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" ../logs
../venv/bin/python -m pip install -r requirements-autodl.txt
../venv/bin/python code/setup_project.py
../venv/bin/python code/prepare_data.py
../venv/bin/python -m pytest -q --basetemp="$TMPDIR/pytest"
nohup ../venv/bin/python -B -m repro.suite > ../logs/suite.log 2>&1 < /dev/null &
```

GPU torch需使用匹配已有驱动的官方wheel；requirements-autodl-lock.txt是本次Python3.10/Linux/CUDA12.4实际环境，不可当所有平台通用锁。29项资产URL/commit/SHA位于sources.lock.json，setup只验证/下载项目路径。网络失败可重试，不能跳过hash或替换数据快照。

## 断点与隔离

深度运行每epoch保存权重、Adam与全部随机状态，恢复前核对配置、数据和模型/engine/lifecycle源码hash；每个run与suite各持文件锁，完成标记最后写。只在无活suite/worker时接续，不清空完成run、不手工并发同一输出目录。GPU单worker、显存最多60%，硬件有其他任务时等待；线程/依赖/数据/cache/输出不与其他实验复用。不要修改已生成结果绑定的核心源码。

进度在../progress.json，当前进程/日志在output/reproduction/active_worker.json，状态必须与PID/日志交叉核实。用户最新偏好是只最后汇报，4小时后台检查不发普通进度。

## 本次统计恢复的实际步骤

原统计优化100+300迭代后存在未收敛/NaN；原证据保留。以训练数据和原参数/有限训练checkpoint作为warm，使用独立UUID目录运行：

```bash
../venv/bin/python -B -m repro.stats_solver sml arimax --initialization exact_diffuse --warm output/reproduction/classic/sml_arimax/fit_details.json
# dataset/kind对应本次六组，各使用自己的原训练来源，不跨组借模型。
../venv/bin/python -B -m repro.repairs sml arimax <新候选目录> code/repro/stats_solver.py --initialization exact_diffuse
```

v1仅能审批原失败/未收敛组。原SML STRX/NAS STRX/NAS ARIMAX有成功flag却未通过事后统一训练score审计，用audited_repairs v2针对固定证书指定身份审批：

```bash
../venv/bin/python -B -m repro.audited_repairs nasdaq arimax <已收敛候选目录> code/repro/stats_solver.py output/reproduction/statistical_audit_certificate_d05027a4c3b9.json code/repro/audit_stats.py --initialization exact_diffuse
```

**证书和policy root只适用于本次原结果身份**。全新重跑、数据/源码/原结果改变时，必须独立生成新的统一审计证书及明确的新版本审批策略；不能改本次固定证书、替换已审批registry或关掉检查。此分支保留本次审批逻辑与真实固定证书，不承诺任意新环境一条命令得到相同统计优化轨迹。新环境重跑主训练后若数值不同，需按同样科学标准重新诊断并记录。

PM ARIMAX只在粗步长方向检验失败，BFGS本身成功且score满足门槛。用repro.stats_certify对原endpoint做固定参数、多步长认证（不优化），通过后source_path换为code/repro/stats_certify.py执行对应审批。真未收敛不能用此认证变成成功。具体来源/不可变版本/失败历史见REPAIR_NEXT和LEDGER。

本次NASDAQ ARIMAX按同一预定流程经历BFGS1200、L-BFGS-B2000均上限失败，Powell300返回flag成功但score仍不够，最终BFGS834收敛；没有根据测试表现选择求解器或降阶，训练约49.7小时。scripts/reproduction/保存本次自写监督器和完整深度审计辅助脚本。

## 最终核验与报告

```bash
../venv/bin/python -B -m repro.audit_stats_multistep
# 六组PASS后，保留真实审计为strict_final_coordinate_audit.json。
../venv/bin/python -B -m repro.verify --deep-audit output/reproduction/deep_audit_<本次真实编号>.json
# 不使用审计复用时不传--deep-audit，会逐一重新前向。
../venv/bin/python -B -m repro.report
```

深度审计复用严格要求完整195集合、相同核验函数/源码/data/artifact proof，并再次核对checkpoint等全部artifact hash，不凭旧PASS跳过完整性检查。经典15组逐点对齐canonical标签/target row并复算原与缩放RMSE/MAE；状态空间审批另做固定物理参数过滤、训练LL/QR映射与未来标签隔离。CSV短十进制经原dtype还原，指标门槛保持rtol1e-9、atol1e-10。

报告生成../delivery，只有7个汇总CSV、2图、详细MD及SHA。生成后还需真实数值/图表复核、许可与敏感信息检查、GitHub发布和Mac SHA核对；delivery_ready不代表所有用户交付自动完成。所有原始结果和失败仍在autoDL，不发布数据、权重、第三方不明许可代码或大日志。新发布commit跳过GitHub云端CI；本次77项测试和完整核验均在已有autoDL实例完成，无额外资源/API费用。
