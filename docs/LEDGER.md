# IMV-LSTM 执行记录

## 2026-10-18 初始化
- 已读取本地17页论文（正文+附录）以及用户仓库README、预处理、训练、原作者配置。
- 远端GPU2080Ti当前空闲；另有CPU任务，保持不动。数据盘仅7.6GB可用，独立项目放系统盘/root/imv_lstm_20261018（28GB可用），避免占用已有实验数据盘。
- SSH专用密钥仅保存在Mac ~/.ssh，远端只添加对应公钥，不记录明文密码。
- GitHub直连超时，使用本项目专用反向SOCKS端口18743从Mac转发完成clone，不改全局网络。
- 保留起点代码89433d8，新分支reproduce/20261018，所有新增工作在全新远端clone；不共享他项目checkout或数据。
- Ruling: 按用户持续自主执行指示省略技能默认的设计/计划确认，直接执行已授权研究范围；不使用逐任务子代理以节省token。
- Ruling: 原作者公开源码SML配置用mse，其他数据用lk；本次论文方法主线使用论文EM等价梯度，MSE独立标注消融。q在每batch以当前参数计算并detach时，第一二项梯度等于混合NLL；第三项只更新全局I，不给模型增加虚构正则梯度。
- Ruling: 论文没有充分定义缩放、PM划分与PLANT原始文件，保留明确可审计的既有时间协议；不宣称原表格严格数值对齐。
- 后台heartbeat已创建（id imv-lstm），每4小时静默推进，遵守用户最新只最终汇报偏好。

## 首次独立代码审查与修复（正式实验启动前）
- 一次只读独立审查发现5项：scheduler绕过完成项的配置/源码校验、result到manifest的崩溃窗口、未收敛统计组被计作成功、orphan worker重复写入、经典预测验证未对齐数据源。
- 修复：worker层持有独立run文件锁；result+manifest后最后原子写complete标记，缺标记从last checkpoint重建；所有重用走配置/源码/数据与完整manifest验证，已存HPO赢家也重新检查候选集合；未收敛统计不计成功，正式验证拒绝；经典预测逐行检查实际数据target_row/label/seed数。
- 生命周期5项回归：观察到5项预期RED（缺失实现），实现后5项GREEN（0.059秒）。完整科学测试仍等待独立依赖安装，不提前宣称通过。
- 统计拟合增加唯一预登记重试：100迭代不收敛则从当前训练参数继续300迭代；仍未收敛保留并停止最终验收，不能假称收敛。
- 调度顺序调整：先完成三数据集核心/解释运行，再做经典组，避免某统计求解器问题阻挡其他公开数据方法实验；分母不变。
- 网络诊断：官方torch直连大文件下载读超时，反向代理大文件也慢；改为独立pip使用公开阿里PyPI镜像，实测约300KB/s。没有额外付费服务。NASDAQ经Mac内存流直接上传，Mac不落数据文件；上传前SHA256已校验，远端完整上传后由bootstrap再次校验。
- 预启动自检：所有新Python源码compileall通过；生命周期5项测试再次通过。科学依赖仍在独立安装，正式训练严格由完整pytest通过之后启动，不把静态检查当数学测试通过。
- 安装watchdog只监控本项目pip，失败时复用本项目下载缓存最多重试3次；bootstrap等待依赖与29项SHA全部通过后执行setup/prepare/完整pytest/suite。
- 本任务预启动审查标记APPROVED_FOR_RUN已生成，含义为允许后台执行强制测试门，不表示实验已成功或论文已复现。

## 2026-10-19 02:49 HKT 后台检查
- 独立环境安装成功，29个固定资产均经setup验证，NASDAQ已完整远端保存；完整科学与回归测试31 passed in 5.16s。当前SML已完成50个HPO，正式重复持续推进，worker存活、日志更新；无重复启动或改动训练代码。正常进度仅记账，不向用户周期汇报。

## 2026-10-19 06:59 HKT 静默检查
- SML已完成50个HPO、35个全变量重复与30个变量选择重训；当前PM2.5搜索持续正常，当前worker存活且验证日志更新。
- SML预先固定seed11的Full/Tensor两项在CPU独立重载核验通过（逐点预测、指标、训练后验I、文件哈希与最优验证轮次）；这只是阶段抽查，最终195项完整核验仍待完成。磁盘剩余约17.8GB，无需资源扩展，无正常任务重启。

## 2026-10-19 10:59 HKT 静默检查
- SML阶段完成后，PM2.5的50个预登记HPO也已完成；当前正式5种子与MSE消融按队列正常推进。累计HPO100、全变量正式38、变量选择30，加审计2共170/363单位。
- suite与当前worker均存活，验证日志持续更新，无失败记录、无异常bootstrap输出；磁盘可用17.6GB。未重复启动任务，未改变搜索/训练协议、源代码或测试集选择规则。继续既有后台队列，不向用户周期汇报。

## 2026-10-19 23:04 HKT 静默阶段核查
- PM2.5全变量35项正式运行已完成，正在变量选择重训。当前216/363单位，worker存活、日志正常，剩余磁盘17.2GB。
- PM2.5固定seed11的Full/Tensor两项CPU独立重载抽查通过：逐点预测/原与缩放指标/训练后验I/哈希/最佳验证轮次均核对；完整195项核验仍待全部队列结束。未改动训练代码、超参选择或已完成证据，未重复启动worker。

## 2026-10-20 07:07 HKT 静默阶段核查
- SML和PM2.5的全部预登记深度实验已完成：共100个HPO、70个正式全变量重复、60个变量选择重训。NASDAQ正在按相同协议搜索，当前nasdaq_full_03，累计251/363单位。
- suite/worker均存活，最近验证日志持续更新，未发现失败或异常bootstrap输出；系统盘可用16.5GB。没有重启正常任务、改动已锁定搜索规则或读取测试结果来调参，继续后台队列。

## 2026-10-20 15:10 HKT 静默阶段核查
- 三数据集全部150项预登记HPO已完成。NASDAQ正在正式5种子与MSE消融阶段，累计全变量正式81项、变量选择60项，整体293/363单位。
- 当前suite与worker存活，日志持续更新，未发现失败；磁盘可用14.6GB。继续原队列，尚不宣称完整实验或最终验收完成，不向用户周期汇报。

## 2026-10-21 03:11 HKT 静默阶段核查
- 三数据集全部150项HPO和105项正式全变量重复已完成；正在NASDAQ变量选择重训，累计321/363单位。正常suite持续运行，无失败记录；磁盘剩余约13GB。
- NASDAQ预登记seed11的Tensor/Full两项CPU独立重载核验通过（预测/指标/后验I/最优轮次/哈希）；三个数据集均完成相同阶段抽查，最终195项完整验证仍待全部实验完成。没有重复启动正常worker或修改已完成训练证据。

## 2026-10-21 15:25 HKT 统计基线未收敛诊断
- 全部150项HPO、105项正式全变量与90项变量选择已完成；经典组推进中，SML ARIMAX(10,0,10)在100+300迭代后仍未收敛，原始结果/参数/警告完整保留且不计成功。
- 训练诊断：3180个有效训练窗口，目标训练minmax尺度SD=0.18427，16维外生矩阵条件数约87.6；原优化器触及300迭代上限，归一梯度仍显著（sigma方向约1.52），不是可直接放宽验收的收敛状态。
- 已核对statsmodels官方SARIMAX.fit文档和本机0.14.4接口：支持optim_score/optim_complex_step。启动独立、只读训练数组的score probe，从未收敛参数继续，依次尝试complex-step梯度的L-BFGS最多1000迭代、BFGS最多500迭代。按训练似然与真实优化器收敛判定，不读验证/测试标签来挑优化方法。
- 诊断位于项目diagnostics/sml_arimax_score_probe，日志logs/sml_arimax_score_probe.log。它不改变原经典实现或任何已完成结果，其他经典任务继续正常运行。即使诊断成功也须独立重放与来源记录后才能作为正式修复，不提前计完成。

## 2026-10-21 16:55 HKT 统计数值故障恢复
- 原suite在PM2.5 ARIMAX拟合产生NaN诊断值时被严格JSON序列化拦截，两次原样重试均失败，停止于354/363；没有把NaN变成指标或伪造成功。
- SML训练专用complex-step延长拟合仍未被求解器判为收敛：L-BFGS触及1000迭代，BFGS因精度损失退出，最大归一梯度约4.7e-5；原参数/所有警告和诊断均保留，不放宽门槛冒称收敛。
- 启动独立恢复协调器diagnostics/continue_classic.py，使用同一suite.lock和原worker/原经典源码，只推进尚未尝试的NAS经典组，跳过已完成及待修统计组；它会持续更新progress/active_worker并保留新失败。深度实验及既有经典证据不重算、不覆盖。
- 后续统计修复将探查方差profile likelihood与外生设计矩阵预条件（相同ARMA阶数/相同变量与时间协议），只依据训练似然和求解器状态，必须进行参数映射等价性与固定参数重放验证。
- 已启动sml/arimax、pm25/arimax、pm25/strx三项训练专用QR预条件/profile探查，均位于diagnostics/*_profile_probe并各持文件锁。ARIMAX将公共方差从训练似然中解析剖面化，降低病态优化维度；外生变量QR是可逆重参数化，包含数值等价断言。所有成功候选必须映回原始参数并核对同一训练似然后才输出候选参数，尚不替换正式结果。
- 资料依据：statsmodels SARIMAX文档concentrate_scale说明（https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html），以及本机0.14.4实现。经典恢复协调器在同一项目后台推进NAS机器学习组，未重算此前已成功组。

### 2026-10-21 numerical diagnostics continuation (training only)
- Recovery coordinator PID615101 holds the suite lock and continues untouched NASDAQ classical units. Old failed or unconverged fits remain preserved. Progress is 354/363 while NASDAQ ENET is running.
- QR/profile probe: PM2.5 STRX optimizer converged and the mapped original training likelihood agrees to <1e-8; it is a candidate only and still requires independent fixed-parameter replay, causal checks and a provenance manifest before acceptance.
- SML ARIMAX profile candidate was rejected: concentrated and original approximate-diffuse training likelihoods did not agree (14699.772 vs14633.348). No success marker was written. The new log-variance diagnostic was also rejected by an independent directional derivative check; its status records the discrepancy. These failed diagnostics must not be promoted.
- Additional immutable exact-diffuse ARIMAX probes were launched for SML and PM2.5. This is an explicitly documented initialization sensitivity variant, not an assertion of equivalence with the original approximate-diffuse likelihood. Orders, predictors, splits and data remain unchanged; no validation/test target values are used by the probes. Both profile-to-unprofiled mapping and convergence must pass before any later consideration.
- Documentation supporting the initialization option: https://www.statsmodels.org/v0.14.4/_modules/statsmodels/tsa/statespace/sarimax.html . Concentrated scale must be frozen at its training estimate for any later validation/test replay.
- PM2.5 exact-diffuse profile probe hit a statsmodels 0.14.4 backend shape error when constructing low-memory results (kalman_filter.update_filter: shapes26260 vs2). Numerical estimates were not accepted. A new immutable v2 probe uses low_memory=False; it keeps the same objective and model, and imports any finite prior training checkpoint. Only our SML diagnostic PID615967 was stopped to avoid the known identical backend path; the recovery coordinator and all registered experiments were left running. Both SML/PM v2 probes were launched. Inspect diagnostics/*_exact_diffuse_v2_probe/status.json on the next check.

### Statistical repair acceptance, independent review and provenance
- All 5 NASDAQ classical groups completed; original converged groups now12/15, progress359/363=98.90%. Original SML ARIMAX, PM2.5 STRX and ARIMAX remain unresolved, their directories are immutable.
- Independent review correctly rejected the initial PM STRX function-stopping flag because max optimizer gopt was about0.479; later strict BFGS attempts still had precision loss. None was accepted or counted. A correct solver return flag alone is insufficient.
- Added `code/repro/repairs.py` sidecar resolver, fixed-variance replay and approval gates; verify/report/suite now resolve approved repairs while preserving original failures. Repairs cannot replace an original successful group or a different prior approval. Hashes bind source/runtime/data/metadata/original failed evidence, precondition matrices, optimizer attempts and archived starting inputs.
- Added `code/repro/stats_solver.py`: recorded training-only start/end provenance, invertible QR, log variances and scaled regression coordinates, nonzero stable full-order ARMA starts, complex-step score validated by independent central directions at start and finish. Solver success requires finite coordinate score<=1e-6 and nonnegative boundary variance KKT violation<=1e-6. Test metrics never enter initializations or stopping rules. Each attempt has a fresh UUID directory; failed solver steps continue from the best finite training point instead of discarding useful iterations.
- New candidates for SML ARIMAX and PM STRX/ARIMAX use exact diffuse initialization. This is an explicit initialization variant; it must be disclosed in the final report, not described as equivalent to the failed approximate-diffuse objective. All replay uses physical parameters and concentrate_scale=False, freezing training variance.
- Meaningful RED→GREEN tests cover one-step current/future-label isolation for both models, no replay fitting, nonfinite/unconverged/weak-score/boundary failure rejection, final-direction rejection and positive acceptance, path escapes, unsupported replacements, missing completion markers, original-success protection, approval immutability, provenance mismatch, coordinate chain-rule correctness and preservation of old attempt evidence. Full suite:55 passed in5.29s (`logs/tests_after_strict_review.log`). Focused independent reviewer then found no remaining blocking issue.
- Independent full deep checkpoint audit PID628399 is running onCPU2threads; evidence `deep_verification_status.json` and `repo/output/reproduction/deep_audit_*.json`, all195 expected. This alone is not final overall verification.

### Strict audit v2, complete deep verification and multistep certification
- All195 official deep runs independently verified PASS, first CPU and remaining free GPU with per-run immutable proof reuse. No retraining or discarded audit rows.
- Implemented/reviewed immutable v2 approval for the three original optimizer-success groups rejected by the common-coordinate training audit; exact certificate/program hashes and policy-root pinning prevent substitution. Consumer strict-complete gates block rejected originals. SML STRX accepted; NAS STRX/ARIMAX remain live training-only optimizations.
- PM ARIMAX BFGS converged (trainLL53795.0174213, score4.460368e-8); coarse h1e-5 central difference had O(h²) truncation error. Fixed endpoint certification on predeclared12-step grid and3fixed directions required three adjacent agreeing steps, without changing parameters/score thresholds or reading test performance. Full curves, original successful attempts, source/data/runtime/start/end provenance and warm-input hashes archived. Independent reviewer found and verified closure of the parent warm-input provenance gap. Accepted PM ARIMAX sidecar pm25_arimax_ea4d3b2c40_d16815aed6.
- Final six-group audit now uses the same multistep rule with actual positive variances unchanged. Independent review required all actual solver/certifier/audit/policy sources pinned at start and rechecked at end; implemented. Supervisor v3 PID632807 will gate verify/report on six PASS.
- Latest full tests74 passed in6.11s, compileall passed (logs/tests_reporting.log), all temp paths isolated. Strict completed360/363 with13classical groups; final report review, GitHub publish and Mac compact delivery remain outstanding.

### 2026-10-21 20:37 UTC compact background check
- NASDAQ STRX BFGS converged in787 iterations, trainLL127789.9103728, max coordinate score3.102014e-8, boundary violation0; independent fixed-parameter/causal/provenance verification passed and v2 sidecar nasdaq_strx_auditv2_742e1bcf_61769b09_d05027a4 approved automatically. Original result remains unchanged.
- Strict classical14/15, public task361/363. NASDAQ ARIMAX PID630303 alive at700 BFGS iterations; supervisor632807 healthy, no acceptance errors. No task restarted or threshold changed. Full overall audit/report/publish/delivery remain pending.

### 2026-10-22 00:37 UTC compact background check
- NASDAQ ARIMAX BFGS hit the predeclared1200-iteration limit (not successful; trainingLL127788.7141923, max coordinate score0.01000886, boundary0). No convergence claim or approval. The unchanged solver automatically continued with its predeclared L-BFGS-B stage, currently50 iterations. PID630303 and supervisor632807 healthy; no duplicate launch or source/config change. Strict14/15,361/363 remain unchanged.

### 2026-10-22 16:46 UTC numerical continuation check
- NASDAQ ARIMAX L-BFGS-B hit its predeclared2000-iteration limit, trainingLL127793.9423893, max coordinate score0.0015393173, boundary0. It improved training likelihood/score relative to the previous endpoint but remains unsuccessful. The immutable solver has entered its predeclared Powell stage, followed by a final BFGS stage. All failure attempts remain archived, no thresholds or parameters were manually changed, no test performance used. PID630303/supervisor632807 healthy; no duplicate work. Strict14/15 unchanged.

### Final statistical convergence and complete independent verification
- NASDAQ ARIMAX completed its unchanged training-only solver sequence: Powell300 returned success but score1.521e-4 was insufficient; final BFGS834 succeeded with trainLL127803.7616708, max coordinate score1.5248436e-8 and boundary0. Final v2 fixed-parameter replay/causality/provenance passed, sidecar nasdaq_arimax_auditv2_742e1bcf_7969af03_d05027a4 approved. About49.7h for this continuation; no order reduction or test-driven method choice.
- All six state-space groups passed the uniform multistep common-coordinate audit. All195 deep and15classical groups then passed complete independent verification. The initial final verification stopped on overly literal CSV64 arithmetic: pandas emits shortest float32 decimals, and XGT/ENET original inverse transforms use float32. Reconstructing original dtype and canonical labels reproduces all metrics within1.42e-14; original metric tolerance unchanged, strict inverse-scale/label roundtrip checks added. Three new parametrized RED→GREEN tests and focused independent review passed. Full release suite77 passed in5.63s; compileall passed (logs/tests_release.log).
- Descriptive diagnostics after evaluation found NASDAQ test targets97.40% above/below training minmax, systematic negative prediction bias, and limited Full ranking stability. These are post-evaluation interpretations, not inputs to new tuning; original predictions, seeds, search and all negative results remain preserved. PM Full is the only IMV/persistence positive comparison; all six IMV Pearson selection means beat posterior selection means.
- Detailed Chinese report and two plots reviewed. all_runs_completed now means budget/checkpoint completion, separately from statistical_converged, avoiding a claim that neural training reached a gradient optimum. XGT five seeds give identical predictions under default full subsampling; no artificial independent uncertainty claim.
