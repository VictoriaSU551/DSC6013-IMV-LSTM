# 最终验收与交付状态

本次150HPO、105全变量、90变量选择、15经典组和195正式深度run核验已全部通过。六组状态空间统一训练score/边界/多步长方向审计全部PASS。NASDAQ ARIMAX最终BFGS834收敛，score1.52e-8；完整失败与接续序列见LEDGER。此前根strict_recovery_failure.json为已修复的CSV精度核验历史失败，不是当前阻塞；副本保留diagnostics/final_verification_csv_precision_failure.json。

最新77项科学测试通过，完整深度审计复用时重查195集合/源码/数据/全部artifact hash；15经典逐点对齐与原精度独立重算通过。原metrics/预测/模型与数据从未为通过核验改写，指标容差保持rtol1e-9/atol1e-10。验证器仅恢复pandas短十进制的原dtype，用canonical标签，新增错误指标/预测篡改反例。详细报告为../delivery/复现实验报告.md，两个图已目视核验。

代码和实验快照在发布流程中，Mac交付与GitHub最终commit/SHA必须实际核对后才结束。公开计划363/363是已执行工作计数，不代表四数据集或论文数值完全一致。若交付未完成，继续发布/回传，不能重复启动已结束训练或把历史失败当新任务。具体当前状态以根progress.json、publication.json和delivery_receipt.json为准。

不可编辑已冻结models/engine/lifecycle/classical、repairs.py(ea4d3b2c40)、stats_solver.py(6d71fea26b)、audited_repairs.py(742e1bcf)、stats_certify.py(fc74db4db3)、configs/statistical_audit_policy.json、audit_stats.py及固定证书statistical_audit_certificate_d05027a4c3b9.json，否则会使真实审批与已完成来源无效。所有原始训练/失败/审批/完整预测和权重都留autoDL唯一根/root/imv_lstm_20261018，不复用其他实验。

实际可运行命令、统计恢复和新运行的证书版本限制见RUNNING_AUTODL.md。PLANT缺失、16非300穷举、PM/缩放协议差异、exact diffuse初始化、事后数值标准、CSV精度、神经网络预算完成不等于梯度收敛、所有负结果必须继续披露。禁止测试调参/挑seed/隐藏负结果。仅最终7CSV+2图+详细MD+SHA回传Mac；验收/发布/回传齐全才暂停heartbeat。
