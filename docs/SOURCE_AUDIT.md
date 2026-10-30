# 文献与数据取得记录

## 固定来源
- 论文：用户提供17页PDF（含附录）；正式出版页面 https://proceedings.mlr.press/v97/guo19b.html ，正文与补充材料可交叉核对。
- 原作者仓库 https://github.com/weilai0980/IMV-LSTM ，固定commit eb7720175975f106ab3b8fd415e764e59bc574b4。README推荐KurochkinAlexey PyTorch实现；作者数据config只列本地 `../../dataset/dataset_ts/*dat`，没有对应文件/下载URL。
- 推荐PyTorch仓库commit c2552282d9995dfb03ef714168c78329e01374cf，公开SML/PM25/NASDAQ。29项资产SHA见sources.lock.json。
- 最终正式实验采用本项目基于论文公式独立编写的`code/repro/`，实现概率混合和冻结后验责任概率训练；MSE是同结构消融。早期第三方MSE短跑已清理，不混入正式结论。未重新发布无明确许可的第三方代码。

## PLANT可得性
2026-10-18调查，2026-10-30重新核查作者仓库、推荐PyTorch仓库、论文Ceci et al. 2017所指来源。未发现可核实为IMV-LSTM论文20842序列、9天气变量的同一原始快照。Ceci论文原始研究含多个意大利PV站点，本论文没有指定足以唯一重建的数据选择/清洗细节。
- 原研究作者托管全文 https://e6.ijs.si/~arashkovska/docs/pub/2017_IEEE_II_PV_Energy_Prediction.pdf
- 大学文献记录 https://ricerca.uniba.it/handle/11586/175028
- 研究组项目说明 https://kdde.di.uniba.it/projects/
本次不购买数据、不联系作者发送消息（用户未授权对外发信），不以其他光伏集代替PLANT。若之后取得可核实快照，可增加独立PLANT协议；当前结果应写“PLANT未复现，原始数据不可核实取得”，不能声称四数据集完整数值复现。

## 原文需要谨慎解释的细节
1. SML文字称每分钟，公开T15及图4明确15分钟；目标用房间温度，与第三方notebook餐厅目标不同。
2. PM2.5原文称41700序列，但未公开划分/缺失策略；图变量7个、文字提风向。当前保留风向4哑变量，合计11维，额外登记此差异。
3. NASDAQ81股+指数82维，训练/验证/测试原始切点35100/37830；不沿用第三方notebook窗口后错位切分。
4. 原作者配置SML loss=mse（与正文概率训练并不一致），其他主数据lk；本次EM为方法复现主线，MSE保留对照。
5. 原文表格未充分指定目标缩放，SML/NASDAQ小数不可自动当成原始摄氏度/指数点。只并列展示原始单位与本次train-minmax，不声称不同口径可直接匹配。
6. 图4的lag9..1对应历史hidden相对最后观测hidden的差距。窗口10时，这些hidden到预测目标实际为10..2步；最后观测通过h_T直接入输出头。
