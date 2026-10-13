# 数据

运行 `python -B code/setup_project.py` 下载到 `data/raw`，随后运行 `python -B code/prepare_data.py` 生成 `data/processed`。

| 数据 | 原始文件 | 下载快照 |
|---|---|---|
| PM2.5 | PRSA_data_2010.1.1-2014.12.31.csv | KurochkinAlexey/IMV_LSTM 的 PM25 目录 |
| SML2010 | NEW-DATA-1.T15.txt、NEW-DATA-2.T15.txt | 同仓库的 SML2010 目录 |
| NASDAQ | nasdaq100_padding.csv | 同仓库的 NASDAQ 目录 |

来源仓库 commit、下载 URL 与 SHA256 全部固定在根目录的 `sources.lock.json`。仓库没有附带 PLANT 数据，本项目也没有伪造或替代 PLANT。

处理后每个数据集保存 `.npz` 和 `.json`。JSON 记录特征顺序、目标、切分、丢弃窗口数以及原始文件/处理代码/数组的校验值。调整协议后必须重新生成；不要将旧模型与新数据混用。
