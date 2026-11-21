# DFDC 跨数据集迁移实验

## 为什么增加 DFDC

当前 Adapter 的开发与评估使用 Celeb-DF v2 官方测试清单中的 518 个视频。
该清单内部按视频和伪造来源分组后，只有 78 个内部测试视频。继续在该清单上
进行方法设计会放大自适应复用和过拟合风险。因此，新的实验协议把 DFDC part 0
作为 Adapter 开发集，并把 Celeb-DF 518 只作为外部测试集。

这不是把 Celeb-DF 的 78 个内部测试改回训练集。全部 Celeb-DF 视频都不参与
DFDC Adapter 的训练、验证或超参数选择。

## 数据来源与限制

使用以下两个 Kaggle 用户转存文件：

- `namht/dfdctrainpart0`：DFDC `dfdc_train_part_0` 的 1,334 个视频，
  12.38 GB；
- `zaharch/train-set-metadata-for-dfdc`：约 43 MB，提供 `filename`、
  `label`、`original` 等字段。

视频转存数据集本身没有附带 `metadata.json`，所以必须另外下载元数据，不能
根据文件名推断标签。两个转存条目的许可证都标为 Unknown；原始数据及其使用
条件来自 DFDC。报告中应明确写成“DFDC part 0 的第三方转存”，并说明这一
限制，不能写成“官方直接下载”。

## 来源分组

DFDC 的 `original` 字段表示伪造视频使用的真实源视频。一个真实视频可能对应
多个伪造版本，因此：

- 真实视频以自己的文件名作为来源组；
- 伪造视频以 `original` 指向的文件名作为来源组；
- 同一来源的全部真实和伪造视频始终进入同一个 split。

导入器将这一来源组写入 `group_id`。特征提取时通过
`--group-manifest` 保留它，避免重新按单个视频编号分组。

## 切分和评估

- DFDC：85% 训练、15% 验证；
- DFDC 不设置内部测试集，避免再次用少量视频挑选方法；
- Celeb-DF 518：最终外部测试，只用于冻结后的评估。

`configs/dfdc.yaml` 固定上述比例。所有 seed 使用同一来源组切分。
当前 part 0 只有 86 个真实视频和 1248 个伪造视频，因此 DFDC 配置启用
逆频率类别权重，使训练时真实和伪造类别的总损失贡献相近；准确率之外仍须
同时报告 balanced accuracy、F1 和 AUC。

## 人脸预处理

每个视频读取八个均匀分布的候选帧，使用 OpenCV YuNet 检测人脸。第一帧选择
面积和置信度乘积最大的脸，后续帧选择与上一检测框中心最近的脸。选择至少
四个成功检测，然后在成功帧中均匀取四个，使用五点 ArcFace 相似变换对齐为
256×256。检测失败视频不会用中心裁剪代替，而是写入 `failed_videos.csv`，
保证报告中的样本数可审计。

## 运行

把视频 ZIP 命名为 `archive.zip`，把元数据文件命名为 `metadata`，放到：

```text
/root/autodl-tmp/CV_ErrorFlip_GenD/data/dfdc/
```

然后执行：

```bash
cd /root/autodl-tmp/CV_ErrorFlip_GenD/code
bash scripts/run_dfdc_pipeline.sh
```

导入器默认只使用两个 OpenCV CPU 线程，避免明显挤占服务器上另一个任务的
CPU 和数据读取资源。需要更低占用时可设置 `DFDC_NUM_THREADS=1`。

先确认管线完成并检查导入摘要、失败列表和 split manifest，再运行：

```bash
SEEDS="2026" bash scripts/run_dfdc_experiments.sh
```

单个 seed 验证流程无误后，再运行三个 seed：

```bash
SEEDS="2026 2027 2028" bash scripts/run_dfdc_experiments.sh
```

## 新结果应如何表述

主结论应比较在 DFDC 上训练的 Adapter 在 Celeb-DF 外部测试集上的表现。
Celeb-DF 上的数字用于检验跨数据集迁移，不与 GenD 论文发布的完整 518 视频
结果直接比较，因为当前人脸裁剪、四帧采样和 mask 协议均不同。

局部退化训练的独立贡献仍应先与 `clean_only` 比较，再与 GenD 和专项退化
Adapter 比较。DFDC 训练只能扩大证据，不能自动证明旧的 78 视频结果显著。
