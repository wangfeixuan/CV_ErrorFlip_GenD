# CV 深伪检测项目当前进度

更新时间：2026-11-21

## 当前结论

项目已经从“退化和校准分析”推进为一个可在 GenD 上训练、可复现的视觉
Adapter 实验，并完成了固定切分、三个随机种子、未见退化和视频组级
bootstrap 分析。

普通局部退化训练显著优于原始 GenD。把 GenD 的全部帧判错位置组合用于
等权训练，比只强调困难位置更稳定；困难位置加权和硬选择没有稳定优势。
这是一个可以正式写入路线 2 报告的负结果，不需要伪造性能提升。

## 服务器位置

项目使用独立目录：

```text
/root/autodl-tmp/CV_ErrorFlip_GenD
```

不要修改 VADE 目录。CV 项目的代码、模型、数据、特征缓存、权重、日志和
结果都在 CV_ErrorFlip_GenD 下。

## 已完成

- GenD 权重 SHA256 校验通过。
- 518 个 Celeb-DF v2 视频、2072 张 256x256 人脸对齐图完成原始/模糊特征缓存。
- 按视频和合成来源对分组，固定为 362 训练、78 验证、78 测试。
- 三种方法各跑三个随机种子，所有种子使用完全相同的 split manifest。
- 比较 GenD、均匀 Adapter、误差样本加权、困难位置硬选择。
- 在 blur k9、未见 JPEG q30、未见 resize 25% 三种条件评价。
- 对六个两帧 mask、全部 16 个 mask、最差位置、翻转率、修复率和伤害率评价。
- 完成视频来源组级 bootstrap 和图表。

## 主要测试结果

测试集为 78 个未进入训练/验证的视频，三个种子取均值。

| 训练条件 | GenD 局部准确率 | 均匀 Adapter | 误差加权 Adapter | 困难位置 Adapter |
|---|---:|---:|---:|---:|
| blur k9（训练过） | 83.12% | 90.38% | 90.31% | 89.67% |
| JPEG q30（未见） | 88.03% | 92.59% | 92.66% | 92.24% |
| resize 25%（未见） | 88.46% | 91.31% | 91.31% | 91.10% |

清晰视频上，GenD 为 85.90%；均匀 Adapter 为 93.59%，
误差加权 Adapter 为 93.59%，困难位置 Adapter 为 92.74%。

blur k9 下，均匀 Adapter 的 GenD 错误修复率为 56.54%，新引入错误率为
2.74%。来源组级 bootstrap 显示均匀 Adapter 相对 GenD 的局部退化增益
约为 +7.26 个百分点，95% CI 为 [1.39, 13.96] 个百分点。

## 必须如实陈述的边界

- 误差加权和困难位置策略没有稳定超过均匀局部退化训练。
- JPEG 与 resize 的增益置信区间包含 0，只能作为未见退化压力测试。
- 测试集只有 78 个视频；16 个 mask 不是 16 倍独立样本。
- Celeb-DF 518 个视频此前已经用于探索，不是完全盲测外部集。
- 来源分组避免同一视频跨切分，但不能证明所有人物身份完全隔离。
- GenD 的原始训练数据重叠情况仍需在报告中说明。
- 提供的 UADFV Google Drive 链接在本地和服务器均连接超时，因此本轮没有
  把它作为额外外部测试集；现有 UADFV 历史 CSV 不能代替新模型的视觉特征
  评测。

## 可复现入口

```bash
cd /root/autodl-tmp/CV_ErrorFlip_GenD/code
../.venv/bin/python -m pytest -q
bash scripts/run_all.sh
```

结果目录：

```text
/root/autodl-tmp/CV_ErrorFlip_GenD/runs_final
```

核心文件：

- `aggregate_all_conditions.md`
- `aggregate_all_conditions.summary.csv`
- `bootstrap_all_conditions.csv`
- `figures/`
- 每个 `<method>_seed<seed>/best.pt`
- 每个 `<method>_seed<seed>/history.csv`
- 每个 `<method>_seed<seed>/eval_test*.json`
- 每个 `<method>_seed<seed>/eval_test*.per_video.csv`
