# 鉴别性残差学习：项目审计与可行性方案

日期：2026-11-25

当前状态：已完成项目审计、研究关系分析，以及 Stage 1-3 的离线残差诊断。
尚未训练任何新模型。这里的诊断使用三个现成的全帧退化缓存，即四帧全部退化；
它不是主报告中“两帧清晰、两帧退化”的 16-mask 评估，两个结果不能直接混用。

## 1. 当前 GenD + Adapter 的实际结构

### 1.1 GenD

- `GenDEncoder` 加载 CLIP ViT-L/14 视觉编码器。
- 视觉编码器输出 1024 维 pooled feature。
- GenD 在编码器内部先执行 L2 normalization，再用一个线性二分类头产生 logits。
- 当前特征缓存保存的是编码后的 normalized feature，不是 CLIP 原始 hidden state。
- 因此所有特征距离、补回和分类都发生在同一个 1024 维单位球面附近。

### 1.2 Adapter

当前最佳主线模型是 Celeb-DF 训练的 `uniform` Adapter，使用
`runs_final/uniform_seed2026/best.pt`。

Adapter 不是 MSE feature autoencoder。实际流程为：

```text
delta = up(dropout(GELU(down(z_degraded))))
z_adapted = normalize(z_degraded + tanh(scale) * delta)
```

其中：

- bottleneck = 128；
- dropout = 0.1；
- scale 初始化为 0；
- 当前 checkpoint 的 scale = 0.05914547；
- 分类头与 Adapter 一起训练；
- 视频概率仍然是四帧 fake probability 的算术平均。

### 1.3 重要术语修正

现有 Adapter 的输出不应默认称为 `z_restored`。它没有显式优化
`||z_original - z_adapted||`，而且诊断显示它平均上并没有更接近清晰特征。
因此后续代码和报告应使用：

```text
z_adapted = A(z_degraded)
r_missing = z_original - z_adapted
```

`r_missing` 只表示“当前 Adapter 输出与清晰 GenD 特征之间的差异”，不能直接
解释为“丢失的、有用的伪造证据”。

## 2. 新研究与原有研究的关系

原有研究回答：

- 冻结 GenD 上训练小 Adapter 是否有用；
- 局部 mask 训练是否优于 clean-only；
- 不同退化 Adapter、混合退化、一致性目标和特征恢复目标的收益与失败案例。

新增研究回答：

- Adapter 成功后，剩余差异由什么构成；
- 这些差异中是否有可恢复、且对分类有用的信息；
- 按分类重要性选择残差，是否优于随机和幅度选择；
- 只有在机制验证成立后，才训练新的残差模块。

两者的关系是“机制诊断 → 条件性方法扩展”，不是替换原 Adapter。新增方法只有在
优于相同训练预算和匹配随机对照时，才构成真实贡献。

## 3. 可直接复用的资源

### 3.1 模型与权重

- `data/model.safetensors`：GenD 编码器和原始分类头；
- `runs_final/uniform_seed2026/best.pt`：当前主 Adapter；
- `runs_final/clean_only_seed2026/best.pt`：Clean-only 对照；
- 三个随机种子的已有 Adapter checkpoint。

### 3.2 特征缓存

- `features_blur_k9.npz`；
- `features_jpeg_q30.npz`；
- `features_resize_25.npz`。

每个缓存包含 518 个视频 x 4 帧 x 1024 维的 clean/degraded features、原始 logits、
标签、来源组、分类头权重和 bias。

### 3.3 数据划分

- 训练：362 个视频；
- 验证：78 个视频；
- 测试：78 个视频；
- 按视频和来源组隔离；
- 当前诊断严格复用原 checkpoint 的同一划分。

### 3.4 退化设置

- Blur：Gaussian 9 x 9，sigma=0，BORDER_REFLECT_101；
- JPEG：quality=30，4:2:0 subsampling，解码后保存 PNG；
- Resize：bicubic 缩小到原尺寸的 25%。

### 3.5 代码

- `src/adapter.py`：Adapter 与分类流程；
- `src/train.py`：原训练目标；
- `src/evaluate.py`：现有评估；
- `src/metrics.py`：ACC/AUC/F1/FPR/FNR；
- `src/extract_features.py`：特征缓存；
- `src/prepare_degradations.py`：退化实现。

## 4. 已新增的离线诊断代码

新增 `src/residual_diagnostics.py`，完全冻结模型，不执行训练。

它执行：

- clean/degraded/adapted 特征状态构建；
- 残差范数、余弦距离和正确类 margin；
- 错误恢复、持续错误和 Adapter harm 分组；
- 只用训练集建立 Random、Magnitude、Classifier-aware 和
  Discriminative-residual 选择器；
- 只用验证集选择补回比例；
- 测试集只评估预先选定的方法；
- 全量补回的 Oracle 实现检查；
- 结果 JSON、逐视频 CSV、重要性权重和诊断图。

## 5. Stage 1-3 实际结果

### 5.1 整体特征相似度并不随检测改善而提高

在测试集上，Adapter 对三个条件都提高了检测准确率，但 clean/adapted cosine
没有提高，反而下降：

| 条件 | GenD 退化 ACC | Adapter ACC | clean-degraded cosine | clean-adapted cosine |
|---|---:|---:|---:|---:|
| Blur | 78.21% | 85.90% | 0.5147 | 0.4937 |
| JPEG | 75.64% | 87.18% | 0.5088 | 0.4991 |
| Resize | 84.62% | 89.74% | 0.5146 | 0.5006 |

cosine gain 与 margin recovery 的 Spearman 相关分别为：

- Blur：-0.589；
- JPEG：-0.198；
- Resize：-0.492。

这直接说明：当前 Adapter 的增益不是通过“更像清晰 GenD 特征”实现的。

### 5.2 未恢复残差幅度不是可靠的鉴别指标

`r_missing` 的平均 L2 norm 接近 1.0，但缺失残差幅度与绝对 margin recovery 的
相关性较弱且不稳定：

- Blur Pearson：0.258；
- JPEG：-0.047；
- Resize：0.103。

因此仅按残差幅度选择不能代表真假分类价值。

### 5.3 Feature add-back 有信号，但鉴别性选择没有胜出

测试集结果如下。比例由验证集选择，测试集未参与选择。

| 条件 | Adapter | Random | Magnitude | Classifier-aware | Discriminative |
|---|---:|---:|---:|---:|---:|
| Blur | 85.90% | 88.46% | **91.03%** | 87.18% | 87.18% |
| JPEG | 87.18% | 91.03% | 92.31% | **93.59%** | 92.31% |
| Resize | 89.74% | 89.74% | **92.31%** | 91.03% | 89.74% |

结论：

- 部分未恢复残差确实能改善当前分类器；
- Magnitude 和 Classifier-aware 选择已有明显收益；
- 当前定义的 Discriminative Residual Selection 没有优于更简单对照；
- 因此现在不能声称“鉴别性残差选择”被验证成功。

### 5.4 反向干预暴露了评估流程冲突

在 `z_original` 上移除 `r_missing` 的操作，在 JPEG 条件下甚至经常提高
Adapter-head 的准确率。原因是 Adapter 与分类头已经联合离开了原始 GenD 的
决策流形；用适配后的 head 评估原始特征，再执行“反向移除”，并不构成同一
决策函数的因果对称实验。

实际全量 add-back 会严格回到 `z_original`，但经过适配后的 head 后只有
85.90% blur accuracy，低于 Adapter 的 85.90% 测试结果和其对 clean 特征
91.03% 的表现。这说明 Oracle 不是可用的性能上限，只能用于数值正确性检查。

## 6. 实验分组

| 组 | 内容 | 目的 |
|---|---|---|
| G0 | Frozen GenD | 原始基线 |
| G1 | 当前最佳 Adapter | 现有方法基线 |
| G2 | G1 相同预算继续训练 | 排除额外训练收益 |
| G3 | Magnitude-weighted restoration | 检验纯残差幅度 |
| G4 | Discriminative-weighted restoration | 检验鉴别权重 |
| G5 | 普通残差分支 | 增加可学习恢复容量 |
| G6 | 鉴别性残差分支 | 检验残差分支与鉴别目标的交互 |
| Oracle | 全量 feature add-back | 仅验证实现正确性，不参与部署排名 |

在进入 G2-G6 前，先修正 Stage 2/3 的选择子，确保候选方法超过匹配随机和
Magnitude 对照。

## 7. 计算成本估计

基于现有 518 视频缓存和 1024 维特征：

| 阶段 | 预计成本 |
|---|---|
| Stage 1-3 离线诊断 | CPU 数分钟，无训练 |
| G2-G4 单 seed 可行性 | 每个模型数分钟，主要 GPU 很小 |
| G5-G6 单 seed | 轻量 MLP，单模型数分钟 |
| 三 seeds + 消融 | 约 15-40 分钟 |
| DFDC 外部验证 | 额外推理数分钟 |

瓶颈是实验组织与统计，不是 CLIP 编码；所有实验应复用特征缓存。

## 8. 风险与修正

1. **“missing”不等于“useful”**

   当前结果已经证明 cosine gap 与检测收益可能相反。必须把残差诊断与检测因果
   贡献分开。

2. **单维选择可能不稳定**

   1024 个维度没有可靠物理语义。下一步应改为 16/32 维 block、残差子空间或
   token-level 选择，并在验证集上检查跨 seed 稳定性。

3. **线性 gradient 对 normalization 和训练后 head 不充分**

   应采用实际有限差分干预：对每个 block 分别添加/移除，测量正确类 margin
   变化，而不是只用一次线性近似。

4. **反向干预必须使用共同决策函数**

   建议分别报告：

   - 原始 GenD head 下的 clean degradation；
   - Adapter head 下的 adapted intervention；
   - 冻结两级模型的有限差分效应。

5. **测试集不能用于选择**

   选择器仅在 train 构建，比例仅在 validation 选择，test 只执行一次。后续
   block 版本也保持同一规则。

6. **Full add-back 不是性能上限**

   Adapter head 已偏离 GenD clean feature calibration。报告应将其标为
   implementation oracle，而非可达到上限。

7. **数据泄漏与域偏移**

   继续按视频、来源组切分；方法确认后在 DFDC 和未见退化上独立评估。

## 9. 调整后的执行计划

当前阶段结论：**研究方向仍可行，但当前提出的一维 Discriminative Residual
Selection 未通过机制验证，不能直接进入 Method A/B 训练。**

建议按以下顺序继续：

1. 将选择从单维改为 16/32 维 block，并用实际有限差分 margin 建立训练集权重；
2. 比较 block random、block magnitude、block sensitivity 和 block
   discriminative intervention；
3. 要求候选方法在 validation 上同时优于 random 和 magnitude，并检查反向移除
   是否产生方向一致的下降；
4. 只有在上述条件成立后，实现 Method A，优先继续训练当前最佳 Adapter；
5. Method A 与等预算继续训练对照相比有稳定增益后，再实现 Method B；
6. 最后进行三 seeds、未见退化、DFDC/Celeb-DF 外部评估和完整消融。

## 10. 本轮诊断工件

结果目录：`diagnostics/residual_20261125/`

- `summary.json`：整体指标；
- `interventions_validation.csv`：验证集补回比例扫描；
- `interventions_test.csv`：测试集最终干预；
- `residual_distance_comparison.png`；
- `missing_residual_distribution.png`；
- `importance_vs_magnitude.png`；
- `addback_ratio_curves.png`；
- `forward_reverse_interventions.png`；
- `real_fake_margin_recovery.png`；
- `roc_comparison_blur.png`。
