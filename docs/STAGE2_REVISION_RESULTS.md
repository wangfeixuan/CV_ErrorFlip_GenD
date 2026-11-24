# Stage 2 Revision：Uniform Adapter 的残差机制与可预测性

日期：2026-11-25

状态：诊断与轻量统计模型拟合已完成；未启动 Method A/B 神经网络训练。

## 1. 结论先行

本轮结果支持“继续研究 Uniform Adapter 的剩余错误”，但还不足以直接启动
Method A/B。

最重要的证据是：

1. 检测增益主要来自特征适配，而不是分类头。同一特征交给 GenD、Uniform
   或 Restoration 头，平均测试准确率差约为 1 个百分点，最大差为 2.56 个百分点。
2. Uniform Adapter 并不把特征拉得更接近清晰特征。测试集
   `cos(z0,zu)` 为 0.490、0.492、0.496；Feature Restoration 的
   `cos(z0,zf)` 为 0.566、0.525、0.556，但 Feature Restoration 检测更差。
3. R3 比 R2 更适合表示 Uniform Adapter 内部仍有变化的退化信息。R3 在
   Blur 和 Resize 上稳定优于 R2，JPEG 上两者接近。整体残差插值在三种条件下
   都优于单维选择器。
4. 真实可部署的 Ridge 只能解释约 17%–18% 的 R3 相对均值方差；分类收益
   不稳定。Oracle 补回仍有明显收益，但无标签条件下的可预测性不足。
5. 持续退化错误 Group B 的视频数很少，R3 范数更大，Oracle R3 能纠正多数
   错误；因此值得继续做独立数据验证，但目前不能据此宣称方法有效。
6. 自适应选择器没有稳定战胜随机和能量匹配随机。PCA/判别方向子空间约提升
   2–4.5 个 balanced-accuracy 点，但 95% 区间在部分条件包含 0，且测试集已
   参与过探索。

推荐下一步：先在独立验证数据上预注册并复核 R3/判别方向补偿，再决定是否进入
Method A。不要直接扩大到 1024 维残差生成网络。

## 2. 评价范围与术语

本文件中的主分析统一使用：

- 518 个 Celeb-DF v2 视频；
- 每个视频 4 帧；
- 362 train / 78 validation / 78 test；
- 四帧全部退化；
- Blur k9、JPEG q30、Resize 25%；
- 测试集只做探索性评估。

因此这些结果不能与主报告中“两帧清晰、两帧退化”的 16-mask 结果直接混用。

记号：

```text
z0  = E(x_clean)
zd  = E(x_degraded)
zu  = U(zd)
zf  = F(zd)
zuc = U(z0)
zfc = F(z0)

R1 = z0  - zd
R2 = z0  - zu
R3 = zuc - zu
R4 = z0  - zf
R5 = zfc - zf
R6 = zu  - zf
```

`zu` 和 `zf` 是 Adapter 输出，不称为“恢复特征”；Feature Restoration 是
特定训练目标产生的机制对照组。

## 3. 诊断实现修正

修正后的实现位于：

- `src/residual_diagnostics.py`
- `src/residual_stage2.py`
- `src/residual_stage2_extended.py`
- `src/adapter_mechanism_stage2.py`

本轮完成：

- R3 的正向干预从 `zu` 出发；反向干预从同一 Adapter 空间中的
  `zuc=A(z0)` 出发，而不是混用 `z0`。
- 原始 GenD 头与 Adapter 头分开报告。
- Oracle 改为真实 clean endpoint，不再把 `zu` 本身误当作 Oracle。
- `signed_discriminative` 进入正式对照，并保留符号。
- 普通随机对照及每个 selector 单独的能量匹配随机对照均报告多次平均值、
  标准差和实际能量匹配误差。
- 所有 selector 使用相同特征数量和相同干预系数比较。
- 增加有限差分 block 子空间、16/32 维 block、PCA 和判别方向。
- 增加 source-group paired bootstrap 和跨退化联合 bootstrap。
- 增加 Numeric tests，覆盖零 mask、全量补回、R3 反向、插值端点和 selector
  确定性。

测试命令：

```bash
python -m pytest -q
```

结果：18 tests passed。

## 4. Adapter 资源核查

三个 seed 的以下资源均存在，已直接复用，没有重新训练：

- `runs_final/uniform_seed{2026,2027,2028}/best.pt`
- `runs_final/restore_seed{2026,2027,2028}/best.pt`
- `runs_final/clean_only_seed{2026,2027,2028}/best.pt`
- 每个 run 的 `run.json`、测试结果和 split manifest。

Uniform、Restoration 与 Clean-only 使用相同 1024→128→1024 Adapter 架构、
相同数据划分和相同视频帧数量。Restoration 额外使用 restoration 与
consistency 目标，因此它们是相同架构下的目标函数对照。

## 5. 三种 Adapter 的同一协议对比

三 seed 平均，四帧全部退化：

| 条件 | Frozen GenD ACC | Uniform ACC | Restoration ACC | Clean-only ACC |
|---|---:|---:|---:|---:|
| Blur | 78.21% | **87.18%** | 85.04% | 82.91% |
| JPEG | 75.64% | 86.75% | 81.62% | **89.74%** |
| Resize | 84.62% | **89.32%** | 88.03% | 88.89% |

Balanced accuracy：

| 条件 | Frozen GenD | Uniform | Restoration | Clean-only |
|---|---:|---:|---:|---:|
| Blur | 72.00% | **83.22%** | 80.43% | 78.79% |
| JPEG | 78.76% | 89.00% | 83.33% | **90.99%** |
| Resize | 84.75% | **87.76%** | 85.33% | 88.60% |

Uniform 明显优于 Restoration 的三个主要解释：

1. Uniform 保留了更多对分类有用的信息，而不是追求像素/特征回归。
2. Restoration 把 `cos(z0,zf)` 提高到约 0.53–0.57，明显高于 Uniform 的
   0.49–0.50，但整体恢复相似度没有转化为更低的 FPR/FNR。
3. Restoration 的剩余残差 R4 和内部残差 R5 仍存在，但更低的残差范数
   不代表更强的真假鉴别信息。

Clean-only 在 JPEG 上表现最好，说明包含退化增强并不自动保证优于普通
fine-tuning。这应作为限制保留下来。

## 6. Adapter 与分类头贡献

交叉 head 诊断在三 seed 上显示：固定特征、替换分类头时，平均 accuracy
变化约 1 个百分点，最大变化 2.56 个百分点。主要差异来自特征状态：

- `zd → zu`：Blur、JPEG、Resize 的测试 balanced accuracy 从
  72.00%、78.76%、84.75% 提高到 83.22%、89.00%、87.76%。
- `z0` 本身仍有约 80.5% balanced accuracy；
- `zf` 在三种条件下均低于 `zu`。

因此，当前 Uniform Adapter 的核心贡献是学习了一个新的可分类特征流形，
不是单纯让 `zu` 接近 `z0`，也不是主要由联合训练的分类头产生。

独立 Logistic Regression 结果与这一判断并不完全一致：

- 独立 head 下 Blur：`z0 90.6%`、`zd 86.8%`、`zu 84.7%`；
- JPEG：`z0 90.6%`、`zd 86.8%`、`zu 91.8%`；
- Resize：`z0 90.6%`、`zd 89.7%`、`zu 86.6%`。

这说明 Adapter 与 head 存在协同适配，不能把交叉 head 当作绝对排名。
但它也排除了“所有收益都只是分类头阈值变化”的简单解释。

## 7. R1–R6 实测

三 seed 平均测试集，Uniform 分类方向上的结果：

| 条件 | 残差 | L2 norm | clean/degraded endpoint cosine | classifier projection |
|---|---|---:|---:|---:|
| Blur | R1 | 0.982 | 0.515 | 0.047 |
| Blur | R2 | 1.006 | 0.490 | -0.179 |
| Blur | R3 | 0.907 | 0.581 | 0.061 |
| Blur | R4 | 0.929 | 0.566 | -0.058 |
| Blur | R5 | 0.905 | 0.587 | 0.036 |
| Blur | R6 | 0.474 | 0.883 | 0.121 |
| JPEG | R2 | 1.004 | 0.492 | -0.099 |
| JPEG | R3 | 0.943 | 0.545 | 0.140 |
| JPEG | R4 | 0.971 | 0.525 | -0.012 |
| JPEG | R5 | 0.956 | 0.538 | 0.082 |
| JPEG | R6 | 0.254 | 0.964 | 0.088 |
| Resize | R2 | 1.000 | 0.496 | -0.149 |
| Resize | R3 | 0.914 | 0.573 | 0.090 |
| Resize | R4 | 0.939 | 0.556 | -0.047 |
| Resize | R5 | 0.918 | 0.574 | 0.047 |
| Resize | R6 | 0.389 | 0.920 | 0.103 |

R2 的平均投影为负，说明它主要是 Uniform 模型相对清晰 GenD 特征产生的
主动调整，不应统一补回。R3 在 Adapter 内部比较，残差更小且投影更接近
正方向，是更合理的后续监督目标。

## 8. R2 与 R3

主 seed 的 validation-selected 整体插值测试结果：

| 条件 | Uniform head + R2 BA | Uniform head + R3 BA |
|---|---:|---:|
| Blur | 85.19% | **86.93%** |
| JPEG | 92.37% | **93.46%** |
| Resize | 84.20% | **89.76%** |

三 seed 选出的 α 和测试结果：

| 条件 | R2 mean BA | R3 mean BA |
|---|---:|---:|
| Blur | 86.09% ± 0.76 | **87.55% ± 2.31** |
| JPEG | **92.74% ± 0.51** | 92.48% ± 1.51 |
| Resize | 86.67% ± 1.75 | **90.63% ± 3.02** |

结论：

- R3 在 2/3 条件明显占优，平均更高；
- JPEG 上 R2 略高，不能宣称 R3 全面压倒 R2；
- R3 仍是当前最值得进入独立验证的目标。

整体插值通常优于复杂单维选择。主 seed 的 R3 单维 selector 测试结果：

| 条件 | Random | Magnitude | Classifier | Signed | ReLU discriminative |
|---|---:|---:|---:|---:|---:|
| Blur | 86.06% | 85.08% | **87.91%** | 87.04% | **87.91%** |
| JPEG | 92.48% | 92.48% | 92.48% | 86.93% | 92.48% |
| Resize | 87.91% | 86.93% | 87.91% | 87.91% | 87.91% |

因此复杂 selector 没有形成稳定优势。普通随机 mask 在 JPEG 和 Resize
已经很强，说明 78 个测试视频上的小幅差异不足以支撑复杂选择策略。

## 9. 子空间与有限差分

R3 在 Adapter 空间中的低维补偿结果：

| 条件 | PCA-16 parallel | PCA-32 parallel | discriminant parallel |
|---|---:|---:|---:|
| Blur | 89.76% | 89.76% | 89.76% |
| JPEG | 92.48% | **93.46%** | **93.46%** |
| Resize | 89.76% | 89.76% | 89.76% |

PCA 高方差方向和分类判别方向在这批数据上给出几乎相同的测试结果。这不是
“相邻 16 维有共同语义”的证明，只说明测试集上可能集中在少数连续方向附近。

有限差分 block selector 使用验证集真实干预排序，测试结果低于 PCA/整体插值。
该结果不支持继续扩大到 1024 维学习式 selector。

## 10. Ridge 可预测性

以 `za` 为输入预测 R3：

| 条件 | Zero BA | Mean BA | Ridge full BA | Ridge 32D BA | Oracle BA |
|---|---:|---:|---:|---:|---:|
| Blur | 80.50% | 82.35% | 83.22% | 83.22% | 89.76% |
| JPEG | 87.58% | 84.20% | 85.08% | 86.06% | 89.76% |
| Resize | 86.06% | 86.06% | 86.93% | 84.20% | 89.76% |

Ridge full 的相对均值 R² 为：

- Blur：0.180；
- JPEG：0.171；
- Resize：0.180。

也就是说，R3 只能被当前退化/适配特征部分预测。更关键的是，预测误差下降没有
稳定转化为分类收益：跨条件平均 BA 从 84.71% 提高到 85.08%，但 JPEG
下降。Ridge 32D 没有带来额外收益。

因此当前证据不支持“先训练一个 1024 维残差生成器”。如果进入 Method A，
应优先使用低维分类方向约束，而不是完整特征回归。

## 11. Error-conditioned Residual Analysis

测试集三 seed 平均的视频数：

| 组 | Blur | JPEG | Resize | 主要含义 |
|---|---:|---:|---:|---|
| A 成功纠错 | 4.0 | 10.3 | 5.0 | clean 对、退化错、Uniform 对 |
| B 持续退化错误 | 4.0 | 4.7 | 3.0 | 最重要但样本极少 |
| C Adapter 引入错误 | 0.0 | 3.7 | 2.3 | 退化原本正确却被改错 |
| D 原始持续错误 | 6.0 | 1.7 | 2.7 | 多数为 Real 被误报 |
| E 稳定正确 | 59.0 | 48.3 | 57.7 | 主要对照组 |

Group B 的平均 R3 范数：

- Blur：1.026；
- JPEG：1.104；
- Resize：1.089。

高于 Group E 的 0.891、0.904、0.891。Group A 约为 0.918、1.029、1.031。
这说明持续错误并非完全随机，存在更大的 A-to-U 表征变化。

Oracle R3 补回在 Group B 上的组内准确率为 91.7%、100%、100%，但每组只有
约 3–5 个视频，不能作为稳定统计证据。Group C 上 Oracle R3 也恢复约
60%–72% 的 Adapter 引入错误。Group D 主要包含 Real 误报，Resize 上
Oracle R3 也未纠正，说明部分错误来自 GenD/分类边界本身。

## 12. 校准与视频级决策

Adapter 的 logit scale 明显大于 GenD：

- Blur：2.879；
- JPEG：2.130；
- Resize：2.498。

验证集温度的 Adapter 值约为 1.44–2.21。温度校准降低视频级 NLL，但没有
稳定提高 balanced accuracy：

- Blur：BA 80.50% → 84.97%；
- JPEG：87.58% → 86.06%；
- Resize：86.06% → 84.20%。

帧平均 logit margin 与四帧平均 probability 的决策规则在 Adapter 上约有
34.6%–37.2% 的视频不一致。必须保留视频级 probability/NLL/margin 分析，
不能只报告 frame logit margin。

## 13. 统计可靠性

测试集只有 78 个视频、52 个来源组。按来源组配对 bootstrap：

| 对比 | 跨三条件平均 accuracy 差 | 95% CI |
|---|---:|---:|
| R3 validation-selected interpolation vs Uniform | +3.42 pp | [0.00, 7.26] |
| Ridge full R3 vs Uniform | +0.85 pp | [-2.16, 4.17] |
| R3 discriminant parallel | +5.56 pp | [1.28, 10.29] |
| R3 block signed | +1.71 pp | [0.40, 3.51] |

跨 seed 的 R3 10% signed mask Jaccard 为：

- Blur：0.907–0.962；
- JPEG：0.925–0.943；
- Resize：0.855–0.925。

mask 索引较稳定，但索引稳定不代表物理语义稳定，也不排除对当前 head 的循环
论证。PCA/判别方向实验只证明对冻结决策函数有贡献。

## 14. 推荐的研究决策

当前满足以下条件：

- 明确证明整体恢复相似度与检测性能解耦；
- 明确证明主要增益来自 Uniform 特征映射；
- 发现 R3 是比 R2 更合理的 Adapter 内残差目标；
- 发现少量持续错误具有更大的 R3；
- 找到稳定的低维判别方向候选。

当前不满足：

- 无标签条件下完整 R3 的可靠预测；
- selector 稳定战胜匹配随机；
- 跨独立数据的确认性收益；
- Method A 相对相同预算继续训练的明确优势。

因此本轮结论是 **Conditional GO：仅进入独立数据复核，不直接启动正式训练**。

## 15. 下一阶段预注册标准

后续独立验证应提前固定以下规则：

1. 主目标优先 R3，而不是完整 R2。
2. 优先低维分类方向或 PCA-16/32 补偿，不增加大型残差网络。
3. 至少三 seed；按来源组做配对 bootstrap。
4. 方法必须在至少 2/3 退化条件优于 Uniform；
5. paired 95% CI 下界在预先指定的主条件上大于 0；
6. 必须同时优于相同训练预算的 G2 继续训练对照组；
7. clean 与 worst-degradation 的下降不得超过预注册容忍度；
8. 不能用测试集重新选择 α、selector、温度或损失权重；
9. 如果只有 Oracle 有效，或 Ridge 收益无法跨条件保持，停止残差生成路线。

后续 Method A 的最小分组保持：

| 组 | 内容 |
|---|---|
| G0 | Frozen GenD |
| G1 | Uniform Adapter |
| G2 | 相同预算普通继续训练 |
| G3 | Uniform-weighted restoration |
| G4 | Magnitude-weighted restoration |
| G5 | Classifier-aware restoration |
| G6 | Discriminative residual learning |
| G7/G8 | 残差分支，仅在 G6 显示能力不足时考虑 |

## 16. 产物索引

`diagnostics/residual_stage2_final_20261125/`

- `single_selector_*`
- `matched_random_*`
- `block_*`
- `subspace_*`
- `finite_difference_subspace.csv`
- `ridge_predictability.csv`
- `selector_transfer.csv`
- `seed_stability.csv`
- `class_balance_audit.csv`
- `bootstrap.csv`

`diagnostics/adapter_mechanism_stage2_20261125/`

- `resource_audit.json`
- `adapter_summary*.csv`
- `residual_metrics_*.csv`
- `cross_head_*.csv`
- `independent_head*.csv`
- `error_groups_*.csv`
- `error_transitions.csv`

`diagnostics/stage2_revision_figures_20261125/`

- `adapter_mechanism_comparison.png`
- `feature_head_cross_diagnosis.png`
- `residual_target_comparison.png`
- `whole_residual_interpolation.png`
- `selector_vs_matched_random.png`
- `subspace_comparison.png`
- `ridge_predictability.png`
- `error_conditioned_residuals.png`
- `calibration_and_video_decision.png`
- `paired_bootstrap_intervals.png`
