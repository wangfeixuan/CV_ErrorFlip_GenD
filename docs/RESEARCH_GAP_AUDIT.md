# Research Gap Audit

检索日期：2026-11-25

本文核验三个拟议 Gap 是否仍然成立。检索优先使用论文正文、出版社页面和作者公开版本，重点关注 2023-2026 年直接相关的方法，而不是只依据综述或二手摘要。

## 结论摘要

| 拟议 Gap | 结论 | 原因 |
|---|---|---|
| 视觉质量恢复不等于伪造证据保留 | 宽泛 Gap 不成立 | 已有工作明确优化取证痕迹补偿、频域证据保留和检测对齐的恢复目标 |
| 视频帧不应简单平均，应按取证可靠性聚合 | 宽泛 Gap 不成立 | 已有 attention temporal pooling、residual-guided pooling 和 RFEA |
| 频域分析 + 证据恢复 + 自适应帧聚合尚未系统整合 | 部分成立，但高度拥挤 | 三者两两组合已经存在，三者的具体整合和交互验证仍较少 |

因此，原始表述“超越视觉质量，做证据保留和自适应帧融合”不能再作为完整的创新声明。仍可研究的是更窄的问题：

> 在冻结的通用 deepfake 检测器上，定位不同局部退化实际破坏了哪些频段/特征方向，并只恢复对分类有因果贡献的取证信息；当同一视频内帧退化类型不同且位置未知时，再按可恢复的取证证据强度做帧级聚合。

## Gap 1：视觉质量与取证证据是否脱钩

结论：动机成立，但“发现两者可能不一致”本身已不是新 Gap。

- Ke 与 Wang 的 DF-UDetector 已明确针对低质量 deepfake，把退化特征转换成更高质量特征，而不是只做图像增强。
- GTCP 直接提出“general trace compensation”，同时去除对抗噪声并补偿高频取证纹理。
- FA-IPNet 明确提出以取证为目标的 rescaling，保留 manipulation-sensitive high-frequency cues，而不是最大化 perceptual reconstruction quality。
- 2026 年 IEEE IoT Journal 的 Quality-Agnostic 工作进一步提出 blind face restoration、saliency-guided restoration 和 restoration-similarity-aware adaptive fusion，直接覆盖了“恢复服务于检测”与“按输入质量自适应融合”的组合。
- “Opposite Effects of Compression Augmentation”表明压缩增强在 quality shift 下可能有效，但在 manipulation shift 下可能伤害检测，说明退化、重构和真假证据之间确实存在方向冲突。
- “Preserving Forgery Artifacts”则说明固定尺寸缩放本身会破坏高频伪造线索。

仍然可成立的窄 Gap：

1. 现有工作通常证明“取证感知恢复有效”，但很少给出跨 blur、JPEG、resize 的逐频段因果归因；
2. 很多方法仍用视觉相似度或间接特征距离约束恢复，缺少“该频段被移除后分类损失变化多少”的检测敏感度度量；
3. 尚缺少统一比较 generic restoration、feature restoration、forensic-trace compensation 和 selective restoration 在相同测试协议下的证据保留曲线。

## Gap 2：帧级取证证据可靠性聚合

结论：宽泛 Gap 不成立，已有直接竞争工作。

- RFEA 的完整名称就是 Reliability-Filtered Evidence Aggregation。它明确处理 evidence-source reliability、response-scale alignment 和 temporally concentrated evidence，指出 indiscriminate fusion 会把五数据集平均 video-level AUC 从 92.05% 降到 85.58%。
- ForensicFlow 使用 frequency stream 和 attention-based temporal pooling，明确优先选择 high-evidence frames。
- RGTP 以 reconstruction residual magnitude 驱动 temporal pooling，解决 mean pooling 对信息帧稀释的问题。
- “Beyond Frame-Level Accuracy”直接指出现有 video-level aggregation 仍常使用 mean pooling，并报告了 GRU temporal aggregation 的负结果。
- Frame-wise Heterogeneous Deepfake Detection 研究了帧级异构伪造类型与 multi-attention aggregation，虽然其“异构”主要指伪造标签而不是退化类型，但名称和任务会压缩我们的概念空间。

仍然可成立的窄 Gap：

现有工作多聚合“不同检测器/不同取证线索的响应”，或把质量/置信度作为权重。较少工作同时控制以下条件：

- 同一短视频内每帧是不同的退化类型；
- 退化位置和类型在推理时未知；
- 权重由“该帧是否能提供可恢复、可分类的取证证据”决定，而不只是由图像质量或模型置信度决定；
- 显式报告 clean-frame weight、degraded-frame weight、weight collapse 和 worst-position accuracy。

这仍可形成研究点，但属于组合和协议层面的中等新意，不能只说“我们提出自适应帧融合”。

## Gap 3：频域、证据恢复和帧聚合的统一

结论：部分成立，但必须证明交互收益。

已有工作已经覆盖大部分两两组合：

- ForensicFlow：RGB + texture + frequency + adaptive temporal pooling；
- Quality-Agnostic：restoration + detection-aligned saliency + adaptive fusion；
- FA-IPNet：frequency-aware forensic-preserving preprocessing + dual-branch forensics；
- GTCP：frequency trace compensation + downstream deepfake detection；
- RFEA：heterogeneous forensic responses + temporal evidence concentration。

同时，*Data Diversity, Not Frequency Invariance* 的受控研究在其实验设置下没有观察到
frequency streams 相对 backbone-matched spatial control 的稳定增益，并发现真实
H.264 压缩级训练优于 JPEG 增强。这说明“加入频域分支”本身不是可靠创新，频域必须
承担可验证的诊断或证据选择作用。

目前较难找到一篇工作同时做到：

1. 用频域诊断解释局部退化实际破坏了什么；
2. 只恢复与真假分类有因果关系的频段/特征；
3. 在帧间异构退化下按恢复后的取证证据强度聚合。

但这个“三合一”只是组合空白，不等于天然创新。至少需要：

- generic restoration、feature restoration、frequency-selective restoration 的严格对照；
- frequency masking / band removal 的因果消融；
- 帧均匀平均、置信度加权、学习权重和 forensic evidence gating 的对照；
- 证明三模块组合优于任意两两组合；
- 在 blur、JPEG、resize 和未见混合退化下同时报告 clean、mean、worst-position 和 false-positive rate。

## 推荐的研究主线

建议把“证据保留”先做成测量问题，再把恢复和融合作为解决手段：

1. 在冻结 GenD 上建立 frequency/feature attribution；
2. 对同一视频分别测量 clean、degraded 和 restored 的取证证据保留率；
3. 只恢复对检测有正贡献且从退化输入中可预测的成分；
4. 在帧间异构退化下，用可选择退出的 forensic reliability gate 聚合；
5. 用 clean-only oracle、generic restore、full feature restore、confidence weighting 和 shuffled-evidence 控制实验验证。

更精确的暂定标题：

**Which Evidence Survives? Frequency-Attributed Selective Restoration and Reliability-Aware Aggregation for Deepfake Video Detection**

## 直接相关文献

- Lu, S., Zhou, Y., Chang, Z., Hao, Q., & Meng, X. (2026). RFEA: Reliable forensic evidence aggregation for deepfake video detection. *Applied Sciences, 16*(20), 10170. https://doi.org/10.3390/app162010170
- Si, X., Li, L., Tang, Z., Li, B., Zhang, L., Yuan, K., Liu, H., & Tian, Q. (2026). Quality-agnostic deepfake detection with saliency-guided restoration and adaptive fusion. *IEEE Internet of Things Journal, 13*(7), 12837-12850. https://doi.org/10.1109/JIOT.2026.3653480
- Romani, M. (2025). ForensicFlow: A tri-modal adaptive network for robust deepfake detection. arXiv:2511.14554. https://arxiv.org/abs/2511.14554
- Al-Fehani, M., & Al-Kuwari, S. (2026). RGTP: A residual-guided temporal pooling framework for federated deepfake video detection. *IEEE Access, 14*, 85109-85120. https://doi.org/10.1109/ACCESS.2026.3700056
- Ke, J., & Wang, L. (2023). DF-UDetector: An effective method towards robust deepfake detection via feature restoration. *Neural Networks, 160*, 216-226. https://doi.org/10.1016/j.neunet.2023.01.001
- Amin, M. A., Ni, J., Xu, P., Yu, Z., & Fu, D. (2026). Trustworthy deepfake defense for consumer applications via frequency-preserving resizing and dual-branch forensics. *IEEE Transactions on Consumer Electronics*. https://ieeexplore.ieee.org/document/11550162
- GTCP: A general traces compensation purifier for enhancing the adversarial robustness of deepfake detection. (2026). *Machine Intelligence Research*. https://doi.org/10.1007/s11633-025-1616-7
- Hasan, S. A., Hera, M. P., Mim, A. L., Hossain, M. M. J., Ali, H., & Bulbul, M. F. (2026). Beyond frame-level accuracy: Analyzing video-level stability in transformer-based deepfake forensics. arXiv:2611.10249. https://arxiv.org/abs/2611.10249
- Li, Z., et al. (2026). Preserving forgery artifacts: AI-generated video detection at native scale. arXiv:2604.04634. https://arxiv.org/abs/2604.04634
- Opposite effects of compression augmentation under quality shift and manipulation shift in deepfake video detection. (2026). IEEE. https://ieeexplore.ieee.org/document/11744081
- Data diversity, not frequency invariance: A controlled and self-audited study of compression-robust deepfake detection. (2026). arXiv:2608.28685. https://arxiv.org/abs/2608.28685
