# Error-Flip-Aware Visual Adaptation for GenD under Sparse Local Degradation

## Abstract

Short-video deepfake detectors often aggregate a small number of face frames.
We study a failure mode in which a fixed number of frames are degraded but
the position of degradation changes the final video decision. Starting from
the frozen GenD detector, we insert a lightweight residual adapter after the
1024-dimensional CLIP vision features and train it on all six two-of-four
blur masks. A second variant up-weights masks on which GenD was correct on the
clean video but wrong on the degraded video; a third selects the hardest
eligible mask. On a fixed source-group split of Celeb-DF v2, all three
adapters substantially improve over GenD. A clean-only adapter control shows
that generic fine-tuning accounts for most of this gain: local-mask training
adds 1.50 points on blur, with a source-group 95% interval of [-0.15, 3.43].
Error-focused weighting and hard-mask selection do not stably outperform
uniform mask training. The supported conclusion is that lightweight visual
adaptation improves performance, while the independent benefit of sparse
mask training is modest and not yet conclusive. Mixed blur/JPEG/resize
training also does not outperform blur-specific adaptation.

## 1. Introduction

Deepfake videos are commonly compressed, resized, and blurred during online
propagation. The degradation is often local: only some sampled frames are
seriously damaged. Previous analysis in this project found 61 of 518 videos
whose predicted label changes under different two-of-four blur masks, even
though the degradation type, strength, and number of blurred frames remain
fixed. This motivates an error-flip formulation: can a small trainable visual
extension recover decisions that the original detector loses under sparse
local degradation?

The primary experiment fixes two blurred frames. This is a problem-driven
choice: the prior audit found 61 videos with position-dependent decisions
under exactly two blurred frames, and `k=2` gives six possible positions while
retaining enough clear visual evidence. It is not assumed to be the optimal
training budget.

We make three contributions. First, we implement a parameter-efficient visual
adapter on frozen GenD features. Second, we construct a controlled comparison
that includes clean-only fine-tuning, uniform mask training, error-weighted
training, and hard-mask training under the same architecture and split. Third,
we evaluate video-level errors, worst-mask performance, and two degradation
types not used for training. We report the independent contribution of
local-mask supervision, the negative result for error-specific weighting, and
the negative result for multi-degradation training.

## 2. Related Work

GenD is the upstream detector and is used with its published CLIP ViT-L/14
vision encoder and two-class linear head. Its main implementation is available
at <https://github.com/yermandy/GenD>.

QAD studies quality-aware joint representation and prediction learning for
deepfake detection. DCPT studies paired clean/degraded consistency. Attack-aware
deepfake detection uses worst-of-K image transformations, and sparse video
perturbation work shows that a small number of altered frames can change a
video decision. Our hard-mask and error-weighted objectives are therefore
related to existing robust-training ideas. We do not claim a new transform
selection principle. The narrower comparison here is the choice among sparse
frame-position masks for a frozen video feature extractor.

## 3. Method

### 3.1 Frozen GenD features

Each video is represented by four aligned face crops. GenD produces a
1024-dimensional L2-normalized feature for every frame. We cache the clean
features, the blur-k9 features, and the original GenD logits. The CLIP encoder
is not updated in the adapter experiments.

### 3.2 Residual adapter

For a frame feature $z_i$, the adapter computes

$$
z'_i =
\mathrm{Normalize}
\left(
z_i + \tanh(\alpha)\,W_2\,\mathrm{GELU}(W_1 z_i)
\right).
$$

The bottleneck is 128 dimensions, dropout is 0.1, and the output is
re-normalized because GenD's published classifier expects L2-normalized
features. The scale parameter starts at zero, so the initial model exactly
reproduces the frozen GenD classifier.

Per-frame class probabilities are averaged into a video probability. The
threshold is fixed at 0.5. The adapter and classifier head are trained; the
CLIP encoder is frozen.

### 3.3 Training objectives

The clean-only control trains the same adapter with the full combined loss
weight on clean videos and never reads degraded features. The other variants
evaluate all six masks containing exactly two blurred frames in every
optimization step.

- Clean-only: generic visual fine-tuning control with no degradation supervision.
- Uniform: average the six video-level binary cross-entropy losses.
- Error-weighted: assign weight 3 to masks that GenD classified incorrectly
  when its clean prediction was correct, then normalize the weights per video.
- Hard-mask: select the currently highest-loss eligible mask per video.

Every method also uses a clean-video loss with weight 0.5. The degraded loss
has weight 1.0. Models are selected only on validation loss.

## 4. Experimental Setup

### Data

We use the 518 videos in the provided Celeb-DF v2 face bank: 178 real and 340
fake. Each video has four aligned 256x256 face crops. The two degradation
banks contain the same 2,072 paths.

The deterministic split contains 362 training videos (124 real, 238 fake),
78 validation videos (27 real, 51 fake), and 78 test videos (27 real, 51 fake).
All experiment runs use the same split manifest hash. A group contains the exact
video and all files generated from the same synthetic identity pair. No video
or group crosses splits.

### Conditions

Training uses the supplied blur-k9 bank. Evaluation additionally uses two
deterministically generated unseen conditions:

- JPEG quality 30, decoded and stored as PNG;
- bicubic resize to 25% of width and height.

We additionally run a mask-count sweep for training `k=1,2,3,4`. Each sample
draws one mask of the selected count per optimization step. This keeps the
number of degraded frame exposures and forward steps equal across sweep arms.
The main `k=2` experiment instead exhaustively averages all six masks and is
reported separately.

For mixed-degradation training, one arm samples one of blur, JPEG or resize
per video and step. A higher-exposure arm evaluates all three degradation
types in every step. Both use the same clean data, split, adapter and optimizer.

### Training

We train three seeds (`2026`, `2027`, `2028`) for every method. The optimizer
is AdamW with learning rate 0.001, weight decay 0.0001, batch size 32, maximum
60 epochs, and patience 12. Early stopping uses the average validation loss
over the clean input and six local masks.

### Metrics

We report clean accuracy, average accuracy and AUC over the six two-frame
masks, worst-mask accuracy, prediction flip rate, recovery of GenD mask-level
errors, and newly introduced mask-level errors. The 16 masks are repeated
measurements of the same videos. Confidence intervals therefore use bootstrap
resampling over the 52 held-out source groups rather than over masks.

## 5. Results

### Main comparison

![Local accuracy](../results_final/figures/local_accuracy_by_condition.png)

| Degradation | GenD | Clean-only | Uniform adapter | Error-weighted | Hard-mask |
|---|---:|---:|---:|---:|---:|
| Blur k9, trained | 83.12% | 88.89% | 90.38 ± 0.46% | 90.31 ± 0.53% | 89.67 ± 0.66% |
| JPEG q30, unseen | 88.03% | 91.88% | 92.59 ± 0.71% | 92.66 ± 0.61% | 92.24 ± 1.05% |
| Resize 25%, unseen | 88.46% | 91.17% | 91.31 ± 0.56% | 91.31 ± 0.56% | 91.10 ± 0.73% |

Averaged over seeds, clean accuracy is 85.90% for GenD, 91.88% for the
clean-only adapter, 93.59% for the uniform adapter, 93.59% for error
weighting, and 92.74% for hard-mask training. The clean-only result shows
that most of the improvement is generic visual adaptation.

### Error recovery and harm

| Condition | Method | Recovery | New errors | Net recovered mask events |
|---|---|---:|---:|---:|
| Blur | Clean-only | 50.21% | 3.26% | 27.00 |
| Blur | Uniform | 56.54% | 2.74% | 34.00 |
| Blur | Error-weighted | 56.54% | 2.83% | 33.67 |
| Blur | Hard-mask | 55.27% | 3.34% | 30.67 |
| JPEG | Clean-only | 59.52% | 3.72% | 18.00 |
| JPEG | Uniform | 66.67% | 3.88% | 21.33 |
| Resize | Clean-only | 50.62% | 3.54% | 12.67 |
| Resize | Uniform | 53.70% | 3.78% | 13.33 |

![Recovery and harm](../results_final/figures/recovery_and_harm.png)

On the trained blur condition, clean-only adaptation improves local accuracy
by 5.77 points over GenD. Adding mask training raises it by another 1.50
points. The independent mask-training source-group 95% interval is
[-0.15, 3.43], so the estimate is directionally positive but not
statistically conclusive.

For unseen JPEG, clean-only adaptation gains 3.85 points over GenD and mask
training adds 0.71. For unseen resize, clean-only gains 2.71 and mask training
adds 0.14. These results do not support a strong claim of mask-training
generalization with 52 test groups.

### Error-focused objectives

Error-weighted training is 0.07 points below uniform on blur (95% interval
[-0.32, 0.15]), 0.07 above uniform on JPEG (interval [0.00, 0.23]), and equal
on resize. The small differences are not a stable method advantage.
Hard-mask training is 0.71 points below uniform on blur, with an interval
[-1.34, -0.16]. Thus selecting only the current hardest mask is less stable
than averaging all local positions.

The residual adapter contains 265,347 trainable parameters, including the
two-class head. The 304M-parameter CLIP encoder remains frozen. After feature
caching, each update evaluates six mask variants of a batch but updates only
the small adapter and head.

### Position sensitivity

![Position sensitivity](../results_final/figures/position_sensitivity.png)

Hard-mask training reduces the mean local accuracy but slightly raised some
worst-position results. Its blur worst-mask accuracy is 86.75%, versus 88.03%
for uniform and 76.92% for GenD. The adapter reduces position sensitivity,
but the hard-selection objective does not provide a reliable additional gain.

### Blur-count follow-up

![Mask-count sweep](../results_final/figures/k_sweep.png)

| Trained k | Clean | Test k=1 | Test k=2 | Test k=3 | Test k=4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 91.45% | 90.28% | 89.10% | 86.43% | 84.19% |
| 2 | 91.45% | 89.96% | 89.10% | 87.29% | 83.76% |
| 3 | **94.02%** | **91.99%** | **89.89%** | **88.03%** | **86.75%** |
| 4 | 91.88% | 91.03% | 89.81% | **88.03%** | **86.75%** |

Training at `k=3` has the strongest point estimates on blur, while `k=4`
is close but weaker under unseen JPEG. Relative to random-mask `k=2`,
training `k=3` is +0.78 points at test `k=2`, with a paired source-group 95%
interval of [-0.68, 2.78]. Training `k=4` is +0.71 points, with interval
[-1.54, 3.54]. Neither result supports a claim that another `k` is superior.
The notable pattern is a graceful decline as the number of test-time degraded
frames increases.

### Six-method follow-up

| Method | Clean | Blur local | JPEG local | Resize local |
|---|---:|---:|---:|---:|
| GenD | 85.90% | 83.12% | 88.03% | 88.46% |
| Clean-only | 91.88% | 88.89% | 91.88% | 91.17% |
| Blur-only | **93.59%** | 90.38% | **92.59%** | **91.31%** |
| JPEG-only | 92.74% | **90.81%** | 91.31% | 91.45% |
| Resize-only | **93.59%** | 89.25% | 93.02% | 90.31% |
| Mixed-all | 90.60% | 88.53% | 90.10% | 89.60% |

There is no clean one-specialist-per-condition pattern. JPEG-only reaches
90.81% on blur, while Resize-only does not lead on resize. Blur-only remains
strongest on JPEG. Mixed-all is below every specialist on all three conditions
and also lowers clean accuracy. The low-cost mixed arm is also below
blur-only, reaching 88.89%, 91.03% and 90.46% on blur, JPEG and resize.
With this data and one small adapter, balancing the objectives yields a
conservative compromise rather than complementary robustness.

Paired source-group intervals confirm the direction. The one-type mixed arm
is -1.50 points versus blur-specific on blur [-3.33, 0.00], -1.57 on JPEG
[-6.93, 2.70], and -0.85 on resize [-2.32, 0.41]. The all-types arm is -1.85
on blur [-3.61, -0.40], -2.49 on JPEG [-8.37, 2.21], and -1.71 on resize
[-3.80, 0.07]. Multi-degradation training is therefore not supported.

## 6. Discussion and Limitations

The strongest supported intervention is visual adaptation itself. Clean-only
fine-tuning accounts for 5.77 of the 7.26 points gained over GenD on trained
blur. Local-mask training contributes an additional 1.50 points. The data do
not support a claim that error weighting or hard-position selection is a
superior training rule. This is still a useful result: it separates generic
fine-tuning, local-mask supervision, and error-selection heuristics.
Mixed-degradation training also fails to improve the tested conditions,
supporting the decision to keep blur-specific training as the main method.

Hard-mask selection may be unstable for three reasons: the selected mask can
change as the adapter updates, the hardest current mask may emphasize label
noise, and the objective discards the regularizing signal from the other five
masks. Error weighting retains all masks but does not create a meaningful
advantage. These are hypotheses, not established mechanisms.

The test split has only 78 videos and 52 source groups. Masks are not
independent samples. The Celeb-DF videos were used in earlier exploratory
analysis and are not a fully blind external test. The source grouping keeps
all frames from a video and the same synthetic identity pair together, but it
does not prove complete person-identity isolation across the whole dataset.
GenD's pretraining overlap must also be checked. Finally, the experiment uses
four fixed face crops and controlled degradation; it does not reproduce the
full online video-transcoding pipeline.

The supplied UADFV download link timed out from both the local and server
environments. Existing UADFV score tables were not reused as an adapter result
because they do not contain the new visual features.

## 7. Conclusion

A lightweight adapter on frozen GenD features substantially improves
performance. The clean-only control shows that most of this gain is generic
visual fine-tuning. Local-mask training adds a modest, directionally positive
but statistically inconclusive increment, while rare error-flip position
selection does not consistently improve over uniform masks. A fair Track 2
contribution is therefore the controlled attribution analysis, including the
negative results for harder-position emphasis and multi-degradation training.

## Reproducibility

The code, model hash, fixed manifests, three-seed checkpoints, per-video
predictions, condition-level summaries, bootstrap output, and figures are
stored in this project. The server workspace is
`/root/autodl-tmp/CV_ErrorFlip_GenD`.

## Appendix: Artifact checks

- The feature cache stores clean/degraded features, logits, head weights,
  labels, source groups, and source-archive hashes.
- All fixed-split manifests have one identical SHA256 digest.
- Every checkpoint is accompanied by validation history, test JSON, and
  per-video/per-mask predictions.
- Source-group bootstrap resamples the 52 held-out groups, not masks.
- Unseen-condition caches repeat the clean side and replace only the degraded
  side. The reported condition comparison uses baseline and adapter predictions
  from the same cache execution.

## References

1. A. Yermakov, J. Cech, J. Matas, and M. Fritz. *Deepfake Detection that
   Generalizes Across Benchmarks.* WACV 2026. arXiv:2508.06248.
2. P. Charitidis, G. Kordopatis-Zilos, S. Papadopoulos, and I. Kompatsiaris.
   *Investigating the Impact of Pre-processing and Prediction Aggregation on
   the DeepFake Detection Task.* arXiv:2006.07084, 2020.
3. B. M. Le and S. S. Woo. *Quality-Agnostic Deepfake Detection with
   Intra-model Collaborative Learning.* ICCV 2023. arXiv:2309.05911.
4. D. Zhang, Y. Li, J. Zhou, and Y. Li. *DPL: Cross-quality DeepFake
   Detection via Dual Progressive Learning.* ACCV 2024. arXiv:2410.07633.
5. Z. Yang, Y. Hou, and X. Yang. *Degradation-Consistent Paired Training for
   Robust AI-Generated Image Detection.* arXiv:2604.10102, 2026.
6. N. Fatima, H. F. Khan, and M. Behzad. *Attack-Aware Deepfake Detection
   under Counter-Forensic Manipulations.* arXiv:2512.22303, 2025.
7. Official GenD implementation:
   <https://github.com/yermandy/GenD>.
