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
adapters substantially improve over GenD. However, error-focused weighting
and hard-mask selection do not stably outperform uniform mask training. The
main finding is therefore precise: lightweight visual adaptation improves
local-degradation robustness, while training only harder degraded positions
is not sufficient.

## 1. Introduction

Deepfake videos are commonly compressed, resized, and blurred during online
propagation. The degradation is often local: only some sampled frames are
seriously damaged. Previous analysis in this project found 61 of 518 videos
whose predicted label changes under different two-of-four blur masks, even
though the degradation type, strength, and number of blurred frames remain
fixed. This motivates an error-flip formulation: can a small trainable visual
extension recover decisions that the original detector loses under sparse
local degradation?

We make three contributions. First, we implement a parameter-efficient visual
adapter on frozen GenD features. Second, we construct a controlled comparison
in which uniform, error-weighted, and hard-mask training receive the same six
masks and the same inference aggregation. Third, we evaluate video-level
errors, worst-mask performance, and performance under two degradation types
not used for training. We report the negative result that error-specific
weighting does not consistently beat uniform augmentation.

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

For each video, all six masks containing exactly two blurred frames are
evaluated in every optimization step. This keeps forward computation equal
across variants.

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
All nine runs use the same split manifest hash. A group contains the exact
video and all files generated from the same synthetic identity pair. No video
or group crosses splits.

### Conditions

Training uses the supplied blur-k9 bank. Evaluation additionally uses two
deterministically generated unseen conditions:

- JPEG quality 30, decoded and stored as PNG;
- bicubic resize to 25% of width and height.

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

| Degradation | GenD | Uniform adapter | Error-weighted | Hard-mask |
|---|---:|---:|---:|---:|
| Blur k9, trained | 83.12% | 90.38 ± 0.46% | 90.31 ± 0.53% | 89.67 ± 0.66% |
| JPEG q30, unseen | 88.03% | 92.59 ± 0.71% | 92.66 ± 0.61% | 92.24 ± 1.05% |
| Resize 25%, unseen | 88.46% | 91.31 ± 0.56% | 91.31 ± 0.56% | 91.10 ± 0.73% |

Averaged over seeds, clean accuracy is 85.90% for GenD, 93.59% for the
uniform adapter, 93.59% for error weighting, and 92.74% for hard-mask
training. The clean gain is a genuine effect of adapting the visual
representation and classifier, not purely a local-mask effect.

### Error recovery and harm

| Condition | Method | Recovery | New errors | Net recovered mask events |
|---|---|---:|---:|---:|
| Blur | Uniform | 56.54% | 2.74% | 34.00 |
| Blur | Error-weighted | 56.54% | 2.83% | 33.67 |
| Blur | Hard-mask | 55.27% | 3.34% | 30.67 |
| JPEG | Uniform | 66.67% | 3.88% | 21.33 |
| Resize | Uniform | 53.70% | 3.78% | 13.33 |

![Recovery and harm](../results_final/figures/recovery_and_harm.png)

On the trained blur condition, uniform adaptation improves local accuracy by
7.26 percentage points relative to GenD. The source-group bootstrap 95%
interval is [1.39, 13.96]. The adapter recovers more errors than it creates,
and the net gain is positive.

For unseen JPEG, uniform adaptation has a +4.56 point mean gain but the 95%
interval includes zero [-1.65, 10.61]. For unseen resize, the gain is +2.85
points with interval [-2.44, 8.33]. These results are promising but not
statistically conclusive with 52 test groups.

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

## 6. Discussion and Limitations

The strongest supported intervention is uniform local-mask adaptation of the
visual representation. The data do not support a claim that error weighting
or hard-position selection is a superior training rule. This is still a
useful result: it separates the benefit of visual adaptation from the benefit
of the proposed error-selection heuristic.

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

A lightweight adapter on frozen GenD features substantially improves local
blur robustness and clean accuracy. Training on rare error-flip positions does
not consistently improve over training on the complete set of two-frame
masks. A fair Track 2 contribution is therefore the controlled method and
analysis, including the negative result for harder-position emphasis.

## Reproducibility

The code, model hash, fixed manifests, three-seed checkpoints, per-video
predictions, condition-level summaries, bootstrap output, and figures are
stored in this project. The server workspace is
`/root/autodl-tmp/CV_ErrorFlip_GenD`.

## Appendix: Artifact checks

- The feature cache stores clean/degraded features, logits, head weights,
  labels, source groups, and source-archive hashes.
- All nine fixed-split manifests have one identical SHA256 digest.
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
