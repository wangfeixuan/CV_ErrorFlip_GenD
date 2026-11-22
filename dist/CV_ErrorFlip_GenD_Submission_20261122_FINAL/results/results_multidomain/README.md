# Domain-balanced multi-domain Adapter

Status: complete.

This follow-up addresses the negative transfer observed when an Adapter was
trained on DFDC only. One model is now trained jointly on:

- Celeb-DF training videos;
- DFDC training videos;
- balanced real/fake samples from each domain in every batch.

Celeb-DF validation and DFDC validation are used for early stopping. The final
Celeb-DF test split contains 78 videos, and the DFDC test split contains 200
videos. Neither test split participates in optimization.

## Celeb-DF held-out results

The best robust model is `multidomain_uniform`, which uses six two-of-four blur
masks across both domains.

| Condition | Frozen GenD local accuracy | Multi-domain Uniform | Difference |
|---|---:|---:|---:|
| Blur k9 | 83.12% | **88.82%** | +5.70 |
| JPEG q30 | 88.03% | **90.60%** | +2.57 |
| Resize 25% | 88.46% | **90.95%** | +2.49 |

Clean Celeb-DF accuracy changes from 85.90% to 91.03%. Worst-mask accuracy on
the trained blur condition rises from 77.41% to 82.48%.

The paired source-group bootstrap intervals against GenD are wide because the
Celeb-DF test set contains only 78 videos and 52 groups. The point estimates
are consistently positive across all tested degradations, but the blur and
JPEG intervals still include zero.

## DFDC addition is not a free improvement

The gain is relative to frozen GenD and to the failed DFDC-only adapter. It is
not a gain over training on Celeb-DF alone. Holding the method fixed to
`uniform`, adding DFDC changes the Celeb-DF test result as follows:

| Condition | Celeb-DF-only | Celeb-DF + DFDC | DFDC effect |
|---|---:|---:|---:|
| Clean | **93.59%** | 91.03% | -2.56 |
| Blur k9 | **90.38%** | 88.82% | -1.57 |
| JPEG q30 | **92.59%** | 90.60% | -1.99 |
| Resize 25% | **91.31%** | 90.95% | -0.36 |

Therefore, the current experiment does not establish that DFDC data provides
an independent accuracy benefit. Joint training recovers most of the severe
zero-shot negative transfer caused by DFDC, but remains slightly worse than the
target-domain-only Adapter on the target-domain test.

The scientific result should be stated as a data-composition and negative
transfer finding, not as "adding more data improves the model." A method claim
would require a domain-invariance component to beat the Celeb-DF-only control.

## Mask-training contribution

On Celeb-DF held-out test, Uniform mask training exceeds Clean-only by:

- Blur: +2.07 percentage points, 95% interval [-1.94, 5.98];
- JPEG: +2.78 percentage points, 95% interval [0.44, 5.56];
- Resize: +2.14 percentage points, 95% interval [-0.71, 5.31].

The JPEG interval excludes zero. The other two are directionally positive but
inconclusive. This is stronger evidence than the DFDC-only model, where
domain shift overwhelmed any mask-training effect.

## Identity regularization

A cosine near-identity penalty of 0.05 was tested as an additional variant. It
did not improve results: Uniform without the penalty is slightly better on
Celeb-DF clean accuracy and all three degraded conditions. The penalty is
therefore retained as a negative ablation, not as the final method.

## DFDC held-out sanity check

The same model also improves DFDC:

| Condition | Frozen GenD local accuracy | Multi-domain Uniform |
|---|---:|---:|
| Blur k9 | 86.92% | **95.53%** |
| JPEG q30 | 83.58% | **88.53%** |
| Resize 25% | 84.08% | **94.14%** |

This shows that joint training retains strong in-domain performance while
recovering Celeb-DF accuracy.

## Files

- `aggregate_celebdf.summary.csv` and `.md`: Celeb-DF seed aggregation;
- `aggregate_dfdc.summary.csv` and `.md`: DFDC seed aggregation;
- `aggregate_celebdf.per_seed.csv`: all Celeb-DF evaluation runs;
- `aggregate_dfdc.per_seed.csv`: all DFDC evaluation runs;
- `bootstrap_celebdf.csv`: paired source-group confidence intervals.

## Protocol limitation

This model uses Celeb-DF training videos, so the result is no longer a
zero-shot cross-dataset transfer claim. It is a fair multi-domain model
development result: Celeb-DF training and validation are used for fitting and
early stopping, while the fixed 78-video Celeb-DF test split is used only for
final evaluation.
