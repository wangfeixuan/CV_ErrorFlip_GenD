# DFDC cross-dataset experiment

Status: complete.

- DFDC `train_part_0`: 1,334 videos imported;
- labels: 86 real and 1,248 fake;
- source groups: 86, following DFDC's `original` field;
- failed face imports: 0;
- training: 8 methods x 3 seeds;
- evaluation: 72 files (8 methods x 3 seeds x 3 conditions);
- external test: all 518 Celeb-DF v2 videos.

The class-imbalanced DFDC set uses inverse-frequency class weighting during
training. DFDC is used only for train/validation; Celeb-DF remains external.

## Main observation

Frozen GenD is more accurate than every DFDC-trained adapter on the
out-of-domain Celeb-DF evaluation. The adapters fit their DFDC validation set
well, but that gain does not transfer across dataset, face-crop and generation
domains.

| Condition | Frozen GenD local accuracy | Best adapter local accuracy | Best method |
|---|---:|---:|---|
| Blur k9 | 83.72% | 76.36% | Flip-weighted |
| JPEG q30 | 84.75% | 73.11% | JPEG-only |
| Resize 25% | 84.91% | 73.64% | Clean-only |

Clean Celeb-DF accuracy falls from 85.14% for frozen GenD to at most 81.40%
after DFDC adaptation. AUC also declines, so the drop is not explained by a
single classification threshold.

DFDC validation accuracy is high (roughly 91-97%), while external Celeb-DF
accuracy is much lower. This is evidence of domain-specific adaptation rather
than a failed training run.

## Controlled local-degradation result

On blur, uniform mask training exceeds clean-only by 1.35 percentage points,
with a source-group bootstrap 95% interval of [-0.14, 2.90] points. The effect
is directionally positive but not statistically conclusive. The corresponding
differences on JPEG and resize include zero and are slightly negative.

These external results do not replace the earlier in-domain experiment. They
show that the local-degradation training effect is not reliably transported to
a different deepfake dataset with a different face pipeline.

## Files

- `aggregate_celebdf.summary.csv`: seed-level aggregate metrics;
- `aggregate_celebdf.per_seed.csv`: one row per evaluation file;
- `aggregate_celebdf.md`: human-readable aggregate table;
- `bootstrap_celebdf.csv`: paired source-group bootstrap intervals.
