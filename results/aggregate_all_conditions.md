# Seed aggregation

Evaluation files: `36`; test videos per seed: `78`.

| Condition | Method | Seeds | Clean Acc | Clean AUC | Local Acc | Local AUC | Worst Acc | Recovery | Harm | Net recovered masks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blur_k9 | clean_only | 3 | 91.88% | 0.9799 | 88.89% | 0.9674 | 85.47% | 50.21% | 3.26% | 27.00 |
| blur_k9 | flip_hard | 3 | 92.74% | 0.9804 | 89.67% | 0.9702 | 86.75% | 55.27% | 3.34% | 30.67 |
| blur_k9 | flip_weighted | 3 | 93.59% | 0.9821 | 90.31% | 0.9699 | 88.03% | 56.54% | 2.83% | 33.67 |
| blur_k9 | uniform | 3 | 93.59% | 0.9821 | 90.38% | 0.9702 | 88.03% | 56.54% | 2.74% | 34.00 |
| jpeg_q30 | clean_only | 3 | 91.88% | 0.9799 | 91.88% | 0.9762 | 86.75% | 59.52% | 3.72% | 18.00 |
| jpeg_q30 | flip_hard | 3 | 92.74% | 0.9804 | 92.24% | 0.9788 | 85.47% | 68.45% | 4.53% | 19.67 |
| jpeg_q30 | flip_weighted | 3 | 93.59% | 0.9821 | 92.66% | 0.9791 | 86.32% | 67.26% | 3.88% | 21.67 |
| jpeg_q30 | uniform | 3 | 93.59% | 0.9821 | 92.59% | 0.9792 | 85.90% | 66.67% | 3.88% | 21.33 |
| resize_25 | clean_only | 3 | 91.88% | 0.9799 | 91.17% | 0.9722 | 86.75% | 50.62% | 3.54% | 12.67 |
| resize_25 | flip_hard | 3 | 92.74% | 0.9804 | 91.10% | 0.9749 | 87.18% | 54.32% | 4.11% | 12.33 |
| resize_25 | flip_weighted | 3 | 93.59% | 0.9821 | 91.31% | 0.9745 | 86.75% | 53.70% | 3.78% | 13.33 |
| resize_25 | uniform | 3 | 93.59% | 0.9821 | 91.31% | 0.9748 | 86.75% | 53.70% | 3.78% | 13.33 |

Results are test-split summaries after seed-level early stopping on validation.
The 16 masks are repeated measurements of the same videos, not independent samples.
Use the per-seed table and paired differences for the final report.
