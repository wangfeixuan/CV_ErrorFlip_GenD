# Seed aggregation

Evaluation files: `36`; test videos per seed: `78`.

| Condition | Method | Train k | Seeds | Clean Acc | Clean AUC | Local Acc | Local AUC | Worst Acc | Recovery | Harm | Net recovered masks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blur_k9 | multidomain_clean | 2 | 3 | 92.74% | 0.9763 | 86.75% | 0.9382 | 77.78% | 60.34% | 7.88% | 17.00 |
| blur_k9 | multidomain_flip_identity | 2 | 3 | 91.03% | 0.9712 | 88.75% | 0.9428 | 82.48% | 53.16% | 4.03% | 26.33 |
| blur_k9 | multidomain_uniform | 2 | 3 | 91.03% | 0.9714 | 88.82% | 0.9446 | 82.48% | 52.74% | 3.86% | 26.67 |
| blur_k9 | multidomain_uniform_identity | 2 | 3 | 90.60% | 0.9714 | 88.75% | 0.9429 | 82.48% | 52.74% | 3.94% | 26.33 |
| jpeg_q30 | multidomain_clean | 2 | 3 | 92.74% | 0.9763 | 87.82% | 0.9547 | 77.78% | 70.83% | 9.87% | -1.00 |
| jpeg_q30 | multidomain_flip_identity | 2 | 3 | 91.03% | 0.9712 | 90.31% | 0.9561 | 81.20% | 70.24% | 6.96% | 10.67 |
| jpeg_q30 | multidomain_uniform | 2 | 3 | 91.03% | 0.9714 | 90.60% | 0.9580 | 81.20% | 70.24% | 6.63% | 12.00 |
| jpeg_q30 | multidomain_uniform_identity | 2 | 3 | 90.60% | 0.9714 | 90.38% | 0.9563 | 80.77% | 70.24% | 6.88% | 11.00 |
| resize_25 | multidomain_clean | 2 | 3 | 92.74% | 0.9763 | 88.82% | 0.9587 | 80.34% | 59.26% | 7.33% | 1.67 |
| resize_25 | multidomain_flip_identity | 2 | 3 | 91.03% | 0.9712 | 90.60% | 0.9530 | 84.19% | 53.09% | 4.51% | 10.00 |
| resize_25 | multidomain_uniform | 2 | 3 | 91.03% | 0.9714 | 90.95% | 0.9534 | 84.62% | 53.09% | 4.11% | 11.67 |
| resize_25 | multidomain_uniform_identity | 2 | 3 | 90.60% | 0.9714 | 91.03% | 0.9530 | 85.04% | 54.32% | 4.19% | 12.00 |

Results are summaries of the evaluation files selected by --glob.
The 16 masks are repeated measurements of the same videos, not independent samples.
Use the per-seed table and paired differences for the final report.
