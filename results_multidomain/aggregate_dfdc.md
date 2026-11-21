# Seed aggregation

Evaluation files: `36`; test videos per seed: `200`.

| Condition | Method | Train k | Seeds | Clean Acc | Clean AUC | Local Acc | Local AUC | Worst Acc | Recovery | Harm | Net recovered masks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blur_k9 | multidomain_clean | 2 | 3 | 96.83% | 0.9815 | 92.72% | 0.9596 | 82.17% | 86.84% | 6.39% | 69.67 |
| blur_k9 | multidomain_flip_identity | 2 | 3 | 98.00% | 0.9781 | 95.39% | 0.9665 | 87.67% | 90.02% | 3.80% | 101.67 |
| blur_k9 | multidomain_uniform | 2 | 3 | 97.83% | 0.9795 | 95.53% | 0.9690 | 88.50% | 90.87% | 3.77% | 103.33 |
| blur_k9 | multidomain_uniform_identity | 2 | 3 | 97.83% | 0.9783 | 95.36% | 0.9678 | 88.17% | 90.66% | 3.93% | 101.33 |
| jpeg_q30 | multidomain_clean | 2 | 3 | 96.83% | 0.9815 | 86.36% | 0.9580 | 66.33% | 85.11% | 13.39% | 33.33 |
| jpeg_q30 | multidomain_flip_identity | 2 | 3 | 98.00% | 0.9781 | 87.39% | 0.9509 | 66.83% | 81.73% | 11.50% | 45.67 |
| jpeg_q30 | multidomain_uniform | 2 | 3 | 97.83% | 0.9795 | 88.53% | 0.9512 | 69.33% | 84.26% | 10.63% | 59.33 |
| jpeg_q30 | multidomain_uniform_identity | 2 | 3 | 97.83% | 0.9783 | 88.03% | 0.9511 | 68.50% | 83.42% | 11.07% | 53.33 |
| resize_25 | multidomain_clean | 2 | 3 | 96.83% | 0.9815 | 92.22% | 0.9437 | 84.83% | 92.84% | 7.90% | 97.67 |
| resize_25 | multidomain_flip_identity | 2 | 3 | 98.00% | 0.9781 | 94.08% | 0.9457 | 88.00% | 94.07% | 5.91% | 120.00 |
| resize_25 | multidomain_uniform | 2 | 3 | 97.83% | 0.9795 | 94.14% | 0.9470 | 88.17% | 94.07% | 5.85% | 120.67 |
| resize_25 | multidomain_uniform_identity | 2 | 3 | 97.83% | 0.9783 | 94.00% | 0.9465 | 87.50% | 94.07% | 6.01% | 119.00 |

Results are summaries of the evaluation files selected by --glob.
The 16 masks are repeated measurements of the same videos, not independent samples.
Use the per-seed table and paired differences for the final report.
