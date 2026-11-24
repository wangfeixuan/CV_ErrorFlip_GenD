# Heterogeneous degradation aggregation

| Suite | Method | Seeds | Base clean | Adapter clean | Base mixed | Adapter mixed | Adapter worst | Base oracle | Adapter oracle | Clean weight | Degraded weight |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seen_blur_jpeg | hda_confidence | 3 | 85.90% | 92.74% | 86.22% | 90.71% | 89.32% | 85.04% | 91.88% | 0.296 | 0.204 |
| seen_blur_jpeg | hda_learned | 3 | 85.90% | 91.03% | 86.22% | 90.85% | 89.32% | 85.04% | 90.67% | 0.395 | 0.105 |
| seen_blur_jpeg | hda_uniform | 3 | 85.90% | 92.31% | 86.22% | 90.03% | 88.89% | 85.04% | 91.60% | 0.250 | 0.250 |
| unseen_blur_resize | hda_confidence | 3 | 85.90% | 92.74% | 85.79% | 90.95% | 89.74% | 85.04% | 91.88% | 0.293 | 0.207 |
| unseen_blur_resize | hda_learned | 3 | 85.90% | 91.03% | 85.79% | 91.45% | 89.74% | 85.04% | 90.67% | 0.393 | 0.107 |
| unseen_blur_resize | hda_uniform | 3 | 85.90% | 92.31% | 85.79% | 89.99% | 88.46% | 85.04% | 91.60% | 0.250 | 0.250 |
| unseen_jpeg_resize | hda_confidence | 3 | 85.90% | 92.74% | 87.93% | 91.70% | 89.74% | 85.04% | 91.88% | 0.305 | 0.195 |
| unseen_jpeg_resize | hda_learned | 3 | 85.90% | 91.03% | 87.93% | 90.92% | 88.46% | 85.04% | 90.67% | 0.393 | 0.107 |
| unseen_jpeg_resize | hda_uniform | 3 | 85.90% | 92.31% | 87.93% | 90.71% | 89.32% | 85.04% | 91.60% | 0.250 | 0.250 |
