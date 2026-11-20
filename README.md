# Error-Flip-Aware GenD Adapter

This project is a Track 2 study of robust short-video deepfake detection. It
starts from the published GenD CLIP ViT-L/14 detector, freezes its vision
encoder, and trains a small residual adapter on the 1024-dimensional frame
features.

The central experiment compares training objectives under identical
architecture and data:

- `clean_only`: train the adapter on clean videos only, with the same total
  loss weight and optimization schedule.
- `uniform`: average the video-level loss over all six two-frame blur masks.
- `flip_weighted`: retain all six masks but up-weight masks on which the
  original GenD prediction is wrong while its clean prediction is correct.
- `flip_hard`: an ablation that backpropagates through the currently hardest
  eligible mask.

The goal is not to claim that training harder examples necessarily improves
accuracy. The goal is to measure whether the adapter recovers GenD errors
caused by sparse local degradation without harming clean videos.

## Current results

The fixed Celeb-DF v2 split contains 362 training, 78 validation, and 78 test
videos. All three model seeds use the same split manifest. Mean accuracy over
the six two-of-four degradation masks is:

| Condition | GenD | Clean-only adapter | Uniform adapter | Error-weighted | Hard-mask |
|---|---:|---:|---:|---:|---:|
| Blur k9, trained | 83.12% | 88.89% | **90.38%** | 90.31% | 89.67% |
| JPEG q30, unseen | 88.03% | 91.88% | 92.59% | **92.66%** | 92.24% |
| Resize 25%, unseen | 88.46% | 91.17% | **91.31%** | **91.31%** | 91.10% |

Clean-only adaptation shows that generic fine-tuning explains most of the
gain over GenD. After controlling for it, local-mask training adds 1.50 points
on blur, 0.71 on unseen JPEG, and 0.14 on unseen resize. The blur difference
has a source-group bootstrap interval of [-0.15, 3.43], so the independent
mask-training effect is directionally positive but not statistically
conclusive. Error weighting and hard-mask selection do not provide a stable
additional gain.

### Why k=2 is the primary setting

The primary experiment fixes two blurred frames because the earlier audit
found 61 videos whose decision changed across exactly two-of-four blur masks,
and `k=2` provides six position permutations while avoiding the near-global
information loss of full-video blur. This is a problem-driven choice, not a
claim that `k=2` is optimal.

As a follow-up, we separately train random-mask adapters with `k=1,2,3,4`
under the same per-sample mask exposure and optimization budget. On the
trained blur condition, test-at-`k=2` accuracy is 89.10%, 89.10%, 89.89%, and
89.81%, respectively. Training at `k=3` or `k=4` is slightly higher than
`k=2`, but paired source-group intervals include zero. The follow-up supports
a mask-count sensitivity analysis, not a claim that another `k` is superior.

### Mixed-degradation follow-up

We also train a unified adapter on blur, JPEG q30, and resize 25%. The
low-cost arm chooses one type per sample and step; the full-exposure arm
evaluates all three types in every step. Neither improves a tested condition.

| Adapter training | Clean | Blur | JPEG | Resize |
|---|---:|---:|---:|---:|
| Blur-specific | 93.59% | **90.38%** | **92.59%** | **91.31%** |
| Mixed, one type | 93.16% | 88.89% | 91.03% | 90.46% |
| Mixed, all types | 90.60% | 88.53% | 90.10% | 89.60% |

The mixed variants are retained as negative controls. With this data size and
one small adapter, balancing three degradation objectives creates a
conservative compromise rather than complementary robustness.

### Six-method comparison

The primary comparison now includes individual JPEG and resize adapters:

| Method | Clean | Blur | JPEG | Resize |
|---|---:|---:|---:|---:|
| GenD | 85.90% | 83.12% | 88.03% | 88.46% |
| Clean-only | 91.88% | 88.89% | 91.88% | 91.17% |
| Blur-only | **93.59%** | 90.38% | **92.59%** | **91.31%** |
| Mask-consistency | 92.74% | 89.89% | 92.74% | 91.38% |
| JPEG-only | 92.74% | **90.81%** | 91.31% | 91.45% |
| Resize-only | **93.59%** | 89.25% | 93.02% | 90.31% |
| Mixed-all | 90.60% | 88.53% | 90.10% | 89.60% |

No specialist dominates its matching condition, which is consistent with the
small dataset and high overlap between the three deepfake-centric degradation
objectives. The full mixed arm remains a negative result.

Mask-consistency adds a variance penalty across the six blur-position
predictions. It loses 0.49 points on blur and 1.28 points in worst-mask
accuracy relative to Blur-only, while gaining only 0.14 on JPEG and 0.07 on
resize. The consistency objective remains a negative result.

### Static weight tuning

The already-trained Clean, Blur, JPEG, and Resize adapters were combined
without retraining. A validation-only sweep selected Clean 0%, Blur 20%, JPEG
40%, and Resize 40%. On test this ensemble reaches 92.31% clean, 89.96% blur,
92.52% JPEG, and 91.45% resize accuracy. It does not beat Blur-only on the
trained blur or JPEG conditions; the resize gain is only 0.14 points.

The final course report is [`reports/CV_ErrorFlip_GenD_Report.pdf`](reports/CV_ErrorFlip_GenD_Report.pdf).
Aggregate tables and figures are under [`results/`](results/).

## Data

The primary face banks in this workspace are:

- `selected_faces.zip`
- `selected_faces_blur_k9.zip`

They contain the same 2,072 paths, representing 518 Celeb-DF v2 videos and
four aligned 256x256 face crops per video.

Training and evaluation use video-level source groups. All frames from a
video and the corresponding original/blur pair remain in the same split.
The source-group key also keeps all synthesized videos generated from the
same identity pair together. It does not prove complete identity isolation,
which must remain an explicit limitation in the report.

## Server setup

Use only the dedicated directory:

```text
/root/autodl-tmp/CV_ErrorFlip_GenD
```

Do not write to or clean the VADE directory. The project expects this layout:

```text
CV_ErrorFlip_GenD/
├── code/
├── data/
│   ├── model.safetensors
│   ├── selected_faces.zip
│   └── selected_faces_blur_k9.zip
├── artifacts/
└── runs/
```

Install the small set of Python dependencies:

```bash
cd /root/autodl-tmp/CV_ErrorFlip_GenD/code
python -m pip install -r requirements.txt
```

The manuscript should cite the original GenD repository:
<https://github.com/yermandy/GenD>.

## Pipeline

### 1. Extract frozen GenD features

```bash
python -m src.extract_features \
  --original-zip ../data/selected_faces.zip \
  --blur-zip ../data/selected_faces_blur_k9.zip \
  --model ../data/model.safetensors \
  --output ../artifacts/features_blur_k9.npz \
  --device cuda \
  --batch-size 16
```

The cache contains normalized original features, normalized blur features,
the original GenD per-frame logits, and the exact publication classification
head. Later adapter training does not need to rerun the CLIP encoder.

### 2. Train the two controlled variants

```bash
python -m src.train \
  --features ../artifacts/features_blur_k9.npz \
  --method uniform \
  --seed 2026 \
  --output-dir ../runs/uniform_seed2026 \
  --device cuda
```

```bash
python -m src.train \
  --features ../artifacts/features_blur_k9.npz \
  --method flip_hard \
  --seed 2026 \
  --output-dir ../runs/flip_hard_seed2026 \
  --device cuda
```

Run three seeds only after the account owner confirms that the GPU is free.
The training script never evaluates or tunes on the test split.

### 3. Evaluate each frozen checkpoint

```bash
python -m src.evaluate \
  --features ../artifacts/features_blur_k9.npz \
  --checkpoint ../runs/flip_hard_seed2026/best.pt \
  --split test \
  --output ../runs/flip_hard_seed2026/eval_test.json
```

Evaluation covers all 16 original/blur masks, but treats the 16 masks as
repeated measurements of the same video. The output separately reports:

- clean performance;
- mean and worst predictive performance over all 16 masks;
- the six-mask, exactly-two-blur condition used in training;
- mask-event recovery and harm counts;
- video-level flip rate and position-score variance.

Do not multiply the 518 videos by 16 as if they were independent samples.

## Tests

```bash
python -m pytest -q
python -m src.smoke_test
```

The smoke test creates synthetic GenD features and checks both training
objectives, checkpoint loading, and evaluation without downloading a model or
using a GPU.

## Scientific boundaries

- This is an extension of GenD, not a new foundation model.
- The existing Celeb-DF videos were used during exploratory analysis and are
  not a fully blind external test.
- The adapter can only use information retained in the frozen features.
- A negative result is a valid result. The report must distinguish total
  accuracy from error recovery and newly introduced errors.
- The two-frame training condition can always be extended to all masks only
  after the initial comparison is complete.

## Repository contents

This repository includes source code, configuration, tests, the final report,
aggregate metrics, bootstrap results, and figures. The following large or
restricted artifacts are intentionally excluded:

- the GenD `model.safetensors` weights;
- Celeb-DF and UADFV face archives;
- cached feature tensors;
- individual checkpoint binaries and per-video prediction dumps.

The model can be downloaded from the official GenD project, while datasets
must be obtained under their original access terms.
