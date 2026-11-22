# WildDeepfake migration

## Purpose

Train the existing GenD Adapter variants on an independent, real-world face
sequence dataset while keeping the Celeb-DF v2 518-video list untouched for
final external evaluation.

## Access

The official WildDeepfake repository requires users to complete its usage
agreement. Complete the agreement before downloading images:

<https://forms.gle/o8vy9Q8fQ5mQZ4Qk6>

The official public dataset is hosted by its authors at:

<https://huggingface.co/datasets/xingjunm/WildDeepfake>

## Fixed subset

`manifests/wilddeepfake_subset_shards.csv` selects a deterministic subset of
the official training shards:

- 142 real shards;
- 128 fake shards;
- 15.998 GiB compressed in total, below the 20 GiB limit.

Only four frames are retained from each face sequence. Downloading the
full 58.6 GiB training partition is explicitly avoided.

## Execution

After the usage agreement has been completed:

```bash
cd /root/autodl-tmp/CV_ErrorFlip_GenD/code
bash scripts/download_wilddeepfake_subset.sh
bash scripts/run_wilddeepfake_pipeline.sh
bash scripts/run_wilddeepfake_experiments.sh
```

The pipeline performs:

1. streamed extraction of four frames per sequence;
2. OpenCV blur k9, JPEG q30, and resize-25 degradation generation;
3. frozen GenD feature extraction;
4. 85% WildDeepfake training and 15% validation split;
5. clean-only, blur-only, mask-consistency, feature-restore, JPEG-only,
   resize-only, and mixed-degradation Adapter training;
6. evaluation on all Celeb-DF v2 518 videos through `--split external`.

## Boundaries

- WildDeepfake releases face sequences rather than complete videos.
- Its face-crop and alignment pipeline differs from the one used for the
  current Celeb-DF crops.
- WildDeepfake does not provide a clean counterpart for every face sequence.
- Source-sequence groups remain together, but complete identity isolation
  cannot be guaranteed.
- The Celeb-DF results are external to Adapter training but are not
  researcher-blind because the project previously explored them.
