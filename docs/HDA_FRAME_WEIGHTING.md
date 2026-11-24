# Heterogeneous Degradation-Aware Frame Weighting

## Research question

Can a detector learn frame reliability when the four frames of one video are
degraded differently and the degraded positions are unknown?

The training schedule assigns one blur, one JPEG, and two clean frames to each
video. All 12 position assignments are exposed. The learned model receives no
degradation-type label at inference; it predicts frame weights from adapted
GenD features.

## Compared aggregators

- `uniform`: arithmetic mean of the four frame probabilities;
- `confidence`: inverse frame-entropy weighting;
- `learned`: an MLP reliability scorer followed by softmax;
- `oracle`: fixed weight on the two clean positions, used only as an upper
  bound because clean positions are unavailable in deployment.

The learned scorer is trained with the video label, a reliability-ranking loss
that weakly prefers clean over degraded frames, and a KL prior that discourages
weight collapse.

## Evaluation suites

- Seen: one blur, one JPEG, two clean frames;
- Unseen degradation pair: one blur, one resize, two clean frames;
- Unseen degradation pair: one JPEG, one resize, two clean frames.

Each suite contains all 12 positions of the two degraded frames. Reported
metrics include clean accuracy, mean mixed-condition accuracy, worst-position
accuracy, Oracle accuracy, and mean weights assigned to clean and degraded
frames.

## Commands

```bash
cd /root/autodl-tmp/CV_ErrorFlip_GenD/code
bash scripts/run_hda_experiments.sh
```

The script trains uniform, confidence, and learned aggregation with seeds 2026,
2027, and 2028. Outputs are written to `runs_hda/`.

This extension is tracked separately from the final course report until the
server experiment completes and the result is reviewed.
