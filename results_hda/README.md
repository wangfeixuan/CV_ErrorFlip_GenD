# Heterogeneous degradation frame-weighting results

Status: complete.

## Protocol

Each four-frame video contains two clean frames, one blurred frame, and one
JPEG-compressed frame. Training and the seen test suite use all 12 position
assignments. Two held-out suites replace JPEG with resize, or blur with resize.

Three seeds were trained for each aggregator:

- uniform averaging;
- inverse-entropy confidence weighting;
- learned reliability weighting from adapted GenD features.

The learned model receives no degradation-type labels at inference. Its weights
are regularized by a clean-versus-degraded ranking loss and a KL prior against
uniform collapse.

## Results

| Suite | Uniform mixed | Confidence mixed | Learned mixed | Learned vs uniform |
|---|---:|---:|---:|---:|
| Seen: blur + JPEG | 90.03% | **90.71%** | 90.85% | +0.82 pp |
| Unseen: blur + resize | 89.99% | 90.95% | **91.45%** | +1.46 pp |
| Unseen: JPEG + resize | 90.71% | **91.70%** | 90.92% | +0.21 pp |

The learned weighting improves over uniform averaging for every seed in every
suite. The paired per-seed gains range from 0.21 to 1.82 percentage points.

Simple confidence weighting remains stronger on clean Celeb-DF and on the
JPEG-plus-resize suite. It also avoids the learned scorer's clean-accuracy
cost:

- learned clean accuracy: 91.03%;
- uniform clean accuracy: 92.31%;
- confidence clean accuracy: 92.74%.

The learned scorer gives clean frames about 39-40% weight and degraded frames
about 10-11%, so it learns the intended quality ordering without collapsing to
one frame. The stronger clean weighting nevertheless costs 1.28 percentage
points on clean input.

## Interpretation

The experiment supports a limited claim: learned reliability weighting improves
video aggregation when frame degradation is heterogeneous and mixed. It does
not establish that learned weights are universally better than a simple
confidence rule. The strongest benefit occurs on the unseen blur-plus-resize
combination.

The Adapter clean-position Oracle is not a strict upper bound because degraded
frames can sometimes supply complementary evidence. The learned model exceeds
its Oracle on the seen suite, which means the classifier benefits from
combining clean and degraded frames rather than ignoring degraded ones.

## Files

- `aggregate_hda.summary.csv`: seed-level aggregate metrics;
- `aggregate_hda.per_seed.csv`: individual seed results;
- `aggregate_hda.md`: readable result table;
- `hda_comparison.png`: mixed-condition accuracy comparison.
