# Result summary

This directory contains aggregate three-seed results for the fixed held-out
split and the source-group bootstrap analysis.

- `aggregate_all_conditions.summary.csv`: condition and method means/standard
  deviations over three seeds;
- `aggregate_all_conditions.per_seed.csv`: one row per method, condition, and
  seed;
- `bootstrap_all_conditions.csv`: paired video-group bootstrap confidence
  intervals;
- `k_sweep.csv` and `k_sweep_bootstrap.csv`: training blur-count sensitivity;
- `aggregate_all_conditions.*`: includes blur-specific, mixed, clean-only,
  error-weighted, hard-mask, and random-mask sweep results;
- `figures/`: report-ready comparison, position-sensitivity, and error
  recovery/harm plots.

The 16 masks are repeated measurements of the same videos. The confidence
intervals use resampling over the 52 held-out source groups, not masks.

Individual checkpoints, training logs, and per-video prediction dumps are
generated artifacts and are intentionally excluded from the repository. The
pipeline can regenerate them from the commands in the root README.
