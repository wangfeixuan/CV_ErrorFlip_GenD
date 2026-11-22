# CV Error-Flip GenD Submission

This bundle contains the final report, editable Word version, LaTeX source,
figures, code, and aggregate experiment outputs. It excludes raw videos, model
weights, feature caches, checkpoints, and other large data files.

## Reports

- `report/CV_ErrorFlip_GenD_Report_20261122.pdf`: final LaTeX PDF using a
  two-column NeurIPS-style reference layout.
- `report/CV_ErrorFlip_GenD_Report_20261122.docx`: editable single-column Word
  report.
- `report/CV_ErrorFlip_GenD_Report_20261122.tex`: LaTeX source.
- `report/archive/`: previous report and draft retained for rollback.

The final PDF has five pages. Main content occupies pages 1-4, and references
begin on page 5. The Word version contains the same results with native
editable tables and text.

## Main result

The final method is the Domain-Balanced Uniform Adapter. On the fixed 78-video
Celeb-DF test split it reaches 91.03% clean accuracy and 88.82%/90.60%/90.95%
on blur/JPEG/resize. It improves over frozen GenD but remains below the
Celeb-DF-only Adapter. DFDC-only transfer is reported as a negative result.

## Data

Raw data is intentionally absent. Dataset reproduction instructions are in
`docs/DFDC_MIGRATION.md`. DFDC part 0 is a third-party Kaggle mirror and remains
subject to the original DFDC terms.

## Verification

- Local tests: `9 passed`.
- DFDC import: 1,334 videos, zero failed face imports.
- DFDC-only experiments: 24 models and 72 external evaluation files.
- Multi-domain experiments: 12 models and 72 evaluation files.
- Bootstrap and aggregation outputs are under `results_dfdc/` and
  `results_multidomain/`.
