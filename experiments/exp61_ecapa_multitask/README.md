# EXP 6.1: ECAPA multitask baseline

This experiment trains and evaluates language, dialect, and gender predictions
on the 1-bit and 16-bit recording domains. The matched and cross-precision
evaluation results are preserved under `reports/ecapa_7p2m/exp61/`, together
with aggregate metrics, confusion matrices, learning histories, and completion
records.

Start with:

- `configs/exp61/experiment.yaml` for the training configuration;
- `reports/ecapa_7p2m/exp61/tables/table_02_matched_results.csv` for matched
  test metrics;
- `reports/ecapa_7p2m/exp61/tables/table_04_cross_precision.csv` for
  cross-precision evaluation;
- `README_UNZIP_AND_VALIDATE.md` for the historical WTC9 integration procedure.

The historical server paths in launch scripts are not portable. Raw audio,
derived feature caches, and model checkpoints are not distributed here.
