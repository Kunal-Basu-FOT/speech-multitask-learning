# Multitask speech representation experiments

This repository documents a set of speech representation experiments developed
during an INSA–NASI–IASc summer fellowship at IISc. The study compares language
identification, dialect and gender tasks across recording domains, model
frontends, task groupings, and training strategies. It preserves the source,
configs, aggregate results, figures, and completion records from the EXP 6.x
series assembled on 2026-08-08. Raw audio, checkpoints, embeddings, and
server-only caches are excluded.

## Selected results

- In the ECAPA multitask baseline (EXP 6.1), mean macro-F1 across the three
  tasks was **0.664** in the 1-bit domain and **0.817** in the 16-bit domain,
  each evaluated on 17,550 examples. See the [matched-results table](experiments/exp61_ecapa_multitask/reports/ecapa_7p2m/exp61/tables/table_02_matched_results.csv).
- The reduced language–dialect grouping experiment (EXP 6.2) found negative
  language-task affinity in both domains under its defined macro-F1 contrast.
  See the [EXP 6.2 report](experiments/exp62_task_grouping/reports/exp62_reduced/README_RESULTS.md).
- The duration and frontend study (EXP 6.9) compares waveform log-mel and
  IndicWav2Vec2 features across duration conditions and recording domains.
  Start with its [interpretation](experiments/exp69_waveform_duration_ablation/reports/exp69/RESULTS_INTERPRETATION.md)
  and [tables](experiments/exp69_waveform_duration_ablation/reports/exp69/tables/).

These are results from distinct protocols and should not be pooled into a
single accuracy claim. The [experiment-status record](docs/EXPERIMENT_STATUS.md)
separates completed studies from pending or abandoned ones.

## Experiment map

| Experiment | Topic | Status | Directory |
|---|---|---|---|
| EXP 6.1 | ECAPA multi-task baseline and cross-precision evaluation | Complete | `experiments/exp61_ecapa_multitask/` |
| EXP 6.2 | Reduced LID/dialect task grouping | Complete | `experiments/exp62_task_grouping/` |
| EXP 6.3 | Pending experiment material | Placeholder; insert later | `experiments/exp63_placeholder/` |
| EXP 6.4 | Abandoned experiment | Abandoned/deleted; documentation only | `experiments/exp64_abandoned/` |
| EXP 6.5 | Embedding/probe and geometry analysis | Complete | `experiments/exp65_exp68_embedding_analysis/` |
| EXP 6.6 | Gradient interference, PCGrad, and GradNorm | Complete | `experiments/exp66_gradient_interference/` |
| EXP 6.7 | Hierarchical LID/dialect modeling | Complete | `experiments/exp67_hierarchical_lid_dialect/` |
| EXP 6.8 | Few-shot gender transfer analysis | Complete | `experiments/exp65_exp68_embedding_analysis/` |
| EXP 6.9 | Waveform-duration ablation with ECAPA | Complete | `experiments/exp69_waveform_duration_ablation/` |

See `docs/EXPERIMENT_STATUS.md` for completion evidence, `docs/ARTIFACT_INDEX.md` for entry points, `docs/SOURCE_MAP.md` for provenance, and `docs/EXCLUSIONS.md` for the cleanup boundary.

## Reproducibility notes

- Each completed experiment retains its own original README, requirements, config, source, tests, scripts, validation metadata, reports, and lightweight result evidence.
- `environment.yml` is the captured server environment specification. Recreating it may require platform-specific adjustments.
- `project_tools/manifest_builders/` contains the small scripts used to build dataset manifests. The path-bearing manifest CSVs and all raw audio remain outside this archive.
- The original code expects external datasets, caches, pretrained weights, and/or checkpoints in some workflows. Those large dependencies are documented but not vendored.
- Evidence JSON may contain original absolute server paths. It is preserved verbatim for provenance; use the repository-relative locations described in the index for offline browsing.
- The captured `environment.yml` documents the training environment. Some
  experiment scripts require external audio manifests, pretrained weights, and
  GPU resources; follow each experiment's README and config rather than
  expecting a single command to rerun the entire series.

## Quick offline review

Start with the report README or interpretation document in each experiment's `reports/` directory. Aggregate tables are CSV files, figures are PNG files, evaluation summaries are `metrics.json`, and retained training histories are `runtime/runs/**/history.csv`.

The [artifact index](docs/ARTIFACT_INDEX.md) links each experiment to its
configuration, report, and result files. The [validation record](docs/VALIDATION.md)
describes the original archive checks and their limits. After downloading,
`python scripts/verify_archive.py` checks the SHA-256 manifest for all released
files.
