# EXP 6.2 — Reduced LID–Dialect Task Grouping Report

## Scope

This is the reduced cache-compatible EXP 6.2 study.

Conditions:

- `lid_only`
- `dialect_only`
- `lid_dialect`

Domains:

- `one_bit`
- `sixteen_bit`

Gender, domain ID, and accent classification are not part of this experiment.

## Directional macro-F1 affinity

Affinity is defined as:

`A(target | partner) = joint metric − single-task metric`

| Domain | Target | Partner | Single-task macro-F1 | Joint macro-F1 | Affinity |
|---|---|---|---:|---:|---:|
| 1-bit | language | dialect | 63.2115% | 45.2750% | -17.9365 pp |
| 1-bit | dialect | language | 46.8517% | 45.2750% | -1.5766 pp |
| 16-bit | language | dialect | 89.6309% | 63.8838% | -25.7472 pp |
| 16-bit | dialect | language | 62.4992% | 63.8838% | +1.3845 pp |

## Matched test metrics

| Domain | Condition | Language accuracy | Language macro-F1 | Dialect language-macro accuracy | Dialect language-macro F1 | Selection macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| 1-bit | LID only | 63.2934% | 63.2115% | — | — | 63.2115% |
| 1-bit | Dialect only | — | 46.8517% | 46.9151% | 46.8517% | 46.8517% |
| 1-bit | LID + dialect | 62.4274% | 45.2750% | 45.4365% | 45.2750% | 45.2750% |
| 16-bit | LID only | 89.6695% | 89.6309% | — | — | 89.6309% |
| 16-bit | Dialect only | — | 62.4992% | 62.5049% | 62.4992% | 62.4992% |
| 16-bit | LID + dialect | 90.2792% | 63.8838% | 63.9335% | 63.8838% | 63.8838% |

## Interpretation rules

- Positive affinity indicates beneficial transfer.
- Negative affinity indicates task interference.
- Macro-F1 affinity is the primary transfer measure.
- Accuracy affinity is retained as the secondary measure.
- Matched and cross-precision scores are reported separately.
- Test results were not used for checkpoint selection.

## Generated figures

- `figures/figure_01_one_bit_matched_macro_f1.png`
- `figures/figure_02_sixteen_bit_matched_macro_f1.png`
- `figures/figure_03_one_bit_affinity_heatmap.png`
- `figures/figure_04_sixteen_bit_affinity_heatmap.png`
- `figures/figure_05_cross_precision_transfer.png`

## Generated tables

- `tables/table_01_condition_configuration.csv`
- `tables/table_02_matched_results.csv`
- `tables/table_03_directional_macro_f1_affinity.csv`
- `tables/table_04_directional_accuracy_affinity.csv`
- `tables/table_05_cross_precision_results.csv`
- `tables/table_06_per_language_dialect.csv`

Confusion-matrix artifacts remain in the original evaluation directories under `results/exp62_reduced/`.
