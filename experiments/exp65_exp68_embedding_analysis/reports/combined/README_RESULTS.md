# EXP 6.5 + EXP 6.8 Generated Results Summary

Generated: 2026-08-06T08:09:36.057662+00:00

## Scope

This report analyzes frozen 192-dimensional ECAPA representations. No encoder was trained in EXP 6.5 or EXP 6.8.
The available project labels are LID, language-local dialect, and gender; domain and accent are not tested.

## EXP 6.5 — joint versus single-task representation

- one_bit lid: joint minus comparison = -1.24 percentage points macro-F1.
- one_bit dialect: joint minus comparison = -0.06 percentage points macro-F1.
- one_bit gender: joint minus comparison = +0.59 percentage points macro-F1.
- sixteen_bit lid: joint minus comparison = -0.42 percentage points macro-F1.
- sixteen_bit dialect: joint minus comparison = +2.47 percentage points macro-F1.
- sixteen_bit gender: joint minus comparison = +0.80 percentage points macro-F1.

## EXP 6.5 — precision effects

- Largest absolute 16-bit minus 1-bit probe gap: lid_dialect / lid = +29.52 percentage points macro-F1.

## EXP 6.8 — held-out gender transfer

- one_bit LID+dialect encoder: gender macro-F1 91.78%, gain over random +0.01 points, random-adjusted supervised recovery 0.48%.
- sixteen_bit LID+dialect encoder: gender macro-F1 87.51%, gain over random -5.88 points, random-adjusted supervised recovery -304.36%.

## Interpretation limits

- Frozen probes quantify linearly recoverable information; they do not prove causal disentanglement.
- Dialect metrics are computed inside each language and macro-averaged, because dialect labels are language-local.
- Cross-precision results use the same probe without target-domain refitting.
- EXP 6.8 is held-out-task few-shot probing, not literal zero-label classification.

## Output inventory

See `reports/exp65/tables`, `reports/exp65/figures`, `reports/exp68/tables`, and `reports/exp68/figures`.
