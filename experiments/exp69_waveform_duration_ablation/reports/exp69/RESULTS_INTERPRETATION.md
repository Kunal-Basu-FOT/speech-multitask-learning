# EXP 6.9 — Automatically Generated Results Interpretation

This document reports measured outputs only. It compares a waveform-derived 80-bin log-Mel front end against the completed full-utterance IndicWav2Vec2 layer-11 reference from EXP 6.1.

## one-bit duration trend
- **Language macro-F1:** best at `9s` (47.09%). Changes 3→7, 7→9 and 9→full: +5.28, +1.45, -4.30 percentage points.
- **Dialect language-macro F1:** best at `9s` (40.10%). Changes 3→7, 7→9 and 9→full: +0.22, +1.04, -2.94 percentage points.
- **Gender macro-F1:** best at `3s` (93.45%). Changes 3→7, 7→9 and 9→full: -0.57, +0.34, -0.58 percentage points.
- **Three-task mean macro-F1:** best at `9s` (60.14%). Changes 3→7, 7→9 and 9→full: +1.65, +0.94, -2.61 percentage points.

## sixteen-bit duration trend
- **Language macro-F1:** best at `9s` (66.65%). Changes 3→7, 7→9 and 9→full: +16.61, +1.29, -8.37 percentage points.
- **Dialect language-macro F1:** best at `7s` (46.11%). Changes 3→7, 7→9 and 9→full: +4.18, -0.44, -1.20 percentage points.
- **Gender macro-F1:** best at `3s` (94.78%). Changes 3→7, 7→9 and 9→full: -0.23, -0.34, +0.27 percentage points.
- **Three-task mean macro-F1:** best at `9s` (68.85%). Changes 3→7, 7→9 and 9→full: +6.85, +0.17, -3.10 percentage points.

## Full-length IndicWav2Vec2 versus log-Mel
- **one-bit Language macro-F1:** IndicWav2Vec2 minus log-Mel = +18.48 percentage points.
- **one-bit Dialect language-macro F1:** IndicWav2Vec2 minus log-Mel = +7.86 percentage points.
- **one-bit Gender macro-F1:** IndicWav2Vec2 minus log-Mel = +0.31 percentage points.
- **one-bit Three-task mean macro-F1:** IndicWav2Vec2 minus log-Mel = +8.88 percentage points.
- **sixteen-bit Language macro-F1:** IndicWav2Vec2 minus log-Mel = +31.12 percentage points.
- **sixteen-bit Dialect language-macro F1:** IndicWav2Vec2 minus log-Mel = +16.97 percentage points.
- **sixteen-bit Gender macro-F1:** IndicWav2Vec2 minus log-Mel = -0.14 percentage points.
- **sixteen-bit Three-task mean macro-F1:** IndicWav2Vec2 minus log-Mel = +15.99 percentage points.

## Paired speaker-bootstrap evidence
- `one_bit` / `language_macro_f1`: mean difference +18.48 pp, 95% CI [+17.44, +19.54] pp, positive in 100.0% of bootstrap samples.
- `one_bit` / `dialect_language_macro_f1`: mean difference +7.83 pp, 95% CI [+6.55, +9.07] pp, positive in 100.0% of bootstrap samples.
- `one_bit` / `gender_macro_f1`: mean difference +0.30 pp, 95% CI [-0.17, +0.75] pp, positive in 89.7% of bootstrap samples.
- `one_bit` / `three_task_mean_macro_f1`: mean difference +8.87 pp, 95% CI [+8.29, +9.46] pp, positive in 100.0% of bootstrap samples.
- `sixteen_bit` / `language_macro_f1`: mean difference +31.14 pp, 95% CI [+29.98, +32.31] pp, positive in 100.0% of bootstrap samples.
- `sixteen_bit` / `dialect_language_macro_f1`: mean difference +16.95 pp, 95% CI [+15.46, +18.45] pp, positive in 100.0% of bootstrap samples.
- `sixteen_bit` / `gender_macro_f1`: mean difference -0.14 pp, 95% CI [-0.59, +0.29] pp, positive in 27.0% of bootstrap samples.
- `sixteen_bit` / `three_task_mean_macro_f1`: mean difference +15.98 pp, 95% CI [+15.27, +16.74] pp, positive in 100.0% of bootstrap samples.

## Interpretation boundary

A positive full-length gap supports the claim that the frozen pretrained IndicWav2Vec2 final-layer representation is more useful than this conventional log-Mel baseline for the same three-task ECAPA classifier. It does not isolate pretraining, contextualisation, the convolutional feature encoder, and the final transformer layer as separate causal factors.
