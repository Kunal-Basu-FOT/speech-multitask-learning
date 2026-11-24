# EXP 6.6 — Task Interference, PCGrad, and GradNorm

## Purpose

This repository runs the project-specific EXP 6.6 study on WTC10 physical GPU 0.
It measures pairwise gradient interaction among the three implemented tasks and
compares three optimization methods using the same hard-sharing ECAPA model:

```text
standard_logged
pcgrad
gradnorm
```

The implemented tasks are:

```text
language identification
oracle language-routed dialect identification
gender classification
```

The original broad problem statement also discusses domain and accent. Those
labels are not present in the frozen cache used here, so this implementation does
not claim to test domain-versus-dialect or accent-related conflict.

## Queue order — important

The queue deliberately does **not** train all six models first and report later.
It follows this exact sequence:

```text
ONE-BIT STAGE
  1. standard_logged / one_bit
     train → matched test → sixteen-bit cross-precision test
  2. pcgrad / one_bit
     train → matched test → sixteen-bit cross-precision test
  3. gradnorm / one_bit
     train → matched test → sixteen-bit cross-precision test
  4. save every one-bit result, table, and figure
  5. write one_bit/DOMAIN_COMPLETE.json

ONLY AFTER STEP 5:

SIXTEEN-BIT STAGE
  6. standard_logged / sixteen_bit
     train → matched test → one-bit cross-precision test
  7. pcgrad / sixteen_bit
     train → matched test → one-bit cross-precision test
  8. gradnorm / sixteen_bit
     train → matched test → one-bit cross-precision test
  9. save every sixteen-bit result, table, and figure
 10. write sixteen_bit/DOMAIN_COMPLETE.json

FINAL STAGE
 11. generate combined endpoint comparison
 12. write EXPERIMENT_COMPLETE.json
```

A failure in one-bit report generation blocks the first sixteen-bit training run.
This is the requested operational difference from a six-model train-first queue.

## Frozen scientific controls

```text
cache: /data/Kunal/shared_cache/exp61_indicwav2vec_final_layer_fp16
train/dev/test: 146,359 / 11,507 / 17,550
representations: full IndicWav2Vec final-layer sequences
feature dimension: 768
cache dtype: float16
cropping/truncation: none
model: hard-sharing ECAPA-TDNN
embedding dimension: 192
head hidden dimension: 256
local dialect heads: 8 heads, 33 total dialect classes
routing: ground-truth language selects local dialect head
training: 40 epochs = 5 + 30 + 5
seed: 1337
optimizer: AdamW
AMP: enabled with recoverable overflow handling
selection: matched-dev mean of LID, dialect, and gender macro-F1
GPU: WTC10 physical GPU 0 only
```

Validated total model parameter count:

```text
7,723,170
```

GradNorm freezes the three unused uncertainty log-variance parameters, so its
trainable model parameter count is 7,723,167 plus three external GradNorm task
weights.

## Method definitions

### standard_logged

Uses the EXP 6.1 homoscedastic uncertainty-weighted objective. Pairwise shared
encoder gradients are sampled every 50 batches for diagnostics. The sampling
frequency is frozen in the run contract and reported; this condition does not
pretend that every batch was measured.

### pcgrad

Uses the same uncertainty-weighted task scaling as `standard_logged`, then applies
Projected Conflicting Gradient surgery to the **full shared input normalization +
ECAPA encoder parameter set** at every encoder-training optimizer step.
Task-specific heads retain their ordinary objective gradients. Pre- and
post-projection cosines and projection counts are logged.

Phase 3 freezes the encoder, so no encoder surgery is applied during the five
heads-only epochs.

### gradnorm

Replaces uncertainty weighting with three positive task weights normalized to sum
to three. The GradNorm controller is updated from per-task gradient norms measured
with respect to the shared 192-dimensional utterance embedding. This is an
explicit, computationally stable proxy for full-encoder gradient magnitude.

Pairwise full shared-encoder gradient diagnostics are sampled every 50 batches.
The GradNorm weights are saved every epoch. During phase 3 the encoder is frozen;
the final learned weights remain fixed while heads adapt.

## Gradient outputs

Each condition/domain stores:

```text
runtime/runs/exp66/<domain>/<condition>/
├── RUN_CONTRACT.json
├── CONFIG_SNAPSHOT.json
├── progress.json
├── history.csv
├── gradient_samples.csv
├── gradient_epoch_summary.csv
├── gradnorm_weights.csv          # GradNorm only
├── best_dev_metrics.json
├── TRAINING_COMPLETE.json
└── checkpoints/
    ├── last.pt
    └── best.pt
```

Gradient pair statistics cover:

```text
LID ↔ dialect
LID ↔ gender
dialect ↔ gender
```

Recorded quantities include:

```text
pre-intervention cosine
pre-intervention conflict indicator (cosine < 0)
post-PCGrad cosine
per-task gradient norms
PCGrad projection count
per-epoch weighted mean cosine
per-epoch conflict rate
```

The package does not store full gradient vectors because that would be
prohibitively large. It stores the scientifically relevant pairwise statistics.

## Evaluation outputs

Each trained condition is evaluated immediately before the next condition:

```text
results/exp66/<train_domain>/<condition>/test_<test_domain>/
├── metrics.json
├── predictions.csv
├── EVALUATION_COMPLETE.json
└── confusion/
    ├── lid_counts.csv
    ├── lid_row_normalized.csv
    ├── lid.png
    ├── gender_counts.csv
    ├── gender_row_normalized.csv
    ├── gender.png
    ├── dialect_<language>_counts.csv
    ├── dialect_<language>_row_normalized.csv
    └── dialect_bn.png
```

For one-bit-trained models:

```text
one_bit test       matched
sixteen_bit test   cross precision
```

For sixteen-bit-trained models:

```text
sixteen_bit test   matched
one_bit test       cross precision
```

## Per-domain report outputs

After all three methods for a domain complete, the queue generates:

```text
reports/exp66/<domain>/
├── table_01_matched_and_cross_precision.csv
├── table_02_per_language_dialect.csv
├── table_03_gradient_conflict_aggregate.csv
├── table_04_gradient_epoch_summary.csv
├── table_05_training_history.csv
├── table_06_parameter_counts.csv
├── table_07_gradnorm_weights.csv
├── table_08_method_configuration.csv
├── figure_01_matched_macro_f1_by_method.png
├── figure_02_matched_vs_cross_precision.png
├── figure_03_gradient_conflict_rate_by_method.png
├── figure_04_mean_gradient_cosine_by_method.png
├── figure_gradient_epochs_standard_logged.png
├── figure_gradient_epochs_pcgrad.png
├── figure_gradient_epochs_gradnorm.png
├── figure_08_development_selection_curves.png
├── figure_09_loss_curves.png
├── figure_10_gradnorm_task_weights.png
├── figure_11_pcgrad_projection_counts.png
└── DOMAIN_REPORT_COMPLETE.json
```

The final combined report is stored under:

```text
reports/exp66/final/
```

## Installation

Run from the extracted upload package:

```bash
bash INSTALL_ON_WTC10.sh
```

Expected:

```text
EXP66_WTC10_INSTALL_PASS
```

## Validation gates

```bash
cd /data/Kunal/FORMAL_EXP6_6_GradientInterference_IndicWav2Vec_FinalLayer
bash scripts/00_verify_environment.sh
bash scripts/01_validate_exp66.sh
bash scripts/02_real_cache_preflight.sh
bash scripts/03_gpu0_smoke.sh
```

Required markers:

```text
EXP66_WTC10_ENVIRONMENT_PASS
EXP66_PCGRAD_GEOMETRY_VALIDATION_PASS
EXP66_STANDARD_LOGGED_SYNTHETIC_PASS
EXP66_PCGRAD_SYNTHETIC_PASS
EXP66_GRADNORM_SYNTHETIC_PASS
EXP66_SYNTHETIC_VALIDATION_PASS
EXP66_VALIDATION_PASS
EXP66_REAL_CACHE_PREFLIGHT_PASS
EXP66_GPU0_SMOKE_PASS
```

## Launch

```bash
bash scripts/10_launch_exp66_gpu0.sh
bash scripts/12_attach_exp66_gpu0.sh
```

Detach without stopping:

```text
Ctrl+A, then D
```

Status:

```bash
bash scripts/11_status_exp66_gpu0.sh
```

Graceful stop:

```bash
bash scripts/13_stop_exp66_gpu0.sh
```

Resume is exact at completed epoch boundaries. An interrupted incomplete epoch is
restarted; completed epochs and completed condition evaluations are skipped.
