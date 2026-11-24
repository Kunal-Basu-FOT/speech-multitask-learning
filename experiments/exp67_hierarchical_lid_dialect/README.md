# EXP 6.7 — Hierarchical Language → Dialect Conditioning

## Purpose

This repository runs the project-specific EXP 6.7 study on WTC7 physical GPU 1.
It tests whether a predicted language posterior helps local dialect identification,
while preserving the validated EXP 6.1 data, ECAPA, training, checkpointing, and
evaluation contracts.

## Scientific intervention

The flat EXP 6.1 dialect pathway is:

```text
shared embedding [192]
→ ground-truth language selects a local dialect head
→ local dialect prediction
```

The EXP 6.7 pathway is:

```text
shared embedding [192]
→ LID head
→ 8-class LID softmax probabilities
→ stop-gradient
→ concatenate with shared embedding [192 + 8 = 200]
→ ground-truth language selects a local dialect head
→ local dialect prediction
```

The LID probabilities condition the dialect classifier, but they do not select the
local dialect head. Local-head routing remains oracle/ground-truth-language based,
matching EXP 6.1.

## Models and queue

Two new models train sequentially on physical GPU 1:

```text
1. hierarchical_lid_dialect / one_bit
2. hierarchical_lid_dialect / sixteen_bit
```

Each model is evaluated on both test representations:

```text
one_bit-trained model:
  one_bit test       matched
  sixteen_bit test   cross precision

sixteen_bit-trained model:
  sixteen_bit test   matched
  one_bit test       cross precision
```

The full queue therefore produces four test cells.

## Frozen controls inherited from EXP 6.1

```text
cache: full final-layer IndicWav2Vec sequences
feature dimension: 768
stored dtype: float16
full utterances: yes
crop/truncation: none
train rows: 146,359
dev rows: 11,507
test rows: 17,550
languages: bh, bn, ch, kn, mg, mr, mt, te
tasks: LID, local dialect, gender
ECAPA channels: [480, 480, 480, 480, 1440]
embedding: 192
head hidden dimension: 256
seed: 1337
schedule: 40 epochs = 5 + 30 + 5
optimizer: AdamW
batching: stable frame-budget batching
AMP: enabled with recoverable overflow handling
selection: three-task mean macro-F1 on matched development data
```

## Parameter contract

The validated hierarchical model has:

```text
encoder parameters: 7,214,196
task parameters:      525,486
total parameters:   7,739,682
```

The increase over the flat EXP 6.1 baseline comes from changing all local dialect
heads from 192-dimensional input to 200-dimensional input.

## Stop-gradient contract

Dialect loss is allowed to update:

```text
shared input normalization
shared ECAPA encoder
dialect heads
```

Dialect loss is not allowed to update the LID head through the posterior branch.
The formal validation explicitly tests this by backpropagating dialect-only loss
and rejecting any nonzero LID-head gradient.

The independent LID classification loss still trains the LID head normally.

## Evaluation outputs

For each of the four train/test domain cells:

```text
metrics.json
predictions.csv
EVALUATION_COMPLETE.json
confusion/lid_counts.csv
confusion/lid_row_normalized.csv
confusion/gender_counts.csv
confusion/gender_row_normalized.csv
confusion/dialect_<language>_counts.csv
confusion/dialect_<language>_row_normalized.csv
representative LID, gender, and Bengali dialect PNGs
lid_confidence_dialect_analysis.json
lid_confidence_dialect_analysis.csv
lid_confidence_vs_dialect_accuracy.png
```

`predictions.csv` includes:

```text
LID prediction and correctness
maximum LID posterior
true-language posterior
LID entropy
dialect prediction and correctness
gender prediction
```

This supports the later analysis of whether confident LID predictions provide a
more useful dialect scaffold.

## Automatic report generated after the queue

```text
reports/exp67/table_01_matched_and_cross_precision.csv
reports/exp67/table_02_per_language_dialect.csv
reports/exp67/table_03_parameter_count.csv
reports/exp67/figure_01_matched_1bit_vs_16bit.png
reports/exp67/figure_02_cross_precision_delta.png
reports/exp67/REPORT_COMPLETE.json
```

The flat-versus-hierarchical EXP 6.1 comparison is intentionally deferred until
the completed EXP 6.1 results are imported. No hard-sharing model is retrained in
this repository.

## Validation

```bash
bash scripts/00_verify_environment.sh
bash scripts/01_validate_exp67.sh
bash scripts/02_real_cache_preflight.sh
bash scripts/03_gpu1_smoke.sh
```

Required markers:

```text
EXP67_WTC7_ENVIRONMENT_PASS
EXP67_STOP_GRADIENT_VALIDATION_PASS
EXP67_SYNTHETIC_VALIDATION_PASS
EXP67_VALIDATION_PASS
EXP67_REAL_CACHE_PREFLIGHT_PASS
EXP67_GPU1_SMOKE_PASS
```

## Launch and operation

Launch:

```bash
bash scripts/10_launch_exp67_gpu1.sh
```

Attach:

```bash
bash scripts/12_attach_exp67_gpu1.sh
```

Detach without stopping:

```text
Ctrl+A, then D
```

Status:

```bash
bash scripts/11_status_exp67_gpu1.sh
```

Graceful stop:

```bash
bash scripts/13_stop_exp67_gpu1.sh
```

Resume is exact at completed epoch boundaries. An interrupted incomplete epoch
restarts from its beginning; completed epochs are not repeated.

Regenerate the internal EXP 6.7 report after all four evaluation cells exist:

```bash
bash scripts/20_generate_report.sh
```
