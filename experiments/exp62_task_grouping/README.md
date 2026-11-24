# EXP 6.2 — Reduced LID–Dialect Task Grouping

## Purpose

This is the upload-ready implementation of the reduced six-model EXP 6.2 study on WTC9.

It trains three conditions at each precision:

```text
lid_only
dialect_only
lid_dialect
```

Domains:

```text
one_bit
sixteen_bit
```

Total formal training: **6 models**.

Every best checkpoint is evaluated on both test domains:

```text
matched test
cross-precision test
```

Total test evaluations: **12**.

## Frozen controls inherited from EXP 6.1

- Cache: `/data/Kunal/shared_cache/exp61_indicwav2vec_final_layer_fp16`
- Population: 146,359 train / 11,507 dev / 17,550 test
- Full IndicWav2Vec final-layer sequences, 768 dimensions
- No crop and no truncation
- Compact SpeechBrain ECAPA-TDNN, 192-dimensional embedding
- Language-conditioned local dialect heads
- Ground-truth language routing for primary dialect evaluation
- Stable frame-budget batching: 12,000 frames, maximum batch size 32
- 40 epochs: 5 warm-up + 30 full training + 5 heads-only
- AdamW, AMP, gradient clipping, seed 1337
- Matched-domain development checkpoint selection

## Confusion matrices

Every evaluation creates one three-panel overview:

```text
LID | Gender | Bengali dialect
```

Only trained tasks contain matrices. Inactive tasks are explicitly marked `Task not trained — N/A`; no random or untrained head is reported.

Valid detailed matrices are also saved separately. For dialect-active conditions, all eight language-specific dialect matrices are generated; Bengali is the representative main-report matrix.

## Report outputs

After all 12 evaluations, run:

```bash
bash scripts/20_generate_exp62_report.sh
```

This creates:

- condition parameter-count table;
- complete matched/cross metrics table;
- directional macro-F1 affinity table and figure;
- directional accuracy affinity table and figure;
- cross-precision change table and figure.

Until EXP 6.1 finishes, the report writes:

```text
REPORT_WAITING_FOR_EXP61.json
```

After EXP 6.1 finishes, first try the guarded importer:

```bash
bash scripts/22_import_exp61_matched_metrics.sh
```

Open the generated JSON and verify the recorded `_provenance` paths are the matched test outputs from the completed best checkpoints. If automatic discovery is ambiguous, create and fill the manual template:

```bash
bash scripts/21_create_exp61_summary_template.sh
```

Then rerun the report command. It creates separate 1-bit and 16-bit comparison figures for accuracy and macro-F1, with:

```text
single task
LID + dialect
LID + dialect + gender (EXP 6.1)
```

shown separately for LID and dialect.

## Install

The outer upload package contains `INSTALL_ON_WTC9.sh`. Follow `README_UPLOAD_FIRST.md` in that package.

## Validate before launch

```bash
cd /data/Kunal/FORMAL_EXP6_2_TaskGrouping_IndicWav2Vec_FinalLayer
bash scripts/00_verify_environment.sh
bash scripts/01_validate_exp62.sh
bash scripts/02_real_cache_preflight.sh
bash scripts/03_gpu0_smoke.sh
```

Do not launch unless all commands pass.

## Launch on physical GPU 0

```bash
bash scripts/10_launch_exp62_gpu0.sh
```

Attach:

```bash
bash scripts/12_attach_exp62_gpu0.sh
```

Detach without stopping:

```text
Ctrl+A, then D
```

Status:

```bash
bash scripts/11_status_exp62_gpu0.sh
```

Stop safely:

```bash
bash scripts/13_stop_exp62_gpu0.sh
```

An interrupted incomplete epoch restarts from its beginning; completed epochs resume from `last.pt`.

## Queue order

```text
one_bit/lid_only       → matched + cross evaluation
one_bit/dialect_only   → matched + cross evaluation
one_bit/lid_dialect    → matched + cross evaluation
sixteen_bit/lid_only   → matched + cross evaluation
sixteen_bit/dialect_only → matched + cross evaluation
sixteen_bit/lid_dialect  → matched + cross evaluation
```

## Output roots

```text
runtime/runs/exp62_reduced/
results/exp62_reduced/
reports/exp62_reduced/
```

Each condition is isolated and trains from scratch unless resuming its own `last.pt`.

## Scientific limitation

This is a focused LID–dialect cooperation study. It must not be described as an exhaustive task-grouping analysis over LID, dialect, and gender.
