# EXP 6.1 configurations

## Common configuration

- `common/project.yaml`: repository, runtime, GPU and provenance settings
- `common/data.yaml`: manifests, labels, tasks and cache rules
- `common/training.yaml`: curriculum, loss, batching and evaluation settings
- `common/report.yaml`: final figure and table definitions

## Model

- `models/ecapa_compact_7m.yaml`: compact SpeechBrain ECAPA-TDNN

The encoder uses:

- input dimension: 768
- channels: 480, 480, 480, 480, 1440
- Res2Net scale: 8
- SE channels: 128
- attention channels: 128
- utterance embedding: 192
- target encoder parameter count: 7,212,660

## Formal runs

- `exp61/one_bit/run.yaml`
- `exp61/sixteen_bit/run.yaml`

The loader resolves the files listed under `includes` relative to each run
configuration and merges them before training.

## Scientific design

- full cached feature sequence
- no temporal cropping or truncation
- LID: one 8-class head
- gender: one 2-class head
- dialect: eight language-specific heads
- primary dialect routing: oracle ground-truth language
- 40 epochs: 5 + 30 + 5
- uncertainty-weighted multitask loss
- frame-budget dynamic batching
- matched and crossed precision evaluation
