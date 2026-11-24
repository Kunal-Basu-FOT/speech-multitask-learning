# Scientific protocol

## Independent variables

1. Encoder supervision: LID-only, dialect-only, LID+dialect.
2. Representation precision: one-bit or sixteen-bit cache.

## Frozen controls

- Cache-aligned rows: 146,359 train, 11,507 dev, 17,550 test.
- Full final-layer IndicWav2Vec sequences, 768 dimensions, no crop or truncation.
- Compact ECAPA encoder, 192-dimensional shared representation.
- Fixed seed 1337.
- Official speaker-disjoint train/dev/test partitions.

## EXP 6.5 claims

Linear probes measure recoverable information in a frozen representation. Geometry metrics measure separation and compactness. Neither proves causal or complete disentanglement.

## Dialect protocol

Dialect labels are local to language. Primary probing and geometry are therefore performed independently inside each language and macro-averaged across the eight languages.

## EXP 6.8 terminology

The EXP 6.2 encoders receive zero gender supervision, while the downstream gender probe uses labels. The precise description is:

> Held-out-task representation transfer with few-shot linear probing.

It is not literal zero-label classification.

## Model selection

Probe regularization is selected on development data. Test data is used only for final evaluation.

## Cross precision

A probe fitted on the encoder's matched training-input representation is applied unchanged to the opposite-precision test representation. No target-precision refitting is included in the primary cross-precision result.
