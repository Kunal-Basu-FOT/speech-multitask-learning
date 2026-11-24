# EXP 6.1 common model package

## Encoder

The shared encoder is a compact SpeechBrain ECAPA-TDNN receiving complete
IndicWav2Vec final-layer sequences of shape `[batch, time, 768]`.

Frozen architecture:

- channels: `[480, 480, 480, 480, 1440]`
- kernels: `[5, 3, 3, 3, 1]`
- dilations: `[1, 2, 3, 4, 1]`
- Res2Net scale: 8
- squeeze-excitation channels: 128
- attention channels: 128
- embedding dimension: 192
- encoder dropout: 0.1
- ECAPA parameters: 7,212,660

A learnable input LayerNorm contributes 1,536 parameters. Padded frames are
masked back to zero after normalization.

## Classifiers

All heads use:

```text
LayerNorm(192) → Linear(192, 256) → ReLU → Dropout(0.1) → Linear
```

Classifier modules:

- one 8-class language head
- one 2-class gender head
- eight language-specific dialect heads
- 33 dialect classes across the eight local label spaces

Dialect routing uses the ground-truth language label. Predicted-language
routing is not used as the primary EXP 6.1 result.

## Task weighting

Three learned log variances implement homoscedastic uncertainty weighting
for language, dialect, and gender losses. Initial values are zero and the
effective values are clamped to `[-6, 6]` during loss calculation.

## Parameter contract

- SpeechBrain ECAPA: 7,212,660
- input normalization: 1,536
- all classifier heads: 508,971
- uncertainty parameters: 3
- total model: 7,723,170
