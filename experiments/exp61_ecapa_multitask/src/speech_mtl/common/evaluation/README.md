# EXP 6.1 common evaluation package

## Evaluation population

The package evaluates every utterance supplied by a dataloader. For the
cache-aligned EXP 6.1 manifests this means:

- development: 11,507 utterances
- test: 17,550 utterances

The same evaluator is used for matched-precision and cross-precision runs.

## Metrics

Language:

- accuracy
- macro precision
- macro recall
- macro F1
- per-class metrics and support

Gender:

- accuracy
- macro precision
- macro recall
- macro F1
- per-class metrics and support

Dialect:

- metrics for each of the eight language-specific heads
- sample-weighted accuracy across all dialect examples
- language-macro accuracy
- language-macro F1

The development checkpoint-selection metric is:

```text
mean(
    language macro F1,
    dialect language-macro F1,
    gender macro F1
)
```

## Oracle dialect routing

Dialect predictions are obtained from the head selected by each example's
ground-truth language. Predicted-language routing is not used for the primary
EXP 6.1 result.

## Confusion matrices

Raw count matrices and row-normalized percentage matrices are produced for:

- one 8-class language matrix
- one 2-class gender matrix
- eight language-specific dialect matrices

Rows are true labels and columns are predicted labels.

## Prediction records

The optional streaming CSV stores one row per evaluated utterance with true
and predicted numeric labels for all three tasks. This supports later error
analysis without repeating model inference.
