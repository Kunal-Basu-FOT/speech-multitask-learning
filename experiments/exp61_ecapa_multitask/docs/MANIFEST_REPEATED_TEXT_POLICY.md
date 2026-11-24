# Manifest repeated-text policy

## Observed state

The EXP 6.1 training manifest contains:

- 146,666 rows
- 146,666 distinct utterance IDs
- 6 repeated `(language, text)` groups
- 12 rows participating in those groups

The repeated rows are not identical recordings. They have distinct
utterance IDs and distinct speakers.

Two groups contain the same prompt under different dialect labels:

- Kannada: `kn_D1` and `kn_D3`
- Maithili: `mt_D2` and `mt_D4`

Other groups contain distinct speakers, and in several cases different
gender labels, reading the same prompt.

## Formal policy

- `utt_id` is the primary uniqueness key.
- Duplicate `utt_id` values are fatal.
- Repeated transcript strings with distinct utterance IDs are retained.
- No source manifest row is deleted or rewritten.
- The six repeated-text groups are preserved in the validation bundle.
- Speaker-disjoint train, development, and test splits remain mandatory.

## Reporting wording

The corpus uses a reduced unique-sentence-oriented manifest. Six
within-language transcript strings occur twice across distinct speakers
or dialect realizations. These 12 valid utterances were retained.
