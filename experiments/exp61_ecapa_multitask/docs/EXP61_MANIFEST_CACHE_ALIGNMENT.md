# EXP 6.1 manifest and cache alignment

The active manifests were regenerated directly from the existing
IndicWav2Vec final-layer cache indexes.

## Active counts

- Train: 146,359
- Development: 11,507
- Test: 17,550

## Alignment guarantee

For each split:

- manifest audio paths exactly equal cache index audio paths;
- manifest row order exactly equals cache index row order;
- 1-bit and 16-bit indexes are identical;
- no features were regenerated;
- no cache files were modified;
- full utterances remain intact.

## Previous manifest backup

`/data/Kunal/manifests/archive_before_cache_alignment_20260805T170525+0530`

## Active manifest files

- `/data/Kunal/manifests/respin_unique_sentence_train.csv`
- `/data/Kunal/manifests/respin_unique_sentence_dev.csv`
- `/data/Kunal/manifests/respin_unique_sentence_test.csv`

The filenames retain their previous names for compatibility, but they
now represent the cache-native EXP 6.1 population.
