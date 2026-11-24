# Cache-native data package

This package reads the existing EXP 6.1 IndicWav2Vec final-layer cache:

```text
/data/Kunal/shared_cache/exp61_indicwav2vec_final_layer_fp16/
├── one_bit/
│   ├── train/
│   ├── dev/
│   └── test/
└── sixteen_bit/
    ├── train/
    ├── dev/
    └── test/
```

Each split must contain:

```text
index.csv
features.f16
metadata.json
COMPLETE.json
```

`features.f16` is treated as a contiguous row-major float16 matrix with
shape `[total_frames, 768]`. The loader uses `start_frame` and `num_frames`
from `index.csv` and never crops or truncates an utterance.

The historical `features_path` inside metadata JSON is deliberately ignored.
The binary beside the split index is always used.

Expected cache-aligned split sizes:

- train: 146,359
- dev: 11,507
- test: 17,550
