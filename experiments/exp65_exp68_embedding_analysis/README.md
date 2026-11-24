# EXP 6.5 + EXP 6.8 Combined Analysis Package

This is a code-only, upload-ready package for the cache-compatible three-task SPEECH_MTL project.
It performs **no new encoder training**.

## Scientific scope

- **EXP 6.5:** frozen representation probing and geometry for the six reduced EXP 6.2 encoders:
  `lid_only`, `dialect_only`, and `lid_dialect`, each at one-bit and sixteen-bit.
- **EXP 6.8:** held-out gender transfer with nested 10%, 25%, 50%, and 100% linear probes.
- **References:** completed EXP 6.1 gender-supervised three-task encoders.
- **Control:** one deterministic random ECAPA encoder.
- **Precision protocol:** matched and direct cross-precision evaluation.

The implemented project scope is LID, language-local dialect, and gender. It does not test domain or accent.

## Expected machine paths

```text
Combined root:
/data/Kunal/FORMAL_EXP6_5_EXP6_8_Combined_Embedding_Analysis_IndicWav2Vec_FinalLayer

EXP 6.1:
/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK

EXP 6.2:
/data/Kunal/FORMAL_EXP6_2_TaskGrouping_IndicWav2Vec_FinalLayer

Shared cache:
/data/Kunal/shared_cache/exp61_indicwav2vec_final_layer_fp16

Python:
/data/Kunal/env_speech_mtl/bin/python
```

## Existing EXP 6.1 reference extraction

This package preserves and consumes:

```text
runtime/reference_embeddings/exp61/
```

The ZIP intentionally contains no `runtime`, `results`, `reports`, checkpoints, caches, or `.npy` files.
Unzipping it over the existing combined repository therefore does not overwrite the GPU-1 extraction.

## Install

From `/data/Kunal`:

```bash
unzip -o EXP65_EXP68_COMBINED_COMPLETE_UPLOAD_READY_2026-08-06.zip
cd /data/Kunal/FORMAL_EXP6_5_EXP6_8_Combined_Embedding_Analysis_IndicWav2Vec_FinalLayer
bash scripts/00_install_validate.sh
```

## Run after EXP 6.2 completes

```bash
cd /data/Kunal/FORMAL_EXP6_5_EXP6_8_Combined_Embedding_Analysis_IndicWav2Vec_FinalLayer
bash RUN_ALL_AFTER_EXP62.sh
```

This launches a detached screen named:

```text
exp65_exp68_gpu0
```

Attach with:

```bash
screen -r exp65_exp68_gpu0
```

Detach with `Ctrl+A`, then `D`.

## Final outputs

```text
runtime/embeddings/exp62/          frozen EXP 6.2 embeddings
runtime/embeddings/random/         random-encoder control embeddings
results/exp65/                     machine-readable EXP 6.5 results
results/exp68/                     machine-readable EXP 6.8 results
reports/exp65/figures/             EXP 6.5 figures
reports/exp65/tables/              EXP 6.5 tables
reports/exp68/figures/             EXP 6.8 figures
reports/exp68/tables/              EXP 6.8 tables
reports/combined/README_RESULTS.md  generated interpretation
reports/combined/REPORT_COMPLETE.json
runtime/EXPERIMENT_COMPLETE.json
```

## Main analyses

### EXP 6.5

- Frozen ridge-linear probes for LID, gender, and eight language-local dialect problems.
- Native-head versus frozen-probe comparison where EXP 6.2 metrics are available.
- Joint-versus-single-task probe deltas.
- One-bit versus sixteen-bit representation deltas.
- PCA, UMAP, t-SNE, silhouette, centroid separation, and within/between scatter.
- Dialect geometry is computed within language and macro-averaged; it is not treated as one naive global label.

### EXP 6.8

- Gender probes with nested, language×gender-stratified fractions: 10%, 25%, 50%, 100%.
- LID-only, dialect-only, and LID+dialect frozen sources.
- Random frozen encoder control.
- EXP 6.1 gender-supervised three-task reference.
- Matched and direct cross-precision evaluation without target-precision refitting.
- Transfer gain over random and fraction of supervised performance recovered.

## Safety behavior

The launcher fails closed when:

- any EXP 6.2 condition is incomplete;
- the EXP 6.1 12-cell reference marker is missing;
- physical GPU 0 is occupied;
- cache indexes differ between precisions;
- a checkpoint cannot be mapped exactly onto the frozen ECAPA encoder;
- row counts, utterance IDs, embedding dimensions, or finite-value checks fail.

Completed embedding cells are immutable and skipped on rerun. Incomplete cell directories are not silently overwritten.
