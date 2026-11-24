# EXP 6.1 common training package

## Frozen schedule

The training schedule contains 40 global epochs:

| Phase | Epochs | Encoder | Task heads | Learning-rate policy |
|---|---:|---|---|---|
| Phase 1 | 1–5 | trainable | trainable | linear warmup from 1e-5 to 1e-4 |
| Phase 2 | 6–35 | trainable | trainable | cosine decay from 1e-4 to 5e-6 |
| Phase 3 | 36–40 | frozen | trainable | task LR 2e-5; encoder LR 0 |

The three learned uncertainty parameters remain trainable throughout.

## Optimizer

- AdamW
- peak learning rate: 1e-4
- weight decay: 1e-4
- betas: 0.9, 0.999
- epsilon: 1e-8
- separate encoder and task parameter groups

## Loss

Three raw cross-entropy losses are calculated:

- language identification
- oracle-routed language-specific dialect identification
- gender classification

Dialect cross-entropy is sample-weighted across the eight local heads.
The three task losses are combined by learned homoscedastic uncertainty
weighting.

## Runtime features

- CUDA AMP with float16 autocast and GradScaler
- gradient accumulation
- global gradient clipping at 1.0
- non-finite loss and gradient rejection
- visible tqdm progress
- atomic progress JSON
- append-only epoch history CSV
- atomic `last.pt` and `best.pt`
- model, optimizer, scheduler, AMP, RNG, and progress state resumption
- explicit `weights_only=False` checkpoint loading for PyTorch 2.6

This package supplies reusable training mechanics. The experiment-specific
runner that joins training, development evaluation, checkpoint selection,
and the sequential queue is populated later under `speech_mtl/exp61/`.
