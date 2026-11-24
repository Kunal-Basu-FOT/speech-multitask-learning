# EXP 6.1 integration layer

## Sequential queue

The production queue uses WTC9 physical GPU 1 only:

```text
preflight
→ train 1-bit model
→ matched 1-bit dev/test evaluation
→ 1-bit model on 16-bit test
→ train 16-bit model
→ matched 16-bit dev/test evaluation
→ 16-bit model on 1-bit test
→ figures and tables
```

All CUDA work sees only process-local `cuda:0` through:

```text
CUDA_VISIBLE_DEVICES=1
```

CPU fallback is rejected.

## Training

- complete final-layer feature sequences, no cropping
- stable frame-budget batches
- 40 epochs with 5 + 30 + 5 phases
- AMP FP16
- AdamW
- homoscedastic three-task loss
- visible tqdm
- one development evaluation per epoch
- best checkpoint selected by three-task mean macro-F1
- epoch-boundary `last.pt` resume

An interruption inside an epoch restarts that epoch from its beginning. Completed
epochs, optimizer state, scheduler state, AMP state, model state, and RNG state
are preserved.

## Outputs

Training:

```text
runtime/runs/ecapa_7p2m/exp61/{one_bit,sixteen_bit}/
```

Evaluations:

```text
results/ecapa_7p2m/exp61/evaluations/
```

Report artifacts:

```text
reports/ecapa_7p2m/exp61/
```
