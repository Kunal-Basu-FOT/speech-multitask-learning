# EXP 6.9 Scientific Protocol

## Title

**Utterance-Duration and Front-End Ablation of EXP 6.1**

## Question

This experiment asks two linked questions:

1. How much waveform duration is required by a conventional waveform-derived ECAPA baseline?
2. At full duration, how much does the completed IndicWav2Vec2 layer-11 representation improve the same three-task problem?

## New conditions

Eight new models are trained:

- one-bit log-Mel: 3 s, 7 s, 9 s, full;
- sixteen-bit log-Mel: 3 s, 7 s, 9 s, full.

The completed EXP 6.1 one-bit and sixteen-bit full-utterance models are reused as frozen reference results. No IndicWav2Vec2 model is retrained.

## Front end

The conventional baseline is:

```text
quantized waveform
→ 80-bin log-Mel filterbanks
→ Linear(80, 768)
→ the same compact ECAPA-TDNN used by EXP 6.1
→ 192-dimensional shared embedding
→ LID, oracle-routed dialect and gender heads
```

Feature extraction is fixed to:

- 16 kHz audio;
- 25 ms window (`win_length=400`);
- 10 ms shift (`hop_length=160`);
- 512-point FFT;
- 80 Mel bins;
- HTK Mel scale;
- no DCT;
- `center=False`;
- natural logarithm after a `1e-10` floor.

The one-bit waveform is exactly `+1` when `x > 0`, otherwise `-1`. Zero therefore maps to `-1`, matching Scheme 1 in the shared framework.

## Duration policy

The full waveform is decoded once and converted into a full log-Mel sequence. Duration caps are applied to that sequence with exact 25 ms / 10 ms geometry:

- 3 s → 298 frames;
- 7 s → 698 frames;
- 9 s → 898 frames;
- full → no cap.

Long training utterances use a deterministic epoch-dependent contiguous crop. Development and test use a fixed centre crop. Short utterances remain complete; they are neither discarded nor repeated.

## Data population

The source metadata is the active, cache-aligned EXP 6.1 population:

- train: 146,359;
- dev: 11,507;
- test: 17,550.

The one-bit and sixteen-bit source indexes must be byte-identical. No new split is constructed.

## Training controls

The eight models retain the EXP 6.1 controls:

- 40 epochs;
- epochs 1–5: joint warm-up, LR `1e-5 → 1e-4`;
- epochs 6–35: joint cosine decay to `5e-6`;
- epochs 36–40: encoder frozen, task LR `2e-5`;
- AdamW, weight decay `1e-4`;
- AMP with GradScaler;
- recoverable isolated AMP overflow skips;
- global gradient clipping at 1.0;
- homoscedastic task weighting;
- seed 1337;
- best model selected by the mean of language macro-F1, dialect language-macro F1 and gender macro-F1.

## Evaluation

Every condition produces:

- matched development evaluation;
- matched test evaluation;
- same-precision test evaluation at all four duration caps;
- direct cross-precision test evaluation at the training duration;
- prediction CSVs for every test cell;
- language, gender and eight local dialect confusion matrices.

## Main comparison

The principal front-end claim uses only the full-duration cells:

```text
full log-Mel ECAPA
versus
full IndicWav2Vec2 layer-11 ECAPA from completed EXP 6.1
```

The package calculates paired speaker-bootstrap confidence intervals for the difference. This comparison measures the combined benefit of pretrained/contextual IndicWav2Vec2 representations relative to the specified log-Mel baseline. It does not isolate the effects of pretraining, contextualisation and architecture individually.

## RNG safety

All training checkpoints are loaded with `map_location="cpu"`. CPU and CUDA RNG states are converted explicitly to contiguous CPU `torch.uint8` tensors before restoration. Evaluation loads model weights only and never restores RNG state. The GPU preflight performs a real checkpoint save/load/RNG roundtrip before production work begins.
