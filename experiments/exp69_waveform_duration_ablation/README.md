# EXP 6.9 — Waveform/Log-Mel Duration Ablation of EXP 6.1

This package trains eight conventional waveform-derived baselines on WTC9 physical GPU 1 and compares their full-duration performance with the completed EXP 6.1 IndicWav2Vec2 layer-11 models.

## Repository path

```text
/data/Kunal/FORMAL_EXP6_9_Waveform_Duration_Ablation_ECAPA
```

## Conditions

```text
one_bit:     3s, 7s, 9s, full
sixteen_bit: 3s, 7s, 9s, full
```

Each model uses:

```text
raw quantized waveform
→ 80-bin log-Mel
→ Linear(80,768)
→ EXP 6.1 compact ECAPA-TDNN
→ shared 192-D embedding
→ LID + oracle-routed dialect + gender
```

The complete scientific protocol is in `docs/SCIENTIFIC_PROTOCOL.md`.

## Safety properties

- The shared RESPIN audio tree is read-only.
- The completed EXP 6.1 repository is read-only.
- The package creates a new log-Mel cache under `/data/Kunal/shared_cache/exp69_waveform_logmel80`.
- All checkpoints are loaded on CPU first.
- RNG tensors are forced to CPU `torch.uint8` before restoration.
- Evaluation never restores RNG state.
- The GPU preflight executes a real RNG checkpoint roundtrip on physical GPU 1.
- Completed conditions are skipped; interrupted training resumes at completed epoch boundaries.
- Isolated AMP overflows are skipped and counted instead of terminating the queue.

## Install

```bash
cd /data/Kunal/FORMAL_EXP6_9_Waveform_Duration_Ablation_ECAPA
bash scripts/00_install_validate.sh
```

## Launch on WTC9 physical GPU 1

Wait until GPU 1 is free, then run:

```bash
cd /data/Kunal/FORMAL_EXP6_9_Waveform_Duration_Ablation_ECAPA
bash RUN_ALL_GPU1.sh
```

Screen and log:

```text
screen: exp69_gpu1
log: runtime/logs/exp69_gpu1.screen.log
```

Monitor:

```bash
bash scripts/11_status.sh
```

Attach:

```bash
bash scripts/12_attach.sh
```

Detach with `Ctrl+A`, then `D`.

## Output layout

```text
runtime/runs/exp69/<domain>/<duration>/
results/exp69/<train-domain>/<train-duration>/input_<domain>/<split>_<duration>/
reports/exp69/tables/
reports/exp69/figures/
reports/exp69/RESULTS_INTERPRETATION.md
```

Final markers:

```text
runtime/EXPERIMENT_COMPLETE.json
runtime/runs/exp69/EXPERIMENT_COMPLETE.json
reports/exp69/REPORT_COMPLETE.json
```

## Main report outputs

Eight CSV tables include experimental configuration, matched duration results, full front-end comparison, IndicWav2Vec2 gains, cross-precision results, duration-transfer results, runtime, and paired speaker-bootstrap intervals.

Eight 300-DPI figures include the four-panel duration/front-end graph, two duration-transfer heatmaps, full-front-end gain, cross-precision robustness, runtime, and full-condition confusion composites.
