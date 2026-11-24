# Source map and canonical-copy decisions

All reads came from `/data/Kunal`; all originals remain untouched. Material was copied into this clean project.

| Clean directory | Canonical source | Selection evidence |
|---|---|---|
| `experiments/exp61_ecapa_multitask/` | `/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK/` | Complete code/config/test tree plus experiment, training, evaluation, and report markers |
| `experiments/exp62_task_grouping/` | `/data/Kunal/FORMAL_EXP6_2_TaskGrouping_IndicWav2Vec_FinalLayer/` | Completed reduced six-condition experiment and report |
| `experiments/exp65_exp68_embedding_analysis/` | `/data/Kunal/FORMAL_EXP6_5_EXP6_8_Combined_Embedding_Analysis_IndicWav2Vec_FinalLayer/` | Combined EXP 6.5/6.8 completion marker and complete reports |
| `experiments/exp66_gradient_interference/` | `/data/Kunal/Notebooks/FORMAL_EXP6_6_GradientInterference_IndicWav2Vec_FinalLayer/` | Only formal EXP 6.6 source found; final report and run completion markers present |
| `experiments/exp67_hierarchical_lid_dialect/` | `/data/Kunal/Notebooks/FORMAL_EXP6_7_Hierarchical_LID_Dialect_IndicWav2Vec_FinalLayer/` | Only formal EXP 6.7 source found; report and run completion markers present |
| `experiments/exp69_waveform_duration_ablation/` | `/data/Kunal/FORMAL_EXP6_9_Waveform_Duration_Ablation_ECAPA/` | Completed top-level copy selected; recursive comparison showed no differences from the duplicate under `Notebooks/` |
| `project_tools/manifest_builders/` | `/data/Kunal/scripts/` | Lightweight manifest-generation source code |
| `environment.yml` | `/data/Kunal/speech_mtl_environment.yml` | Captured environment specification |

## Duplicate source preserved by provenance

EXP 6.9 also exists at `/data/Kunal/Notebooks/FORMAL_EXP6_9_Waveform_Duration_Ablation_ECAPA/`. A recursive `diff -qr` produced no differences at cleanup time. To avoid duplicating approximately 2 GB of identical source evidence, the clean project contains one canonical lightweight copy and records both paths here. Both originals remain in place.

## Shared dependencies

`shared_src`, `shared_pretrained`, and `shared_cache` were treated as external server dependencies and were not copied. The shared source tree includes a Fairseq checkout identified by its source directory as `fairseq_5d8d0674`; experiment requirements and imports remain the authoritative dependency instructions.
