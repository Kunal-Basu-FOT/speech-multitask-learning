# Exclusions

The archive intentionally omits large or machine-specific artifacts while retaining their small completion/metric metadata where useful.

| Excluded class | Examples/reason |
|---|---|
| Model checkpoints | `*.pt`, `*.pth`, `*.ckpt`, `checkpoints/`; typically about 87-94 MB per file |
| Runtime logs | `runtime/logs/`, `*.log`, screen logs; some source logs exceed 400 MB |
| Embedding arrays and caches | `runtime/embeddings/`, `runtime/reference_embeddings/`, `*.npy`, cache directories; source embedding tree is multi-gigabyte |
| Raw gradient samples | `gradient_samples.csv`; large per-step samples, with aggregate gradient reports retained |
| Per-example predictions | `predictions.csv`; aggregate `metrics.json`, evaluation markers, and confusion matrices retained |
| Python/Jupyter caches | `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ipynb_checkpoints/` |
| Environment/runtime installations | `/data/Kunal/env_speech_mtl/`, `/data/Kunal/tools/` |
| Pretrained/shared source trees | `/data/Kunal/shared_pretrained/`, `/data/Kunal/shared_src/`; external dependencies, not project-authored experiment material |
| Shared computed cache | `/data/Kunal/shared_cache/`; server-only derived data |
| Dataset manifests and audio | `/data/Kunal/manifests/*.csv` and all raw audio; manifests are path-bearing dataset inventory (about 68 MB total), while their builder scripts are retained |
| Root/Jupyter operational logs | `/data/Kunal/jupyter_wtc9.log` and other operational logs |

The source archive preserved the original experiment files. The publication
copy adds reader-facing documentation and an archive checker. EXP 6.4 is
documented as abandoned rather than reconstructed. EXP 6.3 remains a
placeholder and is not counted as a completed result.
