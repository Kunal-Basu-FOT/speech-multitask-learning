from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMBINED_ROOT = Path(
    "/data/Kunal/FORMAL_EXP6_5_EXP6_8_Combined_Embedding_Analysis_IndicWav2Vec_FinalLayer"
)
EXP61_ROOT = Path("/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK")
EXP62_ROOT = Path(
    "/data/Kunal/FORMAL_EXP6_2_TaskGrouping_IndicWav2Vec_FinalLayer"
)
CACHE_ROOT = Path(
    "/data/Kunal/shared_cache/exp61_indicwav2vec_final_layer_fp16"
)
PYTHON = Path("/data/Kunal/env_speech_mtl/bin/python")


@dataclass(frozen=True)
class AnalysisConfig:
    seed: int
    feature_dimension: int
    embedding_dimension: int
    domains: tuple[str, ...]
    splits: tuple[str, ...]
    conditions: tuple[str, ...]
    expected_rows: dict[str, int]
    frame_budget: int
    maximum_batch_size: int
    workers: int
    fewshot_fractions: tuple[float, ...]
    ridge_lambdas: tuple[float, ...]
    umap_samples: int
    tsne_samples: int
    silhouette_samples: int
    output_dpi: int

    @classmethod
    def load(cls, path: Path | None = None) -> "AnalysisConfig":
        path = path or COMBINED_ROOT / "configs/exp65_68.json"
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            seed=int(raw["seed"]),
            feature_dimension=int(raw["feature_dimension"]),
            embedding_dimension=int(raw["embedding_dimension"]),
            domains=tuple(raw["domains"]),
            splits=tuple(raw["splits"]),
            conditions=tuple(raw["conditions"]),
            expected_rows={str(k): int(v) for k, v in raw["expected_rows"].items()},
            frame_budget=int(raw["frame_budget"]),
            maximum_batch_size=int(raw["maximum_batch_size"]),
            workers=int(raw["workers"]),
            fewshot_fractions=tuple(float(x) for x in raw["fewshot_fractions"]),
            ridge_lambdas=tuple(float(x) for x in raw["ridge_lambdas"]),
            umap_samples=int(raw["umap_samples"]),
            tsne_samples=int(raw["tsne_samples"]),
            silhouette_samples=int(raw["silhouette_samples"]),
            output_dpi=int(raw["output_dpi"]),
        )
