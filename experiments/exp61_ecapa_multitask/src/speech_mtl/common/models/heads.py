from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


DEFAULT_DIALECT_CLASS_COUNTS = {
    "bh": 3,
    "bn": 5,
    "ch": 4,
    "kn": 5,
    "mg": 4,
    "mr": 4,
    "mt": 4,
    "te": 4,
}


class ClassificationHead(nn.Module):
    """Layer-normalized two-layer MLP classifier."""

    def __init__(
        self,
        *,
        embedding_dimension: int,
        hidden_dimension: int,
        number_of_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if hidden_dimension <= 0:
            raise ValueError("hidden_dimension must be positive")
        if number_of_classes <= 1:
            raise ValueError("number_of_classes must be at least two")

        self.embedding_dimension = int(embedding_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.number_of_classes = int(number_of_classes)

        self.network = nn.Sequential(
            nn.LayerNorm(self.embedding_dimension),
            nn.Linear(
                self.embedding_dimension,
                self.hidden_dimension,
            ),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(
                self.hidden_dimension,
                self.number_of_classes,
            ),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings must have shape [batch, embedding], "
                f"received {tuple(embeddings.shape)}"
            )

        return self.network(embeddings)


class OracleDialectHeads(nn.Module):
    """Eight language-specific dialect heads with oracle language routing."""

    def __init__(
        self,
        *,
        embedding_dimension: int = 192,
        hidden_dimension: int = 256,
        dialect_class_counts: dict[str, int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        counts = dict(
            dialect_class_counts or DEFAULT_DIALECT_CLASS_COUNTS
        )

        if set(counts) != set(DEFAULT_DIALECT_CLASS_COUNTS):
            raise ValueError(
                "Dialect heads must be defined exactly for: "
                f"{sorted(DEFAULT_DIALECT_CLASS_COUNTS)}"
            )

        if sum(counts.values()) != 33:
            raise ValueError(
                f"Expected 33 dialect classes across heads, "
                f"received {sum(counts.values())}"
            )

        self.class_counts = counts
        self.heads = nn.ModuleDict(
            {
                language: ClassificationHead(
                    embedding_dimension=embedding_dimension,
                    hidden_dimension=hidden_dimension,
                    number_of_classes=counts[language],
                    dropout=dropout,
                )
                for language in sorted(counts)
            }
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        oracle_languages: Sequence[str],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch_size = int(embeddings.shape[0])

        if len(oracle_languages) != batch_size:
            raise ValueError(
                f"Expected {batch_size} oracle language labels, "
                f"received {len(oracle_languages)}"
            )

        unknown = sorted(
            {
                str(language)
                for language in oracle_languages
                if str(language) not in self.heads
            }
        )

        if unknown:
            raise ValueError(
                f"Unknown oracle language labels: {unknown}"
            )

        logits_by_language: dict[str, torch.Tensor] = {}
        indices_by_language: dict[str, torch.Tensor] = {}

        for language, head in self.heads.items():
            selected = [
                index
                for index, value in enumerate(oracle_languages)
                if str(value) == language
            ]

            if not selected:
                continue

            indices = torch.tensor(
                selected,
                dtype=torch.long,
                device=embeddings.device,
            )

            indices_by_language[language] = indices
            logits_by_language[language] = head(
                embeddings.index_select(0, indices)
            )

        selected_total = sum(
            int(indices.numel())
            for indices in indices_by_language.values()
        )

        if selected_total != batch_size:
            raise RuntimeError(
                "Oracle dialect routing did not cover the entire batch"
            )

        return logits_by_language, indices_by_language
