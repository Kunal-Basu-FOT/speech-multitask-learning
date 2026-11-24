from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .ecapa_encoder import CompactECAPAEncoder
from .heads import (
    DEFAULT_DIALECT_CLASS_COUNTS,
    ClassificationHead,
    OracleDialectHeads,
)
from .uncertainty import HomoscedasticTaskWeighting


@dataclass
class MultitaskOutput:
    embeddings: torch.Tensor
    language_logits: torch.Tensor
    gender_logits: torch.Tensor
    dialect_logits: dict[str, torch.Tensor]
    dialect_batch_indices: dict[str, torch.Tensor]


class MultitaskECAPA(nn.Module):
    """Three-task ECAPA-TDNN model with oracle dialect routing."""

    def __init__(
        self,
        *,
        feature_dimension: int = 768,
        embedding_dimension: int = 192,
        head_hidden_dimension: int = 256,
        head_dropout: float = 0.1,
        encoder_dropout: float = 0.1,
        dialect_class_counts: dict[str, int] | None = None,
    ) -> None:
        super().__init__()

        self.feature_dimension = int(feature_dimension)
        self.embedding_dimension = int(embedding_dimension)

        self.encoder = CompactECAPAEncoder(
            input_size=self.feature_dimension,
            embedding_dimension=self.embedding_dimension,
            dropout=encoder_dropout,
        )

        self.language_head = ClassificationHead(
            embedding_dimension=self.embedding_dimension,
            hidden_dimension=head_hidden_dimension,
            number_of_classes=8,
            dropout=head_dropout,
        )

        self.gender_head = ClassificationHead(
            embedding_dimension=self.embedding_dimension,
            hidden_dimension=head_hidden_dimension,
            number_of_classes=2,
            dropout=head_dropout,
        )

        self.dialect_heads = OracleDialectHeads(
            embedding_dimension=self.embedding_dimension,
            hidden_dimension=head_hidden_dimension,
            dialect_class_counts=(
                dialect_class_counts
                or DEFAULT_DIALECT_CLASS_COUNTS
            ),
            dropout=head_dropout,
        )

        self.task_weighting = HomoscedasticTaskWeighting(
            initial_log_variance=0.0,
            minimum_log_variance=-6.0,
            maximum_log_variance=6.0,
        )

    def forward(
        self,
        features: torch.Tensor,
        relative_lengths: torch.Tensor,
        oracle_languages: Sequence[str],
        valid_mask: torch.Tensor | None = None,
    ) -> MultitaskOutput:
        embeddings = self.encoder(
            features=features,
            relative_lengths=relative_lengths,
            valid_mask=valid_mask,
        )

        language_logits = self.language_head(embeddings)
        gender_logits = self.gender_head(embeddings)

        dialect_logits, dialect_batch_indices = self.dialect_heads(
            embeddings,
            oracle_languages,
        )

        return MultitaskOutput(
            embeddings=embeddings,
            language_logits=language_logits,
            gender_logits=gender_logits,
            dialect_logits=dialect_logits,
            dialect_batch_indices=dialect_batch_indices,
        )

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze input normalization and ECAPA together."""

        self.encoder.set_trainable(trainable)

    def encoder_parameters(self):
        yield from self.encoder.parameters()

    def task_parameters(self):
        yield from self.language_head.parameters()
        yield from self.gender_head.parameters()
        yield from self.dialect_heads.parameters()
        yield from self.task_weighting.parameters()
