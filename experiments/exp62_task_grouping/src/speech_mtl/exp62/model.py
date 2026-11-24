"""Compact ECAPA encoder with condition-specific LID and dialect heads."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

from .conditions import Condition
from .config import ModelConfig
from .labels import LabelSpace


class ClassifierHead(nn.Sequential):
    def __init__(self, embedding_dim: int, hidden: int, dropout: float, classes: int):
        super().__init__(
            OrderedDict(
                [
                    ("norm", nn.LayerNorm(embedding_dim)),
                    ("linear1", nn.Linear(embedding_dim, hidden)),
                    ("relu", nn.ReLU()),
                    ("dropout", nn.Dropout(dropout)),
                    ("linear2", nn.Linear(hidden, classes)),
                ]
            )
        )


class ReducedMTLModel(nn.Module):
    def __init__(self, cfg: ModelConfig, labels: LabelSpace, condition: Condition):
        super().__init__()
        self.cfg = cfg
        self.labels = labels
        self.condition = condition
        self.input_norm = nn.LayerNorm(cfg.input_dim)
        self.ecapa = ECAPA_TDNN(
            input_size=cfg.input_dim,
            lin_neurons=cfg.embedding_dim,
            channels=list(cfg.channels),
            kernel_sizes=list(cfg.kernels),
            dilations=list(cfg.dilations),
            attention_channels=cfg.attention_channels,
            res2net_scale=cfg.res2net_scale,
            se_channels=cfg.se_channels,
            global_context=cfg.global_context,
            groups=list(cfg.groups),
            dropout=cfg.dropout,
        )
        if "lid" in condition.active_tasks:
            self.lid_head = ClassifierHead(
                cfg.embedding_dim, cfg.head_hidden, cfg.head_dropout, labels.num_lid_classes
            )
        else:
            self.lid_head = None
        if "dialect" in condition.active_tasks:
            self.dialect_heads = nn.ModuleDict(
                {
                    lang: ClassifierHead(
                        cfg.embedding_dim,
                        cfg.head_hidden,
                        cfg.head_dropout,
                        labels.num_dialect_classes(lang),
                    )
                    for lang in labels.languages
                }
            )
        else:
            self.dialect_heads = nn.ModuleDict()
        if len(condition.active_tasks) > 1:
            self.log_vars = nn.ParameterDict(
                {task: nn.Parameter(torch.zeros(())) for task in condition.active_tasks}
            )
        else:
            self.log_vars = nn.ParameterDict()

    def encode(
        self,
        features: torch.Tensor,
        relative_lengths: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.input_norm(features)
        normalized = normalized.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        embedding = self.ecapa(normalized, lengths=relative_lengths)
        if embedding.ndim == 3 and embedding.shape[1] == 1:
            embedding = embedding[:, 0, :]
        if embedding.ndim != 2:
            raise RuntimeError(f"Unexpected ECAPA output shape: {tuple(embedding.shape)}")
        return embedding

    def forward(self, batch: dict) -> dict:
        embedding = self.encode(
            batch["features"], batch["relative_lengths"], batch["valid_mask"]
        )
        output: dict = {"embedding": embedding}
        if self.lid_head is not None:
            output["lid_logits"] = self.lid_head(embedding)
        if self.dialect_heads:
            routed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for lang in self.labels.languages:
                indices = [i for i, item in enumerate(batch["language"]) if item == lang]
                if not indices:
                    continue
                index_tensor = torch.tensor(indices, dtype=torch.long, device=embedding.device)
                routed[lang] = (index_tensor, self.dialect_heads[lang](embedding[index_tensor]))
            output["dialect_logits"] = routed
        return output

    def encoder_parameters(self):
        yield from self.input_norm.parameters()
        yield from self.ecapa.parameters()

    def task_parameters(self):
        if self.lid_head is not None:
            yield from self.lid_head.parameters()
        yield from self.dialect_heads.parameters()
        yield from self.log_vars.parameters()

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder_parameters():
            parameter.requires_grad_(trainable)
        if trainable:
            self.input_norm.train()
            self.ecapa.train()
        else:
            self.input_norm.eval()
            self.ecapa.eval()

    @property
    def parameter_counts(self) -> dict[str, int]:
        encoder = sum(p.numel() for p in self.encoder_parameters())
        tasks = sum(p.numel() for p in self.task_parameters())
        return {
            "encoder": encoder,
            "tasks": tasks,
            "total": encoder + tasks,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
