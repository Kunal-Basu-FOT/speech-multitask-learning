"""Hard-sharing ECAPA model used by all EXP 6.6 optimization conditions."""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

from .config import ModelConfig
from .labels import LabelSpace

TASKS = ("lid", "dialect", "gender")
CONDITIONS = ("standard_logged", "pcgrad", "gradnorm")


class ClassifierHead(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, output_dim: int) -> None:
        super().__init__(
            OrderedDict(
                [
                    ("norm", nn.LayerNorm(input_dim)),
                    ("linear1", nn.Linear(input_dim, hidden_dim)),
                    ("relu", nn.ReLU()),
                    ("dropout", nn.Dropout(dropout)),
                    ("linear2", nn.Linear(hidden_dim, output_dim)),
                ]
            )
        )


class HardSharingMTLModel(nn.Module):
    def __init__(self, cfg: ModelConfig, labels: LabelSpace) -> None:
        super().__init__()
        self.cfg = cfg
        self.labels = labels
        self.input_norm = nn.LayerNorm(cfg.input_dim)
        self.encoder = ECAPA_TDNN(
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
        self.lid_head = ClassifierHead(
            cfg.embedding_dim, cfg.head_hidden, cfg.head_dropout, labels.num_lid_classes
        )
        self.gender_head = ClassifierHead(
            cfg.embedding_dim, cfg.head_hidden, cfg.head_dropout, labels.num_gender_classes
        )
        self.dialect_heads = nn.ModuleDict(
            {
                language: ClassifierHead(
                    cfg.embedding_dim,
                    cfg.head_hidden,
                    cfg.head_dropout,
                    labels.num_dialect_classes(language),
                )
                for language in labels.languages
            }
        )
        self.log_vars = nn.ParameterDict(
            {task: nn.Parameter(torch.zeros(())) for task in TASKS}
        )

    def forward(self, batch: dict) -> dict:
        x = self.input_norm(batch["features"])
        x = x.masked_fill(~batch["valid_mask"].unsqueeze(-1), 0.0)
        embedding = self.encoder(x, lengths=batch["relative_lengths"])
        if embedding.ndim == 3:
            embedding = embedding[:, 0, :]
        lid_logits = self.lid_head(embedding)
        routed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for language in self.labels.languages:
            indices = [
                index
                for index, observed_language in enumerate(batch["language"])
                if observed_language == language
            ]
            if indices:
                index_tensor = torch.tensor(indices, dtype=torch.long, device=embedding.device)
                routed[language] = (
                    index_tensor,
                    self.dialect_heads[language](embedding[index_tensor]),
                )
        return {
            "embedding": embedding,
            "lid_logits": lid_logits,
            "lid_probabilities": torch.softmax(lid_logits, dim=-1),
            "dialect_logits": routed,
            "gender_logits": self.gender_head(embedding),
        }

    def encoder_parameters(self):
        yield from self.input_norm.parameters()
        yield from self.encoder.parameters()

    def task_parameters(self, condition: str):
        yield from self.lid_head.parameters()
        yield from self.gender_head.parameters()
        yield from self.dialect_heads.parameters()
        if condition in {"standard_logged", "pcgrad"}:
            yield from self.log_vars.parameters()

    def configure_condition(self, condition: str) -> None:
        if condition not in CONDITIONS:
            raise ValueError(condition)
        train_log_vars = condition in {"standard_logged", "pcgrad"}
        for parameter in self.log_vars.parameters():
            parameter.requires_grad_(train_log_vars)

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder_parameters():
            parameter.requires_grad_(trainable)
        self.input_norm.train(trainable)
        self.encoder.train(trainable)

    @property
    def parameter_counts(self) -> dict[str, int]:
        encoder = sum(parameter.numel() for parameter in self.encoder_parameters())
        tasks = sum(
            parameter.numel()
            for module in (self.lid_head, self.gender_head, self.dialect_heads)
            for parameter in module.parameters()
        )
        log_vars = sum(parameter.numel() for parameter in self.log_vars.parameters())
        return {
            "encoder": encoder,
            "task_heads": tasks,
            "log_variances": log_vars,
            "total": encoder + tasks + log_vars,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
