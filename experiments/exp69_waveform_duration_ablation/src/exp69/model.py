from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class LogMelProjectedMultitaskECAPA(nn.Module):
    """80-bin log-Mel front end projected into the frozen EXP 6.1 ECAPA shape."""

    def __init__(self, feature_dimension: int = 80, projected_dimension: int = 768) -> None:
        super().__init__()
        if feature_dimension != 80:
            raise ValueError(f"EXP 6.9 expects 80 log-Mel bins, got {feature_dimension}")
        if projected_dimension != 768:
            raise ValueError(f"EXP 6.9 expects projection dimension 768, got {projected_dimension}")

        # Imported lazily so package unit tests can run without SpeechBrain.
        from speech_mtl.common.models.factory import build_exp61_model

        self.frontend_projection = nn.Linear(feature_dimension, projected_dimension)
        nn.init.xavier_uniform_(self.frontend_projection.weight)
        nn.init.zeros_(self.frontend_projection.bias)
        self.base = build_exp61_model()

    def forward(
        self,
        *,
        features: torch.Tensor,
        relative_lengths: torch.Tensor,
        oracle_languages: list[str],
        valid_mask: torch.Tensor,
    ):
        if features.ndim != 3 or features.shape[-1] != 80:
            raise ValueError(f"Expected [B,T,80] log-Mel features, received {tuple(features.shape)}")
        projected = self.frontend_projection(features)
        projected = projected.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return self.base(
            features=projected,
            relative_lengths=relative_lengths,
            oracle_languages=oracle_languages,
            valid_mask=valid_mask,
        )

    def encoder_parameters(self) -> list[nn.Parameter]:
        values = list(self.frontend_projection.parameters())
        values.extend(list(self.base.encoder_parameters()))
        return values

    def task_parameters(self) -> list[nn.Parameter]:
        encoder_ids = {id(parameter) for parameter in self.encoder_parameters()}
        values = [parameter for parameter in self.parameters() if id(parameter) not in encoder_ids]
        if not values:
            raise RuntimeError("No task parameters were identified")
        return values

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.frontend_projection.parameters():
            parameter.requires_grad_(trainable)
        self.base.set_encoder_trainable(trainable)
        if trainable:
            self.frontend_projection.train()
            self.base.encoder.train()
        else:
            self.frontend_projection.eval()
            self.base.encoder.eval()

    @property
    def embedding_dimension(self) -> int:
        return int(self.base.embedding_dimension)


def parameter_report(model: LogMelProjectedMultitaskECAPA) -> dict[str, int]:
    encoder_ids = {id(value) for value in model.encoder_parameters()}
    total = sum(parameter.numel() for parameter in model.parameters())
    encoder = sum(parameter.numel() for parameter in model.parameters() if id(parameter) in encoder_ids)
    tasks = total - encoder
    projection = sum(parameter.numel() for parameter in model.frontend_projection.parameters())
    return {
        "total": int(total),
        "encoder_including_projection": int(encoder),
        "task_parameters": int(tasks),
        "frontend_projection": int(projection),
    }

# ---------------------------------------------------------------------------
# EXP69_SINGLETON_SAFE_BATCHNORM_PATCH_V2
# ---------------------------------------------------------------------------
#
# SpeechBrain BatchNorm1d supports two tensor conventions:
#
#   skip_transpose=False: [batch, time, channels]
#   skip_transpose=True:  [batch, channels, time]
#
# ECAPA uses skip_transpose=True. After attentive statistics pooling,
# asp_bn can therefore receive [1, channels, 1]. PyTorch BatchNorm rejects
# this during training because only one value exists per channel.
#
# Only for that exact degenerate case, use the accumulated running
# statistics. All ordinary batches retain the original forward path.
#
def _install_exp69_singleton_safe_batchnorm() -> None:
    from speechbrain.nnet.normalization import (
        BatchNorm1d as SpeechBrainBatchNorm1d,
    )

    if getattr(
        SpeechBrainBatchNorm1d,
        "_exp69_singleton_safe_v2_installed",
        False,
    ):
        return

    original_forward = SpeechBrainBatchNorm1d.forward

    def singleton_safe_forward(self, x):
        shape_original = x.shape

        # Reproduce SpeechBrain's layout conversion so that the number
        # of values per normalized channel is calculated correctly.
        if self.combine_batch_time:
            if x.ndim == 3:
                x_for_norm = x.reshape(
                    shape_original[0] * shape_original[1],
                    shape_original[2],
                )
            else:
                x_for_norm = x.reshape(
                    shape_original[0] * shape_original[1],
                    shape_original[3],
                    shape_original[2],
                )
        elif not self.skip_transpose:
            x_for_norm = x.transpose(-1, 1)
        else:
            x_for_norm = x

        channels = int(x_for_norm.shape[1])
        values_per_channel = (
            int(x_for_norm.numel()) // channels
        )

        singleton = (
            self.training
            and values_per_channel == 1
        )

        if not singleton:
            return original_forward(self, x)

        if (
            not self.norm.track_running_stats
            or self.norm.running_mean is None
            or self.norm.running_var is None
        ):
            raise RuntimeError(
                "Singleton BatchNorm fallback requires running "
                "statistics, but this layer does not track them."
            )

        norm_training_before = self.norm.training
        self.norm.eval()

        try:
            normalized = self.norm(x_for_norm)
        finally:
            self.norm.train(norm_training_before)

        if self.combine_batch_time:
            normalized = normalized.reshape(shape_original)
        elif not self.skip_transpose:
            normalized = normalized.transpose(1, -1)

        return normalized

    SpeechBrainBatchNorm1d.forward = singleton_safe_forward
    SpeechBrainBatchNorm1d._exp69_singleton_safe_v2_installed = True


_install_exp69_singleton_safe_batchnorm()
