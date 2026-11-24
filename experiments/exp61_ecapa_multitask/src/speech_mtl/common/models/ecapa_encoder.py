from __future__ import annotations

import torch
from torch import nn

try:
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
except ImportError as exc:
    raise ImportError(
        "SpeechBrain is required for CompactECAPAEncoder. "
        "Install speechbrain==1.0.3 in the project environment."
    ) from exc


class CompactECAPAEncoder(nn.Module):
    """Input normalization plus a compact SpeechBrain ECAPA-TDNN.

    Input shape:
        [batch, time, 768]

    Output shape:
        [batch, 192]

    Padded frames are masked back to zero after input normalization so
    LayerNorm affine parameters cannot introduce nonzero padding.
    """

    FEATURE_DIMENSION = 768
    EMBEDDING_DIMENSION = 192
    EXPECTED_ECAPA_PARAMETERS = 7_212_660

    def __init__(
        self,
        *,
        input_size: int = FEATURE_DIMENSION,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if input_size != self.FEATURE_DIMENSION:
            raise ValueError(
                f"EXP 6.1 expects input_size={self.FEATURE_DIMENSION}, "
                f"received {input_size}"
            )

        if embedding_dimension != self.EMBEDDING_DIMENSION:
            raise ValueError(
                "EXP 6.1 expects embedding_dimension="
                f"{self.EMBEDDING_DIMENSION}, received {embedding_dimension}"
            )

        self.input_size = int(input_size)
        self.embedding_dimension = int(embedding_dimension)

        self.input_normalization = nn.LayerNorm(self.input_size)

        self.ecapa = ECAPA_TDNN(
            input_size=self.input_size,
            lin_neurons=self.embedding_dimension,
            channels=[480, 480, 480, 480, 1440],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            res2net_scale=8,
            se_channels=128,
            global_context=True,
            groups=[1, 1, 1, 1, 1],
            dropout=float(dropout),
        )

    @staticmethod
    def _mask_from_relative_lengths(
        relative_lengths: torch.Tensor,
        maximum_frames: int,
    ) -> torch.Tensor:
        if relative_lengths.ndim != 1:
            raise ValueError(
                "relative_lengths must have shape [batch], "
                f"received {tuple(relative_lengths.shape)}"
            )

        if torch.any(relative_lengths <= 0) or torch.any(relative_lengths > 1):
            raise ValueError("relative_lengths values must be in (0, 1]")

        absolute_lengths = torch.clamp(
            torch.round(relative_lengths * maximum_frames).to(torch.long),
            min=1,
            max=maximum_frames,
        )

        frame_ids = torch.arange(
            maximum_frames,
            device=relative_lengths.device,
        ).unsqueeze(0)

        return frame_ids < absolute_lengths.unsqueeze(1)

    def forward(
        self,
        features: torch.Tensor,
        relative_lengths: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                "features must have shape [batch, time, feature], "
                f"received {tuple(features.shape)}"
            )

        batch_size, maximum_frames, feature_dimension = features.shape

        if feature_dimension != self.input_size:
            raise ValueError(
                f"Expected feature dimension {self.input_size}, "
                f"received {feature_dimension}"
            )

        if relative_lengths.shape != (batch_size,):
            raise ValueError(
                "relative_lengths must have shape "
                f"({batch_size},), received {tuple(relative_lengths.shape)}"
            )

        if valid_mask is None:
            valid_mask = self._mask_from_relative_lengths(
                relative_lengths=relative_lengths,
                maximum_frames=maximum_frames,
            )
        else:
            if valid_mask.shape != (batch_size, maximum_frames):
                raise ValueError(
                    "valid_mask must have shape "
                    f"({batch_size}, {maximum_frames}), "
                    f"received {tuple(valid_mask.shape)}"
                )
            valid_mask = valid_mask.to(dtype=torch.bool)

        normalized = self.input_normalization(features)
        normalized = normalized.masked_fill(
            ~valid_mask.unsqueeze(-1),
            0.0,
        )

        embeddings = self.ecapa(
            normalized,
            lengths=relative_lengths,
        )

        if embeddings.shape != (
            batch_size,
            1,
            self.embedding_dimension,
        ):
            raise RuntimeError(
                "Unexpected SpeechBrain ECAPA output shape: "
                f"{tuple(embeddings.shape)}"
            )

        return embeddings.squeeze(1)

    def set_trainable(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)
