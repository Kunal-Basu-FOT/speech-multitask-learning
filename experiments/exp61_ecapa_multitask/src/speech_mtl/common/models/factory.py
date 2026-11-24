from __future__ import annotations

from .heads import DEFAULT_DIALECT_CLASS_COUNTS
from .multitask import MultitaskECAPA


def build_exp61_model() -> MultitaskECAPA:
    """Build the frozen compact ECAPA-TDNN architecture for EXP 6.1."""

    return MultitaskECAPA(
        feature_dimension=768,
        embedding_dimension=192,
        head_hidden_dimension=256,
        head_dropout=0.1,
        encoder_dropout=0.1,
        dialect_class_counts=DEFAULT_DIALECT_CLASS_COUNTS,
    )
