"""Compact ECAPA-TDNN multitask model for SPEECH_MTL EXP 6.1."""

from .audit import ModelParameterReport, audit_model_parameters
from .ecapa_encoder import CompactECAPAEncoder
from .heads import ClassificationHead, OracleDialectHeads
from .multitask import MultitaskECAPA, MultitaskOutput
from .uncertainty import HomoscedasticTaskWeighting

__all__ = [
    "ClassificationHead",
    "CompactECAPAEncoder",
    "HomoscedasticTaskWeighting",
    "ModelParameterReport",
    "MultitaskECAPA",
    "MultitaskOutput",
    "OracleDialectHeads",
    "audit_model_parameters",
]
