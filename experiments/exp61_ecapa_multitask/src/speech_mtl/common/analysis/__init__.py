"""Report-generation utilities for SPEECH_MTL EXP 6.1."""

from .confusion_plots import save_composite_confusion_figure
from .cross_precision import save_cross_precision_summary
from .curves import save_training_curves
from .report_tables import (
    save_cross_precision_table,
    save_experimental_configuration_table,
    save_matched_results_table,
    save_per_language_dialect_table,
    save_runtime_table,
)

__all__ = [
    "save_composite_confusion_figure",
    "save_cross_precision_summary",
    "save_cross_precision_table",
    "save_experimental_configuration_table",
    "save_matched_results_table",
    "save_per_language_dialect_table",
    "save_runtime_table",
    "save_training_curves",
]
