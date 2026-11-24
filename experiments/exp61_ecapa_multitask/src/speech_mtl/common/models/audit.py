from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn

from .multitask import MultitaskECAPA


@dataclass(frozen=True)
class ModelParameterReport:
    input_normalization: int
    ecapa: int
    encoder_total: int
    language_head: int
    gender_head: int
    dialect_heads: int
    all_classifier_heads: int
    task_weighting: int
    total: int
    trainable: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def audit_model_parameters(
    model: MultitaskECAPA,
    *,
    expected_ecapa_parameters: int = 7_212_660,
    expected_total_parameters: int = 7_723_170,
    enforce_exact: bool = True,
) -> ModelParameterReport:
    input_normalization = count_parameters(
        model.encoder.input_normalization
    )
    ecapa = count_parameters(model.encoder.ecapa)
    language_head = count_parameters(model.language_head)
    gender_head = count_parameters(model.gender_head)
    dialect_heads = count_parameters(model.dialect_heads)
    task_weighting = count_parameters(model.task_weighting)

    report = ModelParameterReport(
        input_normalization=input_normalization,
        ecapa=ecapa,
        encoder_total=input_normalization + ecapa,
        language_head=language_head,
        gender_head=gender_head,
        dialect_heads=dialect_heads,
        all_classifier_heads=(
            language_head + gender_head + dialect_heads
        ),
        task_weighting=task_weighting,
        total=count_parameters(model),
        trainable=count_trainable_parameters(model),
    )

    if enforce_exact:
        if report.ecapa != expected_ecapa_parameters:
            raise RuntimeError(
                "SpeechBrain ECAPA parameter count differs from the "
                "frozen EXP 6.1 architecture: "
                f"expected={expected_ecapa_parameters:,}, "
                f"actual={report.ecapa:,}"
            )

        if report.total != expected_total_parameters:
            raise RuntimeError(
                "Total model parameter count differs from the frozen "
                "EXP 6.1 architecture: "
                f"expected={expected_total_parameters:,}, "
                f"actual={report.total:,}"
            )

    return report
