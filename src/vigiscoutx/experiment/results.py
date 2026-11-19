"""Ergebnis der quantitativen Evaluation (ARC-043, außerhalb SOI-001)."""

from typing import Self

from pydantic import Field, StrictFloat, model_validator

from vigiscoutx.domain.types import FrozenModel, Identifier, Sha256Hex
from vigiscoutx.experiment.types import EvaluationMetric


class MetricValue(FrozenModel):
    metric: EvaluationMetric
    value: StrictFloat


class EvaluationResult(FrozenModel):
    evaluation_id: Identifier
    freeze_id: Sha256Hex
    method_id: Identifier
    metrics: tuple[MetricValue, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_metrics(self) -> Self:
        names = [m.metric for m in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("jede Evaluationsmetrik darf nur einmal vorkommen")
        return self
