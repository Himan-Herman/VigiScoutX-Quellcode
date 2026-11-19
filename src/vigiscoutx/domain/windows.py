"""Stichprobenfenster und Modellausgaben.

`MonitoringWindow` trägt bewusst keinen Shift-Typ, keine Severity und keine experimentelle
Ground Truth; diese Informationen gehören ausschließlich zum externen `ExperimentCase` (DEC-029).
"""

import math
from typing import Self

from pydantic import Field, StrictFloat, model_validator

from vigiscoutx.domain.refs import SampleRef
from vigiscoutx.domain.types import FrozenModel, Identifier, NonNegativeInt

PROBABILITY_SUM_TOLERANCE = 1e-5
"""Numerische Validierungstoleranz für Softmax-Zeilen; kein wissenschaftlicher Parameter."""


def _require_unique_samples(samples: tuple[SampleRef, ...]) -> None:
    ids = [s.sample_id for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id darf innerhalb eines Fensters nur einmal vorkommen")


class ReferenceWindow(FrozenModel):
    window_id: Identifier
    samples: tuple[SampleRef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        _require_unique_samples(self.samples)
        return self


class MonitoringWindow(FrozenModel):
    window_id: Identifier
    window_index: NonNegativeInt
    samples: tuple[SampleRef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        _require_unique_samples(self.samples)
        return self


def _require_matrix(sample_ids: tuple[str, ...], rows: tuple[tuple[float, ...], ...]) -> int:
    if len(rows) != len(sample_ids):
        raise ValueError("Anzahl Zeilen muss der Anzahl sample_ids entsprechen")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids müssen eindeutig sein")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError("alle Zeilen müssen dieselbe Dimension haben")
    return widths.pop()


class ModelOutputBatch(FrozenModel):
    window_id: Identifier
    sample_ids: tuple[Identifier, ...] = Field(min_length=1)
    probabilities: tuple[tuple[StrictFloat, ...], ...]

    @model_validator(mode="after")
    def _valid_softmax(self) -> Self:
        if _require_matrix(self.sample_ids, self.probabilities) < 2:
            raise ValueError("mindestens zwei Klassen erforderlich")
        for row in self.probabilities:
            if any(p < 0.0 or p > 1.0 for p in row):
                raise ValueError("Wahrscheinlichkeiten müssen in [0, 1] liegen")
            if not math.isclose(sum(row), 1.0, abs_tol=PROBABILITY_SUM_TOLERANCE):
                raise ValueError("Wahrscheinlichkeiten einer Zeile müssen sich zu 1 summieren")
        return self


class RepresentationBatch(FrozenModel):
    window_id: Identifier
    sample_ids: tuple[Identifier, ...] = Field(min_length=1)
    embeddings: tuple[tuple[StrictFloat, ...], ...]

    @model_validator(mode="after")
    def _valid_matrix(self) -> Self:
        if _require_matrix(self.sample_ids, self.embeddings) < 1:
            raise ValueError("Embedding-Dimension muss mindestens 1 sein")
        return self
