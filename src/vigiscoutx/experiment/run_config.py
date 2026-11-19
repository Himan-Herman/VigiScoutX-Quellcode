"""RunConfig eines Experimentlaufs: wissenschaftlicher Inhalt plus laufbezogene Felder."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from vigiscoutx.config.loader import load_model, parse_model
from vigiscoutx.config.scientific_config import ScientificConfig, calibration_gaps
from vigiscoutx.domain.refs import SeedBundle
from vigiscoutx.domain.types import FrozenModel, Identifier, PositiveInt, Sha256Hex
from vigiscoutx.experiment.types import EvaluationMetric, Phase


class RunConfig(FrozenModel):
    schema_version: Literal[1]
    run_id: Identifier
    phase: Phase
    freeze_id: Sha256Hex | None
    scientific: ScientificConfig
    repetitions: PositiveInt
    seeds: SeedBundle
    evaluation_metrics: tuple[EvaluationMetric, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _phase_rules(self) -> Self:
        if len(set(self.evaluation_metrics)) != len(self.evaluation_metrics):
            raise ValueError("evaluation_metrics müssen eindeutig sein")
        if self.phase is Phase.FROZEN:
            raise ValueError("FROZEN ist ein Guard-Zustand und keine Run-Phase")
        if self.phase is Phase.DEVELOPMENT_CALIBRATION and self.freeze_id is not None:
            raise ValueError("DEVELOPMENT_CALIBRATION-Runs dürfen keine freeze_id tragen")
        if self.phase is Phase.HELD_OUT_EVALUATION:
            missing = calibration_gaps(self.scientific)
            if self.freeze_id is None:
                missing.append("freeze_id")
            if missing:
                raise ValueError(f"HELD_OUT_EVALUATION erfordert gesetzte Werte: {missing}")
        return self


def parse_run_config(text: str) -> RunConfig:
    return parse_model(text, RunConfig)


def load_run_config(path: Path) -> RunConfig:
    return load_model(path, RunConfig)
