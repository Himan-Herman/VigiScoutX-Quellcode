"""Inhalt eines Split-Manifests (generisch, ohne Datensatzbezug)."""

from typing import Self

from pydantic import Field, model_validator

from vigiscoutx.domain.types import FrozenModel, Identifier
from vigiscoutx.experiment.types import ModelDevelopmentRole, Partition


class SplitAssignment(FrozenModel):
    """Zuordnung eines Samples; `model_development_role` ist genau für MODEL_DEVELOPMENT gesetzt."""

    sample_id: Identifier
    group_id: Identifier
    partition: Partition
    model_development_role: ModelDevelopmentRole | None

    @model_validator(mode="after")
    def _role_only_in_model_development(self) -> Self:
        in_md = self.partition is Partition.MODEL_DEVELOPMENT
        if in_md and self.model_development_role is None:
            raise ValueError("MODEL_DEVELOPMENT erfordert TRAIN oder MODEL_VALIDATION")
        if not in_md and self.model_development_role is not None:
            raise ValueError("model_development_role ist nur in MODEL_DEVELOPMENT zulässig")
        return self


class SplitManifest(FrozenModel):
    """Zuordnung aller Samples zu Hauptpartitionen und Model-Development-Unterrollen.

    Leakage-Regeln werden explizit in `vigiscoutx.experiment.split_integrity` geprüft, damit
    Verstöße als `LeakageError` erscheinen.
    """

    manifest_id: Identifier
    assignments: tuple[SplitAssignment, ...] = Field(min_length=1)
