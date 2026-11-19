"""Gemeinsame Basistypen und Aufzählungen der Domain-Contracts."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StringConstraints

Sha256Hex = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
GitCommitHash = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$")]
Identifier = Annotated[
    str,
    StringConstraints(
        strict=True, min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"
    ),
]
Code = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Z][A-Z0-9_]{1,63}$")]

# Strikte Skalare: keine stille Umwandlung von "3.5" → float, "yes" → bool oder 3.0 → int.
PositiveInt = Annotated[StrictInt, Field(gt=0)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
NonNegativeFloat = Annotated[StrictFloat, Field(ge=0.0)]
UnitInterval = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]


class FrozenModel(BaseModel):
    """Unveränderlicher, geschlossener Contract ohne unbekannte Felder.

    `model_copy(update=...)` validiert die Kopie vollständig neu, damit Invarianten nicht über
    Kopien umgangen werden können. `model_construct` bleibt ein bewusster Validierungs-Bypass.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        if not update:
            return super().model_copy(deep=deep)
        data = {name: getattr(self, name) for name in type(self).model_fields}
        data.update(update)
        return self.model_validate(data)


class SignalGroup(StrEnum):
    G1_INPUT = "G1"
    G2_DATA_QUALITY = "G2"
    G3_PREDICTIVE = "G3"
    G4_PERFORMANCE = "G4"


PRIMARY_GROUPS: frozenset[SignalGroup] = frozenset(
    {SignalGroup.G1_INPUT, SignalGroup.G2_DATA_QUALITY, SignalGroup.G3_PREDICTIVE}
)
"""Einzige Gruppen, die im Primary-Pfad zulässig sind (DEC-021, DEC-030)."""


class PerformanceMetric(StrEnum):
    BALANCED_ACCURACY = "BALANCED_ACCURACY"


def require_primary_group(group: SignalGroup) -> SignalGroup:
    if group not in PRIMARY_GROUPS:
        raise ValueError(f"Gruppe {group.value} ist im Primary-Pfad unzulässig (nur G1/G2/G3)")
    return group
