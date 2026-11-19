"""Wissenschaftlicher Konfigurationsinhalt des SOI-Cores (DEC-032).

Alle wissenschaftlich wirksamen Parameter sind Pflichtfelder ohne Default. Werte, die erst im
Calibration-Pilot bestimmt werden, sind explizit als `null` anzugeben, solange sie offen sind.
Laufbezogene Experimentfelder (Phase, Freeze, Seeds, Wiederholungen, Evaluationsmetriken) liegen
in `vigiscoutx.experiment.run_config`.
"""

from enum import StrEnum
from typing import Annotated, Final, Self

from pydantic import Field, StrictFloat, field_validator, model_validator

from vigiscoutx.domain.frozen_json import ScientificParams
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.refs import (
    DatasetSnapshotRef,
    ModelSnapshotRef,
    SplitManifestRef,
)
from vigiscoutx.domain.types import (
    FrozenModel,
    Identifier,
    NonNegativeFloat,
    PerformanceMetric,
    PositiveInt,
    SignalGroup,
    require_primary_group,
)
from vigiscoutx.errors import PolicyResolutionError


class FusionRule(StrEnum):
    """Vorab definierter Kandidatenraum für den primären Alarm (DEC-026)."""

    A_ANY_PRIMARY_GROUP = "A_ANY_PRIMARY_GROUP"
    B_AT_LEAST_TWO_PRIMARY_GROUPS = "B_AT_LEAST_TWO_PRIMARY_GROUPS"


class PreprocessingConfig(FrozenModel):
    pipeline_id: Identifier
    input_height: PositiveInt
    input_width: PositiveInt
    color_mode: Identifier
    params: ScientificParams


class ThresholdConfig(FrozenModel):
    kind: Identifier
    value: StrictFloat


class SignalConfig(FrozenModel):
    signal_id: Identifier
    group: SignalGroup
    method: Identifier
    params: ScientificParams
    threshold: ThresholdConfig | None

    @field_validator("group")
    @classmethod
    def _primary_group(cls, group: SignalGroup) -> SignalGroup:
        return require_primary_group(group)


class MonitoringConfig(FrozenModel):
    window_size: PositiveInt
    signals: tuple[SignalConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_signals(self) -> Self:
        ids = [s.signal_id for s in self.signals]
        if len(ids) != len(set(ids)):
            raise ValueError("signal_id muss eindeutig sein")
        return self


class FusionConfig(FrozenModel):
    rule: FusionRule | None
    groups: tuple[SignalGroup, ...] = Field(min_length=1)

    @field_validator("groups")
    @classmethod
    def _primary_groups(cls, groups: tuple[SignalGroup, ...]) -> tuple[SignalGroup, ...]:
        for group in groups:
            require_primary_group(group)
        if len(groups) != len(set(groups)):
            raise ValueError("Fusionsgruppen müssen eindeutig sein")
        return groups


class RelevanceConfig(FrozenModel):
    performance_metric: PerformanceMetric
    clean_quantile: Annotated[StrictFloat, Field(gt=0.0, lt=1.0)]
    delta_rel: NonNegativeFloat | None


class PolicyRef(FrozenModel):
    policy_id: Identifier
    version: Identifier


RISK_POLICY_REF: Final = PolicyRef(policy_id="vsx-risk-r0-r3", version="1.0.0")
"""Einzige eingefrorene Risk-Policy (DEC-074); von `vigiscoutx.risk.policy` aufgelöst."""

RECOMMENDATION_POLICY_REF: Final = PolicyRef(policy_id="vsx-recommendation-review", version="1.0.0")
"""Einzige eingefrorene Recommendation-Policy (DEC-075)."""


def require_frozen_policy_refs(risk: PolicyRef, recommendation: PolicyRef) -> None:
    """Nur die eingefrorenen v1.0.0-Policies sind bindbar; jede andere ID/Version = harter Fehler.

    Liegt in der Konfigurationsschicht, damit Freeze-Pfade die Bindung prüfen können, ohne die
    Risk-/Recommendation-Module zu importieren (Calibration-Grenze, DEC-076).
    """
    for kind, ref, frozen in (
        ("Risk", risk, RISK_POLICY_REF),
        ("Recommendation", recommendation, RECOMMENDATION_POLICY_REF),
    ):
        if ref != frozen:
            raise PolicyResolutionError(
                f"unbekannte {kind}-Policy {ref.policy_id}@{ref.version} "
                f"(eingefroren: {frozen.policy_id}@{frozen.version})"
            )


class ScientificConfig(FrozenModel):
    dataset: DatasetSnapshotRef
    split: SplitManifestRef
    model: ModelSnapshotRef
    preprocessing: PreprocessingConfig
    monitoring: MonitoringConfig
    fusion: FusionConfig
    relevance: RelevanceConfig
    risk_policy: PolicyRef
    recommendation_policy: PolicyRef


def signal_gaps(signals: tuple[SignalConfig, ...]) -> list[str]:
    """Offene Signalparameter; dieselbe Semantik für `calibration_gaps` und das V2-Freeze-Gate.

    - Signal mit gesetzter `ThresholdConfig`: vollständig.
    - Registriertes methodenentschiedenes Signal (G1–G3, `METHOD_DECIDED_SIGNALS`) ohne
      Threshold: vollständig genau dann, wenn seine Parameter die eingefrorene
      Methodenkonfiguration vollständig validieren; sonst `…method_config` offen.
    - Jedes andere Signal ohne Threshold: `…threshold` offen.
    """
    from vigiscoutx.signals.config import method_decided  # Methodenschicht, kein Importzyklus

    gaps: list[str] = []
    for signal in signals:
        if signal.threshold is not None:
            continue
        decided = method_decided(signal)
        if decided is None:
            gaps.append(f"monitoring.signals[{signal.signal_id}].threshold")
        elif not decided:
            gaps.append(f"monitoring.signals[{signal.signal_id}].method_config")
    return gaps


def calibration_gaps(config: ScientificConfig) -> list[str]:
    """Liefert die noch nicht festgelegten Calibration-Parameter."""
    gaps = signal_gaps(config.monitoring.signals)
    if config.fusion.rule is None:
        gaps.append("fusion.rule")
    if config.relevance.delta_rel is None:
        gaps.append("relevance.delta_rel")
    return gaps


def scientific_config_sha256(config: ScientificConfig) -> str:
    return canonical_sha256(config)
