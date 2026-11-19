"""Immutable Definition der Risk-Policy v1.0.0 und eindeutige `PolicyRef`-Auflösung (DEC-074)."""

from typing import Final, Literal, Self

from pydantic import model_validator

from vigiscoutx.config.scientific_config import RISK_POLICY_REF as RISK_POLICY_REF
from vigiscoutx.config.scientific_config import PolicyRef
from vigiscoutx.domain.assessments import PrimaryRiskState, SignalStatus
from vigiscoutx.domain.types import FrozenModel, NonNegativeInt, SignalGroup
from vigiscoutx.errors import PolicyResolutionError


class AlarmCountState(FrozenModel):
    min_alarms: NonNegativeInt
    max_alarms: NonNegativeInt
    state: PrimaryRiskState


class RiskPolicyDefinition(FrozenModel):
    """Alle Regeln sind Pflichtfelder; die eingefrorenen Werte werden erzwungen."""

    ref: PolicyRef
    mandatory_groups: tuple[SignalGroup, ...]
    status_precedence: tuple[SignalStatus, ...]
    primary_states: tuple[AlarmCountState, ...]
    primary_r3: Literal["FORBIDDEN"]
    secondary_r3: Literal["VALID_PRIMARY_AND_VALID_G4_WITH_RELEVANT_DEGRADATION"]
    secondary_not_relevant: Literal["PRIMARY_STATE_PRESERVED"]
    secondary_invalid_g4: Literal["PRIMARY_UNCHANGED_NO_R3"]
    fusion_coupling: Literal["INDEPENDENT_OF_FUSION_A_B"]

    @model_validator(mode="after")
    def _frozen(self) -> Self:
        expected = {
            "ref": RISK_POLICY_REF,
            "mandatory_groups": (
                SignalGroup.G1_INPUT,
                SignalGroup.G2_DATA_QUALITY,
                SignalGroup.G3_PREDICTIVE,
            ),
            "status_precedence": (
                SignalStatus.ERROR,
                SignalStatus.NOT_EVALUABLE,
                SignalStatus.VALID,
            ),
            "primary_states": (
                AlarmCountState(min_alarms=0, max_alarms=0, state=PrimaryRiskState.R0),
                AlarmCountState(min_alarms=1, max_alarms=1, state=PrimaryRiskState.R1),
                AlarmCountState(min_alarms=2, max_alarms=3, state=PrimaryRiskState.R2),
            ),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} ist durch Risk-Policy 1.0.0 eingefroren")
        return self

    def state_for(self, alarms: int) -> PrimaryRiskState:
        for rule in self.primary_states:
            if rule.min_alarms <= alarms <= rule.max_alarms:
                return rule.state
        raise ValueError(f"keine Zustandsregel für {alarms} alarmierende Gruppen")


RISK_POLICY_V1: Final = RiskPolicyDefinition(
    ref=RISK_POLICY_REF,
    mandatory_groups=(SignalGroup.G1_INPUT, SignalGroup.G2_DATA_QUALITY, SignalGroup.G3_PREDICTIVE),
    status_precedence=(SignalStatus.ERROR, SignalStatus.NOT_EVALUABLE, SignalStatus.VALID),
    primary_states=(
        AlarmCountState(min_alarms=0, max_alarms=0, state=PrimaryRiskState.R0),
        AlarmCountState(min_alarms=1, max_alarms=1, state=PrimaryRiskState.R1),
        AlarmCountState(min_alarms=2, max_alarms=3, state=PrimaryRiskState.R2),
    ),
    primary_r3="FORBIDDEN",
    secondary_r3="VALID_PRIMARY_AND_VALID_G4_WITH_RELEVANT_DEGRADATION",
    secondary_not_relevant="PRIMARY_STATE_PRESERVED",
    secondary_invalid_g4="PRIMARY_UNCHANGED_NO_R3",
    fusion_coupling="INDEPENDENT_OF_FUSION_A_B",
)


def resolve_risk_policy(ref: PolicyRef) -> RiskPolicyDefinition:
    """Exakt `vsx-risk-r0-r3` / `1.0.0`; jede andere ID oder Version ist ein harter Fehler."""
    if ref != RISK_POLICY_REF:
        raise PolicyResolutionError(
            f"unbekannte Risk-Policy {ref.policy_id}@{ref.version} "
            f"(eingefroren: {RISK_POLICY_REF.policy_id}@{RISK_POLICY_REF.version})"
        )
    return RISK_POLICY_V1
