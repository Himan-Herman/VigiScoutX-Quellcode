"""Immutable Definition der Recommendation-Policy v1.0.0 und `PolicyRef`-Auflösung (DEC-075)."""

from enum import StrEnum
from typing import Final, Self

from pydantic import model_validator

from vigiscoutx.config.scientific_config import (
    RECOMMENDATION_POLICY_REF as RECOMMENDATION_POLICY_REF,
)
from vigiscoutx.config.scientific_config import (
    PolicyRef,
    ScientificConfig,
)
from vigiscoutx.domain.types import Code, FrozenModel
from vigiscoutx.errors import PolicyResolutionError
from vigiscoutx.risk.policy import RiskPolicyDefinition, resolve_risk_policy


class Outcome(StrEnum):
    """Bewertungsergebnisse, denen die Policy genau eine Empfehlung zuordnet."""

    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    PRIMARY_NOT_EVALUABLE = "PRIMARY_NOT_EVALUABLE"
    PRIMARY_ERROR = "PRIMARY_ERROR"
    PRIMARY_CONTRACT_ERROR = "PRIMARY_CONTRACT_ERROR"
    R3 = "R3"
    G4_NOT_EVALUABLE = "G4_NOT_EVALUABLE"
    G4_ERROR = "G4_ERROR"
    G4_CONTRACT_ERROR = "G4_CONTRACT_ERROR"


class RecommendationRule(FrozenModel):
    outcome: Outcome
    code: Code
    reason_codes: tuple[Code, ...]


_V1_RULES: Final = (
    (Outcome.R0, "REC_CONTINUE_MONITORING", ("RSN_R0_NO_GROUP_ALARM",)),
    (Outcome.R1, "REC_FURTHER_MONITORING", ("RSN_R1_SINGLE_GROUP_ALARM",)),
    (Outcome.R2, "REC_CAUSE_ANALYSIS", ("RSN_R2_MULTI_GROUP_ALARM",)),
    (
        Outcome.PRIMARY_NOT_EVALUABLE,
        "REC_REVIEW_EVALUABILITY",
        ("RSN_REQUIRED_INPUT_NOT_EVALUABLE",),
    ),
    (Outcome.PRIMARY_ERROR, "REC_TECHNICAL_REVIEW", ("RSN_REQUIRED_INPUT_ERROR",)),
    (
        Outcome.PRIMARY_CONTRACT_ERROR,
        "REC_TECHNICAL_REVIEW",
        ("RSN_REQUIRED_INPUT_ERROR", "RSN_POLICY_INPUT_CONTRACT_ERROR"),
    ),
    (Outcome.R3, "REC_ESCALATE_REVIEW", ("RSN_R3_RELEVANT_DEGRADATION_CONFIRMED",)),
    (Outcome.G4_NOT_EVALUABLE, "REC_REVIEW_EVALUABILITY", ("RSN_G4_NOT_EVALUABLE",)),
    (Outcome.G4_ERROR, "REC_TECHNICAL_REVIEW", ("RSN_G4_ERROR",)),
    (
        Outcome.G4_CONTRACT_ERROR,
        "REC_TECHNICAL_REVIEW",
        ("RSN_G4_ERROR", "RSN_POLICY_INPUT_CONTRACT_ERROR"),
    ),
)
"""Regelzuordnung für `G4_CONTRACT_ERROR`."""

ACTION_CODES: Final = frozenset(code for _, code, _ in _V1_RULES)


class RecommendationPolicyDefinition(FrozenModel):
    ref: PolicyRef
    rules: tuple[RecommendationRule, ...]

    @model_validator(mode="after")
    def _frozen(self) -> Self:
        if self.ref != RECOMMENDATION_POLICY_REF:
            raise ValueError("ref ist durch Recommendation-Policy 1.0.0 eingefroren")
        actual = tuple((r.outcome, r.code, r.reason_codes) for r in self.rules)
        if actual != _V1_RULES:
            raise ValueError("Regeln sind durch Recommendation-Policy 1.0.0 eingefroren")
        return self

    def rule(self, outcome: Outcome) -> RecommendationRule:
        return next(r for r in self.rules if r.outcome is outcome)


RECOMMENDATION_POLICY_V1: Final = RecommendationPolicyDefinition(
    ref=RECOMMENDATION_POLICY_REF,
    rules=tuple(
        RecommendationRule(outcome=outcome, code=code, reason_codes=reasons)
        for outcome, code, reasons in _V1_RULES
    ),
)


def resolve_recommendation_policy(ref: PolicyRef) -> RecommendationPolicyDefinition:
    """Exakt `vsx-recommendation-review` / `1.0.0`; sonst harter Fehler."""
    if ref != RECOMMENDATION_POLICY_REF:
        raise PolicyResolutionError(
            f"unbekannte Recommendation-Policy {ref.policy_id}@{ref.version} "
            f"(eingefroren: {RECOMMENDATION_POLICY_REF.policy_id}@"
            f"{RECOMMENDATION_POLICY_REF.version})"
        )
    return RECOMMENDATION_POLICY_V1


def resolve_configured_policies(
    scientific: ScientificConfig,
) -> tuple[RiskPolicyDefinition, RecommendationPolicyDefinition]:
    """Policies einer RunConfig bzw. eines FreezeManifests; unbekannte Refs = harter Fehler."""
    return (
        resolve_risk_policy(scientific.risk_policy),
        resolve_recommendation_policy(scientific.recommendation_policy),
    )
