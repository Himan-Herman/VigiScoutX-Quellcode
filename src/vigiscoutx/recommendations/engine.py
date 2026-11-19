"""RecommendationEngine (ARC-013, DEC-075): genau eine Primary-Empfehlung je Primary-Ergebnis
und höchstens eine separate sekundäre Empfehlung; die Primary-Empfehlung wird nie verändert."""

from vigiscoutx.domain.assessments import (
    AnalysisLevel,
    LabelAwareAssessment,
    LabelAwareFailure,
    LabelAwareRiskState,
    PrimaryRiskAssessment,
    PrimaryRiskFailure,
    Recommendation,
    SignalStatus,
)
from vigiscoutx.domain.refs import (
    PRIMARY_LOCAL_ID,
    SECONDARY_LOCAL_ID,
    window_evidence,
)
from vigiscoutx.recommendations.policy import Outcome, RecommendationPolicyDefinition


def _recommend(
    window_id: str,
    level: AnalysisLevel,
    outcome: Outcome,
    policy: RecommendationPolicyDefinition,
    evidence: tuple[object, ...],
) -> Recommendation:
    rule = policy.rule(outcome)
    return Recommendation.model_validate(
        {
            "window_id": window_id,
            "level": level,
            "code": rule.code,
            "reason_codes": rule.reason_codes,
            "evidence_refs": evidence,
        }
    )


def primary_recommendation(
    outcome: PrimaryRiskAssessment | PrimaryRiskFailure, policy: RecommendationPolicyDefinition
) -> Recommendation:
    if isinstance(outcome, PrimaryRiskAssessment):
        ref = window_evidence("primary_risk_assessment", outcome.window_id, PRIMARY_LOCAL_ID)
        return _recommend(
            outcome.window_id,
            AnalysisLevel.PRIMARY,
            Outcome(outcome.state.value),
            policy,
            (ref, *outcome.evidence_refs),
        )
    if outcome.contract_violation:
        result = Outcome.PRIMARY_CONTRACT_ERROR
    elif outcome.status is SignalStatus.ERROR:
        result = Outcome.PRIMARY_ERROR
    else:
        result = Outcome.PRIMARY_NOT_EVALUABLE
    ref = window_evidence("primary_risk_failure", outcome.window_id, PRIMARY_LOCAL_ID)
    return _recommend(
        outcome.window_id, AnalysisLevel.PRIMARY, result, policy, (ref, *outcome.evidence_refs)
    )


def secondary_recommendation(
    outcome: LabelAwareAssessment | LabelAwareFailure | None,
    policy: RecommendationPolicyDefinition,
) -> Recommendation | None:
    """R3 bzw. G4-Fehler → eigene sekundäre Empfehlung; gültiges G4 ohne Relevanz → keine."""
    if outcome is None:
        return None
    if isinstance(outcome, LabelAwareAssessment):
        if outcome.state is not LabelAwareRiskState.R3:
            return None
        ref = window_evidence("label_aware_assessment", outcome.window_id, SECONDARY_LOCAL_ID)
        return _recommend(
            outcome.window_id,
            AnalysisLevel.SECONDARY,
            Outcome.R3,
            policy,
            (ref, *outcome.evidence_refs[1:]),
        )
    if outcome.contract_violation:
        result = Outcome.G4_CONTRACT_ERROR
    elif outcome.status is SignalStatus.ERROR:
        result = Outcome.G4_ERROR
    else:
        result = Outcome.G4_NOT_EVALUABLE
    ref = window_evidence("label_aware_failure", outcome.window_id, SECONDARY_LOCAL_ID)
    return _recommend(
        outcome.window_id,
        AnalysisLevel.SECONDARY,
        result,
        policy,
        (ref, *outcome.evidence_refs[1:]),
    )
