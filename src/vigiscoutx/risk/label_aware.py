"""Sekundäre label-aware Analyse: einzige Quelle von R3 (ARC-012, DEC-074, DEC-076).

R3 nur bei gültigem Primary-Ergebnis und gültigem G4 (Reference und Monitoring `VALID` und
evaluierbar, `delta_ba` und Relevanz bestimmt) mit `relevant_degradation = true` unter der
gebundenen Relevanzkonfiguration; sonst bleibt der Primary-Zustand erhalten. G4-Fehler erzeugen
einen `LabelAwareFailure` und verändern das Primary-Ergebnis nie.
"""

from vigiscoutx.config.scientific_config import RelevanceConfig
from vigiscoutx.domain.assessments import (
    LabelAwareAssessment,
    LabelAwareFailure,
    LabelAwareRiskState,
    PerformanceAssessment,
    PrimaryRiskAssessment,
    PrimaryRiskFailure,
    SignalStatus,
)
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.refs import (
    PERFORMANCE_LOCAL_ID,
    PRIMARY_LOCAL_ID,
    relevance_evidence,
    window_evidence,
)
from vigiscoutx.risk.policy import RiskPolicyDefinition

type SecondaryOutcome = LabelAwareAssessment | LabelAwareFailure


def relevance_config_sha256(relevance: RelevanceConfig) -> str:
    return canonical_sha256(relevance)


def _g4_status(performance: PerformanceAssessment) -> SignalStatus:
    evaluations = (performance.reference, performance.monitoring)
    if any(e.status is SignalStatus.ERROR for e in evaluations):
        return SignalStatus.ERROR
    if any(e.status is not SignalStatus.VALID or not e.performance_evaluable for e in evaluations):
        return SignalStatus.NOT_EVALUABLE
    return SignalStatus.VALID


def _contract_violation(
    primary: PrimaryRiskAssessment, performance: PerformanceAssessment, relevance: RelevanceConfig
) -> bool:
    if performance.window_id != primary.window_id:
        return True
    if performance.reference.window_id != primary.reference_window_id:
        return True
    if relevance.delta_rel is None or performance.metric is not relevance.performance_metric:
        return True
    if performance.delta_rel != relevance.delta_rel:
        return True
    # Beide Evaluationen gültig, aber Relevanz unbestimmt (Baseline 5.3).
    return _g4_status(performance) is SignalStatus.VALID and (
        performance.delta_ba is None or performance.relevant_degradation is None
    )


def assess_label_aware(
    primary: PrimaryRiskAssessment | PrimaryRiskFailure,
    performance: PerformanceAssessment | None,
    relevance: RelevanceConfig,
    policy: RiskPolicyDefinition,
) -> SecondaryOutcome | None:
    """`None`, wenn kein G4 vorliegt oder das Primary-Ergebnis ungültig ist (kein Fehler)."""
    if performance is None or isinstance(primary, PrimaryRiskFailure):
        return None
    failure_refs = (
        window_evidence("primary_risk_assessment", primary.window_id, PRIMARY_LOCAL_ID),
        # G4-Eingabe der sekundären Analyse dieses Windows; nie eine windowfremde Referenz.
        window_evidence("performance_assessment", primary.window_id, PERFORMANCE_LOCAL_ID),
    )
    if _contract_violation(primary, performance, relevance):
        return LabelAwareFailure(
            window_id=primary.window_id,
            primary=primary,
            performance=performance,
            status=SignalStatus.ERROR,
            contract_violation=True,
            evidence_refs=failure_refs,
        )
    status = _g4_status(performance)
    if status is not SignalStatus.VALID:
        return LabelAwareFailure(
            window_id=primary.window_id,
            primary=primary,
            performance=performance,
            status=status,
            contract_violation=False,
            evidence_refs=failure_refs,
        )
    relevant = performance.relevant_degradation is True
    digest = relevance_config_sha256(relevance)
    return LabelAwareAssessment(
        window_id=primary.window_id,
        primary=primary,
        performance=performance,
        relevance_config_sha256=digest,
        state=LabelAwareRiskState.R3 if relevant else LabelAwareRiskState(primary.state.value),
        evidence_refs=(*failure_refs, relevance_evidence(digest)),
    )
