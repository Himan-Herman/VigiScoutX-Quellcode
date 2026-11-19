"""P4-03C HIGH-1/HIGH-2: eindeutige, windowgebundene Evidence und semantische Run-Konsistenz.

Ausschließlich synthetische Fixtures. Widersprüchliche Ergebnisse werden direkt (bzw. über
`model_construct`) konstruiert und nicht über RiskMapper oder RecommendationEngine erzeugt; die
Ablehnung erfolgt allein durch die verbindliche Validierung des `MonitoringRunResult`.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from signal_factories import (
    g4_error_assessment,
    performance_assessment,
    primary_group,
    primary_groups,
    signal_chain,
)
from vigiscoutx.config.scientific_config import (
    RECOMMENDATION_POLICY_REF,
    RISK_POLICY_REF,
    PolicyRef,
    RelevanceConfig,
)
from vigiscoutx.domain.assessments import (
    AnalysisLevel,
    GroupAssessment,
    LabelAwareAssessment,
    LabelAwareFailure,
    LabelAwareRiskState,
    PerformanceAssessment,
    PrimaryRiskAssessment,
    PrimaryRiskFailure,
    PrimaryRiskState,
    Recommendation,
    SignalResult,
    SignalStatus,
)
from vigiscoutx.domain.refs import EvidenceRef, relevance_evidence, window_evidence
from vigiscoutx.domain.types import PerformanceMetric, SignalGroup
from vigiscoutx.errors import PolicyResolutionError
from vigiscoutx.monitoring import MonitoringRunResult
from vigiscoutx.monitoring.run_result import resolve_run_evidence
from vigiscoutx.recommendations.engine import primary_recommendation, secondary_recommendation
from vigiscoutx.recommendations.policy import RECOMMENDATION_POLICY_V1
from vigiscoutx.risk.label_aware import assess_label_aware, relevance_config_sha256
from vigiscoutx.risk.mapper import assess_primary
from vigiscoutx.risk.policy import RISK_POLICY_V1

G1, G2, G3 = SignalGroup.G1_INPUT, SignalGroup.G2_DATA_QUALITY, SignalGroup.G3_PREDICTIVE
RELEVANCE = RelevanceConfig(
    performance_metric=PerformanceMetric.BALANCED_ACCURACY, clean_quantile=0.95, delta_rel=0.1
)
RELEVANCE_SHA = relevance_config_sha256(RELEVANCE)
RISK, REC = RISK_POLICY_V1, RECOMMENDATION_POLICY_V1
NE = SignalStatus.NOT_EVALUABLE


def _refs(window_id: str, *groups: SignalGroup) -> tuple[EvidenceRef, ...]:
    return tuple(window_evidence("group_assessment", window_id, g.value) for g in groups)


def _primary_ref(window_id: str, kind: str = "primary_risk_assessment") -> EvidenceRef:
    return EvidenceRef(kind=kind, ref_id=f"{kind}:{window_id}:primary")  # type: ignore[arg-type]


def _with_window(performance: PerformanceAssessment, window_id: str) -> PerformanceAssessment:
    monitoring = performance.monitoring.model_copy(update={"window_id": window_id})
    return performance.model_copy(update={"monitoring": monitoring})


def _run(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "run_id": "run",
        "config_sha256": "0" * 64,
        "freeze_id": None,
        "risk_policy": RISK_POLICY_REF,
        "recommendation_policy": RECOMMENDATION_POLICY_REF,
        "relevance": RELEVANCE,
        "signal_results": (),
        "group_assessments": (),
        "primary_assessments": (),
        "primary_failures": (),
        "label_aware_assessments": (),
        "label_aware_failures": (),
        "recommendations": (),
        "artifact_manifest_sha256": None,
    }
    return {**base, **fields}


def _valid_run() -> dict[str, Any]:
    """Sieben Windows: R0, R2+R3, R1+G4-ERROR, R1+G4 nicht relevant, NOT_EVALUABLE, fehlende
    Gruppe und windowfremde Gruppe (beide fail-closed als Vertragsverletzung)."""
    signals: list[SignalResult] = []
    groups: list[GroupAssessment] = []
    chains = [
        signal_chain((False, False, False), "w0"),
        signal_chain((True, True, False), "w3"),
        signal_chain((True, False, False), "wg"),
        signal_chain((False, True, False), "wv"),
        signal_chain(window_id="wn", statuses=(NE, SignalStatus.VALID, SignalStatus.VALID)),
        signal_chain(window_id="wc", groups=(G1, G2)),  # G3 fehlt
        signal_chain(window_id="wx", groups=(G1,)),  # G1 im fremden Window
        signal_chain(window_id="wf", groups=(G2, G3)),
    ]
    for chain_signals, chain_groups in chains:
        signals += chain_signals
        groups += chain_groups
    by_window: dict[str, list[GroupAssessment]] = {}
    for group in groups:
        by_window.setdefault(group.window_id, []).append(group)
    primary = {w: assess_primary(w, gs, RISK) for w, gs in by_window.items()}
    secondary = {
        "w3": assess_label_aware(
            primary["w3"], performance_assessment(True, window_id="w3"), RELEVANCE, RISK
        ),
        "wg": assess_label_aware(
            primary["wg"], g4_error_assessment(window_id="wg"), RELEVANCE, RISK
        ),
        "wv": assess_label_aware(
            primary["wv"], performance_assessment(False, window_id="wv"), RELEVANCE, RISK
        ),
    }
    recommendations = [primary_recommendation(o, REC) for o in primary.values()]
    for outcome in secondary.values():
        derived = secondary_recommendation(outcome, REC)
        if derived is not None:
            recommendations.append(derived)
    return _run(
        signal_results=tuple(signals),
        group_assessments=tuple(groups),
        primary_assessments=tuple(
            o for o in primary.values() if isinstance(o, PrimaryRiskAssessment)
        ),
        primary_failures=tuple(o for o in primary.values() if isinstance(o, PrimaryRiskFailure)),
        label_aware_assessments=tuple(
            o for o in secondary.values() if isinstance(o, LabelAwareAssessment)
        ),
        label_aware_failures=tuple(
            o for o in secondary.values() if isinstance(o, LabelAwareFailure)
        ),
        recommendations=tuple(recommendations),
    )


def _by_window(data: dict[str, Any], field: str, window_id: str) -> Any:
    return next(item for item in data[field] if item.window_id == window_id)


def _replace(data: dict[str, Any], field: str, window_id: str, item: Any) -> dict[str, Any]:
    kept = tuple(x for x in data[field] if x.window_id != window_id)
    return {**data, field: (*kept, item)}


def _replace_recommendation(
    data: dict[str, Any], window_id: str, level: AnalysisLevel, item: Any | None
) -> dict[str, Any]:
    kept = tuple(r for r in data["recommendations"] if (r.window_id, r.level) != (window_id, level))
    return {**data, "recommendations": kept if item is None else (*kept, item)}


def _with_chain(
    data: dict[str, Any], window_id: str, alarms: tuple[bool, bool, bool]
) -> dict[str, Any]:
    """Ersetzt SignalResults und Gruppen eines Windows durch eine konsistente neue Kette."""
    chain_signals, chain_groups = signal_chain(alarms, window_id)
    return {
        **data,
        "signal_results": (
            *(x for x in data["signal_results"] if x.monitoring_window_id != window_id),
            *chain_signals,
        ),
        "group_assessments": (
            *(g for g in data["group_assessments"] if g.window_id != window_id),
            *chain_groups,
        ),
    }


def _rejected(data: dict[str, Any], match: str | None = None) -> None:
    with pytest.raises(ValidationError, match=match):
        MonitoringRunResult.model_validate(data)


# --- Positivfall ---------------------------------------------------------------------------------


def test_valid_multi_window_run_resolves_each_reference_once_in_its_own_window() -> None:
    run = MonitoringRunResult.model_validate(_valid_run())
    assert {(r.window_id, r.level.value) for r in run.recommendations} == {
        *((w, "PRIMARY") for w in ("w0", "w3", "wg", "wv", "wn", "wc", "wx", "wf")),
        ("w3", "SECONDARY"),
        ("wg", "SECONDARY"),
    }
    owners: list[tuple[str, tuple[EvidenceRef, ...]]] = [
        *((o.window_id, o.evidence_refs) for o in run.primary_assessments),
        *((o.window_id, o.evidence_refs) for o in run.primary_failures),
        *((o.window_id, o.evidence_refs) for o in run.recommendations),
    ]
    for owner_window, refs in owners:
        for ref in refs:
            if ref.kind != "relevance_config":
                assert ref.ref_id.split(":")[1] == owner_window
    resolve_run_evidence(run)


def test_missing_and_foreign_window_groups_stay_traceable_fail_closed() -> None:
    run = MonitoringRunResult.model_validate(_valid_run())
    failures = {f.window_id: f for f in run.primary_failures}
    missing = failures["wc"]
    assert missing.contract_violation and missing.status is SignalStatus.ERROR
    assert missing.invalid_groups == (G3,) and missing.evidence_refs == _refs("wc", G1, G2)
    # Windowfremdes G1 (`wx`) wird nie im Window `wf` referenziert: beide Windows fail-closed.
    assert failures["wf"].invalid_groups == (G1,)
    assert failures["wf"].evidence_refs == _refs("wf", G2, G3)
    assert failures["wx"].invalid_groups == (G2, G3)
    assert failures["wx"].evidence_refs == _refs("wx", G1)
    for window in ("wc", "wf", "wx"):
        rec = next(r for r in run.recommendations if r.window_id == window)
        assert rec.reason_codes == ("RSN_REQUIRED_INPUT_ERROR", "RSN_POLICY_INPUT_CONTRACT_ERROR")


# --- HIGH-1: eindeutige, windowgebundene Evidence ----------------------------------------------


def test_two_g1_results_with_the_same_identity_are_rejected() -> None:
    data = _valid_run()
    twin = primary_group(G1, window_id="w0", alarm=True)
    _rejected({**data, "group_assessments": (*data["group_assessments"], twin)}, "Mehrfachbelegung")
    exact = _by_window(data, "group_assessments", "w0")
    _rejected({**data, "group_assessments": (*data["group_assessments"], exact)})


def test_one_reference_hitting_two_objects_is_ambiguous_not_lenient() -> None:
    data = _valid_run()
    twin = primary_group(G1, window_id="wc")
    tampered = MonitoringRunResult.model_construct(
        **{**data, "group_assessments": (*data["group_assessments"], twin)}
    )
    with pytest.raises(ValueError, match="mehrdeutig"):
        resolve_run_evidence(tampered)


def test_contract_failure_with_duplicate_groups_never_cites_an_ambiguous_reference() -> None:
    groups = (*primary_groups(window_id="w0"), primary_group(G1, window_id="w0", alarm=True))
    failure = assess_primary("w0", groups, RISK)
    assert isinstance(failure, PrimaryRiskFailure) and failure.contract_violation
    assert failure.invalid_groups == (G1,)
    assert failure.evidence_refs == _refs("w0", G2, G3)
    with pytest.raises(ValidationError, match="eindeutige Gruppenreferenzen"):
        failure.model_copy(update={"evidence_refs": _refs("w0", G1, G2, G3)})
    data = _valid_run()
    data = {
        **data,
        "group_assessments": (*data["group_assessments"], groups[-1]),
        "primary_assessments": tuple(a for a in data["primary_assessments"] if a.window_id != "w0"),
        "primary_failures": (*data["primary_failures"], failure),
    }
    data = _replace_recommendation(
        data, "w0", AnalysisLevel.PRIMARY, primary_recommendation(failure, REC)
    )
    _rejected(data, "Mehrfachbelegung")


def test_fabricated_missing_group_contract_failure_is_rejected() -> None:
    data = _valid_run()
    claimed = PrimaryRiskFailure(
        window_id="wc",
        status=SignalStatus.ERROR,
        contract_violation=True,
        invalid_groups=(G2,),
        evidence_refs=_refs("wc", G1, G3),
    )
    _rejected(_replace(data, "primary_failures", "wc", claimed), "folgt nicht aus Gruppen")


def test_recommendation_for_window_a_citing_only_window_b_is_rejected() -> None:
    data = _valid_run()
    foreign = (_primary_ref("w3"), *_refs("w3", G1, G2, G3))
    with pytest.raises(ValidationError, match="fremdem Window"):
        Recommendation(
            window_id="w0",
            level=AnalysisLevel.PRIMARY,
            code="REC_CONTINUE_MONITORING",
            reason_codes=("RSN_R0_NO_GROUP_ALARM",),
            evidence_refs=foreign,
        )
    bypass = Recommendation.model_construct(
        window_id="w0",
        level=AnalysisLevel.PRIMARY,
        code="REC_CONTINUE_MONITORING",
        reason_codes=("RSN_R0_NO_GROUP_ALARM",),
        evidence_refs=foreign,
    )
    _rejected(_replace_recommendation(data, "w0", AnalysisLevel.PRIMARY, bypass), "fremdem Window")
    tampered = MonitoringRunResult.model_construct(
        **_replace_recommendation(data, "w0", AnalysisLevel.PRIMARY, bypass)
    )
    with pytest.raises(ValueError, match="fremdem Window"):
        resolve_run_evidence(tampered)


@pytest.mark.parametrize(
    ("ref", "match"),
    [
        pytest.param(
            EvidenceRef(kind="group_assessment", ref_id="group_assessment:w0:G9"),
            "nicht auflösbar",
            id="unknown",
        ),
        pytest.param(
            EvidenceRef(kind="primary_risk_failure", ref_id="primary_risk_failure:w0:primary"),
            "nicht auflösbar",
            id="wrong-kind-for-window",
        ),
        pytest.param(
            EvidenceRef(kind="primary_risk_failure", ref_id="primary_risk_assessment:w0:primary"),
            "Ergebnisschema",
            id="kind-prefix-mismatch",
        ),
        pytest.param(relevance_evidence("f" * 64), "nicht auflösbar", id="unbound-relevance"),
    ],
)
def test_unknown_or_mistyped_evidence_is_rejected(ref: EvidenceRef, match: str) -> None:
    data = _valid_run()
    rec = _by_window(data, "recommendations", "w0")
    bypass = Recommendation.model_construct(
        **{**dict(rec), "evidence_refs": (*rec.evidence_refs, ref)}
    )
    tampered = _replace_recommendation(data, "w0", AnalysisLevel.PRIMARY, bypass)
    with pytest.raises(ValueError, match=match):
        resolve_run_evidence(MonitoringRunResult.model_construct(**tampered))
    _rejected(tampered)


# --- HIGH-2: semantische Konsistenz (fünf reproduzierte Widersprüche) ----------------------------


def test_r0_although_g1_alarms_is_rejected() -> None:
    data = _with_chain(_valid_run(), "w0", (True, False, False))
    contradiction = PrimaryRiskAssessment(
        window_id="w0",
        reference_window_id="r",
        state=PrimaryRiskState.R0,
        flagged_groups=(),
        evidence_refs=_refs("w0", G1, G2, G3),
    )
    _rejected(data, "folgt nicht aus Gruppen")
    _rejected(_replace(data, "primary_assessments", "w0", contradiction), "folgt nicht aus Gruppen")


def test_primary_error_although_all_groups_valid_is_rejected() -> None:
    data = _valid_run()
    contradiction = PrimaryRiskFailure(
        window_id="w0",
        status=SignalStatus.ERROR,
        contract_violation=False,
        invalid_groups=(G1,),
        evidence_refs=_refs("w0", G1, G2, G3),
    )
    data = {
        **data,
        "primary_assessments": tuple(a for a in data["primary_assessments"] if a.window_id != "w0"),
        "primary_failures": (*data["primary_failures"], contradiction),
    }
    rec = Recommendation(
        window_id="w0",
        level=AnalysisLevel.PRIMARY,
        code="REC_TECHNICAL_REVIEW",
        reason_codes=("RSN_REQUIRED_INPUT_ERROR",),
        evidence_refs=(_primary_ref("w0", "primary_risk_failure"), *_refs("w0", G1, G2, G3)),
    )
    _rejected(_replace_recommendation(data, "w0", AnalysisLevel.PRIMARY, rec), "folgt nicht")


def test_not_evaluable_label_aware_failure_despite_valid_relevant_g4_is_rejected() -> None:
    data = _valid_run()
    r3 = _by_window(data, "label_aware_assessments", "w3")
    contradiction = LabelAwareFailure(
        window_id="w3",
        primary=r3.primary,
        performance=r3.performance,
        status=SignalStatus.NOT_EVALUABLE,
        contract_violation=False,
        evidence_refs=(
            window_evidence("primary_risk_assessment", "w3", "primary"),
            window_evidence("performance_assessment", "w3", "G4"),
        ),
    )
    rec = Recommendation(
        window_id="w3",
        level=AnalysisLevel.SECONDARY,
        code="REC_REVIEW_EVALUABILITY",
        reason_codes=("RSN_G4_NOT_EVALUABLE",),
        evidence_refs=(
            window_evidence("label_aware_failure", "w3", "secondary"),
            window_evidence("performance_assessment", "w3", "G4"),
        ),
    )
    data = {
        **data,
        "label_aware_assessments": tuple(
            a for a in data["label_aware_assessments"] if a.window_id != "w3"
        ),
        "label_aware_failures": (*data["label_aware_failures"], contradiction),
    }
    _rejected(
        _replace_recommendation(data, "w3", AnalysisLevel.SECONDARY, rec),
        "sekundäres Ergebnis w3 folgt nicht",
    )


def test_r2_recommendation_for_r0_assessment_is_rejected() -> None:
    data = _valid_run()
    rec = Recommendation(
        window_id="w0",
        level=AnalysisLevel.PRIMARY,
        code="REC_CAUSE_ANALYSIS",
        reason_codes=("RSN_R2_MULTI_GROUP_ALARM",),
        evidence_refs=(_primary_ref("w0"), *_refs("w0", G1, G2, G3)),
    )
    _rejected(_replace_recommendation(data, "w0", AnalysisLevel.PRIMARY, rec), "weicht von der")


def test_r3_without_mandatory_secondary_recommendation_is_rejected() -> None:
    data = _replace_recommendation(_valid_run(), "w3", AnalysisLevel.SECONDARY, None)
    _rejected(data, "erforderliche Empfehlung fehlt")


# --- HIGH-2: weitere Konsistenzregeln -----------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "reasons"),
    [
        ("REC_INVENTED_ACTION", ("RSN_R0_NO_GROUP_ALARM",)),
        ("REC_CONTINUE_MONITORING", ("RSN_R1_SINGLE_GROUP_ALARM",)),
        ("REC_FURTHER_MONITORING", ("RSN_R0_NO_GROUP_ALARM",)),
    ],
)
def test_invented_or_swapped_codes_are_rejected(code: str, reasons: tuple[str, ...]) -> None:
    data = _valid_run()
    rec = _by_window(data, "recommendations", "w0").model_copy(
        update={"code": code, "reason_codes": reasons}
    )
    _rejected(_replace_recommendation(data, "w0", AnalysisLevel.PRIMARY, rec), "weicht von der")


def test_secondary_recommendation_for_valid_g4_without_relevance_is_rejected() -> None:
    data = _valid_run()
    not_relevant = _by_window(data, "label_aware_assessments", "wv")
    assert not_relevant.state is LabelAwareRiskState.R1  # Primary-Zustand erhalten
    rec = Recommendation(
        window_id="wv",
        level=AnalysisLevel.SECONDARY,
        code="REC_ESCALATE_REVIEW",
        reason_codes=("RSN_R3_RELEVANT_DEGRADATION_CONFIRMED",),
        evidence_refs=(
            window_evidence("label_aware_assessment", "wv", "secondary"),
            window_evidence("performance_assessment", "wv", "G4"),
            relevance_evidence(RELEVANCE_SHA),
        ),
    )
    _rejected(_replace_recommendation(data, "wv", AnalysisLevel.SECONDARY, rec), "ohne ableitbares")


def test_g4_error_can_never_yield_a_valid_r3() -> None:
    data = _valid_run()
    failure = _by_window(data, "label_aware_failures", "wg")
    with pytest.raises(ValidationError):
        LabelAwareAssessment(
            window_id="wg",
            primary=failure.primary,
            performance=failure.performance,
            relevance_config_sha256=RELEVANCE_SHA,
            state=LabelAwareRiskState.R3,
            evidence_refs=(
                window_evidence("primary_risk_assessment", "wg", "primary"),
                window_evidence("performance_assessment", "wg", "G4"),
                relevance_evidence(RELEVANCE_SHA),
            ),
        )
    bypass = LabelAwareAssessment.model_construct(
        window_id="wg",
        primary=failure.primary,
        performance=failure.performance,
        relevance_config_sha256=RELEVANCE_SHA,
        state=LabelAwareRiskState.R3,
        evidence_refs=(
            window_evidence("primary_risk_assessment", "wg", "primary"),
            window_evidence("performance_assessment", "wg", "G4"),
            relevance_evidence(RELEVANCE_SHA),
        ),
    )
    data = {
        **data,
        "label_aware_failures": tuple(
            f for f in data["label_aware_failures"] if f.window_id != "wg"
        ),
        "label_aware_assessments": (*data["label_aware_assessments"], bypass),
    }
    _rejected(data)


def test_secondary_must_embed_the_unchanged_primary_result() -> None:
    data = _valid_run()
    r3 = _by_window(data, "label_aware_assessments", "w3")
    altered = r3.primary.model_copy(update={"state": PrimaryRiskState.R1, "flagged_groups": (G1,)})
    forged = assess_label_aware(altered, r3.performance, RELEVANCE, RISK)
    _rejected(_replace(data, "label_aware_assessments", "w3", forged), "folgt nicht aus Primary")


def test_secondary_without_valid_primary_or_relevance_is_rejected() -> None:
    data = _valid_run()
    _rejected({**data, "relevance": None}, "Relevanzkonfiguration")
    without_primary = {
        **data,
        "primary_assessments": tuple(a for a in data["primary_assessments"] if a.window_id != "w3"),
    }
    _rejected(without_primary, "genau ein Primary-Ergebnis")


def test_nested_model_construct_bypass_is_revalidated() -> None:
    data = _valid_run()
    bypass = PrimaryRiskAssessment.model_construct(
        window_id="w0",
        reference_window_id="r",
        state=PrimaryRiskState.R0,
        flagged_groups=(G1,),
        evidence_refs=_refs("w0", G1, G2, G3),
    )
    _rejected(_replace(data, "primary_assessments", "w0", bypass))
    inconsistent = GroupAssessment.model_construct(
        **{**dict(_by_window(data, "group_assessments", "w0")), "alarm": None}
    )
    groups = tuple(g for g in data["group_assessments"] if (g.window_id, g.group) != ("w0", G1))
    _rejected({**data, "group_assessments": (*groups, inconsistent)})


def test_secondary_window_mismatch_follows_dec_077_and_stays_window_bound() -> None:
    data = _valid_run()
    primary = _by_window(data, "primary_assessments", "wg")
    failure = assess_label_aware(
        primary, _with_window(g4_error_assessment(), "other"), RELEVANCE, RISK
    )
    assert isinstance(failure, LabelAwareFailure) and failure.contract_violation
    assert failure.evidence_refs[1] == window_evidence("performance_assessment", "wg", "G4")
    data = _replace(data, "label_aware_failures", "wg", failure)
    rec = secondary_recommendation(failure, REC)
    assert rec is not None
    assert rec.reason_codes == ("RSN_G4_ERROR", "RSN_POLICY_INPUT_CONTRACT_ERROR")
    run = MonitoringRunResult.model_validate(
        _replace_recommendation(data, "wg", AnalysisLevel.SECONDARY, rec)
    )
    assert _by_window(dict(run), "primary_assessments", "wg") == primary


@pytest.mark.parametrize("field", ["risk_policy", "recommendation_policy"])
def test_run_binds_only_frozen_policies(field: str) -> None:
    data = {**_valid_run(), field: PolicyRef(policy_id="risk-test", version="1")}
    with pytest.raises(PolicyResolutionError):
        MonitoringRunResult.model_validate(data)
