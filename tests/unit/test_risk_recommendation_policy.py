"""Risk-Policy v1.0.0 und Recommendation-Policy v1.0.0 (P4-03B; DEC-074 – DEC-076).

Ausschließlich synthetische Fixtures. Prüft RiskMapper, R3-Gate, RecommendationEngine,
typisierte Failure-Contracts, auflösbare Evidence-Referenzen und die `PolicyRef`-Bindung.
"""

import ast
import copy
import itertools
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import vigiscoutx
from signal_factories import (
    g4_error_assessment,
    g4_not_evaluable_assessment,
    performance_assessment,
    primary_group,
    primary_groups,
)
from vigiscoutx.config.scientific_config import PolicyRef, RelevanceConfig
from vigiscoutx.domain.assessments import (
    AnalysisLevel,
    GroupAssessment,
    LabelAwareAssessment,
    LabelAwareFailure,
    LabelAwareRiskState,
    PrimaryRiskAssessment,
    PrimaryRiskFailure,
    PrimaryRiskState,
    Recommendation,
    SignalStatus,
)
from vigiscoutx.domain.refs import EvidenceRef, window_evidence
from vigiscoutx.domain.types import PerformanceMetric, SignalGroup
from vigiscoutx.errors import PolicyResolutionError
from vigiscoutx.experiment.calibration.config import FusionCandidate
from vigiscoutx.experiment.calibration.statistics import fusion_alarm
from vigiscoutx.experiment.freeze import build_freeze_manifest_v2
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.recommendations.engine import primary_recommendation, secondary_recommendation
from vigiscoutx.recommendations.policy import (
    ACTION_CODES,
    RECOMMENDATION_POLICY_REF,
    RECOMMENDATION_POLICY_V1,
    resolve_configured_policies,
    resolve_recommendation_policy,
)
from vigiscoutx.risk.label_aware import assess_label_aware, relevance_config_sha256
from vigiscoutx.risk.mapper import assess_primary
from vigiscoutx.risk.policy import RISK_POLICY_REF, RISK_POLICY_V1, resolve_risk_policy

G1, G2, G3 = SignalGroup.G1_INPUT, SignalGroup.G2_DATA_QUALITY, SignalGroup.G3_PREDICTIVE
RELEVANCE = RelevanceConfig(
    performance_metric=PerformanceMetric.BALANCED_ACCURACY, clean_quantile=0.95, delta_rel=0.1
)
RISK, REC = RISK_POLICY_V1, RECOMMENDATION_POLICY_V1
SRC = Path(vigiscoutx.__file__).parent


def _assess(groups: tuple[GroupAssessment, ...], window_id: str = "w") -> Any:
    return assess_primary(window_id, groups, RISK)


# --- RiskMapper: R0/R1/R2 ----------------------------------------------------------------------


@pytest.mark.parametrize("alarms", list(itertools.product([False, True], repeat=3)))
def test_all_eight_alarm_combinations(alarms: tuple[bool, bool, bool]) -> None:
    result = _assess(primary_groups(alarms))
    assert isinstance(result, PrimaryRiskAssessment)
    k = sum(alarms)
    assert result.state is {0: PrimaryRiskState.R0, 1: PrimaryRiskState.R1}.get(
        k, PrimaryRiskState.R2
    )
    assert result.flagged_groups == tuple(g for g, a in zip((G1, G2, G3), alarms, strict=True) if a)
    assert result.reference_window_id == "r"
    assert result.evidence_refs == tuple(
        window_evidence("group_assessment", "w", g.value) for g in (G1, G2, G3)
    )


@pytest.mark.parametrize("alarms", list(itertools.product([False, True], repeat=3)))
def test_risk_state_is_independent_of_fusion_candidate(alarms: tuple[bool, bool, bool]) -> None:
    groups = primary_groups(alarms)
    state = _assess(groups).state
    fusion_a = fusion_alarm(FusionCandidate.FUSION_A, groups)
    fusion_b = fusion_alarm(FusionCandidate.FUSION_B, groups)
    assert state is _assess(groups).state  # deterministisch, ohne Fusionsinput
    if sum(alarms) == 1:
        assert fusion_a and not fusion_b and state is PrimaryRiskState.R1
    if sum(alarms) == 0:
        assert not fusion_a and state is PrimaryRiskState.R0


def test_mapper_input_order_does_not_matter() -> None:
    groups = primary_groups((True, False, True))
    assert _assess(tuple(reversed(groups))) == _assess(groups)


# --- RiskMapper: ungültige Gruppen und Vertragsverletzungen --------------------------------------


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ((SignalStatus.ERROR, SignalStatus.VALID, SignalStatus.VALID), SignalStatus.ERROR),
        (
            (SignalStatus.VALID, SignalStatus.NOT_EVALUABLE, SignalStatus.VALID),
            SignalStatus.NOT_EVALUABLE,
        ),
        ((SignalStatus.NOT_EVALUABLE, SignalStatus.ERROR, SignalStatus.VALID), SignalStatus.ERROR),
        ((SignalStatus.NOT_EVALUABLE,) * 3, SignalStatus.NOT_EVALUABLE),
    ],
)
def test_invalid_groups_never_yield_a_risk_state(
    statuses: tuple[SignalStatus, ...], expected: SignalStatus
) -> None:
    groups = tuple(
        primary_group(g, status=s, alarm=True) for g, s in zip((G1, G2, G3), statuses, strict=True)
    )
    result = _assess(groups)
    assert isinstance(result, PrimaryRiskFailure)
    assert result.status is expected and not result.contract_violation
    assert result.invalid_groups == tuple(
        g for g, s in zip((G1, G2, G3), statuses, strict=True) if s is not SignalStatus.VALID
    )
    assert not hasattr(result, "state")


def _foreign_group() -> GroupAssessment:
    valid = primary_group(G1)
    return GroupAssessment.model_construct(**{**dict(valid), "group": SignalGroup.G4_PERFORMANCE})


def _inconsistent_group() -> GroupAssessment:
    valid = primary_group(G2)
    return GroupAssessment.model_construct(**{**dict(valid), "alarm": None})


@pytest.mark.parametrize(
    "groups",
    [
        pytest.param(primary_groups()[:2], id="missing"),
        pytest.param((*primary_groups(), primary_group(G1)), id="duplicate"),
        pytest.param(
            (primary_groups()[0], primary_groups()[0], primary_groups()[2]), id="dup-g1-no-g2"
        ),
        pytest.param((_foreign_group(), *primary_groups()[1:]), id="foreign-identity"),
        pytest.param((primary_group(G1, window_id="other"), *primary_groups()[1:]), id="window"),
        pytest.param(
            (primary_group(G1, reference_window_id="r2"), *primary_groups()[1:]), id="reference"
        ),
        pytest.param(
            (primary_groups()[0], _inconsistent_group(), primary_groups()[2]), id="status"
        ),
        pytest.param((), id="empty"),
    ],
)
def test_contract_violations_are_errors_without_state(groups: tuple[GroupAssessment, ...]) -> None:
    result = _assess(groups)
    assert isinstance(result, PrimaryRiskFailure)
    assert result.status is SignalStatus.ERROR and result.contract_violation
    recommendation = primary_recommendation(result, REC)
    assert recommendation.code == "REC_TECHNICAL_REVIEW"
    assert recommendation.reason_codes == (
        "RSN_REQUIRED_INPUT_ERROR",
        "RSN_POLICY_INPUT_CONTRACT_ERROR",
    )


def test_primary_contracts_cannot_express_r3_or_carry_g4_evidence() -> None:
    assert "R3" not in {s.value for s in PrimaryRiskState}
    perf = window_evidence("performance_assessment", "w", "G4")
    valid = _assess(primary_groups())
    with pytest.raises(ValidationError, match="Performance-Evidenz"):
        valid.model_copy(update={"evidence_refs": (*valid.evidence_refs, perf)})
    failure = _assess(primary_groups()[:2])
    with pytest.raises(ValidationError, match="Performance-Evidenz"):
        failure.model_copy(update={"evidence_refs": (perf,)})
    with pytest.raises(ValidationError):
        PrimaryRiskFailure.model_validate(
            {**failure.model_dump(), "status": SignalStatus.NOT_EVALUABLE}
        )  # Vertragsverletzung ist immer ERROR


def test_primary_mapper_imports_no_label_aware_or_fusion_module() -> None:
    tree = ast.parse((SRC / "risk" / "mapper.py").read_text(encoding="utf-8"))
    modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = ("performance", "label_aware", "experiment", "calibration", "fusion")
    assert not [m for m in modules if any(f in m for f in forbidden)]


# --- Label-aware Analyse (R3-Gate) ---------------------------------------------------------------


@pytest.mark.parametrize(
    "alarms", [(False, False, False), (True, False, False), (True, True, True)]
)
def test_r3_only_with_valid_relevant_g4_else_primary_state_preserved(
    alarms: tuple[bool, bool, bool],
) -> None:
    primary = _assess(primary_groups(alarms))
    relevant = assess_label_aware(primary, performance_assessment(True), RELEVANCE, RISK)
    assert isinstance(relevant, LabelAwareAssessment)
    assert relevant.state is LabelAwareRiskState.R3 and relevant.primary == primary
    assert relevant.relevance_config_sha256 == relevance_config_sha256(RELEVANCE)
    not_relevant = assess_label_aware(primary, performance_assessment(False), RELEVANCE, RISK)
    assert isinstance(not_relevant, LabelAwareAssessment)
    assert not_relevant.state.value == primary.state.value
    assert secondary_recommendation(not_relevant, REC) is None


@pytest.mark.parametrize(
    ("performance", "status", "code", "reasons"),
    [
        (g4_error_assessment, SignalStatus.ERROR, "REC_TECHNICAL_REVIEW", ("RSN_G4_ERROR",)),
        (
            g4_not_evaluable_assessment,
            SignalStatus.NOT_EVALUABLE,
            "REC_REVIEW_EVALUABILITY",
            ("RSN_G4_NOT_EVALUABLE",),
        ),
    ],
)
def test_g4_failures_are_separate_and_never_change_primary(
    performance: Any, status: SignalStatus, code: str, reasons: tuple[str, ...]
) -> None:
    primary = _assess(primary_groups((True, False, False)))
    before = primary_recommendation(primary, REC)
    secondary = assess_label_aware(primary, performance(), RELEVANCE, RISK)
    assert isinstance(secondary, LabelAwareFailure)
    assert secondary.status is status and not secondary.contract_violation
    assert secondary.primary == primary and primary.state is PrimaryRiskState.R1
    assert primary_recommendation(primary, REC) == before
    recommendation = secondary_recommendation(secondary, REC)
    assert recommendation is not None and recommendation.level is AnalysisLevel.SECONDARY
    assert (recommendation.code, recommendation.reason_codes) == (code, reasons)


def test_missing_g4_or_invalid_primary_yield_no_secondary_result() -> None:
    primary = _assess(primary_groups())
    assert assess_label_aware(primary, None, RELEVANCE, RISK) is None
    failure = _assess(primary_groups()[:2])
    assert assess_label_aware(failure, performance_assessment(True), RELEVANCE, RISK) is None


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            lambda: (performance_assessment(True), RELEVANCE.model_copy(update={"delta_rel": 0.2})),
            id="delta-rel",
        ),
        pytest.param(
            lambda: (
                performance_assessment(True),
                RELEVANCE.model_copy(update={"delta_rel": None}),
            ),
            id="unbound",
        ),
        pytest.param(
            lambda: (performance_assessment(True, window_id="other"), RELEVANCE), id="window"
        ),
        pytest.param(
            lambda: (performance_assessment(True, delta_rel=None), RELEVANCE),
            id="relevance-undetermined",
        ),
    ],
)
def test_secondary_contract_violations(case: Any) -> None:
    performance, relevance = case()
    primary = _assess(primary_groups((True, True, False)))
    secondary = assess_label_aware(primary, performance, relevance, RISK)
    assert isinstance(secondary, LabelAwareFailure)
    assert secondary.status is SignalStatus.ERROR and secondary.contract_violation
    assert secondary.primary == primary
    recommendation = secondary_recommendation(secondary, REC)
    assert recommendation is not None
    assert recommendation.code == "REC_TECHNICAL_REVIEW"
    assert recommendation.reason_codes == ("RSN_G4_ERROR", "RSN_POLICY_INPUT_CONTRACT_ERROR")
    assert primary_recommendation(primary, REC).code == "REC_CAUSE_ANALYSIS"


# --- RecommendationEngine ------------------------------------------------------------------------


def test_all_action_codes_and_reasons() -> None:
    seen = {}
    for alarms, code, reason in (
        ((False, False, False), "REC_CONTINUE_MONITORING", "RSN_R0_NO_GROUP_ALARM"),
        ((True, False, False), "REC_FURTHER_MONITORING", "RSN_R1_SINGLE_GROUP_ALARM"),
        ((True, True, False), "REC_CAUSE_ANALYSIS", "RSN_R2_MULTI_GROUP_ALARM"),
    ):
        rec = primary_recommendation(_assess(primary_groups(alarms)), REC)
        assert (rec.code, rec.reason_codes, rec.level) == (code, (reason,), AnalysisLevel.PRIMARY)
        seen[code] = rec
    for status, code, reason in (
        (SignalStatus.NOT_EVALUABLE, "REC_REVIEW_EVALUABILITY", "RSN_REQUIRED_INPUT_NOT_EVALUABLE"),
        (SignalStatus.ERROR, "REC_TECHNICAL_REVIEW", "RSN_REQUIRED_INPUT_ERROR"),
    ):
        groups = (primary_group(G1, status=status, alarm=None), *primary_groups()[1:])
        rec = primary_recommendation(_assess(groups), REC)
        assert (rec.code, rec.reason_codes) == (code, (reason,))
        seen[code] = rec
    r3 = assess_label_aware(
        _assess(primary_groups()), performance_assessment(True), RELEVANCE, RISK
    )
    escalation = secondary_recommendation(r3, REC)
    assert escalation is not None
    rec = escalation
    assert (rec.code, rec.reason_codes) == (
        "REC_ESCALATE_REVIEW",
        ("RSN_R3_RELEVANT_DEGRADATION_CONFIRMED",),
    )
    seen[rec.code] = rec
    assert set(seen) == ACTION_CODES and len(ACTION_CODES) == 6


def test_r3_recommendation_references_g4_and_relevance_config() -> None:
    r3 = assess_label_aware(
        _assess(primary_groups()), performance_assessment(True), RELEVANCE, RISK
    )
    rec = secondary_recommendation(r3, REC)
    assert rec is not None
    assert [r.kind for r in rec.evidence_refs] == [
        "label_aware_assessment",
        "performance_assessment",
        "relevance_config",
    ]
    assert rec.evidence_refs[2].ref_id == f"relevance_config:{relevance_config_sha256(RELEVANCE)}"


def test_r0_is_proven_by_three_valid_non_alarming_groups() -> None:
    rec = primary_recommendation(_assess(primary_groups()), REC)
    assert [r.ref_id for r in rec.evidence_refs] == [
        "primary_risk_assessment:w:primary",
        "group_assessment:w:G1",
        "group_assessment:w:G2",
        "group_assessment:w:G3",
    ]


def test_primary_recommendation_rejects_g4_evidence() -> None:
    with pytest.raises(ValidationError, match="Performance-Evidenz"):
        Recommendation(
            window_id="w",
            level=AnalysisLevel.PRIMARY,
            code="REC_CONTINUE_MONITORING",
            reason_codes=("RSN_R0_NO_GROUP_ALARM",),
            evidence_refs=(window_evidence("performance_assessment", "w", "G4"),),
        )


# --- PolicyRef-Bindung ----------------------------------------------------------------------------


def test_policy_refs_are_exact_and_immutable() -> None:
    assert PolicyRef(policy_id="vsx-risk-r0-r3", version="1.0.0") == RISK_POLICY_REF
    assert (
        PolicyRef(policy_id="vsx-recommendation-review", version="1.0.0")
        == RECOMMENDATION_POLICY_REF
    )
    assert resolve_risk_policy(RISK_POLICY_REF) is RISK_POLICY_V1
    assert resolve_recommendation_policy(RECOMMENDATION_POLICY_REF) is RECOMMENDATION_POLICY_V1
    with pytest.raises(ValidationError):
        RISK_POLICY_V1.model_copy(update={"primary_states": RISK_POLICY_V1.primary_states[:2]})
    with pytest.raises(ValidationError):
        RECOMMENDATION_POLICY_V1.model_copy(update={"rules": RECOMMENDATION_POLICY_V1.rules[1:]})


@pytest.mark.parametrize(
    "ref",
    [
        PolicyRef(policy_id="vsx-risk-r0-r3", version="1.0.1"),
        PolicyRef(policy_id="vsx-risk-r0-r2", version="1.0.0"),
        PolicyRef(policy_id="dummy-policy", version="0"),
    ],
)
def test_unknown_risk_policy_refs_fail_hard(ref: PolicyRef) -> None:
    with pytest.raises(PolicyResolutionError):
        resolve_risk_policy(ref)


@pytest.mark.parametrize(
    "ref",
    [
        PolicyRef(policy_id="vsx-recommendation-review", version="2.0.0"),
        PolicyRef(policy_id="vsx-risk-r0-r3", version="1.0.0"),
    ],
)
def test_unknown_recommendation_policy_refs_fail_hard(ref: PolicyRef) -> None:
    with pytest.raises(PolicyResolutionError):
        resolve_recommendation_policy(ref)


def test_policies_are_compatible_with_run_config_and_freeze_manifest(
    calibrated_config_data: dict[str, Any],
) -> None:
    data = copy.deepcopy(calibrated_config_data)
    data["scientific"]["risk_policy"] = RISK_POLICY_REF.model_dump()
    data["scientific"]["recommendation_policy"] = RECOMMENDATION_POLICY_REF.model_dump()
    data["repetitions"] = 1
    config = RunConfig.model_validate(data)
    assert resolve_configured_policies(config.scientific) == (
        RISK_POLICY_V1,
        RECOMMENDATION_POLICY_V1,
    )
    manifest = build_freeze_manifest_v2(
        config,
        seed_list=(config.seeds,),
        git_commit="a" * 40,
        environment_lock_sha256="e" * 64,
        calibration_record_sha256="f" * 64,
    )
    assert (manifest.risk_policy, manifest.recommendation_policy) == (
        RISK_POLICY_REF,
        RECOMMENDATION_POLICY_REF,
    )
    data["scientific"]["risk_policy"] = {"policy_id": "risk-test", "version": "1"}
    with pytest.raises(PolicyResolutionError):
        resolve_configured_policies(RunConfig.model_validate(data).scientific)


def test_calibration_never_depends_on_risk_or_recommendation_policies() -> None:
    roots = [SRC / "experiment", SRC.parents[1] / "experiments" / "runners"]
    assert all(root.is_dir() for root in roots)
    policy_packages = ("vigiscoutx.risk", "vigiscoutx.recommendations", "vigiscoutx.monitoring")
    offending = []
    for root in roots:
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                if any(m.startswith(policy_packages) for m in modules):
                    offending.append(path.name)
    assert offending == []


def test_evidence_kinds_cover_risk_results() -> None:
    kinds = set(EvidenceRef.model_fields["kind"].annotation.__args__)  # type: ignore[union-attr]
    assert {"primary_risk_assessment", "primary_risk_failure", "label_aware_assessment"} <= kinds
    assert {"label_aware_failure", "relevance_config"} <= kinds
