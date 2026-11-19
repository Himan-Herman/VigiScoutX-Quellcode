"""P4-03C HIGH-4: SignalResults → GroupAssessments im offiziellen `MonitoringRunResult`.

Ausschließlich synthetische Fixtures. Widersprüchliche Signale und Gruppen werden direkt
konstruiert; jeder Negativfall scheitert über `MonitoringRunResult.model_validate`, nicht über
isolierte Aggregationstests. Gültige Basisergebnisse entstehen über die bestehende Aggregation.
"""

import itertools
from typing import Any

import pytest
from pydantic import ValidationError

from signal_factories import aggregate, group_signals, primary_groups, signal_chain, signal_result
from vigiscoutx.config.scientific_config import RECOMMENDATION_POLICY_REF, RISK_POLICY_REF
from vigiscoutx.domain.assessments import (
    AnalysisLevel,
    G3Substatus,
    GroupAssessment,
    PrimaryRiskAssessment,
    PrimaryRiskFailure,
    PrimaryRiskState,
    ReasonCode,
    Recommendation,
    SignalResult,
    SignalStatus,
)
from vigiscoutx.domain.refs import EvidenceRef, window_evidence
from vigiscoutx.domain.types import SignalGroup
from vigiscoutx.monitoring import MonitoringRunResult
from vigiscoutx.recommendations.engine import primary_recommendation
from vigiscoutx.recommendations.policy import RECOMMENDATION_POLICY_V1
from vigiscoutx.risk.mapper import assess_primary
from vigiscoutx.risk.policy import RISK_POLICY_V1
from vigiscoutx.signals.config import G1_SIGNAL_ID, G2_SIGNAL_IDS, G3_CONFIDENCE_SIGNAL_ID

G1, G2, G3 = SignalGroup.G1_INPUT, SignalGroup.G2_DATA_QUALITY, SignalGroup.G3_PREDICTIVE
G2_IDS = tuple(G2_SIGNAL_IDS.values())
VALID, NE, ERR = SignalStatus.VALID, SignalStatus.NOT_EVALUABLE, SignalStatus.ERROR


def _signal_ref(window_id: str, signal_id: str) -> EvidenceRef:
    return window_evidence("signal_result", window_id, signal_id)


def _run(signals: tuple[SignalResult, ...], groups: tuple[GroupAssessment, ...]) -> dict[str, Any]:
    """Run-Payload mit den aus den Gruppen abgeleiteten Primary-Ergebnissen und Empfehlungen."""
    by_window: dict[str, list[GroupAssessment]] = {}
    for group in groups:
        by_window.setdefault(group.window_id, []).append(group)
    primary = [assess_primary(w, gs, RISK_POLICY_V1) for w, gs in by_window.items()]
    return {
        "run_id": "run",
        "config_sha256": "0" * 64,
        "freeze_id": None,
        "risk_policy": RISK_POLICY_REF,
        "recommendation_policy": RECOMMENDATION_POLICY_REF,
        "relevance": None,
        "signal_results": signals,
        "group_assessments": groups,
        "primary_assessments": tuple(p for p in primary if isinstance(p, PrimaryRiskAssessment)),
        "primary_failures": tuple(p for p in primary if isinstance(p, PrimaryRiskFailure)),
        "label_aware_assessments": (),
        "label_aware_failures": (),
        "recommendations": tuple(
            primary_recommendation(p, RECOMMENDATION_POLICY_V1) for p in primary
        ),
        "artifact_manifest_sha256": None,
    }


def _replace_signal(
    signals: tuple[SignalResult, ...], window_id: str, signal_id: str, new: SignalResult | None
) -> tuple[SignalResult, ...]:
    kept = tuple(
        s for s in signals if (s.monitoring_window_id, s.signal_id) != (window_id, signal_id)
    )
    return kept if new is None else (*kept, new)


def _replace_group(
    groups: tuple[GroupAssessment, ...], new: GroupAssessment
) -> tuple[GroupAssessment, ...]:
    kept = tuple(g for g in groups if (g.window_id, g.group) != (new.window_id, new.group))
    return (*kept, new)


def _g1(window_id: str = "w0", **overrides: Any) -> SignalResult:
    data: dict[str, Any] = {
        "signal_id": G1_SIGNAL_ID,
        "group": G1,
        "monitoring_window_id": window_id,
        "reference_window_id": "r",
    }
    return signal_result(**{**data, **overrides})


def _group(group: SignalGroup, window_id: str = "w0", **overrides: Any) -> GroupAssessment:
    """Direkt konstruiertes, in sich gültiges GroupAssessment (unabhängig von der Aggregation)."""
    ids = {G1: (G1_SIGNAL_ID,), G2: G2_IDS, G3: ("G3_OUTPUT_MUKS", G3_CONFIDENCE_SIGNAL_ID)}[group]
    data: dict[str, Any] = {
        "group": group,
        "window_id": window_id,
        "reference_window_id": "r",
        "status": VALID,
        "alarm": False,
        "substatus": G3Substatus.NONE if group is G3 else None,
        "reason_code": None,
        "signal_ids": ids,
        "evidence_refs": tuple(_signal_ref(window_id, i) for i in ids),
    }
    return GroupAssessment.model_validate({**data, **overrides})


def _rejected(data: dict[str, Any], match: str | None = None) -> None:
    with pytest.raises(ValidationError, match=match):
        MonitoringRunResult.model_validate(data)


# --- Integritätsprüfung ------------------------------------------------------------------------


def test_alarming_g1_signal_behind_r0_is_rejected() -> None:
    """G1-Signal VALID/alarm=true, G1-Gruppe VALID/alarm=false, R0, REC_CONTINUE_MONITORING."""
    signals, groups = signal_chain(window_id="w0")
    signals = _replace_signal(signals, "w0", G1_SIGNAL_ID, _g1(alarm=True))
    group_refs = tuple(window_evidence("group_assessment", "w0", g.value) for g in (G1, G2, G3))
    r0 = PrimaryRiskAssessment(
        window_id="w0",
        reference_window_id="r",
        state=PrimaryRiskState.R0,
        flagged_groups=(),
        evidence_refs=group_refs,
    )
    continue_monitoring = Recommendation(
        window_id="w0",
        level=AnalysisLevel.PRIMARY,
        code="REC_CONTINUE_MONITORING",
        reason_codes=("RSN_R0_NO_GROUP_ALARM",),
        evidence_refs=(window_evidence("primary_risk_assessment", "w0", "primary"), *group_refs),
    )
    data = {
        **_run(signals, groups),
        "primary_assessments": (r0,),
        "recommendations": (continue_monitoring,),
    }
    _rejected(data, "Gruppe w0:G1 folgt nicht aus ihren SignalResults")


# --- Pflicht-Negativtests 1–14 ------------------------------------------------------------------


def test_01_alarming_g1_signal_with_non_alarming_group() -> None:
    signals, groups = signal_chain(window_id="w0")
    _rejected(_run(_replace_signal(signals, "w0", G1_SIGNAL_ID, _g1(alarm=True)), groups))


def test_02_non_alarming_g1_signal_with_alarming_group() -> None:
    signals, groups = signal_chain((True, False, False), window_id="w0")
    _rejected(
        _run(_replace_signal(signals, "w0", G1_SIGNAL_ID, _g1(alarm=False)), groups),
        "folgt nicht aus ihren SignalResults",
    )


@pytest.mark.parametrize(
    ("status", "reason"),
    [(ERR, ReasonCode.KS_EXACT_FALLBACK), (NE, ReasonCode.INSUFFICIENT_SAMPLE)],
    ids=["03-error", "04-not-evaluable"],
)
def test_03_04_invalid_signal_behind_valid_group(status: SignalStatus, reason: ReasonCode) -> None:
    signals, groups = signal_chain(window_id="w0")
    invalid = _g1(status=status, alarm=None, reason_code=reason)
    _rejected(
        _run(_replace_signal(signals, "w0", G1_SIGNAL_ID, invalid), groups),
        "folgt nicht aus ihren SignalResults",
    )


def test_05_g2_with_missing_subsignal() -> None:
    signals, groups = signal_chain(window_id="w0")
    three = G2_IDS[:3]
    g2 = _group(G2, signal_ids=three, evidence_refs=tuple(_signal_ref("w0", i) for i in three))
    signals = _replace_signal(signals, "w0", G2_IDS[3], None)
    _rejected(_run(signals, _replace_group(groups, g2)), "verletzen die Aggregation")


def test_06_g2_with_additional_subsignal() -> None:
    signals, groups = signal_chain(window_id="w0")
    extra = signal_result(
        signal_id="G2_EXTRA_INDICATOR_KS",
        group=G2,
        monitoring_window_id="w0",
        reference_window_id="r",
    )
    five = (*G2_IDS, extra.signal_id)
    g2 = _group(G2, signal_ids=five, evidence_refs=tuple(_signal_ref("w0", i) for i in five))
    _rejected(_run((*signals, extra), _replace_group(groups, g2)), "verletzen die Aggregation")


def test_07_g2_with_duplicate_subsignal() -> None:
    signals, groups = signal_chain(window_id="w0")
    doubled = (*G2_IDS[:3], G2_IDS[0])
    bypass = GroupAssessment.model_construct(
        **{
            **dict(_group(G2)),
            "signal_ids": doubled,
            "evidence_refs": tuple(_signal_ref("w0", i) for i in doubled),
        }
    )
    _rejected(_run(signals, _replace_group(groups, bypass)), "signal_ids müssen eindeutig")


def test_08_g3_with_wrong_or_result() -> None:
    signals, groups = signal_chain((False, False, True), window_id="w0")
    assert next(g for g in groups if g.group is G3).alarm is True
    wrong = _group(G3, alarm=False, substatus=G3Substatus.NONE)
    _rejected(_run(signals, _replace_group(groups, wrong)), "Gruppe w0:G3 folgt nicht")


def test_09_g3_with_wrong_substatus() -> None:
    signals, groups = signal_chain((False, False, True), window_id="w0")
    assert next(g for g in groups if g.group is G3).substatus is G3Substatus.OUTPUT_ONLY
    wrong = _group(G3, alarm=True, substatus=G3Substatus.CONFIDENCE_ONLY)
    _rejected(_run(signals, _replace_group(groups, wrong)), "Gruppe w0:G3 folgt nicht")


def test_10_signal_reference_to_another_window() -> None:
    w0_signals, w0_groups = signal_chain(window_id="w0")
    w1_signals, w1_groups = signal_chain(window_id="w1")
    foreign = _group(G1, evidence_refs=(_signal_ref("w1", G1_SIGNAL_ID),))
    signals, groups = (*w0_signals, *w1_signals), (*_replace_group(w0_groups, foreign), *w1_groups)
    _rejected(_run(signals, groups), "Gruppe w0:G1 folgt nicht")
    # Fehlt das eigene G1-Signal, wird das gleichnamige Signal aus w1 nie herangezogen.
    _rejected(
        _run(_replace_signal(signals, "w0", G1_SIGNAL_ID, None), (*w0_groups, *w1_groups)),
        "SignalResult im eigenen Window fehlt",
    )


def test_11_signal_with_another_reference_window() -> None:
    signals, groups = signal_chain(window_id="w0")
    _rejected(
        _run(_replace_signal(signals, "w0", G1_SIGNAL_ID, _g1(reference_window_id="r2")), groups),
        "Gruppe w0:G1 folgt nicht",
    )


def test_12_unknown_signal_reference() -> None:
    signals, groups = signal_chain(window_id="w0")
    unknown = _group(
        G1, signal_ids=("G1_UNKNOWN",), evidence_refs=(_signal_ref("w0", "G1_UNKNOWN"),)
    )
    _rejected(
        _run(signals, _replace_group(groups, unknown)), "SignalResult im eigenen Window fehlt"
    )


def test_13_duplicate_window_signal_identity() -> None:
    signals, groups = signal_chain(window_id="w0")
    _rejected(_run((*signals, _g1()), groups), "Mehrfachbelegung von SignalResult")


@pytest.mark.parametrize(
    "refs",
    [
        pytest.param((), id="no-evidence"),
        pytest.param((_signal_ref("w0", G2_IDS[0]),), id="other-signal"),
        pytest.param(
            (EvidenceRef(kind="signal_result", ref_id=G1_SIGNAL_ID),), id="legacy-ambiguous"
        ),
        pytest.param(
            (EvidenceRef(kind="artifact", ref_id=f"signal_result:w0:{G1_SIGNAL_ID}"),),
            id="wrong-kind",
        ),
    ],
)
def test_14_signal_id_and_evidence_mismatch(refs: tuple[EvidenceRef, ...]) -> None:
    signals, groups = signal_chain(window_id="w0")
    _rejected(_run(signals, _replace_group(groups, _group(G1, evidence_refs=refs))), "folgt nicht")


def test_signal_of_the_wrong_group_is_rejected() -> None:
    signals, groups = signal_chain(window_id="w0")
    misgrouped = _g1(group=G2)
    _rejected(
        _run(_replace_signal(signals, "w0", G1_SIGNAL_ID, misgrouped), groups),
        "verletzen die Aggregation",
    )


def test_orphaned_signals_and_groups_without_signals_fail_closed() -> None:
    signals, groups = signal_chain(window_id="w0", groups=(G1, G2))
    orphan_g3 = group_signals(G3, window_id="w0")
    _rejected(_run((*signals, *orphan_g3), groups), "SignalResult ohne zugehörige Gruppe")
    _rejected(_run((), groups), "SignalResult im eigenen Window fehlt")
    orphan_window = group_signals(G1, window_id="w9")
    _rejected(_run((*signals, *orphan_window), groups), "SignalResult ohne zugehörige Gruppe")


# --- Pflicht-Positivtests 15–16 -----------------------------------------------------------------


def test_15_consistent_signals_and_groups_across_windows_pass() -> None:
    chains = [signal_chain((i % 2 == 0, False, i == 2), window_id=f"w{i}") for i in range(4)]
    signals = tuple(s for chain, _ in chains for s in chain)
    groups = tuple(g for _, chain in chains for g in chain)
    run = MonitoringRunResult.model_validate(_run(signals, groups))
    # Gleiche signal_id in verschiedenen Windows ist zulässig, da windowgebunden eindeutig.
    assert sum(s.signal_id == G1_SIGNAL_ID for s in run.signal_results) == 4
    for group in run.group_assessments:
        assert group.evidence_refs == tuple(
            _signal_ref(group.window_id, i) for i in group.signal_ids
        )


@pytest.mark.parametrize("alarms", list(itertools.product([False, True], repeat=3)))
def test_16_valid_g1_g2_g3_results_are_unchanged(alarms: tuple[bool, bool, bool]) -> None:
    """Fachliche Gruppen-/Risikoergebnisse unverändert; nur die Evidence ist windowgebunden."""
    signals, groups = signal_chain(alarms, window_id="w0")
    reference = primary_groups(alarms, window_id="w0")
    for actual, expected in zip(groups, reference, strict=True):
        fields = ("group", "window_id", "reference_window_id", "status", "alarm", "substatus")
        assert all(getattr(actual, f) == getattr(expected, f) for f in (*fields, "reason_code"))
    assert assess_primary("w0", groups, RISK_POLICY_V1) == assess_primary(
        "w0", reference, RISK_POLICY_V1
    )
    run = MonitoringRunResult.model_validate(_run(signals, groups))
    assert run.primary_assessments == (assess_primary("w0", reference, RISK_POLICY_V1),)


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ((ERR, NE, VALID), (ERR, NE, VALID)),
        ((NE, VALID, ERR), (NE, VALID, ERR)),
    ],
)
def test_status_precedence_is_reproduced_from_signals(
    statuses: tuple[SignalStatus, SignalStatus, SignalStatus],
    expected: tuple[SignalStatus, SignalStatus, SignalStatus],
) -> None:
    signals, groups = signal_chain(window_id="w0", statuses=statuses)
    assert tuple(g.status for g in groups) == expected
    run = MonitoringRunResult.model_validate(_run(signals, groups))
    assert run.primary_failures[0].status is ERR  # ERROR > NOT_EVALUABLE > VALID
    assert aggregate(G1, signals[:1]).status is statuses[0]
