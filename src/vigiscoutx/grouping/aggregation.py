"""Strikte Aggregation verpflichtender Subsignale (DEC-049, MONITORING_METHODS_BASELINE 7.1).

Präzedenz `ERROR > NOT_EVALUABLE > VALID`. Nur wenn alle Children `VALID` sind, ist der Parent
`VALID` mit dem eingefrorenen OR der Child-Alarme; sonst `alarm = null` mit
`SUBSIGNAL_ERROR` bzw. `SUBSIGNAL_NOT_EVALUABLE`. Child-Ergebnisse bleiben unverändert erhalten.
"""

from collections.abc import Sequence
from typing import Protocol

from pydantic import StrictBool

from vigiscoutx.domain.assessments import (
    G3Substatus,
    GroupAssessment,
    ReasonCode,
    SignalResult,
    SignalStatus,
)
from vigiscoutx.domain.refs import window_evidence
from vigiscoutx.domain.types import FrozenModel, SignalGroup
from vigiscoutx.signals.config import (
    G1_SIGNAL_ID,
    G2_SIGNAL_IDS,
    G3_CONFIDENCE_SIGNAL_ID,
    G3_OUTPUT_SIGNAL_ID,
)


class StatusCarrier(Protocol):
    @property
    def status(self) -> SignalStatus: ...

    @property
    def alarm(self) -> bool | None: ...


class AggregateOutcome(FrozenModel):
    status: SignalStatus
    alarm: StrictBool | None
    reason_code: ReasonCode | None


def aggregate_strict(children: Sequence[StatusCarrier]) -> AggregateOutcome:
    if not children:
        raise ValueError("Aggregation erfordert mindestens ein verpflichtendes Subsignal")
    statuses = {child.status for child in children}
    if SignalStatus.ERROR in statuses:
        return AggregateOutcome(
            status=SignalStatus.ERROR, alarm=None, reason_code=ReasonCode.SUBSIGNAL_ERROR
        )
    if SignalStatus.NOT_EVALUABLE in statuses:
        return AggregateOutcome(
            status=SignalStatus.NOT_EVALUABLE,
            alarm=None,
            reason_code=ReasonCode.SUBSIGNAL_NOT_EVALUABLE,
        )
    return AggregateOutcome(
        status=SignalStatus.VALID,
        alarm=any(child.alarm is True for child in children),
        reason_code=None,
    )


def _group_assessment(
    group: SignalGroup,
    results: Sequence[SignalResult],
    outcome: AggregateOutcome,
    substatus: G3Substatus | None,
) -> GroupAssessment:
    if any(r.group is not group for r in results):
        raise ValueError(f"alle Subsignale müssen zur Gruppe {group.value} gehören")
    windows = {(r.monitoring_window_id, r.reference_window_id) for r in results}
    if len(windows) != 1:
        raise ValueError("alle Subsignale einer Gruppe müssen dieselben Fenster betreffen")
    (monitoring_window_id, reference_window_id) = windows.pop()
    return GroupAssessment(
        group=group,
        window_id=monitoring_window_id,
        reference_window_id=reference_window_id,
        status=outcome.status,
        alarm=outcome.alarm,
        substatus=substatus,
        reason_code=outcome.reason_code,
        signal_ids=tuple(r.signal_id for r in results),
        # Windowgebundene, runweit eindeutige Signal-Evidence (P4-03C); fachlich unverändert.
        evidence_refs=tuple(
            window_evidence("signal_result", monitoring_window_id, r.signal_id) for r in results
        ),
    )


def assess_group(group: SignalGroup, results: Sequence[SignalResult]) -> GroupAssessment:
    """G1/G2: strikte Aggregation der verpflichtenden Subsignale ohne Substatus."""
    if group is SignalGroup.G3_PREDICTIVE:
        raise ValueError("G3 wird über assess_g3_group aggregiert")
    if group is SignalGroup.G1_INPUT:
        _require_exact_g1(results)
    if group is SignalGroup.G2_DATA_QUALITY:
        _require_complete_g2(results)
    return _group_assessment(group, results, aggregate_strict(results), None)


def _require_exact_g1(results: Sequence[SignalResult]) -> None:
    """G1 besteht ausschließlich aus genau einem Child `G1_MMD_EMBEDDING` (DEC-042)."""
    ids = [r.signal_id for r in results]
    if ids != [G1_SIGNAL_ID]:
        raise ValueError(f"G1 erfordert exakt das Subsignal {G1_SIGNAL_ID}, erhalten: {ids}")


def _require_complete_g2(results: Sequence[SignalResult]) -> None:
    """G2 besteht verpflichtend aus exakt den vier Qualitätskomponenten (DEC-043, DEC-049).

    Geprüft vor jeder Status-/Alarmaggregation; Reihenfolge ist unerheblich, fehlende,
    zusätzliche, doppelte oder fremde Children sind ein Strukturfehler.
    """
    ids = [r.signal_id for r in results]
    required = set(G2_SIGNAL_IDS.values())
    if len(ids) != len(required) or set(ids) != required:
        raise ValueError(f"G2 erfordert exakt die Subsignale {sorted(required)}, erhalten: {ids}")


def g3_substatus(output_alarm: bool, confidence_alarm: bool) -> G3Substatus:
    if output_alarm and confidence_alarm:
        return G3Substatus.BOTH
    if output_alarm:
        return G3Substatus.OUTPUT_ONLY
    if confidence_alarm:
        return G3Substatus.CONFIDENCE_ONLY
    return G3Substatus.NONE


def assess_g3_group(output: SignalResult, confidence: SignalResult) -> GroupAssessment:
    """`G3_alarm = output_alarm OR confidence_alarm`; Substatus nur bei gültiger Gruppe."""
    if (output.signal_id, confidence.signal_id) != (G3_OUTPUT_SIGNAL_ID, G3_CONFIDENCE_SIGNAL_ID):
        raise ValueError("G3 erwartet G3_OUTPUT_MUKS und G3_CONFIDENCE_KS")
    outcome = aggregate_strict((output, confidence))
    substatus = None
    if outcome.status is SignalStatus.VALID:
        substatus = g3_substatus(output.alarm is True, confidence.alarm is True)
    return _group_assessment(SignalGroup.G3_PREDICTIVE, (output, confidence), outcome, substatus)
