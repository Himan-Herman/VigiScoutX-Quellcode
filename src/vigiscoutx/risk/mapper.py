"""RiskMapper: label-freie Primary-Bewertung R0–R2 aus G1/G2/G3 (ARC-011, DEC-074).

Verwendet ausschließlich die drei Pflicht-`GroupAssessment`s eines Monitoring-Windows; kein G4,
keine Labels, kein Fusion-A/B-Ergebnis. Der Risikozustand ist ein eigener Output: `k = 1` ergibt
R1 unabhängig davon, ob der gewählte Fusionskandidat alarmiert.
"""

from collections.abc import Sequence

from vigiscoutx.domain.assessments import (
    PRIMARY_GROUPS_ORDERED,
    GroupAssessment,
    PrimaryRiskAssessment,
    PrimaryRiskFailure,
    SignalStatus,
)
from vigiscoutx.domain.refs import EvidenceRef, window_evidence
from vigiscoutx.domain.types import SignalGroup
from vigiscoutx.risk.policy import RiskPolicyDefinition

type PrimaryOutcome = PrimaryRiskAssessment | PrimaryRiskFailure


def _status_consistent(group: GroupAssessment) -> bool:
    return (group.status is SignalStatus.VALID) == (group.alarm is not None)


def _unambiguous(window_id: str, groups: Sequence[object]) -> dict[SignalGroup, GroupAssessment]:
    """Pflichtgruppen mit genau einem vertragskonformen Assessment im eigenen Window.

    Mehrfach belegte, windowfremde, fremdtypisierte oder status-/alarm-inkonsistente Einträge
    sind nicht eindeutig referenzierbar und werden nie als Nachweis verwendet (P4-03C).
    """
    in_window = [g for g in groups if isinstance(g, GroupAssessment) and g.window_id == window_id]
    usable: dict[SignalGroup, GroupAssessment] = {}
    for identity in PRIMARY_GROUPS_ORDERED:
        claims = [g for g in in_window if g.group is identity]
        if (
            len(claims) == 1
            and type(claims[0]) is GroupAssessment
            and _status_consistent(claims[0])
        ):
            usable[identity] = claims[0]
    return usable


def _refs(window_id: str, groups: Sequence[SignalGroup]) -> tuple[EvidenceRef, ...]:
    return tuple(
        window_evidence("group_assessment", window_id, g.value)
        for g in PRIMARY_GROUPS_ORDERED
        if g in groups
    )


def assess_primary(
    window_id: str, groups: Sequence[GroupAssessment], policy: RiskPolicyDefinition
) -> PrimaryOutcome:
    """Primary-Ergebnis oder typisierter Failure; nie R0 als Ersatz und nie R3.

    Vertragsverletzung (fehlende, doppelte, windowfremde oder fremde Gruppe, abweichende
    Reference, inkonsistenter Status): `ERROR` mit `contract_violation`; jede nicht eindeutig
    belegte Pflichtgruppe steht in `invalid_groups`, nur eindeutige Gruppen werden referenziert.
    """
    usable = _unambiguous(window_id, groups)
    single_reference = len({g.reference_window_id for g in usable.values()}) <= 1
    if (
        len(groups) != len(PRIMARY_GROUPS_ORDERED)
        or len(usable) != len(PRIMARY_GROUPS_ORDERED)
        or not single_reference
    ):
        return PrimaryRiskFailure(
            window_id=window_id,
            status=SignalStatus.ERROR,
            contract_violation=True,
            invalid_groups=tuple(g for g in PRIMARY_GROUPS_ORDERED if g not in usable),
            evidence_refs=_refs(window_id, tuple(usable)),
        )
    ordered = [usable[g] for g in PRIMARY_GROUPS_ORDERED]
    evidence = _refs(window_id, PRIMARY_GROUPS_ORDERED)
    for status in policy.status_precedence[:-1]:
        if any(g.status is status for g in ordered):
            return PrimaryRiskFailure(
                window_id=window_id,
                status=status,
                contract_violation=False,
                invalid_groups=tuple(
                    g.group for g in ordered if g.status is not SignalStatus.VALID
                ),
                evidence_refs=evidence,
            )
    flagged: tuple[SignalGroup, ...] = tuple(g.group for g in ordered if g.alarm is True)
    return PrimaryRiskAssessment(
        window_id=window_id,
        reference_window_id=ordered[0].reference_window_id,
        state=policy.state_for(len(flagged)),
        flagged_groups=flagged,
        evidence_refs=evidence,
    )
