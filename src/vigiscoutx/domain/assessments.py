"""Ergebnis- und Bewertungscontracts inklusive Primary-/Label-aware-Invariante (DEC-030)."""

import math
from enum import StrEnum
from typing import Final, Self

from pydantic import Field, StrictBool, StrictFloat, field_validator, model_validator

from vigiscoutx.adapters.demonstrator import DEMONSTRATOR_CLASS_ORDER
from vigiscoutx.domain.frozen_json import ScientificParams
from vigiscoutx.domain.refs import (
    PERFORMANCE_LOCAL_ID,
    PRIMARY_LOCAL_ID,
    EvidenceRef,
    ModelSnapshotRef,
    parse_window_evidence,
    reject_performance_evidence,
    relevance_evidence,
    window_evidence,
)
from vigiscoutx.domain.types import (
    Code,
    FrozenModel,
    Identifier,
    NonNegativeFloat,
    NonNegativeInt,
    PerformanceMetric,
    Sha256Hex,
    SignalGroup,
    UnitInterval,
    require_primary_group,
)


class PrimaryRiskState(StrEnum):
    """Zustände der primären label-freien Analyse. R3 existiert hier nicht."""

    R0 = "R0"
    R1 = "R1"
    R2 = "R2"


class LabelAwareRiskState(StrEnum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"


class SignalStatus(StrEnum):
    """Statusvertrag jedes Einzelsignals (MONITORING_METHODS_BASELINE Abschnitt 7, DEC-047)."""

    VALID = "VALID"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    ERROR = "ERROR"


class ReasonCode(StrEnum):
    """Stabile Codes für nicht-valide Ergebnisse; keine UI-Texte."""

    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    NONFINITE_INPUT = "NONFINITE_INPUT"
    INPUT_OUT_OF_RANGE = "INPUT_OUT_OF_RANGE"
    INVALID_INPUT_SHAPE = "INVALID_INPUT_SHAPE"
    INVALID_INPUT_TYPE = "INVALID_INPUT_TYPE"
    INVALID_CLASS_DIMENSION = "INVALID_CLASS_DIMENSION"
    EMBEDDING_DIMENSION_MISMATCH = "EMBEDDING_DIMENSION_MISMATCH"
    KS_EXACT_FALLBACK = "KS_EXACT_FALLBACK"
    KS_COMPUTATION_FAILED = "KS_COMPUTATION_FAILED"
    MMD_COMPUTATION_FAILED = "MMD_COMPUTATION_FAILED"
    INVALID_LABEL = "INVALID_LABEL"
    MISSING_GROUND_TRUTH_CLASS = "MISSING_GROUND_TRUTH_CLASS"
    SUBSIGNAL_ERROR = "SUBSIGNAL_ERROR"
    SUBSIGNAL_NOT_EVALUABLE = "SUBSIGNAL_NOT_EVALUABLE"


class G3Substatus(StrEnum):
    """Nur bei gültiger G3-Gruppe (DEC-044, DEC-049); kein weiterer Substatus."""

    NONE = "NONE"
    OUTPUT_ONLY = "OUTPUT_ONLY"
    CONFIDENCE_ONLY = "CONFIDENCE_ONLY"
    BOTH = "BOTH"


def require_status_consistency(
    status: SignalStatus, alarm: bool | None, reason_code: ReasonCode | None
) -> None:
    """VALID ⇔ alarm gesetzt und kein Reason-Code; sonst alarm = null mit Reason-Code."""
    if status is SignalStatus.VALID:
        if alarm is None or reason_code is not None:
            raise ValueError("VALID erfordert alarm true/false und keinen reason_code")
    elif alarm is not None or reason_code is None:
        raise ValueError(f"{status.value} erfordert alarm = null und einen reason_code")


G3_OUTPUT_SIGNAL_ID: Final = "G3_OUTPUT_MUKS"
"""Einzige Definition der G3-Output-Signal-ID (von `vigiscoutx.signals.config` re-exportiert)."""


class ParametersRef(FrozenModel):
    """Referenz auf die hashbare Methodenkonfiguration eines Signals."""

    config_id: Identifier
    method_version: Identifier
    sha256: Sha256Hex


class ArtifactRef(FrozenModel):
    artifact_id: Identifier
    sha256: Sha256Hex


class SeedRef(FrozenModel):
    """Herkunft eines Zufalls-Seeds aus dem `SeedBundle` (DEC-025, DEC-032)."""

    seed_field: Identifier
    seed: NonNegativeInt
    repeat_id: NonNegativeInt


class ComponentResult(FrozenModel):
    """Komponente eines zusammengesetzten Signals (z. B. je Klasse bei G3 Output)."""

    component_id: Identifier
    status: SignalStatus
    statistics: ScientificParams
    p_value: UnitInterval | None
    adjusted_p_value: UnitInterval | None
    alarm: StrictBool | None
    reason_code: ReasonCode | None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        require_status_consistency(self.status, self.alarm, self.reason_code)
        return self


class SignalResult(FrozenModel):
    """Ergebnis eines Einzelsignals der Gruppen G1–G3 (G4 liefert `PerformanceAssessment`)."""

    signal_id: Identifier
    group: SignalGroup
    method: Code
    method_version: Identifier
    status: SignalStatus
    statistics: ScientificParams
    p_value: UnitInterval | None
    adjusted_p_value: UnitInterval | None
    alpha_or_threshold: StrictFloat | None
    alarm: StrictBool | None
    reason_code: ReasonCode | None
    components: tuple[ComponentResult, ...]
    parameters_ref: ParametersRef
    reference_window_id: Identifier
    monitoring_window_id: Identifier
    reference_n: NonNegativeInt
    monitoring_n: NonNegativeInt
    seed_ref: SeedRef | None
    model_snapshot_ref: ModelSnapshotRef | None
    artifact_refs: tuple[ArtifactRef, ...]
    evidence_refs: tuple[EvidenceRef, ...]

    @field_validator("group")
    @classmethod
    def _primary_group(cls, group: SignalGroup) -> SignalGroup:
        return require_primary_group(group)

    @field_validator("evidence_refs")
    @classmethod
    def _no_performance_evidence(cls, refs: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
        return reject_performance_evidence(refs)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        require_status_consistency(self.status, self.alarm, self.reason_code)
        if self.adjusted_p_value is not None and self.p_value is None:
            raise ValueError("adjusted_p_value erfordert einen p_value")
        ids = [c.component_id for c in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError("component_id muss eindeutig sein")
        if self.signal_id == G3_OUTPUT_SIGNAL_ID:
            _require_g3_output_structure(self)
        return self


def _require_g3_output_structure(result: SignalResult) -> None:
    """G3_OUTPUT_MUKS: exakt sieben kanonische Komponenten, strikt aggregiert (DEC-044, DEC-049).

    Ohne Komponenten ist nur `ERROR` zulässig (z. B. falsche Klassendimension). Mit Komponenten
    muss die ID-Menge exakt der kanonischen Klassenmenge entsprechen (Reihenfolge unerheblich) und
    Status, Alarm und Reason-Code müssen der strikten Aggregation `ERROR > NOT_EVALUABLE > VALID`
    entsprechen; nur bei sieben gültigen Komponenten gilt das OR der Komponentenalarme.
    """
    components = result.components
    if not components:
        if result.status is not SignalStatus.ERROR:
            raise ValueError("G3_OUTPUT_MUKS ohne Komponenten ist nur als ERROR zulässig")
        return
    ids = [c.component_id for c in components]
    if len(ids) != len(DEMONSTRATOR_CLASS_ORDER) or set(ids) != set(DEMONSTRATOR_CLASS_ORDER):
        raise ValueError(
            f"G3_OUTPUT_MUKS erfordert exakt die Komponenten {DEMONSTRATOR_CLASS_ORDER}, "
            f"erhalten: {ids}"
        )
    statuses = {c.status for c in components}
    expected: tuple[SignalStatus, bool | None, ReasonCode | None]
    if SignalStatus.ERROR in statuses:
        expected = (SignalStatus.ERROR, None, ReasonCode.SUBSIGNAL_ERROR)
    elif SignalStatus.NOT_EVALUABLE in statuses:
        expected = (SignalStatus.NOT_EVALUABLE, None, ReasonCode.SUBSIGNAL_NOT_EVALUABLE)
    else:
        expected = (SignalStatus.VALID, any(c.alarm is True for c in components), None)
    if (result.status, result.alarm, result.reason_code) != expected:
        raise ValueError("G3_OUTPUT_MUKS: Status/Alarm müssen der strikten Aggregation entsprechen")


class GroupAssessment(FrozenModel):
    """Verdichtetes Ergebnis je Primary-Gruppe; strikte Statusaggregation (DEC-049)."""

    group: SignalGroup
    window_id: Identifier
    reference_window_id: Identifier
    status: SignalStatus
    alarm: StrictBool | None
    substatus: G3Substatus | None
    reason_code: ReasonCode | None
    signal_ids: tuple[Identifier, ...] = Field(min_length=1)
    evidence_refs: tuple[EvidenceRef, ...]

    @field_validator("group")
    @classmethod
    def _primary_group(cls, group: SignalGroup) -> SignalGroup:
        return require_primary_group(group)

    @field_validator("evidence_refs")
    @classmethod
    def _no_performance_evidence(cls, refs: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
        return reject_performance_evidence(refs)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        require_status_consistency(self.status, self.alarm, self.reason_code)
        if len(self.signal_ids) != len(set(self.signal_ids)):
            raise ValueError("signal_ids müssen eindeutig sein")
        valid_g3 = self.group is SignalGroup.G3_PREDICTIVE and self.status is SignalStatus.VALID
        if not valid_g3:
            if self.substatus is not None:
                raise ValueError("substatus nur bei gültiger G3-Gruppe")
        elif self.substatus is None or (self.substatus is G3Substatus.NONE) == self.alarm:
            raise ValueError("gültige G3-Gruppe: substatus erforderlich; NONE ⇔ alarm = false")
        return self


class PerformanceEvaluation(FrozenModel):
    """G4-Kennzahlen eines Fensters über exakt die feste Klassenliste (DEC-045).

    Fehlt eine Ground-Truth-Klasse, sind Recall dieser Klasse, BA und Macro F1 `null`; es wird
    nicht über vorhandene Klassen neu gemittelt und kein `zero_division`-Default verwendet.
    """

    window_id: Identifier
    status: SignalStatus
    reason_code: ReasonCode | None
    class_order: tuple[Identifier, ...] = Field(min_length=2)
    n: NonNegativeInt
    confusion_matrix: tuple[tuple[NonNegativeInt, ...], ...] | None
    support_per_class: tuple[NonNegativeInt, ...] | None
    recall_per_class: tuple[UnitInterval | None, ...] | None
    f1_per_class: tuple[UnitInterval | None, ...] | None
    balanced_accuracy: UnitInterval | None
    macro_f1: UnitInterval | None
    performance_evaluable: StrictBool
    non_evaluable_classes: tuple[Identifier, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.status is SignalStatus.ERROR:
            if self.reason_code is None or self.performance_evaluable:
                raise ValueError("ERROR erfordert reason_code und performance_evaluable = false")
            if self.balanced_accuracy is not None or self.macro_f1 is not None:
                raise ValueError("ERROR trägt keine Kennzahlen")
            return self
        k = len(self.class_order)
        per_class = (self.support_per_class, self.recall_per_class, self.f1_per_class)
        if self.confusion_matrix is None or any(v is None for v in per_class):
            raise ValueError("Confusion Matrix und Klassenwerte sind erforderlich")
        if len(self.confusion_matrix) != k or any(len(r) != k for r in self.confusion_matrix):
            raise ValueError("Confusion Matrix muss K × K sein")
        if any(v is not None and len(v) != k for v in per_class):
            raise ValueError("Klassenwerte müssen genau K Einträge haben")
        missing = self.status is SignalStatus.NOT_EVALUABLE
        if missing != (not self.performance_evaluable) or missing != bool(
            self.non_evaluable_classes
        ):
            raise ValueError("NOT_EVALUABLE ⇔ performance_evaluable = false ⇔ fehlende Klassen")
        if (self.reason_code is None) != (self.status is SignalStatus.VALID):
            raise ValueError("reason_code genau bei nicht-validem Status")
        if missing and (self.balanced_accuracy is not None or self.macro_f1 is not None):
            raise ValueError("nicht evaluierbar: balanced_accuracy und macro_f1 müssen null sein")
        if not missing and self.balanced_accuracy is None:
            raise ValueError("evaluierbar: balanced_accuracy erforderlich")
        return self


class PerformanceAssessment(FrozenModel):
    """G4, getrennt vom Primary-Pfad (DEC-030); ΔBA nur bei zwei gültigen BA (DEC-045).

    `relevant_degradation` bleibt `null`, solange `delta_rel` (Calibration-Parameter) fehlt.
    """

    metric: PerformanceMetric
    reference: PerformanceEvaluation
    monitoring: PerformanceEvaluation
    reference_n: NonNegativeInt
    monitoring_n: NonNegativeInt
    delta_ba: StrictFloat | None
    delta_rel: NonNegativeFloat | None
    relevant_degradation: StrictBool | None
    parameters_ref: ParametersRef
    model_snapshot_ref: ModelSnapshotRef | None
    artifact_refs: tuple[ArtifactRef, ...]
    evidence_refs: tuple[EvidenceRef, ...]

    @property
    def window_id(self) -> str:
        return self.monitoring.window_id

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.reference_n, self.monitoring_n) != (self.reference.n, self.monitoring.n):
            raise ValueError("reference_n/monitoring_n müssen den Auswertungen entsprechen")
        ba_ref, ba_mon = self.reference.balanced_accuracy, self.monitoring.balanced_accuracy
        if ba_ref is None or ba_mon is None:
            if self.delta_ba is not None:
                raise ValueError("delta_ba nur bei zwei gültigen Balanced Accuracies")
        elif self.delta_ba is None or not math.isclose(
            self.delta_ba, ba_ref - ba_mon, abs_tol=1e-12
        ):
            raise ValueError("delta_ba muss BA_ref − BA_monitoring entsprechen")
        if self.delta_ba is None or self.delta_rel is None:
            if self.relevant_degradation is not None:
                raise ValueError("relevant_degradation bleibt null ohne delta_ba und delta_rel")
        elif self.relevant_degradation != (self.delta_ba >= self.delta_rel):
            raise ValueError("relevant_degradation muss delta_ba ≥ delta_rel entsprechen")
        return self


class AnalysisLevel(StrEnum):
    """Analyseebene eines Ergebnisses bzw. einer Empfehlung (DEC-030, DEC-075)."""

    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"


PRIMARY_GROUPS_ORDERED: Final = (
    SignalGroup.G1_INPUT,
    SignalGroup.G2_DATA_QUALITY,
    SignalGroup.G3_PREDICTIVE,
)
_INVALID_STATUSES: Final = frozenset({SignalStatus.ERROR, SignalStatus.NOT_EVALUABLE})
_STATE_ALARMS: Final = {"R0": (0, 0), "R1": (1, 1), "R2": (2, 3)}


def _group_refs(window_id: str) -> tuple[EvidenceRef, ...]:
    return tuple(
        window_evidence("group_assessment", window_id, g.value) for g in PRIMARY_GROUPS_ORDERED
    )


class PrimaryRiskAssessment(FrozenModel):
    """Gültiges Primary-Ergebnis (R0–R2) aus genau drei gültigen Pflichtgruppen (DEC-074).

    Nachweis: genau die drei `GroupAssessment`-Referenzen des Windows; `flagged_groups` sind die
    alarmierenden Gruppen in kanonischer Reihenfolge, ihre Anzahl bestimmt den Zustand.
    """

    window_id: Identifier
    reference_window_id: Identifier
    state: PrimaryRiskState
    flagged_groups: tuple[SignalGroup, ...]
    evidence_refs: tuple[EvidenceRef, ...]

    @field_validator("flagged_groups")
    @classmethod
    def _primary_only(cls, groups: tuple[SignalGroup, ...]) -> tuple[SignalGroup, ...]:
        for group in groups:
            require_primary_group(group)
        if len(groups) != len(set(groups)):
            raise ValueError("flagged_groups müssen eindeutig sein")
        return groups

    @field_validator("evidence_refs")
    @classmethod
    def _no_performance_evidence(cls, refs: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
        return reject_performance_evidence(refs)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        low, high = _STATE_ALARMS[self.state.value]
        if not low <= len(self.flagged_groups) <= high:
            raise ValueError("Risikozustand passt nicht zur Anzahl alarmierender Gruppen")
        if self.flagged_groups != tuple(
            g for g in PRIMARY_GROUPS_ORDERED if g in self.flagged_groups
        ):
            raise ValueError("flagged_groups in kanonischer Reihenfolge G1, G2, G3")
        if self.evidence_refs != _group_refs(self.window_id):
            raise ValueError("Primary-Nachweis: genau die drei Gruppenreferenzen des Windows")
        return self


class PrimaryRiskFailure(FrozenModel):
    """Ungültige Primary-Auswertung, typisiert und ohne Risikozustand (DEC-076, Präzisierung 1).

    `ERROR` oder `NOT_EVALUABLE` mindestens einer Pflichtgruppe bzw. eine strukturelle
    Verletzung des Policy-Eingabevertrags (`contract_violation`, stets `ERROR`).

    Nachweis (P4-03C): ohne Vertragsverletzung genau die drei Gruppenreferenzen des Windows. Bei
    einer Vertragsverletzung ist jede Pflichtgruppe entweder als `invalid_groups` ausgewiesen
    (fehlend, mehrfach belegt, windowfremd oder vertragswidrig) oder über genau eine eindeutige
    Referenz im eigenen Window nachgewiesen – nie beides, nie eine mehrdeutige Referenz.
    """

    window_id: Identifier
    status: SignalStatus
    contract_violation: StrictBool
    invalid_groups: tuple[SignalGroup, ...]
    evidence_refs: tuple[EvidenceRef, ...]

    @field_validator("evidence_refs")
    @classmethod
    def _no_performance_evidence(cls, refs: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
        return reject_performance_evidence(refs)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.status not in _INVALID_STATUSES:
            raise ValueError("PrimaryRiskFailure nur mit ERROR oder NOT_EVALUABLE")
        if self.contract_violation and self.status is not SignalStatus.ERROR:
            raise ValueError("Vertragsverletzungen werden als ERROR behandelt")
        if not self.contract_violation and not self.invalid_groups:
            raise ValueError("ohne Vertragsverletzung muss mindestens eine Gruppe ungültig sein")
        for group in self.invalid_groups:
            require_primary_group(group)
        if self.invalid_groups != tuple(
            g for g in PRIMARY_GROUPS_ORDERED if g in self.invalid_groups
        ):
            raise ValueError("invalid_groups eindeutig in kanonischer Reihenfolge G1, G2, G3")
        expected = tuple(
            window_evidence("group_assessment", self.window_id, g.value)
            for g in PRIMARY_GROUPS_ORDERED
            if not (self.contract_violation and g in self.invalid_groups)
        )
        if self.evidence_refs != expected:
            raise ValueError(
                "Primary-Fehlernachweis: eindeutige Gruppenreferenzen des eigenen Windows"
            )
        return self


def _require_g4_valid(performance: PerformanceAssessment) -> None:
    evaluations = (performance.reference, performance.monitoring)
    if any(e.status is not SignalStatus.VALID or not e.performance_evaluable for e in evaluations):
        raise ValueError("G4 muss in Reference und Monitoring VALID und evaluierbar sein")
    if performance.delta_ba is None or performance.relevant_degradation is None:
        raise ValueError("G4 erfordert definiertes delta_ba und bestimmte Relevanz")


class LabelAwareAssessment(FrozenModel):
    """Sekundäre Analyse; einzige Stelle, an der R3 existieren darf.

    R3 gilt genau dann, wenn eine relevante Performance-Degradation bestätigt ist (DEC-026); sonst
    entspricht der sekundäre Zustand exakt dem Primary-Zustand (DEC-074). Das Primary-Ergebnis ist
    eingebettet, unveränderlich und wird nicht angepasst. Die Relevanzkonfiguration ist über ihren
    kanonischen Hash gebunden.
    """

    window_id: Identifier
    primary: PrimaryRiskAssessment
    performance: PerformanceAssessment
    relevance_config_sha256: Sha256Hex
    state: LabelAwareRiskState
    evidence_refs: tuple[EvidenceRef, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not (self.window_id == self.primary.window_id == self.performance.window_id):
            raise ValueError("window_id von Primary, Performance und Assessment muss gleich sein")
        if self.performance.relevant_degradation is None:
            raise ValueError("LabelAwareAssessment erfordert eine bestimmte Relevanz (delta_rel)")
        _require_g4_valid(self.performance)
        if (self.state is LabelAwareRiskState.R3) != self.performance.relevant_degradation:
            raise ValueError("R3 gilt genau dann, wenn eine relevante Degradation bestätigt ist")
        if (
            self.state is not LabelAwareRiskState.R3
            and self.state.value != self.primary.state.value
        ):
            raise ValueError("ohne relevante Degradation bleibt der Primary-Zustand erhalten")
        expected = (
            window_evidence("primary_risk_assessment", self.window_id, PRIMARY_LOCAL_ID),
            window_evidence("performance_assessment", self.window_id, PERFORMANCE_LOCAL_ID),
            relevance_evidence(self.relevance_config_sha256),
        )
        if self.evidence_refs != expected:
            raise ValueError("Sekundärnachweis: Primary, G4 und Relevanzkonfiguration")
        return self


class LabelAwareFailure(FrozenModel):
    """Sekundäre G4-Fehler-/Nichtauswertbarkeitsinformation bei gültigem Primary (DEC-076).

    Kein sekundärer Zustand, kein R3; das Primary-Ergebnis bleibt unverändert. Die G4-Referenz
    bezeichnet die eingebettete G4-Eingabe der sekundären Analyse dieses Windows – auch dann, wenn
    eine abweichende G4-Window-ID gerade die Vertragsverletzung ist (DEC-077).
    """

    window_id: Identifier
    primary: PrimaryRiskAssessment
    performance: PerformanceAssessment
    status: SignalStatus
    contract_violation: StrictBool
    evidence_refs: tuple[EvidenceRef, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.window_id != self.primary.window_id:
            raise ValueError("LabelAwareFailure gehört zum Window des Primary-Ergebnisses")
        if self.status not in _INVALID_STATUSES:
            raise ValueError("LabelAwareFailure nur mit ERROR oder NOT_EVALUABLE")
        if self.contract_violation and self.status is not SignalStatus.ERROR:
            raise ValueError("Vertragsverletzungen werden als ERROR behandelt")
        expected = (
            window_evidence("primary_risk_assessment", self.window_id, PRIMARY_LOCAL_ID),
            window_evidence("performance_assessment", self.window_id, PERFORMANCE_LOCAL_ID),
        )
        if self.evidence_refs != expected:
            raise ValueError("Sekundärer Fehlernachweis: Primary und G4 des eigenen Windows")
        return self


RISK_EVIDENCE_KINDS: Final = frozenset(
    {
        "group_assessment",
        "primary_risk_assessment",
        "primary_risk_failure",
        "label_aware_assessment",
        "label_aware_failure",
        "performance_assessment",
        "relevance_config",
    }
)
"""Evidenzarten, die Risk-/Recommendation-Ergebnisse referenzieren dürfen (DEC-076)."""


class Recommendation(FrozenModel):
    """Stabile Codes statt UI-Texten (DEC-031), genau einem Window und einer Ebene zugeordnet.

    Jede windowgebundene Referenz gehört zum Window der Empfehlung; Code und Reason-Codes gegen
    die eingefrorene Policy prüft `vigiscoutx.monitoring.MonitoringRunResult`.
    """

    window_id: Identifier
    level: AnalysisLevel
    code: Code
    reason_codes: tuple[Code, ...] = Field(min_length=1)
    evidence_refs: tuple[EvidenceRef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _level_evidence(self) -> Self:
        if self.level is AnalysisLevel.PRIMARY:
            reject_performance_evidence(self.evidence_refs)
        for ref in self.evidence_refs:
            if ref.kind not in RISK_EVIDENCE_KINDS:
                raise ValueError(f"{ref.kind} ist im Empfehlungsnachweis nicht vorgesehen")
            if ref.kind != "relevance_config" and (parse_window_evidence(ref)[0] != self.window_id):
                raise ValueError(f"Empfehlungsnachweis aus fremdem Window: {ref.ref_id}")
        return self
