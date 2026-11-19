"""Verbindliche Run-Assembly-Validierung des `MonitoringRunResult` (DEC-074 – DEC-077, P4-03C).

Ein `MonitoringRunResult` ist nur gültig, wenn

1. jede Gruppenidentität `(window_id, group)` höchstens einmal belegt ist (keine Mehrfachbelegung);
   jede SignalResult-Identität `(monitoring_window_id, signal_id)` ebenfalls; jedes
   `GroupAssessment` stimmt vollständig mit der Neuableitung der bestehenden Aggregation
   (`assess_group`, `assess_g3_group`; DEC-047, DEC-049) aus genau seinen SignalResults desselben
   Windows überein, und jedes SignalResult gehört zu genau einer Gruppe (keine verwaisten Signale);
2. jedes Window mit Gruppen genau ein Primary-Ergebnis hat und dieses exakt dem entspricht, was die
   gebundene Risk-Policy aus den G1/G2/G3-Gruppen dieses Windows ableitet (Zustand, markierte
   Gruppen, Reference, Failure-Status, Vertragsverletzung);
3. jedes sekundäre Ergebnis exakt der label-aware Ableitung aus dem unveränderten Primary-Ergebnis,
   der eingebetteten G4-Bewertung und der gebundenen Relevanzkonfiguration entspricht;
4. die Empfehlungen exakt der gebundenen Recommendation-Policy entsprechen – genau eine je
   Primary-Ergebnis, genau eine bei R3 oder sekundärem Failure, keine bei gültigem G4 ohne
   relevante Degradation;
5. jede Evidence-Referenz genau ein Objekt des erwarteten Typs im Window des besitzenden
   Ergebnisses identifiziert.

Die Ableitung verwendet die bestehenden Definitionen (Gruppenaggregation, `RiskMapper`,
label-aware Analyse, `RecommendationEngine`), keine zweite Regel- oder Mapping-Tabelle.
Verschachtelte Objekte werden vollständig neu validiert; `model_construct` an einzelnen
Bestandteilen umgeht die Prüfung daher nicht.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Protocol, Self

from pydantic import BaseModel, model_validator

from vigiscoutx.config.scientific_config import PolicyRef, RelevanceConfig
from vigiscoutx.domain.assessments import (
    AnalysisLevel,
    GroupAssessment,
    LabelAwareAssessment,
    LabelAwareFailure,
    PrimaryRiskAssessment,
    PrimaryRiskFailure,
    Recommendation,
    SignalResult,
)
from vigiscoutx.domain.refs import (
    PERFORMANCE_LOCAL_ID,
    PRIMARY_LOCAL_ID,
    SECONDARY_LOCAL_ID,
    EvidenceRef,
    parse_window_evidence,
    relevance_evidence,
    window_evidence,
)
from vigiscoutx.domain.types import FrozenModel, Identifier, Sha256Hex, SignalGroup
from vigiscoutx.grouping.aggregation import assess_g3_group, assess_group
from vigiscoutx.recommendations.engine import primary_recommendation, secondary_recommendation
from vigiscoutx.recommendations.policy import resolve_recommendation_policy
from vigiscoutx.risk.label_aware import (
    SecondaryOutcome,
    assess_label_aware,
    relevance_config_sha256,
)
from vigiscoutx.risk.mapper import PrimaryOutcome, assess_primary
from vigiscoutx.risk.policy import resolve_risk_policy


def _plain(value: object) -> object:
    """Modellinstanzen → Rohdaten, damit Pydantic jedes verschachtelte Objekt neu validiert."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, tuple | list):
        return tuple(_plain(item) for item in value)
    return value


class MonitoringRunResult(FrozenModel):
    """Zusammenhängender Bewertungsnachweis: Signale → Gruppen → Risiko → Empfehlung (ARC-042).

    Bindet die verwendeten Policies und die Relevanzkonfiguration; Ergebnisse, die sich daraus
    nicht exakt ableiten lassen, und mehrdeutige oder windowfremde Referenzen sind unzulässig.
    """

    run_id: Identifier
    config_sha256: Sha256Hex
    freeze_id: Sha256Hex | None
    risk_policy: PolicyRef
    recommendation_policy: PolicyRef
    relevance: RelevanceConfig | None
    signal_results: tuple[SignalResult, ...]
    group_assessments: tuple[GroupAssessment, ...]
    primary_assessments: tuple[PrimaryRiskAssessment, ...]
    primary_failures: tuple[PrimaryRiskFailure, ...]
    label_aware_assessments: tuple[LabelAwareAssessment, ...]
    label_aware_failures: tuple[LabelAwareFailure, ...]
    recommendations: tuple[Recommendation, ...]
    artifact_manifest_sha256: Sha256Hex | None

    @model_validator(mode="before")
    @classmethod
    def _revalidate_nested(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            return {key: _plain(value) for key, value in data.items()}
        return data

    @model_validator(mode="after")
    def _assembly(self) -> Self:
        verify_run_assembly(self)
        return self


class _Windowed(Protocol):
    @property
    def window_id(self) -> str: ...


def _unique_by_window[T: _Windowed](outcomes: Iterable[T], what: str) -> dict[str, T]:
    by_window: dict[str, T] = {}
    for outcome in outcomes:
        if outcome.window_id in by_window:
            raise ValueError(f"je Window höchstens ein {what}: {outcome.window_id}")
        by_window[outcome.window_id] = outcome
    return by_window


def verify_run_assembly(run: MonitoringRunResult) -> None:
    """Prüft Eindeutigkeit, Policy-Ableitung und Evidence-Auflösbarkeit eines Runs (fail-closed)."""
    risk = resolve_risk_policy(run.risk_policy)
    recommendation_policy = resolve_recommendation_policy(run.recommendation_policy)

    identities = Counter((g.window_id, g.group.value) for g in run.group_assessments)
    duplicated = sorted(f"{w}:{g}" for (w, g), n in identities.items() if n > 1)
    if duplicated:
        raise ValueError(f"Mehrfachbelegung von Gruppenidentitäten: {duplicated}")
    verify_signal_to_group(run)
    groups: dict[str, list[GroupAssessment]] = {}
    for group in run.group_assessments:
        groups.setdefault(group.window_id, []).append(group)

    primary: dict[str, PrimaryOutcome] = _unique_by_window(
        [*run.primary_assessments, *run.primary_failures], "Primary-Ergebnis"
    )
    if set(primary) != set(groups):
        raise ValueError("jedes Window mit Gruppen hat genau ein Primary-Ergebnis, keines ohne")
    for window_id, primary_outcome in primary.items():
        if primary_outcome != assess_primary(window_id, groups[window_id], risk):
            raise ValueError(f"Primary-Ergebnis {window_id} folgt nicht aus Gruppen und Policy")

    secondary: dict[str, SecondaryOutcome] = _unique_by_window(
        [*run.label_aware_assessments, *run.label_aware_failures], "sekundäres Ergebnis"
    )
    if secondary and run.relevance is None:
        raise ValueError("sekundäre Ergebnisse erfordern eine gebundene Relevanzkonfiguration")
    for window_id, secondary_outcome in secondary.items():
        base = primary.get(window_id)
        expected = (
            None
            if base is None or run.relevance is None
            else assess_label_aware(base, secondary_outcome.performance, run.relevance, risk)
        )
        if secondary_outcome != expected:
            raise ValueError(
                f"sekundäres Ergebnis {window_id} folgt nicht aus Primary, G4 und Relevanz"
            )

    expected_recommendations: dict[tuple[str, AnalysisLevel], Recommendation] = {
        (w, AnalysisLevel.PRIMARY): primary_recommendation(o, recommendation_policy)
        for w, o in primary.items()
    }
    for window_id, secondary_outcome in secondary.items():
        derived = secondary_recommendation(secondary_outcome, recommendation_policy)
        if derived is not None:
            expected_recommendations[(window_id, AnalysisLevel.SECONDARY)] = derived
    actual: dict[tuple[str, AnalysisLevel], Recommendation] = {}
    for recommendation in run.recommendations:
        key = (recommendation.window_id, recommendation.level)
        if key in actual:
            raise ValueError(f"je Window und Ebene höchstens eine Empfehlung: {key}")
        actual[key] = recommendation
    missing = sorted(f"{w}/{level.value}" for w, level in expected_recommendations.keys() - actual)
    if missing:
        raise ValueError(f"erforderliche Empfehlung fehlt: {missing}")
    extra = sorted(f"{w}/{level.value}" for w, level in actual.keys() - expected_recommendations)
    if extra:
        raise ValueError(f"Empfehlung ohne ableitbares Ergebnis: {extra}")
    for key, recommendation in actual.items():
        if recommendation != expected_recommendations[key]:
            raise ValueError(f"Empfehlung {key[0]}/{key[1].value} weicht von der Policy ab")

    resolve_run_evidence(run)


def _reaggregate(group: SignalGroup, children: list[SignalResult]) -> GroupAssessment:
    """Erwartetes `GroupAssessment` über die bestehende Aggregation; Strukturfehler = Fehler."""
    if group is SignalGroup.G3_PREDICTIVE:
        if len(children) != 2:
            raise ValueError("G3 erwartet G3_OUTPUT_MUKS und G3_CONFIDENCE_KS")
        return assess_g3_group(children[0], children[1])
    return assess_group(group, children)


def verify_signal_to_group(run: MonitoringRunResult) -> None:
    """SignalResults → GroupAssessments: eindeutige Signalidentität, vollständige Kindermenge,
    Window-/Reference-Bindung und exakte Neuableitung jeder Gruppe (fail-closed, P4-03C)."""
    identities = Counter((s.monitoring_window_id, s.signal_id) for s in run.signal_results)
    duplicated = sorted(f"{w}:{i}" for (w, i), n in identities.items() if n > 1)
    if duplicated:
        raise ValueError(f"Mehrfachbelegung von SignalResult-Identitäten: {duplicated}")
    signals = {(s.monitoring_window_id, s.signal_id): s for s in run.signal_results}
    referenced: set[tuple[str, str]] = set()
    for group in run.group_assessments:
        label = f"{group.window_id}:{group.group.value}"
        keys = [(group.window_id, signal_id) for signal_id in group.signal_ids]
        unknown = [k[1] for k in keys if k not in signals]
        if unknown:
            raise ValueError(f"Gruppe {label}: SignalResult im eigenen Window fehlt: {unknown}")
        referenced.update(keys)
        try:
            expected = _reaggregate(group.group, [signals[k] for k in keys])
        except ValueError as error:
            raise ValueError(
                f"Gruppe {label}: Subsignale verletzen die Aggregation: {error}"
            ) from error
        if group != expected:
            raise ValueError(f"Gruppe {label} folgt nicht aus ihren SignalResults (DEC-049)")
    orphaned = sorted(f"{w}:{i}" for w, i in signals.keys() - referenced)
    if orphaned:
        raise ValueError(f"SignalResult ohne zugehörige Gruppe: {orphaned}")


def _evidence_index(run: MonitoringRunResult) -> Counter[EvidenceRef]:
    """Identität jedes referenzierbaren Objekts im Run; Mehrfachzählung = Mehrdeutigkeit."""
    index: Counter[EvidenceRef] = Counter()
    for group in run.group_assessments:
        index[window_evidence("group_assessment", group.window_id, group.group.value)] += 1
    for assessment in run.primary_assessments:
        index[
            window_evidence("primary_risk_assessment", assessment.window_id, PRIMARY_LOCAL_ID)
        ] += 1
    for failure in run.primary_failures:
        index[window_evidence("primary_risk_failure", failure.window_id, PRIMARY_LOCAL_ID)] += 1
    for secondary in run.label_aware_assessments:
        w = secondary.window_id
        index[window_evidence("label_aware_assessment", w, SECONDARY_LOCAL_ID)] += 1
        index[window_evidence("performance_assessment", w, PERFORMANCE_LOCAL_ID)] += 1
    for secondary_failure in run.label_aware_failures:
        w = secondary_failure.window_id
        index[window_evidence("label_aware_failure", w, SECONDARY_LOCAL_ID)] += 1
        index[window_evidence("performance_assessment", w, PERFORMANCE_LOCAL_ID)] += 1
    if run.relevance is not None:
        index[relevance_evidence(relevance_config_sha256(run.relevance))] += 1
    return index


def resolve_run_evidence(run: MonitoringRunResult) -> None:
    """Jede Risk-/Recommendation-Referenz identifiziert genau ein Objekt des erwarteten Typs im
    Window des besitzenden Ergebnisses; keine leniente Mehrfachauflösung (P4-03C)."""
    index = _evidence_index(run)
    owners: list[tuple[str, tuple[EvidenceRef, ...]]] = [
        *((o.window_id, o.evidence_refs) for o in run.primary_assessments),
        *((o.window_id, o.evidence_refs) for o in run.primary_failures),
        *((o.window_id, o.evidence_refs) for o in run.label_aware_assessments),
        *((o.window_id, o.evidence_refs) for o in run.label_aware_failures),
        *((o.window_id, o.evidence_refs) for o in run.recommendations),
    ]
    for owner_window, refs in owners:
        for ref in refs:
            if ref.kind != "relevance_config":
                window_id, _ = parse_window_evidence(ref)
                if window_id != owner_window:
                    raise ValueError(f"Referenz aus fremdem Window {owner_window}: {ref.ref_id}")
            hits = index[ref]
            if hits == 0:
                raise ValueError(f"Referenz nicht auflösbar ({ref.kind}): {ref.ref_id}")
            if hits > 1:
                raise ValueError(f"Referenz mehrdeutig ({hits} Treffer): {ref.ref_id}")
