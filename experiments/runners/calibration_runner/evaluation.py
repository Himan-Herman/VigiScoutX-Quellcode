"""Auswertung eines Reference-/Monitoring-Paars mit den bestehenden Signalmodulen G1–G4.

Alle Ergebnisse entstehen ausschließlich aus `run_g1_mmd`, `run_g2_quality`, `run_g3`,
`evaluate_performance`/`assess_performance`, `assess_group`/`assess_g3_group` und `fusion_alarm`;
p-Werte, Alarme und Statusaggregation werden hier nicht neu berechnet.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from calibration_runner.cases import CaseWindow
from calibration_runner.features import FeatureKind, SampleFeatures, SampleInputFailure
from vigiscoutx.domain.assessments import (
    GroupAssessment,
    PerformanceAssessment,
    PerformanceEvaluation,
    SignalResult,
    SignalStatus,
)
from vigiscoutx.domain.refs import ModelSnapshotRef, SeedBundle
from vigiscoutx.domain.types import SignalGroup
from vigiscoutx.domain.windows import ModelOutputBatch, RepresentationBatch
from vigiscoutx.errors import CalibrationProtocolError, IntegrityError
from vigiscoutx.experiment.calibration.config import R_METHODS, FusionCandidate
from vigiscoutx.experiment.calibration.statistics import (
    ProcedureOutcome,
    ShiftOutcome,
    fusion_alarm,
    procedure_outcome,
)
from vigiscoutx.grouping.aggregation import assess_group
from vigiscoutx.signals.config import G1MmdConfig, G2QualityConfig, G3PredictiveConfig
from vigiscoutx.signals.data_quality.quality import (
    QualityIndicatorBatch,
    QualityIndicators,
    QualityInputFailure,
    run_g2_quality,
)
from vigiscoutx.signals.input.mmd import run_g1_mmd
from vigiscoutx.signals.performance.g4 import (
    G4PerformanceConfig,
    assess_performance,
    evaluate_performance,
)
from vigiscoutx.signals.predictive.g3 import run_g3

_STATUS_PRECEDENCE = (SignalStatus.ERROR, SignalStatus.NOT_EVALUABLE)


@dataclass(frozen=True, slots=True)
class MethodConfigs:
    g1: G1MmdConfig
    g2: G2QualityConfig
    g3: G3PredictiveConfig
    g4: G4PerformanceConfig


# --- Window-Batches aus Per-Sample-Merkmalen ----------------------------------------------------


def representation(
    window_id: str, ids: Sequence[str], rows: Sequence[SampleFeatures]
) -> RepresentationBatch:
    embeddings = [r.embedding for r in rows]
    if any(e is None for e in embeddings):
        raise CalibrationProtocolError("Embeddings fehlen")
    return RepresentationBatch(
        window_id=window_id,
        sample_ids=tuple(ids),
        embeddings=tuple(e for e in embeddings if e is not None),
    )


def model_outputs(
    window_id: str, ids: Sequence[str], rows: Sequence[SampleFeatures]
) -> ModelOutputBatch:
    probabilities = [r.probabilities for r in rows]
    if any(p is None for p in probabilities):
        raise CalibrationProtocolError("Softmax-Ausgaben fehlen")
    return ModelOutputBatch(
        window_id=window_id,
        sample_ids=tuple(ids),
        probabilities=tuple(p for p in probabilities if p is not None),
    )


def quality_batch(
    window_id: str, ids: Sequence[str], rows: Sequence[SampleFeatures]
) -> QualityIndicatorBatch | QualityInputFailure:
    """Entspricht `compute_quality_indicators` über das ganze Window: das erste Sample mit
    verletztem Eingabevertrag bestimmt Status und Reason-Code des Windows."""
    indicators: list[QualityIndicators] = []
    for row in rows:
        quality = row.quality
        if quality is None:
            raise CalibrationProtocolError("G2-Indikatoren fehlen")
        if isinstance(quality, SampleInputFailure):
            return QualityInputFailure(
                window_id=window_id,
                n=len(rows),
                status=quality.status,
                reason_code=quality.reason_code,
            )
        indicators.append(quality)
    return QualityIndicatorBatch(
        window_id=window_id, sample_ids=tuple(ids), indicators=tuple(indicators)
    )


def _argmax(row: Sequence[float]) -> int:
    best = 0
    for index, value in enumerate(row):
        if value > row[best]:
            best = index
    return best


def predictions(outputs: ModelOutputBatch) -> list[int]:
    """Vorhergesagte Klasse = Index der maximalen Softmax-Wahrscheinlichkeit (erster Index bei
    Gleichstand, wie `argmax` im Model Development)."""
    return [_argmax(row) for row in outputs.probabilities]


def performance(
    window_id: str,
    ids: Sequence[str],
    rows: Sequence[SampleFeatures],
    labels: Sequence[int],
    config: G4PerformanceConfig,
) -> PerformanceEvaluation:
    return evaluate_performance(
        window_id, list(labels), predictions(model_outputs(window_id, ids, rows)), config
    )


# --- Reference-Artefakte -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReferenceArtifacts:
    """Unveränderliche Reference-Artefakte (einmal berechnet, über alle Cases wiederverwendet)."""

    window_id: str
    sample_ids: tuple[str, ...]
    representation: RepresentationBatch
    outputs: ModelOutputBatch
    quality: QualityIndicatorBatch | QualityInputFailure
    performance: PerformanceEvaluation


def reference_artifacts(
    window_id: str,
    sample_ids: Sequence[str],
    rows: Sequence[SampleFeatures],
    labels: Sequence[int],
    config: G4PerformanceConfig,
) -> ReferenceArtifacts:
    ids = tuple(sample_ids)
    return ReferenceArtifacts(
        window_id=window_id,
        sample_ids=ids,
        representation=representation(window_id, ids, rows),
        outputs=model_outputs(window_id, ids, rows),
        quality=quality_batch(window_id, ids, rows),
        performance=performance(window_id, ids, rows, labels, config),
    )


# --- Case-Auswertung ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    """Alle Signal-, Gruppen- und G4-Ergebnisse genau eines Cases."""

    case: CaseWindow
    seed: SeedBundle | None
    g1: SignalResult
    g1_group: GroupAssessment
    g2_components: tuple[SignalResult, ...]
    g2_group: GroupAssessment
    g3_output: SignalResult
    g3_confidence: SignalResult
    g3_group: GroupAssessment
    performance: PerformanceAssessment | None

    @property
    def groups(self) -> tuple[GroupAssessment, GroupAssessment, GroupAssessment]:
        return (self.g1_group, self.g2_group, self.g3_group)

    def fusion(self, candidate: FusionCandidate) -> ProcedureOutcome:
        """Fusion nur aus gültigen Gruppen; sonst status-erhaltend (→ STOP bei Verwendung)."""
        statuses = {g.status for g in self.groups}
        for status in _STATUS_PRECEDENCE:
            if status in statuses:
                return ProcedureOutcome(status=status, alarm=None)
        return ProcedureOutcome(
            status=SignalStatus.VALID, alarm=fusion_alarm(candidate, self.groups)
        )

    def primary_outcomes(self) -> dict[str, ProcedureOutcome]:
        """Die neun G1–G3-Verfahren (Signale und Gruppen) ohne Fusion."""
        return {
            self.g1.signal_id: procedure_outcome(self.g1),
            **{c.signal_id: procedure_outcome(c) for c in self.g2_components},
            "G2_GROUP": procedure_outcome(self.g2_group),
            self.g3_output.signal_id: procedure_outcome(self.g3_output),
            self.g3_confidence.signal_id: procedure_outcome(self.g3_confidence),
            "G3_GROUP": procedure_outcome(self.g3_group),
        }

    def procedure_outcomes(self) -> dict[str, ProcedureOutcome]:
        """Exakt die elf Verfahren der R-Stabilität (DEC-058); beide Fusionskandidaten."""
        outcomes: dict[str, ProcedureOutcome] = {
            **self.primary_outcomes(),
            FusionCandidate.FUSION_A.value: self.fusion(FusionCandidate.FUSION_A),
            FusionCandidate.FUSION_B.value: self.fusion(FusionCandidate.FUSION_B),
        }
        if tuple(sorted(outcomes)) != tuple(sorted(R_METHODS)):
            raise CalibrationProtocolError(f"Verfahrensmenge weicht ab: {sorted(outcomes)}")
        return outcomes

    def audit_outcomes(self, candidate: FusionCandidate) -> dict[str, ProcedureOutcome]:
        """AUDIT_CLEAN (Protokoll §13): ausschließlich G1–G3 und der fixierte Fusion Candidate.

        Der nicht gewählte Kandidat wird weder berechnet noch serialisiert.
        """
        return {**self.primary_outcomes(), candidate.value: self.fusion(candidate)}

    def shift_outcomes(self) -> dict[str, ShiftOutcome]:
        """Für relevante/nichtrelevante Shift-Cases; G4 muss vorher geprüft sein."""
        relevant = require_relevance(self)
        return {
            method: ShiftOutcome(status=o.status, alarm=o.alarm, relevant=relevant)
            for method, o in self.procedure_outcomes().items()
        }

    def mandatory_statuses(self) -> list[SignalStatus]:
        """Status aller verpflichtenden G1–G3-Ergebnisse inkl. der sieben MUKS-Komponenten."""
        return [
            self.g1.status,
            self.g1_group.status,
            *(c.status for c in self.g2_components),
            self.g2_group.status,
            *(c.status for c in self.g3_output.components),
            self.g3_output.status,
            self.g3_confidence.status,
            self.g3_group.status,
        ]


def performance_status(assessment: PerformanceAssessment) -> SignalStatus:
    """G4 gültig ⇔ Reference und Monitoring VALID, evaluierbar und ΔBA definiert."""
    evaluations = (assessment.reference, assessment.monitoring)
    for status in _STATUS_PRECEDENCE:
        if any(e.status is status for e in evaluations):
            return status
    if not all(e.performance_evaluable for e in evaluations) or assessment.delta_ba is None:
        return SignalStatus.NOT_EVALUABLE
    return SignalStatus.VALID


def require_relevance(evaluation: CaseEvaluation) -> bool:
    assessment = evaluation.performance
    if assessment is None or assessment.relevant_degradation is None:
        raise CalibrationProtocolError("Relevanz erfordert ein gültiges G4 mit δ_rel")
    return assessment.relevant_degradation


def worst_status(statuses: Sequence[SignalStatus]) -> SignalStatus:
    for status in _STATUS_PRECEDENCE:
        if status in statuses:
            return status
    return SignalStatus.VALID


def case_seed(template: SeedBundle, statistical_seed: int, repeat_id: int) -> SeedBundle:
    """Case-spezifisches SeedBundle: `statistical_seed` und `repeat_id` des Cases; die übrigen
    (in der Calibration nicht verwendeten) Felder stammen aus der Run-Konfiguration."""
    return template.model_copy(
        update={"statistical_seed": statistical_seed, "repeat_id": repeat_id}
    )


def evaluate_case(
    case: CaseWindow,
    reference: ReferenceArtifacts,
    rows: Sequence[SampleFeatures],
    *,
    configs: MethodConfigs,
    seeds: SeedBundle,
    model_ref: ModelSnapshotRef,
    labels: Sequence[int] | None,
    delta_rel: float | None,
) -> CaseEvaluation:
    """G1–G3 (und G4, falls Labels erlaubt) für genau ein Case; genau eine G1-Ausführung."""
    if len(rows) != len(case.sample_ids):
        raise CalibrationProtocolError("Merkmale passen nicht zum Window")
    ids = case.sample_ids
    g1 = run_g1_mmd(
        reference.representation,
        representation(case.window_id, ids, rows),
        configs.g1,
        seeds,
        model_ref,
    )
    seed_ref = g1.seed_ref
    if seed_ref is None or (seed_ref.seed_field, seed_ref.seed, seed_ref.repeat_id) != (
        configs.g1.seed_field,
        seeds.statistical_seed,
        seeds.repeat_id,
    ):
        raise IntegrityError(f"G1-SeedRef weicht vom Case-Seed ab ({case.scenario_key})")
    g2_components, g2_group = run_g2_quality(
        reference.quality, quality_batch(case.window_id, ids, rows), configs.g2
    )
    g3_output, g3_confidence, g3_group = run_g3(
        reference.outputs, model_outputs(case.window_id, ids, rows), configs.g3, model_ref
    )
    assessment = None
    if labels is not None:
        monitoring = performance(case.window_id, ids, rows, labels, configs.g4)
        assessment = assess_performance(
            reference.performance, monitoring, configs.g4, model_ref, delta_rel
        )
    return CaseEvaluation(
        case=case,
        seed=seeds,
        g1=g1,
        g1_group=assess_group(SignalGroup.G1_INPUT, (g1,)),
        g2_components=g2_components,
        g2_group=g2_group,
        g3_output=g3_output,
        g3_confidence=g3_confidence,
        g3_group=g3_group,
        performance=assessment,
    )


def needed_kinds(*, signals: bool) -> frozenset[FeatureKind]:
    if signals:
        return frozenset(FeatureKind)
    return frozenset({FeatureKind.PROBABILITIES})


def summary(
    evaluation: CaseEvaluation, *, audit_candidate: FusionCandidate | None = None
) -> Mapping[str, object]:
    """Kompakte, hashbare Case-Zusammenfassung für die Ergebnisartefakte.

    Mit `audit_candidate` (nur AUDIT_CLEAN) enthält sie ausschließlich G1–G3 und den fixierten
    Kandidaten; sonst die elf Vergleichsverfahren.
    """
    perf = evaluation.performance
    procedures = (
        evaluation.procedure_outcomes()
        if audit_candidate is None
        else evaluation.audit_outcomes(audit_candidate)
    )
    return {
        "procedures": {
            m: {"status": o.status.value, "alarm": o.alarm} for m, o in sorted(procedures.items())
        },
        "g1_p_value": evaluation.g1.p_value,
        "delta_ba": None if perf is None else perf.delta_ba,
        "relevant": None if perf is None else perf.relevant_degradation,
    }
