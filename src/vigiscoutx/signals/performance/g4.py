"""G4 – Performance, ausschließlich label-aware (DEC-045, MONITORING_METHODS_BASELINE 5).

Über exakt die feste Klassenliste: Confusion Matrix, Recall und F1 je Klasse, Balanced Accuracy
(Macro Recall) und Macro F1. Fehlt eine Ground-Truth-Klasse, ist das Fenster `NOT_EVALUABLE`:
Recall dieser Klasse, BA und Macro F1 sind `null`; es wird weder über vorhandene Klassen neu
gemittelt noch ein `zero_division`-Default verwendet. `delta_rel` wird hier nicht bestimmt.
"""

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import model_validator

from vigiscoutx.adapters.demonstrator import require_canonical_class_order
from vigiscoutx.domain.assessments import (
    ParametersRef,
    PerformanceAssessment,
    PerformanceEvaluation,
    ReasonCode,
    SignalStatus,
)
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.refs import ModelSnapshotRef
from vigiscoutx.domain.types import FrozenModel, Identifier, PerformanceMetric


class G4PerformanceConfig(FrozenModel):
    performance_id: Literal["G4_PERFORMANCE"]
    method_version: Literal["monitoring-methods-baseline:1.2"]
    class_order: tuple[Identifier, ...]
    metric: Literal["BALANCED_ACCURACY"]
    missing_class_rule: Literal["NOT_EVALUABLE_NO_REAVERAGING"]
    zero_division: Literal["NULL"]

    @model_validator(mode="after")
    def _frozen(self) -> Self:
        require_canonical_class_order(self.class_order)
        return self

    def parameters_ref(self) -> ParametersRef:
        return ParametersRef(
            config_id=self.performance_id,
            method_version=self.method_version,
            sha256=canonical_sha256(self),
        )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def evaluate_performance(
    window_id: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    config: G4PerformanceConfig,
) -> PerformanceEvaluation:
    """G4-Kennzahlen eines Fensters; Klassenindex i entspricht `config.class_order[i]`."""
    classes = config.class_order
    k = len(classes)

    def invalid(reason: ReasonCode) -> PerformanceEvaluation:
        return PerformanceEvaluation(
            window_id=window_id,
            status=SignalStatus.ERROR,
            reason_code=reason,
            class_order=classes,
            n=len(y_true),
            confusion_matrix=None,
            support_per_class=None,
            recall_per_class=None,
            f1_per_class=None,
            balanced_accuracy=None,
            macro_f1=None,
            performance_evaluable=False,
            non_evaluable_classes=(),
        )

    if len(y_true) != len(y_pred):
        return invalid(ReasonCode.INVALID_INPUT_SHAPE)
    labels = (*y_true, *y_pred)
    if any(type(v) is not int or not 0 <= v < k for v in labels):
        return invalid(ReasonCode.INVALID_LABEL)
    matrix = [[0] * k for _ in range(k)]
    for true, pred in zip(y_true, y_pred, strict=True):
        matrix[true][pred] += 1
    support = [sum(row) for row in matrix]
    tp = [matrix[c][c] for c in range(k)]
    fp = [sum(matrix[t][c] for t in range(k)) - tp[c] for c in range(k)]
    fn = [support[c] - tp[c] for c in range(k)]
    recall = [tp[c] / support[c] if support[c] > 0 else None for c in range(k)]
    f1_denominators = [2 * tp[c] + fp[c] + fn[c] for c in range(k)]
    f1 = [2 * tp[c] / d if d > 0 else None for c, d in enumerate(f1_denominators)]
    missing = tuple(classes[c] for c in range(k) if support[c] == 0)
    evaluable = not missing
    recalls = [r for r in recall if r is not None]
    f1s = [f for f in f1 if f is not None]
    return PerformanceEvaluation(
        window_id=window_id,
        status=SignalStatus.VALID if evaluable else SignalStatus.NOT_EVALUABLE,
        reason_code=None if evaluable else ReasonCode.MISSING_GROUND_TRUTH_CLASS,
        class_order=classes,
        n=len(y_true),
        confusion_matrix=tuple(tuple(row) for row in matrix),
        support_per_class=tuple(support),
        recall_per_class=tuple(recall),
        f1_per_class=tuple(f1),
        balanced_accuracy=_mean(recalls) if evaluable and len(recalls) == k else None,
        macro_f1=_mean(f1s) if evaluable and len(f1s) == k else None,
        performance_evaluable=evaluable,
        non_evaluable_classes=missing,
    )


def assess_performance(
    reference: PerformanceEvaluation,
    monitoring: PerformanceEvaluation,
    config: G4PerformanceConfig,
    model_snapshot_ref: ModelSnapshotRef,
    delta_rel: float | None,
) -> PerformanceAssessment:
    """`delta_ba = BA_ref − BA_monitoring` nur bei zwei gültigen BA; Relevanz nur mit delta_rel."""
    ba_ref, ba_mon = reference.balanced_accuracy, monitoring.balanced_accuracy
    delta_ba = ba_ref - ba_mon if ba_ref is not None and ba_mon is not None else None
    relevant = None if delta_ba is None or delta_rel is None else delta_ba >= delta_rel
    return PerformanceAssessment(
        metric=PerformanceMetric.BALANCED_ACCURACY,
        reference=reference,
        monitoring=monitoring,
        reference_n=reference.n,
        monitoring_n=monitoring.n,
        delta_ba=delta_ba,
        delta_rel=delta_rel,
        relevant_degradation=relevant,
        parameters_ref=config.parameters_ref(),
        model_snapshot_ref=model_snapshot_ref,
        artifact_refs=(),
        evidence_refs=(),
    )
