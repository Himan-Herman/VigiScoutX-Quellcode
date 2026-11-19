"""G4 – Performance, label-aware (DEC-045)."""

import pytest

from signal_factories import MODEL_REF, g4_config
from vigiscoutx.domain.assessments import ReasonCode, SignalStatus
from vigiscoutx.signals.performance.g4 import assess_performance, evaluate_performance

CONFIG = g4_config()


def _all_classes() -> tuple[list[int], list[int]]:
    # Klasse 0: 4 Beispiele, 3 korrekt, 1 → Klasse 1; Klassen 1–6: je 2 Beispiele, korrekt,
    # außer einem Beispiel von Klasse 2, das als Klasse 0 vorhergesagt wird.
    y_true = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    y_pred = [0, 0, 0, 1, 1, 1, 2, 0, 3, 3, 4, 4, 5, 5, 6, 6]
    return y_true, y_pred


def test_confusion_matrix_recall_ba_f1_match_manual_values() -> None:
    result = evaluate_performance("w", *_all_classes(), CONFIG)
    assert result.status is SignalStatus.VALID
    assert result.performance_evaluable is True
    assert result.confusion_matrix is not None
    assert result.confusion_matrix[0] == (3, 1, 0, 0, 0, 0, 0)
    assert result.confusion_matrix[2] == (1, 0, 1, 0, 0, 0, 0)
    assert result.support_per_class == (4, 2, 2, 2, 2, 2, 2)
    recalls = (0.75, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0)
    assert result.recall_per_class == pytest.approx(recalls)
    assert result.balanced_accuracy == pytest.approx(sum(recalls) / 7)
    # F1: Klasse 0: TP 3, FP 1, FN 1 → 6/8; Klasse 1: TP 2, FP 1, FN 0 → 4/5;
    # Klasse 2: TP 1, FP 0, FN 1 → 2/3; übrige 1.
    f1 = (6 / 8, 4 / 5, 2 / 3, 1.0, 1.0, 1.0, 1.0)
    assert result.f1_per_class == pytest.approx(f1)
    assert result.macro_f1 == pytest.approx(sum(f1) / 7)
    assert result.non_evaluable_classes == ()


def test_missing_ground_truth_class_is_not_evaluable_without_reaveraging() -> None:
    y_true = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    y_pred = [0, 6, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    result = evaluate_performance("w", y_true, y_pred, CONFIG)
    assert result.status is SignalStatus.NOT_EVALUABLE
    assert result.reason_code is ReasonCode.MISSING_GROUND_TRUTH_CLASS
    assert result.performance_evaluable is False
    assert result.non_evaluable_classes == ("VASC",)
    assert result.recall_per_class is not None and result.recall_per_class[6] is None
    assert result.balanced_accuracy is None
    assert result.macro_f1 is None
    # Klasse 6 wurde vorhergesagt (FP > 0): F1 ist definiert = 0, kein zero_division-Default.
    assert result.f1_per_class is not None and result.f1_per_class[6] == 0.0


def test_undefined_f1_is_null_not_zero() -> None:
    y_true = [0, 1, 2, 3, 4, 5]
    result = evaluate_performance("w", y_true, list(y_true), CONFIG)
    assert result.f1_per_class is not None and result.f1_per_class[6] is None
    assert result.macro_f1 is None
    assert result.balanced_accuracy is None


def test_macro_denominator_is_always_seven() -> None:
    y_true = [c for c in range(7) for _ in range(3)]
    y_pred = list(y_true)
    y_pred[0] = 1
    result = evaluate_performance("w", y_true, y_pred, CONFIG)
    assert result.recall_per_class is not None
    assert result.balanced_accuracy == pytest.approx((2 / 3 + 6) / 7)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "reason"),
    [
        ([0, 7], [0, 0], ReasonCode.INVALID_LABEL),
        ([0, 1], [0, -1], ReasonCode.INVALID_LABEL),
        ([0, 1], [0], ReasonCode.INVALID_INPUT_SHAPE),
        ([0, True], [0, 1], ReasonCode.INVALID_LABEL),
    ],
)
def test_invalid_labels_are_error(y_true: list[int], y_pred: list[int], reason: ReasonCode) -> None:
    result = evaluate_performance("w", y_true, y_pred, CONFIG)
    assert (result.status, result.reason_code) == (SignalStatus.ERROR, reason)
    assert result.balanced_accuracy is None and result.macro_f1 is None


def test_delta_ba_only_with_two_valid_ba() -> None:
    valid_ref = evaluate_performance("r", list(range(7)), list(range(7)), CONFIG)
    valid_mon = evaluate_performance("w", *_all_classes(), CONFIG)
    missing = evaluate_performance("w", [0, 1], [0, 1], CONFIG)
    both = assess_performance(valid_ref, valid_mon, CONFIG, MODEL_REF, delta_rel=None)
    assert valid_mon.balanced_accuracy is not None
    assert both.delta_ba == pytest.approx(1.0 - valid_mon.balanced_accuracy)
    assert both.relevant_degradation is None
    one_missing = assess_performance(valid_ref, missing, CONFIG, MODEL_REF, delta_rel=None)
    assert one_missing.delta_ba is None
    assert one_missing.relevant_degradation is None
    assert one_missing.monitoring.status is SignalStatus.NOT_EVALUABLE
