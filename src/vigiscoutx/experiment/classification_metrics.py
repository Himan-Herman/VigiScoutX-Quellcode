"""Klassifikationsmetriken für Model Development (Validation)."""

from collections.abc import Sequence


def per_class_recall(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> list[float]:
    """Recall je Klasse; jede Klasse muss in `y_true` vorkommen."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true und y_pred müssen gleich lang und nicht leer sein")
    recalls: list[float] = []
    for cls in range(num_classes):
        positives = [p for t, p in zip(y_true, y_pred, strict=True) if t == cls]
        if not positives:
            raise ValueError(f"Klasse {cls} fehlt in y_true; Balanced Accuracy undefiniert")
        recalls.append(sum(1 for p in positives if p == cls) / len(positives))
    return recalls


def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> float:
    """BA = (1/K) · Σ_c Recall_c (DEC-022)."""
    return sum(per_class_recall(y_true, y_pred, num_classes)) / num_classes


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> float:
    scores: list[float] = []
    for cls in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p != cls)
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return sum(scores) / num_classes
