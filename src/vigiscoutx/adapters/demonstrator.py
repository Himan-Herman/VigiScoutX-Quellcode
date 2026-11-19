"""Zentrale technische Konventionen des HAM10000-/ResNet-18-Demonstrators.

`DEMONSTRATOR_CLASS_ORDER` ist die einzige Definition der Output-Index-Konvention:
Index i des Modell-Outputs entspricht `DEMONSTRATOR_CLASS_ORDER[i]`. Dies ist eine technische
Indexkonvention, keine wissenschaftliche Rangordnung (ARCHITECTURE_BASELINE, Abschnitt 11).
"""

from collections.abc import Sequence

DEMONSTRATOR_CLASS_ORDER: tuple[str, ...] = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")


def require_canonical_class_order(order: Sequence[str]) -> tuple[str, ...]:
    """Akzeptiert ausschließlich exakt die kanonische Reihenfolge (keine Sortierung)."""
    candidate = tuple(order)
    if candidate != DEMONSTRATOR_CLASS_ORDER:
        raise ValueError(
            f"Klassenreihenfolge {candidate} weicht von der kanonischen Reihenfolge "
            f"{DEMONSTRATOR_CLASS_ORDER} ab"
        )
    return candidate
