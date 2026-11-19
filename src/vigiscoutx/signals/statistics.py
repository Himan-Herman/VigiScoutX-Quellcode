"""Gemeinsame statistische Bausteine der Primary-Signale G2/G3 (MONITORING_METHODS_BASELINE 3.5).

Einziger KS-Helper des Projekts: Zwei-Stichproben-KS mit `alternative="two-sided"` und
`method="exact"`. Ein von SciPy gemeldeter Wechsel auf das asymptotische Verfahren wird erkannt
und als `ERROR` zurückgegeben; der asymptotische p-Wert wird verworfen.
"""

import math
import warnings

import numpy as np
import numpy.typing as npt
import scipy.stats

from vigiscoutx.domain.assessments import ReasonCode, SignalStatus
from vigiscoutx.domain.types import FrozenModel, UnitInterval
from vigiscoutx.signals.config import KsSpec

KS_ALTERNATIVE = "two-sided"
KS_METHOD = "exact"


class KsOutcome(FrozenModel):
    status: SignalStatus
    statistic: UnitInterval | None
    p_value: UnitInterval | None
    reason_code: ReasonCode | None


def _failure(status: SignalStatus, reason: ReasonCode) -> KsOutcome:
    return KsOutcome(status=status, statistic=None, p_value=None, reason_code=reason)


def exact_two_sided_ks(
    reference: npt.ArrayLike, monitoring: npt.ArrayLike, spec: KsSpec
) -> KsOutcome:
    """Exakter zweiseitiger Zwei-Stichproben-KS-Test Reference gegen Monitoring."""
    if (spec.alternative, spec.method) != (KS_ALTERNATIVE, KS_METHOD):
        raise ValueError("KS ist auf alternative=two-sided und method=exact eingefroren")
    ref = np.asarray(reference, dtype=np.float64)
    mon = np.asarray(monitoring, dtype=np.float64)
    if ref.ndim != 1 or mon.ndim != 1:
        return _failure(SignalStatus.ERROR, ReasonCode.INVALID_INPUT_SHAPE)
    if ref.size == 0 or mon.size == 0:
        return _failure(SignalStatus.NOT_EVALUABLE, ReasonCode.INSUFFICIENT_SAMPLE)
    if not (np.isfinite(ref).all() and np.isfinite(mon).all()):
        return _failure(SignalStatus.ERROR, ReasonCode.NONFINITE_INPUT)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = scipy.stats.ks_2samp(ref, mon, alternative=KS_ALTERNATIVE, method=KS_METHOD)
        except (ValueError, FloatingPointError, OverflowError, MemoryError):
            return _failure(SignalStatus.ERROR, ReasonCode.KS_COMPUTATION_FAILED)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    if runtime:
        fallback = any("asymp" in str(w.message) for w in runtime)
        reason = ReasonCode.KS_EXACT_FALLBACK if fallback else ReasonCode.KS_COMPUTATION_FAILED
        return _failure(SignalStatus.ERROR, reason)
    statistic, p_value = float(result.statistic), float(result.pvalue)
    if not (math.isfinite(statistic) and math.isfinite(p_value) and 0.0 <= p_value <= 1.0):
        return _failure(SignalStatus.ERROR, ReasonCode.KS_COMPUTATION_FAILED)
    return KsOutcome(
        status=SignalStatus.VALID, statistic=statistic, p_value=p_value, reason_code=None
    )


def is_significant(p_value: float, alpha: float) -> bool:
    """Verbindliche Entscheidungsregel: signifikant genau dann, wenn `p ≤ α`."""
    return p_value <= alpha


def bonferroni_adjust(p_value: float, factor: int) -> float:
    """`adjusted_p = min(1, factor · p)`."""
    return min(1.0, factor * p_value)
