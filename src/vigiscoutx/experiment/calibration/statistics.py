"""Protokollstatistik: D0/δ_rel, Severity-Auswahl, R-Stabilität und Fusion A/B.

Raten werden exakt als Brüche geführt, damit Gleichheitsvergleiche der Auswahlregeln nicht von
Gleitkommarundung abhängen. Nicht-valide Pflichtsignale und undefinierte Raten führen zu einem
STOP, nie zu 0, 1 oder NaN.
"""

import itertools
import math
import statistics
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Self

from pydantic import StrictBool, model_validator

from vigiscoutx.domain.assessments import GroupAssessment, SignalResult, SignalStatus
from vigiscoutx.domain.types import FrozenModel, Identifier, NonNegativeInt, SignalGroup
from vigiscoutx.errors import CalibrationProtocolError, CalibrationStopError
from vigiscoutx.experiment.calibration.config import (
    CalibrationProtocolConfig,
    FusionCandidate,
    StopReason,
)

# --- D0 und δ_rel (DEC-053) ----------------------------------------------------------------------


def d0(ba_ref: float, ba_monitoring: float) -> float:
    """`D0 = BA_ref − BA_monitoring` (identisch mit `ΔBA`)."""
    if not (math.isfinite(ba_ref) and math.isfinite(ba_monitoring)):
        raise ValueError("Balanced Accuracies müssen endlich sein")
    return ba_ref - ba_monitoring


def d0_plus(value: float) -> float:
    return max(0.0, value)


def empirical_quantile(values: Sequence[float], q: float) -> float:
    """`Q_q = x_(⌈qN⌉)` über die sortierten Werte, ohne Interpolation."""
    if not values or not 0.0 < q <= 1.0:
        raise ValueError("nicht leere Werte und 0 < q ≤ 1 erforderlich")
    ordered = sorted(values)
    rank = math.ceil(Fraction(str(q)) * len(ordered))
    return ordered[rank - 1]


class RelevanceThreshold(FrozenModel):
    q090: float
    q095: float
    q099: float
    delta_rel: float
    n: NonNegativeInt


def relevance_threshold(
    d0_values: Sequence[float | None], config: CalibrationProtocolConfig
) -> RelevanceThreshold:
    """`δ_rel = Q0.95(D0⁺)` aus exakt `d0_repeats` gültigen Wiederholungen.

    `None` kennzeichnet einen nicht-validen Repeat (G4 `NOT_EVALUABLE`/`ERROR`) → STOP ohne
    Ersatzdraw.
    """
    if len(d0_values) != config.d0_repeats:
        raise ValueError(f"exakt {config.d0_repeats} D0-Wiederholungen erforderlich")
    if any(v is None or not math.isfinite(v) for v in d0_values):
        raise CalibrationStopError(StopReason.D0_REPEAT_NOT_VALID)
    plus = [d0_plus(v) for v in d0_values if v is not None]
    q090, q095, q099 = (empirical_quantile(plus, q) for q in config.quantiles)
    if not (math.isfinite(q095) and q095 > 0.0):
        raise CalibrationStopError(StopReason.DELTA_REL_INVALID, f"δ_rel = {q095}")
    return RelevanceThreshold(q090=q090, q095=q095, q099=q099, delta_rel=q095, n=len(plus))


# --- Severity-Selection (DEC-054) ---------------------------------------------------------------


def severity_median(degradations: Sequence[float]) -> float:
    """`median_r(BA_ref − BA_shift,r)`; gerade Anzahl: Mittel der beiden mittleren Werte."""
    if not degradations or not all(math.isfinite(v) for v in degradations):
        raise ValueError("endliche Degradationswerte erforderlich")
    return float(statistics.median(degradations))


def select_severity_triple(
    medians: Sequence[float | None], delta_rel: float, config: CalibrationProtocolConfig
) -> tuple[int, int, int]:
    """Triple `i1 < i2 < i3` (physische Indizes) mit minimalem
    `E = |m_i1/δ − 0.5| + |m_i2/δ − 1| + |m_i3/δ − 2|`; Gleichstand → physisch schwächeres Triple.

    `None` kennzeichnet einen technisch ungültigen Kandidaten.
    """
    if len(medians) != 5 or not (math.isfinite(delta_rel) and delta_rel > 0):
        raise ValueError("fünf Kandidaten und δ_rel > 0 erforderlich")
    valid = {i: m for i, m in enumerate(medians) if m is not None and math.isfinite(m)}
    if len(valid) < 3:
        raise CalibrationStopError(StopReason.FEWER_THAN_THREE_VALID_SEVERITIES)
    targets = config.severity_targets

    def objective(triple: tuple[int, ...]) -> float:
        return sum(abs(valid[i] / delta_rel - t) for i, t in zip(triple, targets, strict=True))

    best = min(itertools.combinations(sorted(valid), 3), key=lambda t: (objective(t), t))
    return (best[0], best[1], best[2])


# --- Raten, R-Stabilität (DEC-058) --------------------------------------------------------------


class Rate(FrozenModel):
    successes: NonNegativeInt
    trials: NonNegativeInt

    @model_validator(mode="after")
    def _bounded(self) -> Self:
        if self.successes > self.trials:
            raise ValueError("successes darf trials nicht übersteigen")
        return self

    @property
    def defined(self) -> bool:
        return self.trials > 0

    @property
    def value(self) -> Fraction:
        if not self.defined:
            raise CalibrationStopError(StopReason.SENSITIVITY_UNDEFINED, "trials = 0")
        return Fraction(self.successes, self.trials)

    def mcse(self) -> float:
        """`sqrt(p̂ (1 − p̂) / N)`."""
        p = float(self.value)
        return math.sqrt(p * (1.0 - p) / self.trials)


def count_rate(outcomes: Sequence[bool]) -> Rate:
    return Rate(successes=sum(outcomes), trials=len(outcomes))


class ProcedureOutcome(FrozenModel):
    """Status und Alarm eines verpflichtenden Verfahrens in einem Window (VALID ⇔ alarm gesetzt)."""

    status: SignalStatus
    alarm: StrictBool | None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.status is SignalStatus.VALID) != (self.alarm is not None):
            raise ValueError("VALID erfordert alarm true/false, sonst alarm = null")
        return self


class ShiftOutcome(ProcedureOutcome):
    """Verfahrensergebnis auf einem abrupten Shift-Window; `relevant` ⇔ `ΔBA ≥ δ_rel`."""

    relevant: StrictBool


def require_valid_alarms(outcomes: Sequence[ProcedureOutcome]) -> list[bool]:
    """Nur VALID-Ergebnisse gehen in Raten ein; `ERROR`/`NOT_EVALUABLE` → STOP (nie `false`)."""
    alarms: list[bool] = []
    for outcome in outcomes:
        if outcome.status is not SignalStatus.VALID or outcome.alarm is None:
            raise CalibrationStopError(
                StopReason.MANDATORY_SIGNAL_NOT_VALID, f"Status {outcome.status.value}"
            )
        alarms.append(outcome.alarm)
    return alarms


class MethodOutcomes(FrozenModel):
    """Ergebnisse eines der elf Verfahren je Wiederholung (Index = repeat_id, nested)."""

    method: Identifier
    clean: tuple[ProcedureOutcome, ...]
    shifts: tuple[tuple[ShiftOutcome, ...], ...]

    @model_validator(mode="after")
    def _aligned(self) -> Self:
        if len(self.clean) != len(self.shifts):
            raise ValueError("clean- und shift-Wiederholungen müssen gleich lang sein")
        return self


def sensitivity(outcomes: Sequence[ShiftOutcome]) -> Rate:
    """Gepoolte Sensitivität auf relevanten abrupten Shift-Windows; `TP + FN = 0` → STOP."""
    alarms = require_valid_alarms(outcomes)
    relevant = [a for a, o in zip(alarms, outcomes, strict=True) if o.relevant]
    if not relevant:
        raise CalibrationStopError(StopReason.SENSITIVITY_UNDEFINED, "TP + FN = 0")
    return count_rate(relevant)


def _stable(full: Rate, half: Rate, config: CalibrationProtocolConfig) -> bool:
    return (
        full.mcse() <= config.mcse_max
        and abs(float(full.value) - float(half.value)) <= config.stability_max_difference
    )


def _half(r: int, config: CalibrationProtocolConfig) -> int:
    return config.r_half_reference_at_100 if r == config.r_candidates[0] else r // 2


def validate_method_identities(
    outcomes: Mapping[str, MethodOutcomes], config: CalibrationProtocolConfig
) -> None:
    """Exakt die elf eingefrorenen Verfahren; Schlüssel und `MethodOutcomes.method` identisch.

    Wird vor jeder Status-, Raten- oder MCSE-Berechnung geprüft; Abweichung = Protokollfehler.
    """
    keys = list(outcomes)
    if len(keys) != len(config.r_methods) or set(keys) != set(config.r_methods):
        raise CalibrationProtocolError("R-Stabilität erfordert exakt die elf Verfahren")
    mismatched = sorted(k for k, v in outcomes.items() if v.method != k)
    if mismatched:
        raise CalibrationProtocolError(f"Verfahrensidentität weicht vom Schlüssel ab: {mismatched}")
    identities = [v.method for v in outcomes.values()]
    if len(set(identities)) != len(identities):
        raise CalibrationProtocolError("doppelte Verfahrensidentitäten")


def r_candidate_accepted(
    outcomes: Mapping[str, MethodOutcomes], r: int, config: CalibrationProtocolConfig
) -> bool:
    validate_method_identities(outcomes, config)
    half = _half(r, config)
    for method in config.r_methods:
        data = outcomes[method]
        if len(data.clean) < r:
            raise ValueError(f"{method}: weniger als {r} Wiederholungen")
        clean = require_valid_alarms(data.clean[:r])
        shifts = [o for rep in data.shifts[:r] for o in rep]
        require_valid_alarms(shifts)
        half_shifts = [o for rep in data.shifts[:half] for o in rep]
        if not (
            _stable(count_rate(clean), count_rate(clean[:half]), config)
            and _stable(sensitivity(shifts), sensitivity(half_shifts), config)
        ):
            return False
    return True


def select_r(outcomes: Mapping[str, MethodOutcomes], config: CalibrationProtocolConfig) -> int:
    """Erstes akzeptiertes R in `100 → 200 → 400`; sonst STOP → manuelle Entscheidung."""
    validate_method_identities(outcomes, config)
    for r in config.r_candidates:
        if r_candidate_accepted(outcomes, r, config):
            return r
    raise CalibrationStopError(StopReason.R_NOT_STABLE)


# --- Fusion A/B (DEC-059) -----------------------------------------------------------------------

_PRIMARY = (SignalGroup.G1_INPUT, SignalGroup.G2_DATA_QUALITY, SignalGroup.G3_PREDICTIVE)


def valid_alarm(result: SignalResult | GroupAssessment) -> bool:
    """Alarm eines verpflichtenden Signals; `ERROR`/`NOT_EVALUABLE` → Protokollfehler, STOP."""
    if result.status is not SignalStatus.VALID or result.alarm is None:
        raise CalibrationStopError(
            StopReason.MANDATORY_SIGNAL_NOT_VALID, f"Status {result.status.value}"
        )
    return result.alarm


def procedure_outcome(result: SignalResult | GroupAssessment) -> ProcedureOutcome:
    """Status-erhaltende Sicht auf ein Signal- oder Gruppenergebnis."""
    return ProcedureOutcome(status=result.status, alarm=result.alarm)


def fusion_alarm(candidate: FusionCandidate, groups: Sequence[GroupAssessment]) -> bool:
    """A: ≥ 1 alarmierende gültige Gruppe aus G1/G2/G3; B: ≥ 2."""
    if sorted(g.group for g in groups) != sorted(_PRIMARY):
        raise ValueError("Fusion erfordert genau die Gruppen G1, G2 und G3")
    alarms = sum(valid_alarm(g) for g in groups)
    return alarms >= (1 if candidate is FusionCandidate.FUSION_A else 2)


class FusionMetrics(FrozenModel):
    candidate: FusionCandidate
    clean_far: Rate
    sensitivity: Rate
    nonrelevant_shift_alarm_rate: Rate


def fusion_metrics(
    candidate: FusionCandidate,
    clean: Sequence[ProcedureOutcome],
    shifts: Sequence[ShiftOutcome],
) -> FusionMetrics:
    alarms = require_valid_alarms(shifts)
    return FusionMetrics(
        candidate=candidate,
        clean_far=count_rate(require_valid_alarms(clean)),
        sensitivity=sensitivity(shifts),
        nonrelevant_shift_alarm_rate=count_rate(
            [a for a, s in zip(alarms, shifts, strict=True) if not s.relevant]
        ),
    )


def select_fusion(
    metrics: Sequence[FusionMetrics], config: CalibrationProtocolConfig
) -> FusionCandidate:
    """Nur Kandidaten mit Clean FAR ≤ 0.05; dann höhere Sensitivität, niedrigere FAR,
    niedrigere nonrelevant_shift_alarm_rate; vollständiger Gleichstand → B."""
    if sorted(m.candidate for m in metrics) != sorted(FusionCandidate):
        raise ValueError("Fusion-Auswahl erfordert genau A und B")
    far_max = Fraction(str(config.far_max))
    eligible = [m for m in metrics if m.clean_far.value <= far_max]
    if not eligible:
        raise CalibrationStopError(StopReason.NO_FAR_ELIGIBLE_FUSION)
    if len(eligible) == 1:
        return eligible[0].candidate
    a, b = sorted(eligible, key=lambda m: m.candidate.value)
    if a.sensitivity.value != b.sensitivity.value:
        return max(eligible, key=lambda m: m.sensitivity.value).candidate
    if a.clean_far.value != b.clean_far.value:
        return min(eligible, key=lambda m: m.clean_far.value).candidate
    rates = (a.nonrelevant_shift_alarm_rate, b.nonrelevant_shift_alarm_rate)
    if not all(r.defined for r in rates):
        raise CalibrationStopError(StopReason.NONRELEVANT_RATE_UNDEFINED)
    if rates[0].value != rates[1].value:
        return min(eligible, key=lambda m: m.nonrelevant_shift_alarm_rate.value).candidate
    return FusionCandidate(config.fusion_full_tie)


def audit_passed(audit_far: Rate, config: CalibrationProtocolConfig) -> bool:
    """`FAR_audit ≤ 0.05`; ein Fehlschlag führt ausschließlich zu STOP → manuelle Entscheidung."""
    return audit_far.value <= Fraction(str(config.far_max))
