"""Kennzahlen der HELD_OUT-Evaluation (Protokoll v1.1, §5–7), rein aus Case-Ergebnissen.

- Primäre Evaluationsmenge: ausschließlich die abrupten Shift-Cases (L1–L3); Ground Truth aus dem
  getrennten G4: `y = 1 ⇔ ΔBA ≥ Schwelle` (δ_rel; Sensitivitätsanalyse Q0.90/Q0.99).
- Gepoolte ganzzahlige TP/FP/TN/FN (Micro-Aggregation) und vollständige Zähler/Nenner je Familie,
  Level und Familie × Level. Nenner null → `UNDEFINED`, nie ein Bibliotheksdefault.
- Separate Clean-FAR über die Clean-Windows; Alarme bei t = 0 gesondert.
- Schrittweise Verläufe L0 → L1 → L2 → L3 aus denselben Case-Ergebnissen (§6).
- Je Repeat ein gepaarter Block; Unterschiede zu `FUSION_A` nur deskriptiv (§7), keine
  Konfidenzintervalle oder p-Werte über Windows.

`FUSION_B` wird weder berechnet noch akzeptiert.
"""

import statistics
from collections.abc import Mapping, Sequence
from typing import Final, Literal

from vigiscoutx.domain.types import FrozenModel, Identifier, NonNegativeInt, PositiveInt
from vigiscoutx.errors import IntegrityError

ISOLATED_METHODS: Final = (
    "G1_MMD_EMBEDDING",
    "G2_MEAN_LUMINANCE_KS",
    "G2_LUMINANCE_STD_KS",
    "G2_VARIANCE_OF_LAPLACIAN_KS",
    "G2_IMMERKAER_NOISE_VARIANCE_KS",
    "G3_OUTPUT_MUKS",
    "G3_CONFIDENCE_KS",
)
FUSION: Final = "FUSION_A"
PRIMARY_METHODS: Final = (*ISOLATED_METHODS, FUSION)
DIAGNOSTIC_GROUPS: Final = ("G2_GROUP", "G3_GROUP")
ALL_METHODS: Final = (*PRIMARY_METHODS, *DIAGNOSTIC_GROUPS)
THRESHOLDS: Final = ("DELTA_REL", "Q090", "Q099")


class MetricsInputError(IntegrityError):
    """Case-Ergebnisse unvollständig oder inkonsistent (STOP, keine Ersatzfälle)."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MetricsInputError(message)


class Ratio(FrozenModel):
    numerator: NonNegativeInt
    denominator: NonNegativeInt
    value: float | None
    status: Literal["DEFINED", "UNDEFINED"]


def ratio(numerator: int, denominator: int) -> Ratio:
    if denominator == 0:
        return Ratio(numerator=numerator, denominator=0, value=None, status="UNDEFINED")
    return Ratio(
        numerator=numerator,
        denominator=denominator,
        value=numerator / denominator,
        status="DEFINED",
    )


class Confusion(FrozenModel):
    tp: NonNegativeInt
    fp: NonNegativeInt
    tn: NonNegativeInt
    fn: NonNegativeInt
    sensitivity: Ratio
    specificity: Ratio
    far_relevance: Ratio
    precision: Ratio
    f1: Ratio


def confusion(alarms: Sequence[bool], relevant: Sequence[bool]) -> Confusion:
    _require(len(alarms) == len(relevant), "Alarme und Ground Truth nicht gepaart")
    tp = sum(a and y for a, y in zip(alarms, relevant, strict=True))
    fn = sum(not a and y for a, y in zip(alarms, relevant, strict=True))
    fp = sum(a and not y for a, y in zip(alarms, relevant, strict=True))
    tn = sum(not a and not y for a, y in zip(alarms, relevant, strict=True))
    return Confusion(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        sensitivity=ratio(tp, tp + fn),
        specificity=ratio(tn, tn + fp),
        far_relevance=ratio(fp, tn + fp),
        precision=ratio(tp, tp + fp),
        f1=ratio(2 * tp, 2 * tp + fp + fn),
    )


class ScoredCase(FrozenModel):
    """Label-freie Alarme eines Cases plus das getrennt verknüpfte G4 (`delta_ba`)."""

    case_id: Identifier
    clean: bool
    family: Identifier | None
    level: NonNegativeInt | None
    repeat_id: PositiveInt
    alarms: tuple[tuple[Identifier, bool], ...]
    delta_ba: float | None

    def alarm(self, method: str) -> bool:
        return dict(self.alarms)[method]


class Thresholds(FrozenModel):
    delta_rel: float
    q090: float
    q099: float

    def value(self, name: str) -> float:
        return {"DELTA_REL": self.delta_rel, "Q090": self.q090, "Q099": self.q099}[name]


def _relevant(case: ScoredCase, threshold: float) -> bool:
    _require(case.delta_ba is not None, f"{case.case_id}: G4 ohne ΔBA")
    assert case.delta_ba is not None
    return case.delta_ba >= threshold  # identisch zur G4-Relevanzregel (`assess_performance`)


class StratumConfusion(FrozenModel):
    family: Identifier | None
    level: NonNegativeInt | None
    method: Identifier
    confusion: Confusion


class ThresholdTable(FrozenModel):
    threshold: Literal["DELTA_REL", "Q090", "Q099"]
    threshold_value: float
    n_cases: NonNegativeInt
    n_relevant: NonNegativeInt
    pooled: tuple[StratumConfusion, ...]
    by_family: tuple[StratumConfusion, ...]
    by_level: tuple[StratumConfusion, ...]
    by_family_level: tuple[StratumConfusion, ...]


def _table(name: str, value: float, shifts: Sequence[ScoredCase]) -> ThresholdTable:
    def block(
        cases: Sequence[ScoredCase], family: str | None, level: int | None
    ) -> list[StratumConfusion]:
        relevant = [_relevant(c, value) for c in cases]
        return [
            StratumConfusion(
                family=family,
                level=level,
                method=m,
                confusion=confusion([c.alarm(m) for c in cases], relevant),
            )
            for m in ALL_METHODS
        ]

    families = sorted({c.family for c in shifts if c.family is not None})
    levels = sorted({c.level for c in shifts if c.level is not None})
    return ThresholdTable.model_validate(
        {
            "threshold": name,
            "threshold_value": value,
            "n_cases": len(shifts),
            "n_relevant": sum(_relevant(c, value) for c in shifts),
            "pooled": block(shifts, None, None),
            "by_family": [
                s for f in families for s in block([c for c in shifts if c.family == f], f, None)
            ],
            "by_level": [
                s for lv in levels for s in block([c for c in shifts if c.level == lv], None, lv)
            ],
            "by_family_level": [
                s
                for f in families
                for lv in levels
                for s in block([c for c in shifts if (c.family, c.level) == (f, lv)], f, lv)
            ],
        }
    )


# --- Schrittweise Detektion (§6) ----------------------------------------------------------------


class GradualOutcome(FrozenModel):
    family: Identifier
    repeat_id: PositiveInt
    method: Identifier
    t_rel: NonNegativeInt | None
    t_alarm: PositiveInt | None
    alarm_at_t0: bool
    category: Literal[
        "DETECTED_DELAY",
        "DETECTED_LEAD",
        "NOT_DETECTED",
        "NO_RELEVANT_DEGRADATION",
        "RELEVANT_AT_CLEAN_BASELINE",
    ]
    delay: NonNegativeInt | None
    lead: PositiveInt | None


def gradual_outcome(
    family: str,
    repeat_id: int,
    method: str,
    alarms: Sequence[bool],
    delta_ba: Sequence[float],
    delta_rel: float,
) -> GradualOutcome:
    """`t_rel = min{t : ΔBA_t ≥ δ_rel}`, `t_alarm⁺ = min{t ≥ 1 : alarm_t}` über t = 0 … 3."""
    _require(len(alarms) == len(delta_ba) == 4, "Verlauf erfordert genau t = 0 … 3")
    t_rel = next((t for t, d in enumerate(delta_ba) if d >= delta_rel), None)
    t_alarm = next((t for t in range(1, 4) if alarms[t]), None)
    delay: int | None = None
    lead: int | None = None
    if t_rel is None:
        category = "NO_RELEVANT_DEGRADATION"
    elif t_rel == 0:
        category = "RELEVANT_AT_CLEAN_BASELINE"
    elif t_alarm is None:
        category = "NOT_DETECTED"
    elif t_alarm >= t_rel:
        category, delay = "DETECTED_DELAY", t_alarm - t_rel
    else:
        category, lead = "DETECTED_LEAD", t_rel - t_alarm
    return GradualOutcome.model_validate(
        {
            "family": family,
            "repeat_id": repeat_id,
            "method": method,
            "t_rel": t_rel,
            "t_alarm": t_alarm,
            "alarm_at_t0": alarms[0],
            "category": category,
            "delay": delay,
            "lead": lead,
        }
    )


class GradualSummary(FrozenModel):
    family: Identifier | None
    method: Identifier
    n_trajectories: NonNegativeInt
    categories: tuple[tuple[Identifier, NonNegativeInt], ...]
    delay_counts: tuple[tuple[NonNegativeInt, NonNegativeInt], ...]
    lead_counts: tuple[tuple[PositiveInt, NonNegativeInt], ...]
    not_detected: NonNegativeInt
    alarms_at_t0: NonNegativeInt


_CATEGORIES: Final = (
    "DETECTED_DELAY",
    "DETECTED_LEAD",
    "NOT_DETECTED",
    "NO_RELEVANT_DEGRADATION",
    "RELEVANT_AT_CLEAN_BASELINE",
)


def _gradual_summary(
    family: str | None, method: str, items: Sequence[GradualOutcome]
) -> GradualSummary:
    delays = [o.delay for o in items if o.delay is not None]
    leads = [o.lead for o in items if o.lead is not None]
    return GradualSummary(
        family=family,
        method=method,
        n_trajectories=len(items),
        categories=tuple((c, sum(o.category == c for o in items)) for c in _CATEGORIES),
        delay_counts=tuple((d, delays.count(d)) for d in sorted(set(delays))),
        lead_counts=tuple((lv, leads.count(lv)) for lv in sorted(set(leads))),
        not_detected=sum(o.category == "NOT_DETECTED" for o in items),
        alarms_at_t0=sum(o.alarm_at_t0 for o in items),
    )


# --- Gepaarte Repeat-Blöcke (§7) ----------------------------------------------------------------


class RepeatBlock(FrozenModel):
    repeat_id: PositiveInt
    method: Identifier
    tp: NonNegativeInt
    fp: NonNegativeInt
    tp_minus_fusion: int
    fp_minus_fusion: int


class Spread(FrozenModel):
    """Deskriptive Streuung über Repeats (Populations-SD, ddof = 0); kein Intervall."""

    method: Identifier
    quantity: Literal["tp_minus_fusion", "fp_minus_fusion"]
    n_repeats: PositiveInt
    mean: float
    sd: float
    minimum: int
    maximum: int


def _spread(method: str, quantity: str, values: Sequence[int]) -> Spread:
    return Spread.model_validate(
        {
            "method": method,
            "quantity": quantity,
            "n_repeats": len(values),
            "mean": statistics.fmean(values),
            "sd": statistics.pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    )


# --- Gesamtauswertung ---------------------------------------------------------------------------


class HeldOutMetrics(FrozenModel):
    schema_version: Literal[1]
    thresholds: Thresholds
    n_clean: NonNegativeInt
    n_baselines: NonNegativeInt
    n_abrupt: NonNegativeInt
    n_trajectories: NonNegativeInt
    clean_far: tuple[tuple[Identifier, Ratio], ...]
    baseline_alarms: tuple[tuple[Identifier, NonNegativeInt], ...]
    confusion: tuple[ThresholdTable, ...]
    gradual: tuple[GradualOutcome, ...]
    gradual_summary: tuple[GradualSummary, ...]
    repeat_blocks: tuple[RepeatBlock, ...]
    repeat_spread: tuple[Spread, ...]
    limitations: tuple[str, ...]


LIMITATIONS: Final = (
    "Keine Behauptung unabhängiger Patienten oder Szenarien; keine binomialen Konfidenzintervalle "
    "oder p-Werte über Windows.",
    "Repeat-Streuung beschreibt die Simulation auf dieser endlichen Partition, kein "
    "populationsbezogenes Intervall.",
    "Precision/F1 gelten nur für die konstruierte Prävalenz dieses Benchmarks.",
    "Schrittweise Verläufe sind kontrollierte Pseudozeitverläufe aus denselben L0–L3-Cases.",
)


def held_out_metrics(
    cases: Sequence[ScoredCase],
    thresholds: Thresholds,
    *,
    families: Sequence[str],
    repeats: int,
) -> HeldOutMetrics:
    """Alle Kennzahlen aus genau einem vollständigen Satz Case-Ergebnisse; Lücke → STOP."""
    methods = set(ALL_METHODS)
    for c in cases:
        _require({m for m, _ in c.alarms} == methods, f"{c.case_id}: Verfahrensmenge weicht ab")
    clean = [c for c in cases if c.clean]
    scenario = [c for c in cases if not c.clean]
    _require(len(clean) == repeats, "Clean-Windows fehlen")
    _require(len(scenario) == len(families) * 4 * repeats, "Szenario-Cases fehlen")
    index: dict[tuple[str, int, int], ScoredCase] = {}
    for c in scenario:
        _require(c.family is not None and c.level is not None, f"{c.case_id}: ohne Familie/Level")
        assert c.family is not None and c.level is not None
        key = (c.family, c.repeat_id, c.level)
        _require(key not in index, f"doppelter Case {key}")
        index[key] = c
    for f in families:
        for r in range(1, repeats + 1):
            for lv in range(4):
                _require((f, r, lv) in index, f"Case fehlt: {f}/{r}/L{lv}")
    shifts = [c for c in scenario if c.level != 0]
    baselines = [c for c in scenario if c.level == 0]

    gradual = [
        gradual_outcome(
            f,
            r,
            m,
            [index[(f, r, lv)].alarm(m) for lv in range(4)],
            [_delta(index[(f, r, lv)]) for lv in range(4)],
            thresholds.delta_rel,
        )
        for f in families
        for r in range(1, repeats + 1)
        for m in PRIMARY_METHODS
    ]
    summaries = [
        _gradual_summary(None, m, [o for o in gradual if o.method == m]) for m in PRIMARY_METHODS
    ] + [
        _gradual_summary(f, m, [o for o in gradual if (o.family, o.method) == (f, m)])
        for f in families
        for m in PRIMARY_METHODS
    ]

    blocks: list[RepeatBlock] = []
    for r in range(1, repeats + 1):
        repeat_cases = [c for c in shifts if c.repeat_id == r]
        relevant = [_relevant(c, thresholds.delta_rel) for c in repeat_cases]
        counts = {
            m: confusion([c.alarm(m) for c in repeat_cases], relevant) for m in PRIMARY_METHODS
        }
        fusion = counts[FUSION]
        blocks += [
            RepeatBlock(
                repeat_id=r,
                method=m,
                tp=counts[m].tp,
                fp=counts[m].fp,
                tp_minus_fusion=counts[m].tp - fusion.tp,
                fp_minus_fusion=counts[m].fp - fusion.fp,
            )
            for m in PRIMARY_METHODS
        ]
    spreads = [
        _spread(m, q, [getattr(b, q) for b in blocks if b.method == m])
        for m in ISOLATED_METHODS
        for q in ("tp_minus_fusion", "fp_minus_fusion")
    ]
    return HeldOutMetrics(
        schema_version=1,
        thresholds=thresholds,
        n_clean=len(clean),
        n_baselines=len(baselines),
        n_abrupt=len(shifts),
        n_trajectories=len(families) * repeats,
        clean_far=tuple(
            (m, ratio(sum(c.alarm(m) for c in clean), len(clean))) for m in ALL_METHODS
        ),
        baseline_alarms=tuple((m, sum(c.alarm(m) for c in baselines)) for m in ALL_METHODS),
        confusion=tuple(_table(t, thresholds.value(t), shifts) for t in THRESHOLDS),
        gradual=tuple(gradual),
        gradual_summary=tuple(summaries),
        repeat_blocks=tuple(blocks),
        repeat_spread=tuple(spreads),
        limitations=LIMITATIONS,
    )


def _delta(case: ScoredCase) -> float:
    _require(case.delta_ba is not None, f"{case.case_id}: G4 ohne ΔBA")
    assert case.delta_ba is not None
    return case.delta_ba


def metric_lookup(metrics: HeldOutMetrics) -> Mapping[tuple[str, str], Confusion]:
    """`(threshold, method) → gepoolte Konfusionsmatrix` (Bequemlichkeit für Berichte/Tests)."""
    return {(t.threshold, s.method): s.confusion for t in metrics.confusion for s in t.pooled}
