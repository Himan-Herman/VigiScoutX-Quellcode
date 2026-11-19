"""P4-03AI: HELD_OUT-Kennzahlen (Protokoll v1.1, §5–7) auf synthetischen Case-Ergebnissen."""

from typing import Any

import pytest

from vigiscoutx.evaluation.held_out_metrics import (
    ALL_METHODS,
    FUSION,
    ISOLATED_METHODS,
    MetricsInputError,
    ScoredCase,
    Thresholds,
    confusion,
    gradual_outcome,
    held_out_metrics,
    metric_lookup,
    ratio,
)

FAMILIES = ("NOISE", "CLASS")
THRESHOLDS = Thresholds(delta_rel=0.10, q090=0.05, q099=0.20)


def test_confusion_formulas_and_undefined_denominators() -> None:
    alarms = [True, True, False, False, True]
    relevant = [True, False, True, False, False]
    result = confusion(alarms, relevant)
    assert (result.tp, result.fp, result.tn, result.fn) == (1, 2, 1, 1)
    assert result.sensitivity.value == 1 / 2
    assert result.specificity.value == 1 / 3
    assert result.far_relevance.value == 2 / 3
    assert result.precision.value == 1 / 3
    assert result.f1.value == 2 / (2 + 2 + 1)
    empty = confusion([False, False], [False, False])
    assert empty.sensitivity.status == "UNDEFINED" and empty.sensitivity.value is None
    assert empty.precision.status == "UNDEFINED"
    assert empty.f1.status == "UNDEFINED"
    assert empty.specificity.value == 1.0
    assert ratio(0, 0).status == "UNDEFINED"
    with pytest.raises(MetricsInputError):
        confusion([True], [True, False])


@pytest.mark.parametrize(
    ("alarms", "deltas", "category", "delay", "lead", "t0"),
    [
        ((False, False, True, True), (0.0, 0.0, 0.2, 0.3), "DETECTED_DELAY", 0, None, False),
        ((False, False, False, True), (0.0, 0.0, 0.2, 0.3), "DETECTED_DELAY", 1, None, False),
        ((False, True, True, True), (0.0, 0.0, 0.2, 0.3), "DETECTED_LEAD", None, 1, False),
        ((True, False, False, False), (0.0, 0.0, 0.2, 0.3), "NOT_DETECTED", None, None, True),
        (
            (False, True, True, True),
            (0.0, 0.0, 0.05, 0.09),
            "NO_RELEVANT_DEGRADATION",
            None,
            None,
            False,
        ),
        (
            (False, True, True, True),
            (0.1, 0.2, 0.3, 0.4),
            "RELEVANT_AT_CLEAN_BASELINE",
            None,
            None,
            False,
        ),
    ],
)
def test_gradual_detection_categories(
    alarms: tuple[bool, ...],
    deltas: tuple[float, ...],
    category: str,
    delay: int | None,
    lead: int | None,
    t0: bool,
) -> None:
    outcome = gradual_outcome("NOISE", 1, FUSION, alarms, deltas, 0.10)
    assert (outcome.category, outcome.delay, outcome.lead, outcome.alarm_at_t0) == (
        category,
        delay,
        lead,
        t0,
    )
    assert outcome.t_alarm is None or outcome.t_alarm >= 1  # Alarm bei t = 0 zählt nicht


def _case(
    case_id: str, *, clean: bool, family: str | None, level: int | None, repeat: int, **kw: Any
) -> ScoredCase:
    alarm = kw.get("alarm", {})
    return ScoredCase(
        case_id=case_id,
        clean=clean,
        family=family,
        level=level,
        repeat_id=repeat,
        alarms=tuple(sorted((m, alarm.get(m, False)) for m in ALL_METHODS)),
        delta_ba=kw.get("delta"),
    )


def _cases(repeats: int = 2) -> list[ScoredCase]:
    cases = [
        _case(f"c{r}", clean=True, family=None, level=None, repeat=r, alarm={FUSION: r == 1})
        for r in range(1, repeats + 1)
    ]
    for family in FAMILIES:
        for r in range(1, repeats + 1):
            for level in range(4):
                delta = (0.0, 0.07, 0.15, 0.25)[level]
                alarm = {
                    FUSION: level >= 2,
                    "G1_MMD_EMBEDDING": level >= 1,
                    "G3_CONFIDENCE_KS": level == 3 and r == 1,
                }
                cases.append(
                    _case(
                        f"{family}{r}{level}",
                        clean=False,
                        family=family,
                        level=level,
                        repeat=r,
                        alarm=alarm,
                        delta=delta,
                    )
                )
    return cases


def test_metrics_pool_only_abrupt_cases_and_report_clean_far_separately() -> None:
    metrics = held_out_metrics(_cases(), THRESHOLDS, families=FAMILIES, repeats=2)
    assert (metrics.n_clean, metrics.n_baselines, metrics.n_abrupt) == (2, 4, 12)
    pooled = metric_lookup(metrics)
    fusion = pooled[("DELTA_REL", FUSION)]
    assert (fusion.tp, fusion.fp, fusion.tn, fusion.fn) == (8, 0, 4, 0)
    g1 = pooled[("DELTA_REL", "G1_MMD_EMBEDDING")]
    assert (g1.tp, g1.fp, g1.tn, g1.fn) == (8, 4, 0, 0)
    assert g1.specificity.value == 0.0 and g1.far_relevance.value == 1.0
    # Sensitivitätsanalyse: dieselben Alarme, andere Schwelle
    assert pooled[("Q090", "G1_MMD_EMBEDDING")].tp == 12
    assert pooled[("Q099", FUSION)].tp == 4 and pooled[("Q099", FUSION)].fp == 4
    clean_far = dict(metrics.clean_far)
    assert clean_far[FUSION].numerator == 1 and clean_far[FUSION].denominator == 2
    assert dict(metrics.baseline_alarms)[FUSION] == 0
    table = next(t for t in metrics.confusion if t.threshold == "DELTA_REL")
    total = sum(
        s.confusion.tp + s.confusion.fp + s.confusion.tn + s.confusion.fn
        for s in table.by_family_level
        if s.method == FUSION
    )
    assert total == 12
    assert {s.method for s in table.pooled} == set(ALL_METHODS)
    assert "FUSION_B" not in {s.method for s in table.pooled}


def test_gradual_summary_keeps_non_detections_and_paired_blocks() -> None:
    metrics = held_out_metrics(_cases(), THRESHOLDS, families=FAMILIES, repeats=2)
    assert metrics.n_trajectories == 4
    summary = next(s for s in metrics.gradual_summary if s.family is None and s.method == FUSION)
    assert dict(summary.categories)["DETECTED_DELAY"] == 4
    assert summary.delay_counts == ((0, 4),)
    g3 = next(
        s for s in metrics.gradual_summary if s.family is None and s.method == "G3_CONFIDENCE_KS"
    )
    assert g3.not_detected == 2 and dict(g3.categories)["DETECTED_DELAY"] == 2
    blocks = {(b.repeat_id, b.method): b for b in metrics.repeat_blocks}
    assert all(blocks[(r, FUSION)].tp_minus_fusion == 0 for r in (1, 2))
    assert blocks[(1, "G1_MMD_EMBEDDING")].fp_minus_fusion == 2
    spread = next(
        s
        for s in metrics.repeat_spread
        if (s.method, s.quantity) == ("G3_CONFIDENCE_KS", "tp_minus_fusion")
    )
    assert (spread.minimum, spread.maximum, spread.n_repeats) == (-4, -2, 2)
    assert {s.method for s in metrics.repeat_spread} == set(ISOLATED_METHODS)
    assert metrics.limitations


@pytest.mark.parametrize("defect", ["missing-case", "duplicate", "fusion-b", "no-g4", "clean"])
def test_incomplete_or_inconsistent_case_results_stop(defect: str) -> None:
    cases = _cases()
    if defect == "missing-case":
        cases = cases[:-1]
    elif defect == "duplicate":
        cases[-1] = cases[-2].model_copy(update={"case_id": "dup"})
    elif defect == "fusion-b":
        cases[3] = cases[3].model_copy(update={"alarms": (*cases[3].alarms, ("FUSION_B", True))})
    elif defect == "no-g4":
        cases[4] = cases[4].model_copy(update={"delta_ba": None})
    else:
        cases = cases[1:]
    with pytest.raises(MetricsInputError):
        held_out_metrics(cases, THRESHOLDS, families=FAMILIES, repeats=2)
