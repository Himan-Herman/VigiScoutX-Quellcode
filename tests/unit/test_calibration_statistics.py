"""D0/δ_rel, Severity-Auswahl, R-Stabilität und Fusion A/B (nur synthetische Werte)."""

from typing import Any

import pytest

from calibration_fixtures import CONFIG, clean, method_outcomes, shift
from signal_factories import group_assessment
from vigiscoutx.domain.assessments import SignalStatus
from vigiscoutx.domain.types import SignalGroup
from vigiscoutx.errors import CalibrationProtocolError, CalibrationStopError
from vigiscoutx.experiment.calibration.config import FusionCandidate, StopReason
from vigiscoutx.experiment.calibration.statistics import (
    FusionMetrics,
    MethodOutcomes,
    ProcedureOutcome,
    Rate,
    ShiftOutcome,
    audit_passed,
    count_rate,
    d0,
    d0_plus,
    empirical_quantile,
    fusion_alarm,
    fusion_metrics,
    r_candidate_accepted,
    relevance_threshold,
    select_fusion,
    select_r,
    select_severity_triple,
    sensitivity,
    severity_median,
)

A, B = FusionCandidate.FUSION_A, FusionCandidate.FUSION_B

# --- D0 und Quantile ----------------------------------------------------------------------------


def test_d0_is_reference_minus_monitoring_with_positive_part() -> None:
    assert d0(0.8, 0.7) == pytest.approx(0.1)
    assert d0(0.7, 0.8) == pytest.approx(-0.1)
    assert d0_plus(d0(0.7, 0.8)) == 0.0 and d0_plus(0.1) == 0.1


def test_quantile_is_nearest_rank_without_interpolation() -> None:
    values = [float(v) for v in range(1, 1001)]
    assert empirical_quantile(values, 0.90) == 900.0
    assert empirical_quantile(values, 0.95) == 950.0
    assert empirical_quantile(values, 0.99) == 990.0
    assert empirical_quantile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0  # interpoliert wäre 2.5
    assert empirical_quantile([1.0, 2.0, 3.0], 0.5) == 2.0


def test_relevance_threshold_from_1000_repeats() -> None:
    values = [(i % 100) / 1000 - 0.02 for i in range(1000)]
    result = relevance_threshold(values, CONFIG)
    plus = sorted(max(0.0, v) for v in values)
    assert result.delta_rel == result.q095 == plus[949]
    assert (result.q090, result.q099) == (plus[899], plus[989])
    assert result.n == 1000


def test_relevance_threshold_stop_rules() -> None:
    with pytest.raises(ValueError):
        relevance_threshold([0.1] * 999, CONFIG)
    with pytest.raises(CalibrationStopError) as missing:
        relevance_threshold([0.1] * 999 + [None], CONFIG)
    assert missing.value.reason == StopReason.D0_REPEAT_NOT_VALID
    with pytest.raises(CalibrationStopError) as zero:
        relevance_threshold([-0.1] * 1000, CONFIG)
    assert zero.value.reason == StopReason.DELTA_REL_INVALID


# --- Severity ----------------------------------------------------------------------------------


def test_severity_median_even_count() -> None:
    assert severity_median([0.1, 0.4, 0.2, 0.3]) == pytest.approx(0.25)


def test_severity_triple_matches_targets() -> None:
    assert select_severity_triple([0.01, 0.05, 0.10, 0.2, 0.4], 0.1, CONFIG) == (1, 2, 3)
    # (0,3,4), (1,3,4), (2,3,4) haben dasselbe E = 0.5 → physisch schwächstes Triple.
    assert select_severity_triple([0.0, 0.0, 0.0, 0.05, 0.1], 0.05, CONFIG) == (0, 3, 4)


def test_severity_tie_prefers_physically_weaker_triple() -> None:
    assert select_severity_triple([0.05, 0.05, 0.1, 0.2, 0.2], 0.1, CONFIG) == (0, 2, 3)


def test_severity_invalid_candidates_and_stop() -> None:
    assert select_severity_triple([None, 0.05, 0.1, None, 0.2], 0.1, CONFIG) == (1, 2, 4)
    with pytest.raises(CalibrationStopError) as stop:
        select_severity_triple([None, 0.05, None, None, 0.2], 0.1, CONFIG)
    assert stop.value.reason == StopReason.FEWER_THAN_THREE_VALID_SEVERITIES


# --- R-Stabilität ------------------------------------------------------------------------------


def _outcomes(clean_every: int, relevant: bool = True, n: int = 400) -> dict[str, MethodOutcomes]:
    data = method_outcomes([i % clean_every == 0 for i in range(n)])
    if not relevant:
        shifts = tuple(tuple(shift(True, relevant=False) for _ in range(12)) for _ in range(n))
        data = {m: o.model_copy(update={"shifts": shifts}) for m, o in data.items()}
    return data


def _with_clean(data: dict[str, MethodOutcomes], method: str, alarms: Any) -> None:
    data[method] = data[method].model_copy(update={"clean": tuple(clean(a) for a in alarms)})


def test_mcse_formula() -> None:
    rate = Rate(successes=5, trials=100)
    assert rate.mcse() == pytest.approx((0.05 * 0.95 / 100) ** 0.5)


def test_first_acceptable_r_is_selected() -> None:
    assert select_r(_outcomes(clean_every=20), CONFIG) == 100  # FAR 5 %, MCSE ≈ 0.022
    assert select_r(_outcomes(clean_every=10), CONFIG) == 200  # MCSE(100) = 0.03 > 0.025


def test_p_half_regression_is_accepted_exactly_at_r_400() -> None:
    """Unabhängig spezifiziert: p = 0.5 in jeder Hälfte jedes nested Präfixes.

    R = 100: MCSE = sqrt(0.25/100) = 0.05 > 0.025 → abgelehnt.
    R = 200: MCSE = sqrt(0.25/200) ≈ 0.0354 > 0.025 → abgelehnt.
    R = 400: MCSE = sqrt(0.5 · 0.5 / 400) = 0.025 ≤ 0.025 und |p_400 − p_200| = 0 → akzeptiert.
    """
    data = _outcomes(clean_every=20)
    alarms = [i % 2 == 0 for i in range(400)]
    _with_clean(data, "G1_MMD_EMBEDDING", alarms)
    assert sum(alarms[:400]) / 400 == 0.5 and sum(alarms[:200]) / 200 == 0.5
    assert sum(alarms[:100]) / 100 == 0.5 and sum(alarms[:50]) / 50 == 0.5
    full = Rate(successes=200, trials=400)
    assert full.mcse() == 0.025 == (0.5 * 0.5 / 400) ** 0.5
    assert abs(float(full.value) - 0.5) <= CONFIG.stability_max_difference
    assert not r_candidate_accepted(data, 100, CONFIG)
    assert not r_candidate_accepted(data, 200, CONFIG)
    assert r_candidate_accepted(data, 400, CONFIG)
    assert select_r(data, CONFIG) == 400


def test_r_nested_uses_first_50_as_reference_at_100() -> None:
    data = _outcomes(clean_every=20)
    # erste 50 ohne Clean-Alarm, danach dichter: |p_100 − p_50| = 0.06 > 0.025.
    _with_clean(data, "G1_MMD_EMBEDDING", [i >= 50 and i % 8 == 0 for i in range(400)])
    assert select_r(data, CONFIG) != 100


def test_r_not_stable_is_stop() -> None:
    data = _outcomes(clean_every=20)
    # 10 % in 0–99, 40 % in 100–199, 60 % in 200–399: MCSE bzw. Halbierungsdifferenz scheitern.
    _with_clean(
        data,
        "G2_GROUP",
        [
            (i % 10 == 0) if i < 100 else (i % 10 < 4) if i < 200 else (i % 10 < 6)
            for i in range(400)
        ],
    )
    with pytest.raises(CalibrationStopError) as stop:
        select_r(data, CONFIG)
    assert stop.value.reason == StopReason.R_NOT_STABLE


def test_r_requires_exactly_eleven_methods() -> None:
    missing = _outcomes(clean_every=20)
    missing.pop("FUSION_B")
    with pytest.raises(CalibrationProtocolError):
        select_r(missing, CONFIG)
    extra = _outcomes(clean_every=20)
    extra["EXTRA"] = extra["G1_MMD_EMBEDDING"].model_copy(update={"method": "EXTRA"})
    with pytest.raises(CalibrationProtocolError):
        select_r(extra, CONFIG)


class _RateSpy:
    """Stellt sicher, dass vor der Identitätsprüfung keine Rate berechnet wird."""

    calls = 0


def test_all_values_fake_is_rejected_before_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vigiscoutx.experiment.calibration import statistics as stats

    data = {m: o.model_copy(update={"method": "FAKE"}) for m, o in _outcomes(20).items()}
    original = stats.count_rate

    def spy(outcomes: Any) -> Rate:
        _RateSpy.calls += 1
        return original(outcomes)

    monkeypatch.setattr(stats, "count_rate", spy)
    with pytest.raises(CalibrationProtocolError, match="Verfahrensidentität"):
        select_r(data, CONFIG)
    with pytest.raises(CalibrationProtocolError):
        r_candidate_accepted(data, 100, CONFIG)
    assert _RateSpy.calls == 0


def test_single_mismatch_and_swapped_identities_are_rejected() -> None:
    one = _outcomes(clean_every=20)
    one["G2_GROUP"] = one["G2_GROUP"].model_copy(update={"method": "G3_GROUP"})
    with pytest.raises(CalibrationProtocolError):
        select_r(one, CONFIG)
    swapped = _outcomes(clean_every=20)
    a, b = swapped["G1_MMD_EMBEDDING"], swapped["FUSION_A"]
    swapped["G1_MMD_EMBEDDING"], swapped["FUSION_A"] = b, a
    with pytest.raises(CalibrationProtocolError, match="Verfahrensidentität"):
        select_r(swapped, CONFIG)


def test_exact_identity_mapping_is_accepted() -> None:
    data = _outcomes(clean_every=20)
    assert all(key == value.method for key, value in data.items())
    assert select_r(data, CONFIG) == 100


@pytest.mark.parametrize("status", ["ERROR", "NOT_EVALUABLE"])
@pytest.mark.parametrize("where", ["clean", "shift"])
def test_non_valid_mandatory_outcome_stops_r(status: str, where: str) -> None:
    data = _outcomes(clean_every=20)
    bad = ProcedureOutcome(status=SignalStatus(status), alarm=None)
    method = data["G3_OUTPUT_MUKS"]
    if where == "clean":
        clean_outcomes = (*method.clean[:10], bad, *method.clean[11:])
        data["G3_OUTPUT_MUKS"] = method.model_copy(update={"clean": clean_outcomes})
    else:
        bad_shift = ShiftOutcome(status=SignalStatus(status), alarm=None, relevant=True)
        shifts = ((*method.shifts[0][:-1], bad_shift), *method.shifts[1:])
        data["G3_OUTPUT_MUKS"] = method.model_copy(update={"shifts": shifts})
    with pytest.raises(CalibrationStopError) as stop:
        select_r(data, CONFIG)
    assert stop.value.reason == StopReason.MANDATORY_SIGNAL_NOT_VALID


def test_outcome_status_and_alarm_are_consistent() -> None:
    with pytest.raises(ValueError):
        ProcedureOutcome(status=SignalStatus.ERROR, alarm=False)
    with pytest.raises(ValueError):
        ProcedureOutcome(status=SignalStatus.VALID, alarm=None)


def test_undefined_sensitivity_stops() -> None:
    with pytest.raises(CalibrationStopError) as stop:
        select_r(_outcomes(clean_every=20, relevant=False), CONFIG)
    assert stop.value.reason == StopReason.SENSITIVITY_UNDEFINED
    with pytest.raises(CalibrationStopError):
        sensitivity([shift(True, relevant=False)])


# --- Fusion ------------------------------------------------------------------------------------


def _groups(*alarms: bool) -> list[Any]:
    groups = (SignalGroup.G1_INPUT, SignalGroup.G2_DATA_QUALITY, SignalGroup.G3_PREDICTIVE)
    result = []
    for group, alarm in zip(groups, alarms, strict=True):
        extra = (
            {"substatus": "BOTH" if alarm else "NONE"} if group is SignalGroup.G3_PREDICTIVE else {}
        )
        result.append(group_assessment(group=group, alarm=alarm, **extra))
    return result


@pytest.mark.parametrize(
    ("alarms", "expected_a", "expected_b"),
    [
        ((False, False, False), False, False),
        ((True, False, False), True, False),
        ((True, False, True), True, True),
        ((True, True, True), True, True),
    ],
)
def test_fusion_a_and_b(alarms: tuple[bool, ...], expected_a: bool, expected_b: bool) -> None:
    groups = _groups(*alarms)
    assert fusion_alarm(A, groups) is expected_a
    assert fusion_alarm(B, groups) is expected_b


def test_non_valid_mandatory_group_is_protocol_failure() -> None:
    groups = _groups(True, False, False)
    groups[1] = group_assessment(
        group=SignalGroup.G2_DATA_QUALITY,
        status=SignalStatus.ERROR,
        alarm=None,
        reason_code="SUBSIGNAL_ERROR",
    )
    with pytest.raises(CalibrationStopError) as stop:
        fusion_alarm(A, groups)
    assert stop.value.reason == StopReason.MANDATORY_SIGNAL_NOT_VALID
    with pytest.raises(ValueError):
        fusion_alarm(A, groups[:2])


def _metrics(
    c: FusionCandidate, far: int, sens: int, nonrel: int, nonrel_n: int = 10
) -> FusionMetrics:
    return FusionMetrics(
        candidate=c,
        clean_far=Rate(successes=far, trials=100),
        sensitivity=Rate(successes=sens, trials=50),
        nonrelevant_shift_alarm_rate=Rate(successes=nonrel, trials=nonrel_n),
    )


def test_fusion_selection_hierarchy() -> None:
    assert select_fusion([_metrics(A, 6, 50, 0), _metrics(B, 5, 10, 0)], CONFIG) is B  # FAR
    assert select_fusion([_metrics(A, 5, 40, 5), _metrics(B, 1, 30, 0)], CONFIG) is A  # Sens.
    assert select_fusion([_metrics(A, 2, 30, 0), _metrics(B, 3, 30, 0)], CONFIG) is A  # FAR
    assert select_fusion([_metrics(A, 3, 30, 1), _metrics(B, 3, 30, 4)], CONFIG) is A  # nonrel
    assert select_fusion([_metrics(A, 3, 30, 2), _metrics(B, 3, 30, 2)], CONFIG) is B  # Tie → B


def test_fusion_selection_stops() -> None:
    with pytest.raises(CalibrationStopError) as none:
        select_fusion([_metrics(A, 6, 50, 0), _metrics(B, 7, 50, 0)], CONFIG)
    assert none.value.reason == StopReason.NO_FAR_ELIGIBLE_FUSION
    with pytest.raises(CalibrationStopError) as undefined:
        select_fusion([_metrics(A, 3, 30, 0, 0), _metrics(B, 3, 30, 0, 0)], CONFIG)
    assert undefined.value.reason == StopReason.NONRELEVANT_RATE_UNDEFINED


def test_fusion_metrics_separate_relevant_and_nonrelevant_windows() -> None:
    shifts = [shift(True), shift(False), shift(True, relevant=False)]
    metrics = fusion_metrics(A, [clean(False), clean(True), clean(False), clean(False)], shifts)
    assert (metrics.sensitivity.successes, metrics.sensitivity.trials) == (1, 2)
    nonrelevant = metrics.nonrelevant_shift_alarm_rate
    assert (nonrelevant.successes, nonrelevant.trials) == (1, 1)
    assert metrics.clean_far == count_rate([False, True, False, False])
    with pytest.raises(CalibrationStopError):
        fusion_metrics(A, [clean(False)], [shift(True, relevant=False)])
    with pytest.raises(CalibrationStopError) as stop:
        fusion_metrics(A, [ProcedureOutcome(status=SignalStatus.ERROR, alarm=None)], shifts)
    assert stop.value.reason == StopReason.MANDATORY_SIGNAL_NOT_VALID


def test_audit_far_boundary() -> None:
    assert audit_passed(Rate(successes=5, trials=100), CONFIG)
    assert not audit_passed(Rate(successes=6, trials=100), CONFIG)
