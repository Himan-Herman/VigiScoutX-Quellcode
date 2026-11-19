"""Strikte Aggregation gemischter Subsignal-Status (DEC-049)."""

import pytest

from signal_factories import canonical_components, g3_output_data, signal_result
from vigiscoutx.domain.assessments import G3Substatus, ReasonCode, SignalResult, SignalStatus
from vigiscoutx.domain.types import SignalGroup
from vigiscoutx.grouping.aggregation import assess_g3_group, assess_group
from vigiscoutx.signals.config import G2_SIGNAL_IDS

V, NE, E = SignalStatus.VALID, SignalStatus.NOT_EVALUABLE, SignalStatus.ERROR


def _child(signal_id: str, status: SignalStatus, alarm: bool | None, group: str) -> SignalResult:
    reason = {V: None, NE: ReasonCode.INSUFFICIENT_SAMPLE, E: ReasonCode.KS_EXACT_FALLBACK}[status]
    return signal_result(
        signal_id=signal_id,
        group=group,
        status=status,
        alarm=alarm,
        reason_code=reason,
        p_value=None if status is not V else 0.5,
    )


G2_IDS = tuple(G2_SIGNAL_IDS.values())


def _g2(*children: tuple[SignalStatus, bool | None]) -> list[SignalResult]:
    return [_child(G2_IDS[i], s, a, "G2") for i, (s, a) in enumerate(children)]


@pytest.mark.parametrize(
    ("children", "expected"),
    [
        (((V, False), (V, False), (V, False), (V, False)), (V, False, None)),
        (((V, False), (V, True), (V, False), (V, False)), (V, True, None)),
        (((V, True), (E, None), (V, False), (V, False)), (E, None, ReasonCode.SUBSIGNAL_ERROR)),
        (
            ((V, True), (NE, None), (V, False), (V, False)),
            (NE, None, ReasonCode.SUBSIGNAL_NOT_EVALUABLE),
        ),
        (((E, None), (NE, None), (V, True), (V, False)), (E, None, ReasonCode.SUBSIGNAL_ERROR)),
    ],
)
def test_g2_strict_aggregation(
    children: tuple[tuple[SignalStatus, bool | None], ...],
    expected: tuple[SignalStatus, bool | None, ReasonCode | None],
) -> None:
    results = _g2(*children)
    group = assess_group(SignalGroup.G2_DATA_QUALITY, results)
    assert (group.status, group.alarm, group.reason_code) == expected
    assert group.substatus is None
    assert group.signal_ids == tuple(r.signal_id for r in results)
    # Child-Ergebnisse bleiben unverändert erhalten, einschließlich valider Alarme.
    assert [(r.status, r.alarm) for r in results] == list(children)


def _g3(
    output: tuple[SignalStatus, bool | None], confidence: tuple[SignalStatus, bool | None]
) -> tuple[SignalResult, SignalResult]:
    out = SignalResult.model_validate(g3_output_data(canonical_components(*output)))
    assert (out.status, out.alarm) == output
    return out, _child("G3_CONFIDENCE_KS", *confidence, group="G3")


@pytest.mark.parametrize(
    ("output", "confidence", "expected"),
    [
        ((V, True), (E, None), (E, None, None)),
        ((V, False), (NE, None), (NE, None, None)),
        ((V, False), (V, False), (V, False, G3Substatus.NONE)),
        ((V, True), (V, False), (V, True, G3Substatus.OUTPUT_ONLY)),
        ((V, False), (V, True), (V, True, G3Substatus.CONFIDENCE_ONLY)),
        ((V, True), (V, True), (V, True, G3Substatus.BOTH)),
        ((E, None), (NE, None), (E, None, None)),
    ],
)
def test_g3_group_strict_aggregation_and_substatus(
    output: tuple[SignalStatus, bool | None],
    confidence: tuple[SignalStatus, bool | None],
    expected: tuple[SignalStatus, bool | None, G3Substatus | None],
) -> None:
    out, conf = _g3(output, confidence)
    group = assess_g3_group(out, conf)
    assert (group.status, group.alarm, group.substatus) == expected
    assert out.alarm is output[1]


def test_g3_requires_its_two_subsignals() -> None:
    out, conf = _g3((V, False), (V, False))
    with pytest.raises(ValueError):
        assess_g3_group(conf, out)
    with pytest.raises(ValueError):
        assess_group(SignalGroup.G3_PREDICTIVE, [out, conf])


def test_group_rejects_foreign_or_mismatched_children() -> None:
    g2 = _g2((V, False), (V, False), (V, False), (V, False))
    with pytest.raises(ValueError):
        assess_group(SignalGroup.G1_INPUT, g2)
    other_window = g2[3].model_copy(update={"monitoring_window_id": "other"})
    with pytest.raises(ValueError):
        assess_group(SignalGroup.G2_DATA_QUALITY, [*g2[:3], other_window])
    with pytest.raises(ValueError):
        assess_group(SignalGroup.G2_DATA_QUALITY, [])


# --- Vollständigkeit des G2-Child-Satzes ------------------------------------------------------


def _four_valid() -> list[SignalResult]:
    return _g2((V, False), (V, True), (V, False), (V, False))


def test_exactly_four_correct_g2_children_pass() -> None:
    group = assess_group(SignalGroup.G2_DATA_QUALITY, _four_valid())
    assert (group.status, group.alarm) == (V, True)


def test_complete_g2_set_in_other_order_passes() -> None:
    children = _four_valid()[::-1]
    group = assess_group(SignalGroup.G2_DATA_QUALITY, children)
    assert (group.status, group.alarm) == (V, True)
    assert group.signal_ids == tuple(c.signal_id for c in children)


def _foreign(signal_id: str) -> SignalResult:
    return _child(signal_id, V, False, "G2")


@pytest.mark.parametrize(
    "children",
    [
        pytest.param(_four_valid()[:1], id="only-one"),
        pytest.param(_four_valid()[:3], id="three-of-four"),
        pytest.param([*_four_valid(), _foreign("G2_EXTRA_KS")], id="five"),
        pytest.param([*_four_valid()[:3], _four_valid()[0]], id="duplicate-id"),
        pytest.param([*_four_valid()[:3], _foreign("G2_WRONG_KS")], id="wrong-id"),
        pytest.param([*_four_valid(), _four_valid()[0]], id="complete-set-plus-duplicate"),
    ],
)
def test_incomplete_or_malformed_g2_child_set_is_rejected(children: list[SignalResult]) -> None:
    with pytest.raises(ValueError, match="G2 erfordert exakt"):
        assess_group(SignalGroup.G2_DATA_QUALITY, children)


def test_incomplete_g2_set_is_rejected_even_with_error_child() -> None:
    children = _g2((E, None), (V, True), (V, False))
    with pytest.raises(ValueError, match="G2 erfordert exakt"):
        assess_group(SignalGroup.G2_DATA_QUALITY, children)
