"""G3 – Output (MUKS) und Confidence (DEC-044) inklusive strikter Aggregation (DEC-049)."""

from typing import Any

import numpy as np
import pytest

from signal_factories import MODEL_REF, g3_config
from vigiscoutx.adapters.demonstrator import DEMONSTRATOR_CLASS_ORDER
from vigiscoutx.domain.assessments import ReasonCode, SignalStatus
from vigiscoutx.domain.types import SignalGroup
from vigiscoutx.domain.windows import ModelOutputBatch
from vigiscoutx.signals.predictive import g3 as g3_module
from vigiscoutx.signals.predictive.g3 import (
    max_softmax_confidence,
    run_g3,
    run_g3_confidence,
    run_g3_output,
)
from vigiscoutx.signals.statistics import KsOutcome, exact_two_sided_ks

CONFIG = g3_config()


def _batch(window: str, logits: np.ndarray) -> ModelOutputBatch:
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return ModelOutputBatch(
        window_id=window,
        sample_ids=tuple(f"{window}-{i}" for i in range(len(probs))),
        probabilities=tuple(tuple(float(v) for v in row) for row in probs),
    )


def _logits(n: int, seed: int, shift_class: int | None = None) -> np.ndarray:
    logits = np.random.default_rng(seed).normal(size=(n, 7))
    if shift_class is not None:
        logits[:, shift_class] += 3.0
    return logits


def test_seven_components_in_canonical_order_with_bonferroni() -> None:
    output = run_g3_output(
        _batch("ref", _logits(20, 0)), _batch("mon", _logits(20, 1, 3)), CONFIG, MODEL_REF
    )
    assert output.signal_id == "G3_OUTPUT_MUKS"
    assert tuple(c.component_id for c in output.components) == DEMONSTRATOR_CLASS_ORDER
    for component in output.components:
        assert component.status is SignalStatus.VALID
        assert component.p_value is not None and component.adjusted_p_value is not None
        assert component.adjusted_p_value == min(1.0, 7 * component.p_value)
        assert component.alarm is (component.adjusted_p_value <= 0.05)
        assert component.statistics["class_id"] == component.component_id
    assert output.status is SignalStatus.VALID
    assert output.alarm is True
    assert output.statistics["component_alpha"] == pytest.approx(1 / 140)
    raw = [c.p_value for c in output.components if c.p_value is not None]
    assert output.statistics["minimum_raw_p"] == min(raw)
    assert output.statistics["minimum_adjusted_p"] == min(1.0, 7 * min(raw))


def test_confidence_is_max_softmax() -> None:
    probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.25, 0.25]])
    assert list(max_softmax_confidence(probs)) == [0.7, 0.5]


def test_confidence_signal() -> None:
    reference = _batch("ref", _logits(15, 2))
    sharper = _batch("mon", _logits(15, 3) * 4.0)
    result = run_g3_confidence(reference, sharper, CONFIG, MODEL_REF)
    assert result.signal_id == "G3_CONFIDENCE_KS"
    assert result.status is SignalStatus.VALID
    assert result.p_value is not None and result.adjusted_p_value is None
    assert result.alarm is (result.p_value <= 0.05)
    assert result.alarm is True
    assert (result.reference_n, result.monitoring_n) == (15, 15)


def test_identical_outputs_give_none_substatus() -> None:
    batch = _batch("ref", _logits(10, 4))
    same = ModelOutputBatch(
        window_id="mon", sample_ids=batch.sample_ids, probabilities=batch.probabilities
    )
    output, confidence, group = run_g3(batch, same, CONFIG, MODEL_REF)
    assert (output.alarm, confidence.alarm) == (False, False)
    assert (group.status, group.alarm, group.substatus.value if group.substatus else None) == (
        SignalStatus.VALID,
        False,
        "NONE",
    )


def test_g3_counts_as_exactly_one_group() -> None:
    reference = _batch("ref", _logits(20, 5))
    monitoring = _batch("mon", _logits(20, 6, 1) * 4.0)
    output, confidence, group = run_g3(reference, monitoring, CONFIG, MODEL_REF)
    assert output.alarm is True and confidence.alarm is True
    assert group.group is SignalGroup.G3_PREDICTIVE
    assert group.alarm is True
    assert group.substatus is not None and group.substatus.value == "BOTH"
    assert group.signal_ids == ("G3_OUTPUT_MUKS", "G3_CONFIDENCE_KS")


def test_wrong_class_dimension_is_error() -> None:
    three = ModelOutputBatch(
        window_id="mon", sample_ids=("a", "b"), probabilities=((0.2, 0.3, 0.5), (0.1, 0.1, 0.8))
    )
    output, confidence, group = run_g3(three, three, CONFIG, MODEL_REF)
    for result in (output, confidence):
        assert result.status is SignalStatus.ERROR
        assert result.reason_code is ReasonCode.INVALID_CLASS_DIMENSION
        assert result.alarm is None
    assert (group.status, group.alarm, group.substatus) == (SignalStatus.ERROR, None, None)


@pytest.mark.parametrize(
    ("status", "reason", "expected_reason"),
    [
        (SignalStatus.ERROR, ReasonCode.KS_EXACT_FALLBACK, ReasonCode.SUBSIGNAL_ERROR),
        (
            SignalStatus.NOT_EVALUABLE,
            ReasonCode.INSUFFICIENT_SAMPLE,
            ReasonCode.SUBSIGNAL_NOT_EVALUABLE,
        ),
    ],
)
def test_output_strict_precedence_over_seven_components(
    monkeypatch: pytest.MonkeyPatch,
    status: SignalStatus,
    reason: ReasonCode,
    expected_reason: ReasonCode,
) -> None:
    original = exact_two_sided_ks
    calls = {"n": 0}

    def one_failing(*args: Any) -> KsOutcome:
        calls["n"] += 1
        if calls["n"] == 7:
            return KsOutcome(status=status, statistic=None, p_value=None, reason_code=reason)
        return original(*args)

    monkeypatch.setattr(g3_module, "exact_two_sided_ks", one_failing)
    output = run_g3_output(
        _batch("ref", _logits(20, 7)), _batch("mon", _logits(20, 8, 0)), CONFIG, MODEL_REF
    )
    assert (output.status, output.alarm, output.reason_code) == (status, None, expected_reason)
    assert "minimum_raw_p" not in output.statistics
    assert output.components[0].status is SignalStatus.VALID
    assert output.components[0].alarm is True
    assert output.components[6].status is status
    assert output.components[6].reason_code is reason
