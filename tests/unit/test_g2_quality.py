"""G2 – Data-Quality-Indikatoren und Window-Vergleich (DEC-043)."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from signal_factories import g2_config
from vigiscoutx.domain.assessments import ReasonCode, SignalStatus
from vigiscoutx.domain.types import SignalGroup
from vigiscoutx.signals.config import G2_SIGNAL_IDS, G2Indicator
from vigiscoutx.signals.data_quality.quality import (
    QualityIndicatorBatch,
    QualityInputFailure,
    compute_quality_indicators,
    image_indicators,
    luma,
    run_g2_quality,
    valid_convolution,
)

CONFIG = g2_config()


def _manual_valid(y: np.ndarray, kernel: tuple[tuple[int, ...], ...]) -> np.ndarray:
    h, w = y.shape
    out = np.zeros((h - 2, w - 2))
    for i in range(h - 2):
        for j in range(w - 2):
            # Faltung: gespiegelte Maske (beide Masken sind punktsymmetrisch).
            out[i, j] = sum(
                kernel[2 - a][2 - b] * y[i + a, j + b] for a in range(3) for b in range(3)
            )
    return out


def test_luma_matches_manual_coefficients() -> None:
    image = np.zeros((3, 2, 2))
    image[0, 0, 0], image[1, 0, 1], image[2, 1, 0] = 1.0, 1.0, 1.0
    image[:, 1, 1] = 0.5
    y = luma(image, CONFIG.luma_coefficients)
    assert y[0, 0] == pytest.approx(0.2126)
    assert y[0, 1] == pytest.approx(0.7152)
    assert y[1, 0] == pytest.approx(0.0722)
    assert y[1, 1] == pytest.approx(0.5)


def test_mean_std_laplacian_and_immerkaer_match_manual_reference() -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(size=(6, 5))
    result = image_indicators(y, CONFIG)
    assert result.mean_luminance == pytest.approx(y.sum() / y.size)
    population_std = math.sqrt(((y - y.mean()) ** 2).sum() / y.size)
    assert result.luminance_std == pytest.approx(population_std)
    lap = _manual_valid(y, ((0, 1, 0), (1, -4, 1), (0, 1, 0)))
    assert lap.shape == (4, 3)
    assert np.allclose(valid_convolution(y, CONFIG.laplacian_kernel), lap)
    assert result.variance_of_laplacian == pytest.approx(((lap - lap.mean()) ** 2).mean())
    imm = _manual_valid(y, ((1, -2, 1), (-2, 4, -2), (1, -2, 1)))
    sigma = math.sqrt(math.pi / 2) * np.abs(imm).sum() / (6 * (6 - 2) * (5 - 2))
    assert result.immerkaer_noise_sigma == pytest.approx(sigma)
    assert result.immerkaer_noise_variance == pytest.approx(sigma**2)
    assert result.immerkaer_noise_variance == result.immerkaer_noise_sigma**2


def test_constant_image_has_zero_laplacian_variance_and_noise() -> None:
    result = image_indicators(np.full((5, 5), 0.3), CONFIG)
    assert result.variance_of_laplacian == 0.0
    assert result.immerkaer_noise_sigma == 0.0
    assert result.luminance_std == pytest.approx(0.0, abs=1e-15)


def _images(n: int, seed: int, brightness: float = 0.0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.6, size=(n, 3, 224, 224))
    return list(np.clip(base + brightness, 0.0, 1.0))


def _indicators(window: str, images: list[np.ndarray]) -> QualityIndicatorBatch:
    batch = compute_quality_indicators(
        window, [f"{window}-{i}" for i in range(len(images))], images, CONFIG
    )
    assert isinstance(batch, QualityIndicatorBatch)
    return batch


def test_four_components_with_bonferroni_and_or_alarm() -> None:
    reference = _indicators("ref", _images(8, 1))
    shifted = _indicators("mon", _images(8, 2, brightness=0.3))
    components, group = run_g2_quality(reference, shifted, CONFIG)
    assert [c.signal_id for c in components] == [G2_SIGNAL_IDS[i] for i in G2Indicator]
    assert all(c.status is SignalStatus.VALID for c in components)
    for component in components:
        assert component.p_value is not None and component.adjusted_p_value is not None
        assert component.adjusted_p_value == min(1.0, 4 * component.p_value)
        assert component.alarm is (component.adjusted_p_value <= 0.05)
        assert component.statistics["component_alpha"] == 0.0125
        assert "ks_d" in component.statistics
    mean_component = components[0]
    assert mean_component.alarm is True
    assert group.group is SignalGroup.G2_DATA_QUALITY
    assert group.status is SignalStatus.VALID
    assert group.alarm is any(c.alarm for c in components)
    assert group.signal_ids == tuple(c.signal_id for c in components)


def test_same_images_give_no_alarm() -> None:
    images = _images(5, 3)
    reference, monitoring = _indicators("ref", images), _indicators("mon", images)
    components, group = run_g2_quality(reference, monitoring, CONFIG)
    assert all(c.alarm is False and c.p_value == 1.0 for c in components)
    assert (group.status, group.alarm) == (SignalStatus.VALID, False)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda a: a[:, :100, :100], ReasonCode.INVALID_INPUT_SHAPE),
        (lambda a: np.where(a > 0.5, 1.5, a), ReasonCode.INPUT_OUT_OF_RANGE),
        (lambda a: np.where(a > 0.5, np.nan, a), ReasonCode.NONFINITE_INPUT),
        (lambda a: (a * 255).astype(np.uint8), ReasonCode.INVALID_INPUT_TYPE),
    ],
)
def test_invalid_input_contract_is_error(mutate: object, reason: ReasonCode) -> None:
    images = _images(2, 4)
    images[1] = mutate(images[1])  # type: ignore[operator]
    failure = compute_quality_indicators("mon", ["a", "b"], images, CONFIG)
    assert isinstance(failure, QualityInputFailure)
    assert (failure.status, failure.reason_code) == (SignalStatus.ERROR, reason)
    components, group = run_g2_quality(_indicators("ref", _images(2, 5)), failure, CONFIG)
    assert all(c.status is SignalStatus.ERROR and c.alarm is None for c in components)
    assert (group.status, group.alarm, group.reason_code) == (
        SignalStatus.ERROR,
        None,
        ReasonCode.SUBSIGNAL_ERROR,
    )


def test_empty_window_is_not_evaluable() -> None:
    failure = compute_quality_indicators("mon", [], [], CONFIG)
    assert isinstance(failure, QualityInputFailure)
    assert failure.status is SignalStatus.NOT_EVALUABLE
    _, group = run_g2_quality(_indicators("ref", _images(2, 6)), failure, CONFIG)
    assert (group.status, group.alarm) == (SignalStatus.NOT_EVALUABLE, None)


def test_indicator_batch_is_deeply_immutable() -> None:
    batch = _indicators("ref", _images(2, 7))
    with pytest.raises(ValidationError):
        batch.indicators[0].mean_luminance = 0.0  # type: ignore[misc]
    assert isinstance(batch.indicators, tuple)
    values = batch.values(G2Indicator.MEAN_LUMINANCE)
    values[0] = -1.0
    assert batch.indicators[0].mean_luminance != -1.0
