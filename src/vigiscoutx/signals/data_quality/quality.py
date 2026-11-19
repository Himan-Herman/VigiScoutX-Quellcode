"""G2 – Data Quality (DEC-043, MONITORING_METHODS_BASELINE 3).

Eingabe je Bild: 224×224 RGB als Float in [0, 1] (CHW), direkt vor der ImageNet-Normalisierung.
Luma `Y = 0.2126 R + 0.7152 G + 0.0722 B` ohne Gamma-Linearisierung (Projektoperationalisierung,
nicht Immerkær zugeschrieben). Je Bild werden fünf Indikatoren berechnet; vier davon werden je
über einen exakten zweiseitigen KS-Test mit Bonferroni `α/4` verglichen. Der Gruppenstatus folgt
der strikten Aggregation (DEC-049).
"""

import math
from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
from pydantic import Field, StrictFloat, model_validator
from scipy.signal import convolve2d

from vigiscoutx.domain.assessments import (
    GroupAssessment,
    ReasonCode,
    SignalResult,
    SignalStatus,
)
from vigiscoutx.domain.frozen_json import FrozenMapping
from vigiscoutx.domain.types import FrozenModel, Identifier, NonNegativeInt, SignalGroup
from vigiscoutx.grouping.aggregation import assess_group
from vigiscoutx.signals.config import G2_SIGNAL_IDS, G2Indicator, G2QualityConfig
from vigiscoutx.signals.statistics import bonferroni_adjust, exact_two_sided_ks, is_significant

FloatArray = npt.NDArray[np.float64]
G2_KS_METHOD = "KS_2SAMP_EXACT_TWO_SIDED_BONFERRONI"


class QualityIndicators(FrozenModel):
    """Die fünf gespeicherten Indikatoren eines Bildes."""

    mean_luminance: StrictFloat
    luminance_std: StrictFloat
    variance_of_laplacian: StrictFloat
    immerkaer_noise_sigma: StrictFloat
    immerkaer_noise_variance: StrictFloat


class QualityIndicatorBatch(FrozenModel):
    """Unveränderliches, cachebares G2-Artefakt eines Fensters (DEC-046)."""

    window_id: Identifier
    sample_ids: tuple[Identifier, ...] = Field(min_length=1)
    indicators: tuple[QualityIndicators, ...]

    @model_validator(mode="after")
    def _aligned(self) -> Self:
        if len(self.sample_ids) != len(self.indicators):
            raise ValueError("je sample_id genau ein Indikatorsatz")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids müssen eindeutig sein")
        return self

    def values(self, indicator: G2Indicator) -> FloatArray:
        return np.array([getattr(row, indicator.value) for row in self.indicators], dtype=float)


class QualityInputFailure(FrozenModel):
    """Eingabe eines Fensters verletzt den G2-Eingabevertrag; es entstehen keine Indikatoren."""

    window_id: Identifier
    n: NonNegativeInt
    status: SignalStatus
    reason_code: ReasonCode


def luma(image: FloatArray, coefficients: Sequence[float]) -> FloatArray:
    """`Y = c_R · R + c_G · G + c_B · B` für ein CHW-Bild."""
    r, g, b = coefficients
    return np.asarray(r * image[0] + g * image[1] + b * image[2], dtype=np.float64)


def valid_convolution(y: FloatArray, kernel: Sequence[Sequence[int]]) -> FloatArray:
    return np.asarray(convolve2d(y, np.array(kernel, dtype=np.float64), mode="valid"))


def image_indicators(y: FloatArray, config: G2QualityConfig) -> QualityIndicators:
    """Indikatoren eines Luma-Bildes der Größe H × W (H, W ≥ 3)."""
    height, width = y.shape
    laplacian = valid_convolution(y, config.laplacian_kernel)
    immerkaer = valid_convolution(y, config.immerkaer_kernel)
    sigma = (
        math.sqrt(math.pi / 2.0)
        * float(np.abs(immerkaer).sum())
        / (6.0 * (height - 2) * (width - 2))
    )
    return QualityIndicators(
        mean_luminance=float(y.mean()),
        luminance_std=float(y.std(ddof=config.std_ddof)),
        variance_of_laplacian=float(laplacian.var(ddof=config.std_ddof)),
        immerkaer_noise_sigma=sigma,
        immerkaer_noise_variance=sigma**2,
    )


def compute_quality_indicators(
    window_id: str,
    sample_ids: Sequence[str],
    images: Sequence[npt.ArrayLike],
    config: G2QualityConfig,
) -> QualityIndicatorBatch | QualityInputFailure:
    """Prüft den Eingabevertrag und berechnet die Indikatoren aller Bilder eines Fensters."""

    def failure(status: SignalStatus, reason: ReasonCode) -> QualityInputFailure:
        return QualityInputFailure(
            window_id=window_id, n=len(images), status=status, reason_code=reason
        )

    if len(sample_ids) != len(images):
        return failure(SignalStatus.ERROR, ReasonCode.INVALID_INPUT_SHAPE)
    if not images:
        return failure(SignalStatus.NOT_EVALUABLE, ReasonCode.INSUFFICIENT_SAMPLE)
    expected = (3, config.input_height, config.input_width)
    rows: list[QualityIndicators] = []
    for image in images:
        array = np.asarray(image)
        if not np.issubdtype(array.dtype, np.floating):
            return failure(SignalStatus.ERROR, ReasonCode.INVALID_INPUT_TYPE)
        if array.shape != expected:
            return failure(SignalStatus.ERROR, ReasonCode.INVALID_INPUT_SHAPE)
        data = array.astype(np.float64)
        if not np.isfinite(data).all():
            return failure(SignalStatus.ERROR, ReasonCode.NONFINITE_INPUT)
        if data.min() < 0.0 or data.max() > 1.0:
            return failure(SignalStatus.ERROR, ReasonCode.INPUT_OUT_OF_RANGE)
        rows.append(image_indicators(luma(data, config.luma_coefficients), config))
    return QualityIndicatorBatch(
        window_id=window_id, sample_ids=tuple(sample_ids), indicators=tuple(rows)
    )


def _component(
    indicator: G2Indicator,
    reference: QualityIndicatorBatch | QualityInputFailure,
    monitoring: QualityIndicatorBatch | QualityInputFailure,
    config: G2QualityConfig,
) -> SignalResult:
    component_alpha = config.alpha / config.bonferroni_factor
    base = {
        "signal_id": G2_SIGNAL_IDS[indicator],
        "group": SignalGroup.G2_DATA_QUALITY,
        "method": G2_KS_METHOD,
        "method_version": config.method_version,
        "components": (),
        "parameters_ref": config.parameters_ref(),
        "reference_window_id": reference.window_id,
        "monitoring_window_id": monitoring.window_id,
        "reference_n": _size(reference),
        "monitoring_n": _size(monitoring),
        "seed_ref": None,
        "model_snapshot_ref": None,
        "artifact_refs": (),
        "evidence_refs": (),
        "alpha_or_threshold": config.alpha,
    }
    fixed = {
        "indicator": indicator.value,
        "bonferroni_factor": config.bonferroni_factor,
        "component_alpha": component_alpha,
    }
    failures = [b for b in (reference, monitoring) if isinstance(b, QualityInputFailure)]
    status: SignalStatus
    reason: ReasonCode | None
    if failures:
        worst = next((f for f in failures if f.status is SignalStatus.ERROR), failures[0])
        status, reason = worst.status, worst.reason_code
    else:
        assert isinstance(reference, QualityIndicatorBatch)
        assert isinstance(monitoring, QualityIndicatorBatch)
        ks = exact_two_sided_ks(
            reference.values(indicator), monitoring.values(indicator), config.ks
        )
        if ks.status is SignalStatus.VALID and ks.p_value is not None:
            adjusted = bonferroni_adjust(ks.p_value, config.bonferroni_factor)
            return SignalResult.model_validate(
                {
                    **base,
                    "status": SignalStatus.VALID,
                    "statistics": FrozenMapping({**fixed, "ks_d": ks.statistic}),
                    "p_value": ks.p_value,
                    "adjusted_p_value": adjusted,
                    "alarm": is_significant(adjusted, config.alpha),
                    "reason_code": None,
                }
            )
        status, reason = ks.status, ks.reason_code
    return SignalResult.model_validate(
        {
            **base,
            "status": status,
            "statistics": FrozenMapping(fixed),
            "p_value": None,
            "adjusted_p_value": None,
            "alarm": None,
            "reason_code": reason,
        }
    )


def _size(batch: QualityIndicatorBatch | QualityInputFailure) -> int:
    return batch.n if isinstance(batch, QualityInputFailure) else len(batch.sample_ids)


def run_g2_quality(
    reference: QualityIndicatorBatch | QualityInputFailure,
    monitoring: QualityIndicatorBatch | QualityInputFailure,
    config: G2QualityConfig,
) -> tuple[tuple[SignalResult, ...], GroupAssessment]:
    """Vier KS-Subsignale und das strikt aggregierte G2-`GroupAssessment`."""
    components = tuple(_component(i, reference, monitoring, config) for i in config.indicators)
    return components, assess_group(SignalGroup.G2_DATA_QUALITY, components)
