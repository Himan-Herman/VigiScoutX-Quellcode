"""Hashbare Methodenkonfigurationen der Primary-Signale G1–G3 (MONITORING_METHODS_BASELINE v1.2).

Jeder wissenschaftlich wirksame Methodenbestandteil ist ein explizites Pflichtfeld. Die in P3-02H
eingefrorenen Werte werden per Validator erzwungen; eine abweichende Konfiguration ist ungültig
und kann daher nicht unbemerkt dieselbe Identität tragen. `sigma` (G1) ist das einzige numerische
Feld, das erst auf CALIBRATION bestimmt und hier nur übergeben wird.
"""

from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, StrictFloat, ValidationError, field_validator, model_validator

from vigiscoutx.adapters.demonstrator import require_canonical_class_order
from vigiscoutx.config.scientific_config import SignalConfig
from vigiscoutx.domain.assessments import G3_OUTPUT_SIGNAL_ID as G3_OUTPUT_SIGNAL_ID
from vigiscoutx.domain.assessments import ParametersRef
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.types import FrozenModel, Identifier, SignalGroup

METHOD_VERSION: Final = "monitoring-methods-baseline:1.2"
BASELINE_ALPHA: Final = 0.05
BASELINE_PERMUTATIONS: Final = 999
"""Projektinterne Reproduzierbarkeits-/Auflösungsentscheidung (DEC-042), kein optimierter Wert."""

LUMA_COEFFICIENTS: Final = (0.2126, 0.7152, 0.0722)
LAPLACIAN_KERNEL: Final = ((0, 1, 0), (1, -4, 1), (0, 1, 0))
IMMERKAER_KERNEL: Final = ((1, -2, 1), (-2, 4, -2), (1, -2, 1))

G1_SIGNAL_ID: Final = "G1_MMD_EMBEDDING"
G3_CONFIDENCE_SIGNAL_ID: Final = "G3_CONFIDENCE_KS"


class G2Indicator(StrEnum):
    """Die vier G2-Testindikatoren in fester Reihenfolge (DEC-043)."""

    MEAN_LUMINANCE = "mean_luminance"
    LUMINANCE_STD = "luminance_std"
    VARIANCE_OF_LAPLACIAN = "variance_of_laplacian"
    IMMERKAER_NOISE_VARIANCE = "immerkaer_noise_variance"


G2_SIGNAL_IDS: Final = {
    G2Indicator.MEAN_LUMINANCE: "G2_MEAN_LUMINANCE_KS",
    G2Indicator.LUMINANCE_STD: "G2_LUMINANCE_STD_KS",
    G2Indicator.VARIANCE_OF_LAPLACIAN: "G2_VARIANCE_OF_LAPLACIAN_KS",
    G2Indicator.IMMERKAER_NOISE_VARIANCE: "G2_IMMERKAER_NOISE_VARIANCE_KS",
}


class AggregationRule(StrEnum):
    """Strikte Statusaggregation ERROR > NOT_EVALUABLE > VALID (DEC-049)."""

    STRICT_ERROR_NOT_EVALUABLE_VALID = "STRICT_ERROR_NOT_EVALUABLE_VALID"


def _require_value(value: object, expected: object, name: str) -> None:
    if value != expected:
        raise ValueError(f"{name} ist eingefroren auf {expected!r} (erhalten: {value!r})")


Alpha = Annotated[StrictFloat, Field(gt=0.0, lt=1.0)]


class KsSpec(FrozenModel):
    """Verbindliche KS-Regel für G2 und G3 (Abschnitt 3.5); kein `auto`, kein Fallback."""

    test: Literal["TWO_SAMPLE_KS"]
    alternative: Literal["two-sided"]
    method: Literal["exact"]
    on_exact_failure: Literal["ERROR"]


class MethodConfig(FrozenModel):
    method_version: Literal["monitoring-methods-baseline:1.2"]

    def config_id(self) -> str:
        raise NotImplementedError

    def parameters_ref(self) -> ParametersRef:
        return ParametersRef(
            config_id=self.config_id(),
            method_version=self.method_version,
            sha256=canonical_sha256(self),
        )


class G1MmdConfig(MethodConfig):
    signal_id: Literal["G1_MMD_EMBEDDING"]
    embedding: Literal["RESNET18_PENULTIMATE"]
    embedding_dim: Literal[512]
    embedding_transform: Literal["NONE"]
    estimator: Literal["MMD2_UNBIASED_QUADRATIC"]
    kernel: Literal["GAUSSIAN_RBF"]
    sigma: Annotated[StrictFloat, Field(gt=0.0)]
    inference: Literal["RANDOM_PERMUTATION_FIXED_GROUP_SIZES"]
    n_permutations: Literal[999]
    tail: Literal["RIGHT"]
    alpha: Alpha
    seed_field: Literal["statistical_seed"]

    @field_validator("alpha")
    @classmethod
    def _frozen_alpha(cls, alpha: float) -> float:
        _require_value(alpha, BASELINE_ALPHA, "alpha")
        return alpha

    def config_id(self) -> str:
        return self.signal_id


class G2QualityConfig(MethodConfig):
    group_id: Literal["G2"]
    input: Literal["RGB_FLOAT_0_1_BEFORE_IMAGENET_NORMALIZATION"]
    input_height: Literal[224]
    input_width: Literal[224]
    luma_coefficients: tuple[StrictFloat, StrictFloat, StrictFloat]
    gamma_linearization: Literal[False]
    std_ddof: Literal[0]
    convolution: Literal["VALID"]
    laplacian_kernel: tuple[tuple[int, ...], ...]
    immerkaer_kernel: tuple[tuple[int, ...], ...]
    indicators: tuple[G2Indicator, ...]
    ks: KsSpec
    alpha: Alpha
    bonferroni_factor: Literal[4]
    aggregation: AggregationRule

    @model_validator(mode="after")
    def _frozen(self) -> Self:
        _require_value(self.luma_coefficients, LUMA_COEFFICIENTS, "luma_coefficients")
        _require_value(self.laplacian_kernel, LAPLACIAN_KERNEL, "laplacian_kernel")
        _require_value(self.immerkaer_kernel, IMMERKAER_KERNEL, "immerkaer_kernel")
        _require_value(self.indicators, tuple(G2Indicator), "indicators")
        _require_value(self.alpha, BASELINE_ALPHA, "alpha")
        return self

    def config_id(self) -> str:
        return self.group_id


class G3PredictiveConfig(MethodConfig):
    group_id: Literal["G3"]
    class_order: tuple[Identifier, ...]
    output_signal_id: Literal["G3_OUTPUT_MUKS"]
    confidence_signal_id: Literal["G3_CONFIDENCE_KS"]
    confidence: Literal["MAX_SOFTMAX"]
    ks: KsSpec
    alpha: Alpha
    output_bonferroni_factor: Literal[7]
    group_rule: Literal["OUTPUT_OR_CONFIDENCE"]
    aggregation: AggregationRule

    @model_validator(mode="after")
    def _frozen(self) -> Self:
        require_canonical_class_order(self.class_order)
        _require_value(self.alpha, BASELINE_ALPHA, "alpha")
        return self

    def config_id(self) -> str:
        return self.group_id


_SIGNAL_CONFIG_METHODS: Final[dict[type[MethodConfig], tuple[str, SignalGroup]]] = {
    G1MmdConfig: ("MMD_U2_RBF_PERMUTATION", SignalGroup.G1_INPUT),
    G2QualityConfig: ("G2_QUALITY_KS_BONFERRONI", SignalGroup.G2_DATA_QUALITY),
    G3PredictiveConfig: ("G3_MUKS_CONFIDENCE_KS", SignalGroup.G3_PREDICTIVE),
}


def method_config_from_signal_config[ConfigT: MethodConfig](
    signal: SignalConfig, model: type[ConfigT]
) -> ConfigT:
    """Validiert `SignalConfig.params` (ScientificConfig) als typisierte Methodenkonfiguration."""
    method, group = _SIGNAL_CONFIG_METHODS[model]
    if (signal.method, signal.group) != (method, group):
        raise ValueError(
            f"SignalConfig {signal.signal_id}: erwartet method={method}, group={group.value}"
        )
    return model.model_validate(dict(signal.params))


METHOD_DECIDED_SIGNALS: Final[dict[tuple[str, SignalGroup, str], type[MethodConfig]]] = {
    (G1_SIGNAL_ID, SignalGroup.G1_INPUT, "MMD_U2_RBF_PERMUTATION"): G1MmdConfig,
    ("G2", SignalGroup.G2_DATA_QUALITY, "G2_QUALITY_KS_BONFERRONI"): G2QualityConfig,
    ("G3", SignalGroup.G3_PREDICTIVE, "G3_MUKS_CONFIDENCE_KS"): G3PredictiveConfig,
}
"""Registrierte Signal-/Methodenkombinationen, deren statistische Entscheidungsregel vollständig
durch die versionsgebundene Methodenkonfiguration bestimmt ist (P4-02E-Contract-Korrektur).

Für sie ist keine separate `ThresholdConfig` vorgesehen; es wird keine Schwelle abgeleitet und
kein Signifikanzniveau als Alarmschwelle interpretiert. Jede andere Kombination bleibt
schwellenpflichtig.
"""


def method_decided(signal: SignalConfig) -> bool | None:
    """`None`: nicht registriert (schwellenpflichtig); `True`: registriert und die Parameter sind
    eine vollständig gültige eingefrorene Methodenkonfiguration genau dieses Signals; `False`:
    registriert, aber die Methodenkonfiguration ist unvollständig oder ungültig."""
    model = METHOD_DECIDED_SIGNALS.get((signal.signal_id, signal.group, signal.method))
    if model is None:
        return None
    try:
        config = method_config_from_signal_config(signal, model)
    except (ValidationError, ValueError):
        return False
    return config.config_id() == signal.signal_id
