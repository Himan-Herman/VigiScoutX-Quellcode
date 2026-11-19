"""Explizite, hashbare Calibration-Protokollkonfiguration (CALIBRATION_PROTOCOL_BASELINE v1.0).

Jede methodisch wirksame Regel ist ein Pflichtfeld; die eingefrorenen Werte werden per Validator
erzwungen. Eine abweichende Konfiguration ist ungültig und kann daher nicht unbemerkt dieselbe
Protokollidentität tragen. Die Konfiguration enthält keine Calibration-Ergebnisse.
"""

from enum import StrEnum
from fractions import Fraction
from typing import Final, Literal, Self

from pydantic import StrictFloat, model_validator

from vigiscoutx.adapters.demonstrator import require_canonical_class_order
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.types import FrozenModel, Identifier

PROTOCOL_VERSION: Final = "calibration-protocol-baseline:1.0"


class CalibrationPool(StrEnum):
    """Interne CALIBRATION-Pools in eingefrorener Blockreihenfolge (DEC-050)."""

    CAL_REF = "CAL_REF"
    CAL_TUNE = "CAL_TUNE"
    CAL_AUDIT = "CAL_AUDIT"


class SamplingNamespace(StrEnum):
    """Namespaces des deterministischen Window-Samplings (DEC-057)."""

    W_VALIDITY = "P4-01|W_VALIDITY"
    D0 = "P4-01|D0"
    SEVERITY = "P4-01|SEVERITY"
    R_STABILITY = "P4-01|R_STABILITY"
    FUSION_CLEAN = "P4-01|FUSION_CLEAN"
    AUDIT_CLEAN = "P4-01|AUDIT_CLEAN"


class TechnicalShift(StrEnum):
    """Die vier technischen Bildshift-Familien (DEC-054)."""

    NOISE = "NOISE"
    CONTRAST = "CONTRAST"
    BLUR = "BLUR"
    RESOLUTION = "RESOLUTION"


class FusionCandidate(StrEnum):
    FUSION_A = "FUSION_A"
    FUSION_B = "FUSION_B"


class StopReason(StrEnum):
    """Alle vorab definierten STOP-Bedingungen des Protokolls (STOP → manuelle Entscheidung)."""

    SIGMA_INVALID = "SIGMA_INVALID"
    NO_ELIGIBLE_WINDOW_SIZE = "NO_ELIGIBLE_WINDOW_SIZE"
    D0_REPEAT_NOT_VALID = "D0_REPEAT_NOT_VALID"
    DELTA_REL_INVALID = "DELTA_REL_INVALID"
    FEWER_THAN_THREE_VALID_SEVERITIES = "FEWER_THAN_THREE_VALID_SEVERITIES"
    CLASS_SHIFT_LEVELS_INVALID = "CLASS_SHIFT_LEVELS_INVALID"
    R_NOT_STABLE = "R_NOT_STABLE"
    SENSITIVITY_UNDEFINED = "SENSITIVITY_UNDEFINED"
    NONRELEVANT_RATE_UNDEFINED = "NONRELEVANT_RATE_UNDEFINED"
    NO_FAR_ELIGIBLE_FUSION = "NO_FAR_ELIGIBLE_FUSION"
    MANDATORY_SIGNAL_NOT_VALID = "MANDATORY_SIGNAL_NOT_VALID"
    AUDIT_FAR_EXCEEDED = "AUDIT_FAR_EXCEEDED"


R_METHODS: Final = (
    "G1_MMD_EMBEDDING",
    "G2_MEAN_LUMINANCE_KS",
    "G2_LUMINANCE_STD_KS",
    "G2_VARIANCE_OF_LAPLACIAN_KS",
    "G2_IMMERKAER_NOISE_VARIANCE_KS",
    "G2_GROUP",
    "G3_OUTPUT_MUKS",
    "G3_CONFIDENCE_KS",
    "G3_GROUP",
    "FUSION_A",
    "FUSION_B",
)
"""Exakt die elf Alarmverfahren der R-Stabilitätsprüfung (DEC-058)."""

_FROZEN: Final[dict[str, object]] = {
    "pool_order": tuple(CalibrationPool),
    "pool_percent": (30, 50, 20),
    "class_order": ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"),
    "window_size_candidates": (128, 256),
    "quantiles": (0.90, 0.95, 0.99),
    "delta_rel_quantile": 0.95,
    "noise_sigma_255": (4, 8, 16, 32, 64),
    "contrast_gamma": (0.90, 0.80, 0.60, 0.40, 0.20),
    "blur_sigma": (0.5, 1.0, 2.0, 4.0, 8.0),
    "resolution_sizes": (192, 160, 128, 96, 64),
    "severity_targets": (0.5, 1.0, 2.0),
    "shift_level_fractions": (0.25, 0.50, 0.75),
    "subgroup_categories": ("male", "female"),
    "subgroup_unusable_values": ("", "null", "missing", "unknown"),
    "r_candidates": (100, 200, 400),
    "r_methods": R_METHODS,
    "mcse_max": 0.025,
    "stability_max_difference": 0.025,
    "far_max": 0.05,
    "sampling_namespaces": tuple(SamplingNamespace),
    "technical_shifts": tuple(TechnicalShift),
    "stop_conditions": tuple(StopReason),
}


class CalibrationProtocolConfig(FrozenModel):
    protocol_version: Literal["calibration-protocol-baseline:1.0"]
    monitoring_methods_version: Literal["monitoring-methods-baseline:1.2"]
    model_id: Literal["resnet18-pilot-P1"]
    # Pools und Canonical Sample (DEC-050)
    pool_order: tuple[CalibrationPool, ...]
    pool_percent: tuple[int, ...]
    pool_apportionment: Literal["LARGEST_REMAINDER_TIE_POOL_ORDER"]
    pool_rank_namespace: Literal["P4-01|POOL"]
    pool_block_assignment: Literal["RANK_BLOCKS_IN_POOL_ORDER"]
    canonical_namespace: Literal["P4-01|CANONICAL"]
    canonical_rule: Literal["MIN_HASH_TIE_LEXICOGRAPHIC_SAMPLE_ID"]
    class_order: tuple[Identifier, ...]
    # Window Size und Reference (DEC-051, DEC-052)
    window_size_candidates: tuple[int, ...]
    class_vector_apportionment: Literal["LARGEST_REMAINDER_TIE_CANONICAL_CLASS_ORDER"]
    window_validity_runs: Literal[100]
    window_size_rule: Literal["SMALLEST_VALID_WINDOW_PRIORITY_FALLBACK"]
    reference_namespace: Literal["P4-01|OFFICIAL_REFERENCE|"]
    candidate_reference_rule: Literal["OFFICIAL_REFERENCE_RANKING"]
    sigma_source: Literal["ALL_CANONICAL_CAL_REF"]
    # D0 und δ_rel (DEC-053)
    d0_repeats: Literal[1000]
    d0_definition: Literal["BA_REF_MINUS_BA_MONITORING"]
    quantile_rule: Literal["NEAREST_RANK_CEIL_QN_NO_INTERPOLATION"]
    quantiles: tuple[StrictFloat, ...]
    delta_rel_quantile: StrictFloat
    # Technische Shifts und Severity-Selection (DEC-054)
    technical_shifts: tuple[TechnicalShift, ...]
    noise_sigma_255: tuple[int, ...]
    noise_seed_namespace: Literal["P4-01|SHIFT_NOISE_EPS|"]
    contrast_gamma: tuple[StrictFloat, ...]
    blur_sigma: tuple[StrictFloat, ...]
    blur_radius_sigmas: Literal[3]
    blur_padding: Literal["reflect"]
    resolution_sizes: tuple[int, ...]
    severity_base_windows: Literal[100]
    severity_targets: tuple[StrictFloat, ...]
    severity_tie_break: Literal["PHYSICALLY_WEAKER_TRIPLE"]
    # Class-/Subgroup-Shift (DEC-055)
    shift_level_fractions: tuple[StrictFloat, ...]
    level_rounding: Literal["ROUND_HALF_UP"]
    subgroup_dimension: Literal["sex"]
    subgroup_categories: tuple[str, ...]
    subgroup_unusable_values: tuple[str, ...]
    subgroup_tie_target: Literal["female"]
    # Schedules (DEC-056)
    abrupt_schedule: Literal["CLEAN_THEN_LEVEL"]
    gradual_schedule: Literal["CLEAN_L1_L2_L3"]
    # Sampling (DEC-057)
    sampling_namespaces: tuple[SamplingNamespace, ...]
    repeat_seed_purpose: Literal["NAMESPACE_WINDOW_SIZE_SCENARIO_KEY"]
    # R (DEC-058)
    r_candidates: tuple[int, ...]
    r_half_reference_at_100: Literal[50]
    r_methods: tuple[Identifier, ...]
    mcse_max: StrictFloat
    stability_max_difference: StrictFloat
    # Fusion und Audit (DEC-059)
    fusion_candidates: tuple[FusionCandidate, ...]
    far_max: StrictFloat
    fusion_full_tie: Literal["FUSION_B"]
    audit_pre_fusion_access: Literal["STRUCTURAL_ONLY"]
    audit_post_fusion_signals: Literal["G1_G2_G3_SELECTED_FUSION_CLEAN_FAR"]
    stop_conditions: tuple[StopReason, ...]

    @model_validator(mode="after")
    def _frozen(self) -> Self:
        require_canonical_class_order(self.class_order)
        for name, expected in _FROZEN.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} ist durch P4-01 eingefroren auf {expected!r}")
        if self.fusion_candidates != tuple(FusionCandidate):
            raise ValueError("Fusion ausschließlich A und B")
        return self

    def pool_fractions(self) -> tuple[Fraction, ...]:
        return tuple(Fraction(p, 100) for p in self.pool_percent)

    def noise_sigmas(self) -> tuple[float, ...]:
        return tuple(v / 255 for v in self.noise_sigma_255)

    def severity_grid(self, shift: TechnicalShift) -> tuple[float, ...]:
        """Kandidaten einer technischen Familie in physischer Ordnung (schwach → stark)."""
        grids: dict[TechnicalShift, tuple[float, ...]] = {
            TechnicalShift.NOISE: self.noise_sigmas(),
            TechnicalShift.CONTRAST: self.contrast_gamma,
            TechnicalShift.BLUR: self.blur_sigma,
            TechnicalShift.RESOLUTION: tuple(float(s) for s in self.resolution_sizes),
        }
        return grids[shift]

    def config_sha256(self) -> str:
        return canonical_sha256(self)


def baseline_protocol_config() -> CalibrationProtocolConfig:
    """Die einzige zulässige Konfiguration gemäß CALIBRATION_PROTOCOL_BASELINE v1.0."""
    return CalibrationProtocolConfig.model_validate(
        {
            **_FROZEN,
            "protocol_version": PROTOCOL_VERSION,
            "monitoring_methods_version": "monitoring-methods-baseline:1.2",
            "model_id": "resnet18-pilot-P1",
            "pool_apportionment": "LARGEST_REMAINDER_TIE_POOL_ORDER",
            "pool_rank_namespace": "P4-01|POOL",
            "pool_block_assignment": "RANK_BLOCKS_IN_POOL_ORDER",
            "canonical_namespace": "P4-01|CANONICAL",
            "canonical_rule": "MIN_HASH_TIE_LEXICOGRAPHIC_SAMPLE_ID",
            "class_vector_apportionment": "LARGEST_REMAINDER_TIE_CANONICAL_CLASS_ORDER",
            "window_validity_runs": 100,
            "window_size_rule": "SMALLEST_VALID_WINDOW_PRIORITY_FALLBACK",
            "reference_namespace": "P4-01|OFFICIAL_REFERENCE|",
            "candidate_reference_rule": "OFFICIAL_REFERENCE_RANKING",
            "sigma_source": "ALL_CANONICAL_CAL_REF",
            "d0_repeats": 1000,
            "d0_definition": "BA_REF_MINUS_BA_MONITORING",
            "quantile_rule": "NEAREST_RANK_CEIL_QN_NO_INTERPOLATION",
            "noise_seed_namespace": "P4-01|SHIFT_NOISE_EPS|",
            "blur_radius_sigmas": 3,
            "blur_padding": "reflect",
            "severity_base_windows": 100,
            "severity_tie_break": "PHYSICALLY_WEAKER_TRIPLE",
            "level_rounding": "ROUND_HALF_UP",
            "subgroup_dimension": "sex",
            "subgroup_tie_target": "female",
            "abrupt_schedule": "CLEAN_THEN_LEVEL",
            "gradual_schedule": "CLEAN_L1_L2_L3",
            "repeat_seed_purpose": "NAMESPACE_WINDOW_SIZE_SCENARIO_KEY",
            "r_half_reference_at_100": 50,
            "fusion_candidates": tuple(FusionCandidate),
            "fusion_full_tie": "FUSION_B",
            "audit_pre_fusion_access": "STRUCTURAL_ONLY",
            "audit_post_fusion_signals": "G1_G2_G3_SELECTED_FUSION_CLEAN_FAR",
        }
    )
