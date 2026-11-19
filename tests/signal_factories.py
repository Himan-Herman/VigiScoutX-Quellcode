"""Gemeinsame Fixture-Bausteine für Signal- und Performance-Contracts (nur synthetische Daten)."""

from typing import Any

from vigiscoutx.adapters.demonstrator import DEMONSTRATOR_CLASS_ORDER
from vigiscoutx.domain.assessments import (
    ComponentResult,
    G3Substatus,
    GroupAssessment,
    LabelAwareAssessment,
    LabelAwareRiskState,
    ParametersRef,
    PerformanceAssessment,
    PrimaryRiskAssessment,
    PrimaryRiskState,
    ReasonCode,
    SignalResult,
    SignalStatus,
)
from vigiscoutx.domain.frozen_json import FrozenMapping
from vigiscoutx.domain.refs import ModelSnapshotRef, relevance_evidence, window_evidence
from vigiscoutx.domain.types import SignalGroup
from vigiscoutx.grouping.aggregation import assess_g3_group, assess_group
from vigiscoutx.signals.config import (
    G1_SIGNAL_ID,
    G2_SIGNAL_IDS,
    G3_CONFIDENCE_SIGNAL_ID,
    AggregationRule,
    G1MmdConfig,
    G2Indicator,
    G2QualityConfig,
    G3PredictiveConfig,
    KsSpec,
)
from vigiscoutx.signals.performance.g4 import (
    G4PerformanceConfig,
    assess_performance,
    evaluate_performance,
)

H = "0" * 64
MODEL_REF = ModelSnapshotRef(model_id="m", architecture="resnet18", uri="local:m", sha256="a" * 64)
PARAMS_REF = ParametersRef(config_id="cfg", method_version="v1", sha256=H)
KS = KsSpec(test="TWO_SAMPLE_KS", alternative="two-sided", method="exact", on_exact_failure="ERROR")


def g1_config(sigma: float = 1.0) -> G1MmdConfig:
    return G1MmdConfig(
        method_version="monitoring-methods-baseline:1.2",
        signal_id="G1_MMD_EMBEDDING",
        embedding="RESNET18_PENULTIMATE",
        embedding_dim=512,
        embedding_transform="NONE",
        estimator="MMD2_UNBIASED_QUADRATIC",
        kernel="GAUSSIAN_RBF",
        sigma=sigma,
        inference="RANDOM_PERMUTATION_FIXED_GROUP_SIZES",
        n_permutations=999,
        tail="RIGHT",
        alpha=0.05,
        seed_field="statistical_seed",
    )


def g2_config() -> G2QualityConfig:
    return G2QualityConfig(
        method_version="monitoring-methods-baseline:1.2",
        group_id="G2",
        input="RGB_FLOAT_0_1_BEFORE_IMAGENET_NORMALIZATION",
        input_height=224,
        input_width=224,
        luma_coefficients=(0.2126, 0.7152, 0.0722),
        gamma_linearization=False,
        std_ddof=0,
        convolution="VALID",
        laplacian_kernel=((0, 1, 0), (1, -4, 1), (0, 1, 0)),
        immerkaer_kernel=((1, -2, 1), (-2, 4, -2), (1, -2, 1)),
        indicators=tuple(G2Indicator),
        ks=KS,
        alpha=0.05,
        bonferroni_factor=4,
        aggregation=AggregationRule.STRICT_ERROR_NOT_EVALUABLE_VALID,
    )


def g3_config() -> G3PredictiveConfig:
    return G3PredictiveConfig(
        method_version="monitoring-methods-baseline:1.2",
        group_id="G3",
        class_order=DEMONSTRATOR_CLASS_ORDER,
        output_signal_id="G3_OUTPUT_MUKS",
        confidence_signal_id="G3_CONFIDENCE_KS",
        confidence="MAX_SOFTMAX",
        ks=KS,
        alpha=0.05,
        output_bonferroni_factor=7,
        group_rule="OUTPUT_OR_CONFIDENCE",
        aggregation=AggregationRule.STRICT_ERROR_NOT_EVALUABLE_VALID,
    )


def g4_config() -> G4PerformanceConfig:
    return G4PerformanceConfig(
        performance_id="G4_PERFORMANCE",
        method_version="monitoring-methods-baseline:1.2",
        class_order=DEMONSTRATOR_CLASS_ORDER,
        metric="BALANCED_ACCURACY",
        missing_class_rule="NOT_EVALUABLE_NO_REAVERAGING",
        zero_division="NULL",
    )


def signal_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "signal_id": "sig-1",
        "group": SignalGroup.G1_INPUT,
        "method": "TEST_METHOD",
        "method_version": "v1",
        "status": SignalStatus.VALID,
        "statistics": FrozenMapping({"d": 0.5}),
        "p_value": 0.5,
        "adjusted_p_value": None,
        "alpha_or_threshold": 0.05,
        "alarm": False,
        "reason_code": None,
        "components": (),
        "parameters_ref": PARAMS_REF,
        "reference_window_id": "r",
        "monitoring_window_id": "w",
        "reference_n": 3,
        "monitoring_n": 3,
        "seed_ref": None,
        "model_snapshot_ref": None,
        "artifact_refs": (),
        "evidence_refs": (),
    }
    data.update(overrides)
    return data


def signal_result(**overrides: Any) -> SignalResult:
    return SignalResult.model_validate(signal_data(**overrides))


_REASONS = {
    SignalStatus.VALID: None,
    SignalStatus.NOT_EVALUABLE: ReasonCode.INSUFFICIENT_SAMPLE,
    SignalStatus.ERROR: ReasonCode.KS_EXACT_FALLBACK,
}


def component(
    component_id: str, status: SignalStatus = SignalStatus.VALID, alarm: bool | None = False
) -> ComponentResult:
    valid = status is SignalStatus.VALID
    return ComponentResult(
        component_id=component_id,
        status=status,
        statistics=FrozenMapping({"class_id": component_id}),
        p_value=0.001 if alarm else (0.5 if valid else None),
        adjusted_p_value=0.007 if alarm else (1.0 if valid else None),
        alarm=alarm if valid else None,
        reason_code=_REASONS[status],
    )


def g3_output_data(components: tuple[ComponentResult, ...], **overrides: Any) -> dict[str, Any]:
    """G3_OUTPUT_MUKS-Rohdaten; Status und Alarm folgen der strikten Aggregation."""
    statuses = {c.status for c in components}
    if SignalStatus.ERROR in statuses:
        status, alarm, reason = SignalStatus.ERROR, None, ReasonCode.SUBSIGNAL_ERROR
    elif SignalStatus.NOT_EVALUABLE in statuses:
        status, alarm, reason = (
            SignalStatus.NOT_EVALUABLE,
            None,
            ReasonCode.SUBSIGNAL_NOT_EVALUABLE,
        )
    else:
        status, alarm, reason = SignalStatus.VALID, any(c.alarm for c in components), None
    data = signal_data(
        signal_id="G3_OUTPUT_MUKS",
        group=SignalGroup.G3_PREDICTIVE,
        status=status,
        alarm=alarm,
        reason_code=reason,
        p_value=None,
        components=components,
    )
    data.update(overrides)
    return data


def canonical_components(
    status: SignalStatus = SignalStatus.VALID, alarm: bool | None = False
) -> tuple[ComponentResult, ...]:
    """Sieben kanonische Komponenten; nur die erste trägt Status bzw. Alarm."""
    first = component(DEMONSTRATOR_CLASS_ORDER[0], status, alarm)
    return (first, *(component(c) for c in DEMONSTRATOR_CLASS_ORDER[1:]))


def group_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "group": SignalGroup.G1_INPUT,
        "window_id": "w",
        "reference_window_id": "r",
        "status": SignalStatus.VALID,
        "alarm": True,
        "substatus": None,
        "reason_code": None,
        "signal_ids": ("s",),
        "evidence_refs": (),
    }
    data.update(overrides)
    return data


def group_assessment(**overrides: Any) -> GroupAssessment:
    return GroupAssessment.model_validate(group_data(**overrides))


def _labels(errors_in_classes: int) -> tuple[list[int], list[int]]:
    """Je Klasse zwei Beispiele; in den ersten `errors_in_classes` Klassen ist eines falsch."""
    y_true = [c for c in range(7) for _ in range(2)]
    y_pred = list(y_true)
    for c in range(errors_in_classes):
        y_pred[2 * c] = (c + 1) % 7
    return y_true, y_pred


def performance_assessment(
    relevant: bool | None, window_id: str = "w", delta_rel: float | None = 0.1
) -> PerformanceAssessment:
    """BA_ref = 1.0; BA_monitoring = 5/7 (relevant) bzw. 13/14 (nicht relevant)."""
    config = g4_config()
    reference = evaluate_performance("r", *_labels(0), config)
    monitoring = evaluate_performance(window_id, *_labels(4 if relevant else 1), config)
    return assess_performance(
        reference, monitoring, config, MODEL_REF, delta_rel if relevant is not None else None
    )


# --- Risk-/Recommendation-Contracts (P4-03B) ----------------------------------------------------

RELEVANCE_SHA = "c" * 64
_FLAGGED = {
    "R0": (),
    "R1": (SignalGroup.G1_INPUT,),
    "R2": (SignalGroup.G1_INPUT, SignalGroup.G2_DATA_QUALITY),
}


def primary_risk(state: str = "R0", window_id: str = "w") -> PrimaryRiskAssessment:
    """Gültiges Primary-Ergebnis mit den drei Gruppenreferenzen des Windows."""
    return PrimaryRiskAssessment(
        window_id=window_id,
        reference_window_id="r",
        state=PrimaryRiskState(state),
        flagged_groups=_FLAGGED[state],
        evidence_refs=tuple(
            window_evidence("group_assessment", window_id, g) for g in ("G1", "G2", "G3")
        ),
    )


def label_aware_assessment(
    primary: PrimaryRiskAssessment, performance: PerformanceAssessment, state: LabelAwareRiskState
) -> LabelAwareAssessment:
    return LabelAwareAssessment(
        window_id=primary.window_id,
        primary=primary,
        performance=performance,
        relevance_config_sha256=RELEVANCE_SHA,
        state=state,
        evidence_refs=(
            window_evidence("primary_risk_assessment", primary.window_id, "primary"),
            window_evidence("performance_assessment", performance.window_id, "G4"),
            relevance_evidence(RELEVANCE_SHA),
        ),
    )


def primary_group(
    group: SignalGroup,
    *,
    window_id: str = "w",
    reference_window_id: str = "r",
    status: SignalStatus = SignalStatus.VALID,
    alarm: bool | None = False,
) -> GroupAssessment:
    """Status-konsistentes Pflicht-`GroupAssessment` (G3 mit passendem Substatus)."""
    valid = status is SignalStatus.VALID
    reason = None if valid else _GROUP_REASONS[status]
    substatus = None
    if group is SignalGroup.G3_PREDICTIVE and valid:
        substatus = G3Substatus.OUTPUT_ONLY if alarm else G3Substatus.NONE
    return GroupAssessment.model_validate(
        group_data(
            group=group,
            window_id=window_id,
            reference_window_id=reference_window_id,
            status=status,
            alarm=alarm if valid else None,
            substatus=substatus,
            reason_code=reason,
            evidence_refs=(),
        )
    )


_GROUP_REASONS = {
    SignalStatus.ERROR: ReasonCode.SUBSIGNAL_ERROR,
    SignalStatus.NOT_EVALUABLE: ReasonCode.SUBSIGNAL_NOT_EVALUABLE,
}


def primary_groups(
    alarms: tuple[bool, bool, bool] = (False, False, False), window_id: str = "w"
) -> tuple[GroupAssessment, ...]:
    return tuple(
        primary_group(g, window_id=window_id, alarm=a)
        for g, a in zip(
            (SignalGroup.G1_INPUT, SignalGroup.G2_DATA_QUALITY, SignalGroup.G3_PREDICTIVE),
            alarms,
            strict=True,
        )
    )


def g4_error_assessment(window_id: str = "w") -> PerformanceAssessment:
    """G4 mit Reference-ERROR (unbekannte Klasse in den Vorhersagen)."""
    config = g4_config()
    reference = evaluate_performance("r", [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 9], config)
    monitoring = evaluate_performance(
        window_id, [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], config
    )
    return assess_performance(reference, monitoring, config, MODEL_REF, 0.1)


def g4_not_evaluable_assessment(window_id: str = "w") -> PerformanceAssessment:
    """G4 mit nicht evaluierbarem Monitoring-Window (fehlende Ground-Truth-Klasse)."""
    config = g4_config()
    reference = evaluate_performance("r", [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], config)
    monitoring = evaluate_performance(window_id, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], config)
    return assess_performance(reference, monitoring, config, MODEL_REF, 0.1)


# --- Vollständige Nachweiskette SignalResult → GroupAssessment (P4-03C) -------------------------


def _child(
    signal_id: str,
    group: SignalGroup,
    window_id: str,
    reference_window_id: str,
    status: SignalStatus,
    alarm: bool,
) -> SignalResult:
    valid = status is SignalStatus.VALID
    return signal_result(
        signal_id=signal_id,
        group=group,
        status=status,
        alarm=alarm if valid else None,
        reason_code=_REASONS[status],
        monitoring_window_id=window_id,
        reference_window_id=reference_window_id,
    )


def group_signals(
    group: SignalGroup,
    *,
    window_id: str = "w",
    reference_window_id: str = "r",
    status: SignalStatus = SignalStatus.VALID,
    alarm: bool = False,
) -> tuple[SignalResult, ...]:
    """Exakt die eingefrorenen Subsignale einer Gruppe; nur das erste trägt Status bzw. Alarm."""
    if group is SignalGroup.G1_INPUT:
        return (_child(G1_SIGNAL_ID, group, window_id, reference_window_id, status, alarm),)
    if group is SignalGroup.G2_DATA_QUALITY:
        ids = list(G2_SIGNAL_IDS.values())
        first = _child(ids[0], group, window_id, reference_window_id, status, alarm)
        rest = (
            _child(i, group, window_id, reference_window_id, SignalStatus.VALID, False)
            for i in ids[1:]
        )
        return (first, *rest)
    output = SignalResult.model_validate(
        g3_output_data(
            canonical_components(status, alarm if status is SignalStatus.VALID else None),
            monitoring_window_id=window_id,
            reference_window_id=reference_window_id,
        )
    )
    confidence = _child(
        G3_CONFIDENCE_SIGNAL_ID, group, window_id, reference_window_id, SignalStatus.VALID, False
    )
    return (output, confidence)


def aggregate(group: SignalGroup, signals: tuple[SignalResult, ...]) -> GroupAssessment:
    """Gruppenergebnis ausschließlich über die bestehende Aggregation (DEC-049)."""
    if group is SignalGroup.G3_PREDICTIVE:
        return assess_g3_group(signals[0], signals[1])
    return assess_group(group, signals)


def signal_chain(
    alarms: tuple[bool, bool, bool] = (False, False, False),
    window_id: str = "w",
    *,
    statuses: tuple[SignalStatus, SignalStatus, SignalStatus] = (SignalStatus.VALID,) * 3,
    groups: tuple[SignalGroup, ...] = (
        SignalGroup.G1_INPUT,
        SignalGroup.G2_DATA_QUALITY,
        SignalGroup.G3_PREDICTIVE,
    ),
    reference_window_id: str = "r",
) -> tuple[tuple[SignalResult, ...], tuple[GroupAssessment, ...]]:
    """SignalResults und daraus aggregierte Gruppen eines Windows (Teilmenge über `groups`)."""
    order = (SignalGroup.G1_INPUT, SignalGroup.G2_DATA_QUALITY, SignalGroup.G3_PREDICTIVE)
    signals: list[SignalResult] = []
    assessed: list[GroupAssessment] = []
    for group, alarm, status in zip(order, alarms, statuses, strict=True):
        if group not in groups:
            continue
        children = group_signals(
            group,
            window_id=window_id,
            reference_window_id=reference_window_id,
            status=status,
            alarm=alarm,
        )
        signals += children
        assessed.append(aggregate(group, children))
    return tuple(signals), tuple(assessed)
