"""G3 – Predictive Behaviour (DEC-044, MONITORING_METHODS_BASELINE 4).

- `G3_OUTPUT_MUKS`: je Softmax-Komponente (kanonische Klassenreihenfolge) ein exakter zweiseitiger
  Zwei-Stichproben-KS-Test, Bonferroni `α/7`; strikte Aggregation über die sieben Komponenten.
- `G3_CONFIDENCE_KS`: `confidence = max` Softmax-Wahrscheinlichkeit, exakter zweiseitiger KS-Test.
- G3-Gruppe: `output_alarm OR confidence_alarm` mit Substatus (nur bei gültiger Gruppe).
"""

import numpy as np
import numpy.typing as npt

from vigiscoutx.domain.assessments import (
    ComponentResult,
    GroupAssessment,
    ReasonCode,
    SignalResult,
    SignalStatus,
)
from vigiscoutx.domain.frozen_json import FrozenMapping
from vigiscoutx.domain.refs import ModelSnapshotRef
from vigiscoutx.domain.types import SignalGroup
from vigiscoutx.domain.windows import ModelOutputBatch
from vigiscoutx.grouping.aggregation import aggregate_strict, assess_g3_group
from vigiscoutx.signals.config import G3PredictiveConfig
from vigiscoutx.signals.statistics import bonferroni_adjust, exact_two_sided_ks, is_significant

FloatArray = npt.NDArray[np.float64]
OUTPUT_METHOD = "MUKS_KS_2SAMP_EXACT_TWO_SIDED_BONFERRONI"
CONFIDENCE_METHOD = "KS_2SAMP_EXACT_TWO_SIDED"


def _probabilities(batch: ModelOutputBatch) -> FloatArray:
    matrix = np.array(batch.probabilities, dtype=np.float64)
    matrix.setflags(write=False)
    return matrix


def max_softmax_confidence(probabilities: FloatArray) -> FloatArray:
    """`confidence_i = max_c softmax_{i,c}`."""
    return np.asarray(probabilities.max(axis=1), dtype=np.float64)


def _base(
    signal_id: str,
    method: str,
    reference: ModelOutputBatch,
    monitoring: ModelOutputBatch,
    config: G3PredictiveConfig,
    model_snapshot_ref: ModelSnapshotRef,
) -> dict[str, object]:
    return {
        "signal_id": signal_id,
        "group": SignalGroup.G3_PREDICTIVE,
        "method": method,
        "method_version": config.method_version,
        "parameters_ref": config.parameters_ref(),
        "reference_window_id": reference.window_id,
        "monitoring_window_id": monitoring.window_id,
        "reference_n": len(reference.sample_ids),
        "monitoring_n": len(monitoring.sample_ids),
        "seed_ref": None,
        "model_snapshot_ref": model_snapshot_ref,
        "artifact_refs": (),
        "evidence_refs": (),
        "alpha_or_threshold": config.alpha,
    }


def _output_component(
    class_id: str, ref: FloatArray, mon: FloatArray, config: G3PredictiveConfig
) -> ComponentResult:
    ks = exact_two_sided_ks(ref, mon, config.ks)
    fixed = {"class_id": class_id}
    if ks.status is SignalStatus.VALID and ks.p_value is not None:
        adjusted = bonferroni_adjust(ks.p_value, config.output_bonferroni_factor)
        return ComponentResult(
            component_id=class_id,
            status=SignalStatus.VALID,
            statistics=FrozenMapping({**fixed, "ks_d": ks.statistic}),
            p_value=ks.p_value,
            adjusted_p_value=adjusted,
            alarm=is_significant(adjusted, config.alpha),
            reason_code=None,
        )
    return ComponentResult(
        component_id=class_id,
        status=ks.status,
        statistics=FrozenMapping(fixed),
        p_value=None,
        adjusted_p_value=None,
        alarm=None,
        reason_code=ks.reason_code,
    )


def run_g3_output(
    reference: ModelOutputBatch,
    monitoring: ModelOutputBatch,
    config: G3PredictiveConfig,
    model_snapshot_ref: ModelSnapshotRef,
) -> SignalResult:
    """G3_OUTPUT_MUKS: sieben klassenweise KS-Komponenten, strikt aggregiert (DEC-049)."""
    base = _base(
        config.output_signal_id, OUTPUT_METHOD, reference, monitoring, config, model_snapshot_ref
    )
    fixed = {
        "bonferroni_factor": config.output_bonferroni_factor,
        "component_alpha": config.alpha / config.output_bonferroni_factor,
    }
    ref, mon = _probabilities(reference), _probabilities(monitoring)
    k = len(config.class_order)
    if ref.shape[1] != k or mon.shape[1] != k:
        return SignalResult.model_validate(
            {
                **base,
                "status": SignalStatus.ERROR,
                "statistics": FrozenMapping(fixed),
                "p_value": None,
                "adjusted_p_value": None,
                "alarm": None,
                "reason_code": ReasonCode.INVALID_CLASS_DIMENSION,
                "components": (),
            }
        )
    components = tuple(
        _output_component(class_id, ref[:, index], mon[:, index], config)
        for index, class_id in enumerate(config.class_order)
    )
    outcome = aggregate_strict(components)
    statistics: dict[str, object] = dict(fixed)
    if outcome.status is SignalStatus.VALID:
        statistics["minimum_raw_p"] = min(c.p_value for c in components if c.p_value is not None)
        statistics["minimum_adjusted_p"] = min(
            c.adjusted_p_value for c in components if c.adjusted_p_value is not None
        )
    return SignalResult.model_validate(
        {
            **base,
            "status": outcome.status,
            "statistics": FrozenMapping(statistics),
            "p_value": None,
            "adjusted_p_value": None,
            "alarm": outcome.alarm,
            "reason_code": outcome.reason_code,
            "components": components,
        }
    )


def run_g3_confidence(
    reference: ModelOutputBatch,
    monitoring: ModelOutputBatch,
    config: G3PredictiveConfig,
    model_snapshot_ref: ModelSnapshotRef,
) -> SignalResult:
    """G3_CONFIDENCE_KS: exakter zweiseitiger KS-Test auf der maximalen Softmax-Ausgabe."""
    base = _base(
        config.confidence_signal_id,
        CONFIDENCE_METHOD,
        reference,
        monitoring,
        config,
        model_snapshot_ref,
    )
    ref, mon = _probabilities(reference), _probabilities(monitoring)
    k = len(config.class_order)
    status: SignalStatus = SignalStatus.ERROR
    reason: ReasonCode | None = ReasonCode.INVALID_CLASS_DIMENSION
    ks = None
    if ref.shape[1] == k and mon.shape[1] == k:
        ks = exact_two_sided_ks(max_softmax_confidence(ref), max_softmax_confidence(mon), config.ks)
        status, reason = ks.status, ks.reason_code
    if ks is not None and ks.status is SignalStatus.VALID and ks.p_value is not None:
        return SignalResult.model_validate(
            {
                **base,
                "status": SignalStatus.VALID,
                "statistics": FrozenMapping({"ks_d": ks.statistic}),
                "p_value": ks.p_value,
                "adjusted_p_value": None,
                "alarm": is_significant(ks.p_value, config.alpha),
                "reason_code": None,
                "components": (),
            }
        )
    return SignalResult.model_validate(
        {
            **base,
            "status": status,
            "statistics": FrozenMapping({}),
            "p_value": None,
            "adjusted_p_value": None,
            "alarm": None,
            "reason_code": reason,
            "components": (),
        }
    )


def run_g3(
    reference: ModelOutputBatch,
    monitoring: ModelOutputBatch,
    config: G3PredictiveConfig,
    model_snapshot_ref: ModelSnapshotRef,
) -> tuple[SignalResult, SignalResult, GroupAssessment]:
    """Output- und Confidence-Subsignal sowie das G3-`GroupAssessment` (zählt als eine Gruppe)."""
    output = run_g3_output(reference, monitoring, config, model_snapshot_ref)
    confidence = run_g3_confidence(reference, monitoring, config, model_snapshot_ref)
    return output, confidence, assess_g3_group(output, confidence)
