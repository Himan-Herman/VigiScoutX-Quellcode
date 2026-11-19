import pytest
from pydantic import ValidationError

from signal_factories import (
    group_data,
    performance_assessment,
    primary_risk,
    signal_data,
)
from vigiscoutx.domain.assessments import (
    GroupAssessment,
    PerformanceAssessment,
    PrimaryRiskAssessment,
    PrimaryRiskState,
    SignalResult,
)
from vigiscoutx.domain.refs import SampleRef, SeedBundle
from vigiscoutx.domain.types import SignalGroup
from vigiscoutx.domain.windows import (
    PROBABILITY_SUM_TOLERANCE,
    ModelOutputBatch,
    ReferenceWindow,
    RepresentationBatch,
)
from vigiscoutx.experiment.results import EvaluationResult, MetricValue
from vigiscoutx.experiment.types import EvaluationMetric

H = "0" * 64


def performance(relevant: bool) -> PerformanceAssessment:
    return performance_assessment(relevant, window_id="w1")


def primary(state: PrimaryRiskState = PrimaryRiskState.R0) -> PrimaryRiskAssessment:
    return primary_risk(state.value, window_id="w1")


def test_contracts_are_immutable() -> None:
    window = ReferenceWindow(window_id="r", samples=(SampleRef(sample_id="s1", group_id="l1"),))
    with pytest.raises(ValidationError):
        window.window_id = "other"  # type: ignore[misc]


def test_window_rejects_duplicate_samples() -> None:
    sample = {"sample_id": "s1", "group_id": "l1"}
    with pytest.raises(ValidationError):
        ReferenceWindow.model_validate({"window_id": "r", "samples": [sample, sample]})


@pytest.mark.parametrize(
    "rows",
    [
        [[0.5, 0.6]],  # Summe ≠ 1
        [[1.2, -0.2]],  # außerhalb [0, 1]
        [[1.0]],  # nur eine Klasse
        [[0.5, 0.5], [0.2, 0.3, 0.5]],  # uneinheitliche Breite
        [[0.5, 0.5 + 2 * PROBABILITY_SUM_TOLERANCE]],  # knapp außerhalb der Toleranz
    ],
)
def test_model_output_batch_rejects_invalid_softmax(rows: list[list[float]]) -> None:
    ids = [f"s{i}" for i in range(len(rows))]
    with pytest.raises(ValidationError):
        ModelOutputBatch.model_validate(
            {"window_id": "w", "sample_ids": ids, "probabilities": rows}
        )


def test_model_output_batch_accepts_valid_softmax() -> None:
    batch = ModelOutputBatch(
        window_id="w", sample_ids=("s1", "s2"), probabilities=((0.25, 0.75), (1.0, 0.0))
    )
    assert len(batch.probabilities) == 2


def test_representation_batch_requires_row_per_sample() -> None:
    with pytest.raises(ValidationError):
        RepresentationBatch.model_validate(
            {"window_id": "w", "sample_ids": ["s1", "s2"], "embeddings": [[0.1, 0.2]]}
        )


@pytest.mark.parametrize("model", [SignalResult, GroupAssessment])
def test_primary_signal_contracts_reject_g4(model: type[SignalResult | GroupAssessment]) -> None:
    data = signal_data(group="G4") if model is SignalResult else group_data(group="G4")
    with pytest.raises(ValidationError):
        model.model_validate(data)


def test_performance_assessment_consistency() -> None:
    assert performance(relevant=True).relevant_degradation is True
    with pytest.raises(ValidationError):
        performance(relevant=True).model_copy(update={"relevant_degradation": False})
    with pytest.raises(ValidationError):
        performance(relevant=True).model_copy(update={"delta_ba": 0.0})


def test_performance_degradation_equal_to_delta_rel_is_relevant() -> None:
    base = performance(relevant=True)
    assert base.delta_ba is not None
    boundary = base.model_copy(update={"delta_rel": base.delta_ba, "relevant_degradation": True})
    assert boundary.relevant_degradation is True
    with pytest.raises(ValidationError):
        boundary.model_copy(update={"relevant_degradation": False})


def test_evaluation_result_requires_freeze_id_and_metrics() -> None:
    result = EvaluationResult(
        evaluation_id="e1",
        freeze_id=H,
        method_id="m1",
        metrics=(MetricValue(metric=EvaluationMetric.F1, value=0.5),),
    )
    assert result.metrics[0].value == 0.5
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(
            {
                "evaluation_id": "e1",
                "freeze_id": None,
                "method_id": "m1",
                "metrics": [{"metric": "F1", "value": 0.5}],
            }
        )
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(
            {
                "evaluation_id": "e1",
                "freeze_id": H,
                "method_id": "m1",
                "metrics": [{"metric": "F1", "value": 0.5}, {"metric": "F1", "value": 0.6}],
            }
        )


def test_seed_bundle_rejects_negative_seeds() -> None:
    with pytest.raises(ValidationError):
        SeedBundle.model_validate(
            {
                "split_seed": -1,
                "model_development_seed": 0,
                "sampling_seed": 0,
                "shift_seed": 0,
                "statistical_seed": 0,
                "repeat_id": 0,
            }
        )


def test_signal_group_values_are_stable() -> None:
    assert [g.value for g in SignalGroup] == ["G1", "G2", "G3", "G4"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("alarm", "yes"), ("reference_n", "3"), ("p_value", "0.01"), ("alarm", 1)],
)
def test_contract_scalars_are_strict(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        SignalResult.model_validate(signal_data(**{field: value}))
