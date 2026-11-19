"""Architekturinvarianten aus DEC-029 – DEC-032."""

import ast
import copy
import importlib
import inspect
import pickle
import pkgutil
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import vigiscoutx
from signal_factories import (
    group_assessment,
    group_data,
    label_aware_assessment,
    performance_assessment,
    primary_risk,
    signal_result,
)
from vigiscoutx.config.scientific_config import SignalConfig
from vigiscoutx.domain.assessments import (
    AnalysisLevel,
    GroupAssessment,
    LabelAwareRiskState,
    PerformanceAssessment,
    PrimaryRiskAssessment,
    PrimaryRiskState,
    Recommendation,
)
from vigiscoutx.domain.frozen_json import FrozenMapping
from vigiscoutx.domain.refs import EvidenceRef, window_evidence
from vigiscoutx.domain.types import FrozenModel, SignalGroup
from vigiscoutx.domain.windows import MonitoringWindow
from vigiscoutx.experiment.freeze import FreezeManifest
from vigiscoutx.experiment.results import EvaluationResult, MetricValue
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.types import EvaluationMetric
from vigiscoutx.ports import DataAccessContext, DatasetPort, LabelPort, ModelPort

SAMPLES = [{"sample_id": "s1", "group_id": "l1"}]
PERF_REF = EvidenceRef(kind="performance_assessment", ref_id="perf-1")
SIGNAL_REF = EvidenceRef(kind="signal_result", ref_id="sig-1")
SRC = Path(vigiscoutx.__file__).parent


def _all_models() -> list[type[FrozenModel]]:
    for module in pkgutil.walk_packages(vigiscoutx.__path__, "vigiscoutx."):
        importlib.import_module(module.name)
    found: list[type[FrozenModel]] = []
    pending: list[type[FrozenModel]] = [FrozenModel]
    while pending:
        cls = pending.pop()
        for sub in cls.__subclasses__():
            found.append(sub)
            pending.append(sub)
    return found


def _performance(relevant: bool) -> PerformanceAssessment:
    return performance_assessment(relevant)


def _primary(state: PrimaryRiskState = PrimaryRiskState.R2) -> PrimaryRiskAssessment:
    return primary_risk(state.value)


# --- MonitoringWindow ohne Experimentwissen -------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["shift_type", "shift", "severity", "ground_truth", "labels", "label", "relevant", "scenario"],
)
def test_monitoring_window_rejects_experiment_information(field: str) -> None:
    with pytest.raises(ValidationError):
        MonitoringWindow.model_validate(
            {"window_id": "m", "window_index": 0, "samples": SAMPLES, field: "x"}
        )


def test_monitoring_window_fields_are_exactly_defined() -> None:
    assert set(MonitoringWindow.model_fields) == {"window_id", "window_index", "samples"}


# --- Primary / R3 / G4 ------------------------------------------------------------------------


def test_primary_risk_assessment_rejects_r3_and_g4() -> None:
    base = {"window_id": "w", "flagged_groups": [], "evidence_refs": []}
    assert "R3" not in {s.value for s in PrimaryRiskState}
    with pytest.raises(ValidationError):
        PrimaryRiskAssessment.model_validate({**base, "state": "R3"})
    with pytest.raises(ValidationError):
        PrimaryRiskAssessment.model_validate({**base, "state": "R1", "flagged_groups": ["G4"]})


def test_model_copy_cannot_create_r3_in_primary() -> None:
    primary = _primary(PrimaryRiskState.R0)
    for value in (LabelAwareRiskState.R3, "R3"):
        with pytest.raises(ValidationError):
            primary.model_copy(update={"state": value})


def test_model_copy_cannot_create_g4_in_primary() -> None:
    with pytest.raises(ValidationError):
        _primary().model_copy(update={"flagged_groups": (SignalGroup.G4_PERFORMANCE,)})


@pytest.mark.parametrize(
    ("instance", "update"),
    [
        (
            group_assessment(),
            {"group": SignalGroup.G4_PERFORMANCE},
        ),
        (
            signal_result(),
            {"p_value": 2.0},
        ),
        (_performance(relevant=True), {"relevant_degradation": False}),
        (
            Recommendation(
                window_id="w",
                level=AnalysisLevel.PRIMARY,
                code="REVIEW",
                reason_codes=("RSN_X",),
                evidence_refs=(window_evidence("primary_risk_assessment", "w", "primary"),),
            ),
            {"code": "bitte prüfen"},
        ),
    ],
)
def test_model_copy_revalidates_updates(instance: FrozenModel, update: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        instance.model_copy(update=update)


def test_model_copy_revalidates_freeze_manifest(freeze_manifest: FreezeManifest) -> None:
    with pytest.raises(ValidationError):
        freeze_manifest.model_copy(update={"repetitions": freeze_manifest.repetitions + 1})


def test_model_copy_without_update_and_valid_update_still_work() -> None:
    primary = _primary(PrimaryRiskState.R0)
    assert primary.model_copy() == primary
    update = {"state": PrimaryRiskState.R1, "flagged_groups": (SignalGroup.G1_INPUT,)}
    assert primary.model_copy(update=update).state is PrimaryRiskState.R1
    with pytest.raises(ValidationError):
        primary.model_copy(update={"state": PrimaryRiskState.R1})  # Zustand ≠ alarmierende Gruppen


@pytest.mark.parametrize("model", [PrimaryRiskAssessment, GroupAssessment])
def test_primary_contracts_reject_performance_evidence(
    model: type[PrimaryRiskAssessment | GroupAssessment],
) -> None:
    if model is PrimaryRiskAssessment:
        valid = primary_risk("R1").model_dump(mode="json")
        base = {k: v for k, v in valid.items() if k != "evidence_refs"}
        allowed = valid["evidence_refs"]
    else:
        base = {k: v for k, v in group_data().items() if k != "evidence_refs"}
        allowed = [SIGNAL_REF.model_dump()]
    assert model.model_validate({**base, "evidence_refs": allowed})
    with pytest.raises(ValidationError, match="Performance-Evidenz"):
        model.model_validate({**base, "evidence_refs": [*allowed, PERF_REF.model_dump()]})


def test_label_aware_assessment_references_performance_and_relevance() -> None:
    assessment = label_aware_assessment(
        _primary(), _performance(relevant=True), LabelAwareRiskState.R3
    )
    assert [r.kind for r in assessment.evidence_refs] == [
        "primary_risk_assessment",
        "performance_assessment",
        "relevance_config",
    ]
    with pytest.raises(ValidationError, match="Sekundärnachweis"):
        assessment.model_copy(update={"evidence_refs": (PERF_REF,)})


def test_label_aware_r3_if_and_only_if_relevant_degradation() -> None:
    primary = _primary(PrimaryRiskState.R2)
    ok = label_aware_assessment(primary, _performance(relevant=True), LabelAwareRiskState.R3)
    assert ok.state is LabelAwareRiskState.R3
    assert ok.primary == primary
    with pytest.raises(ValidationError):
        label_aware_assessment(primary, _performance(relevant=False), LabelAwareRiskState.R3)
    for state in (LabelAwareRiskState.R0, LabelAwareRiskState.R1, LabelAwareRiskState.R2):
        with pytest.raises(ValidationError):
            label_aware_assessment(primary, _performance(relevant=True), state)
    # Ohne relevante Degradation bleibt exakt der Primary-Zustand erhalten (DEC-074).
    assert (
        label_aware_assessment(primary, _performance(relevant=False), LabelAwareRiskState.R2).state
        is LabelAwareRiskState.R2
    )
    for state in (LabelAwareRiskState.R0, LabelAwareRiskState.R1):
        with pytest.raises(ValidationError, match="Primary-Zustand"):
            label_aware_assessment(primary, _performance(relevant=False), state)


# --- Recommendation ------------------------------------------------------------------------


def test_recommendation_carries_codes_not_ui_text() -> None:
    assert set(Recommendation.model_fields) == {
        "window_id",
        "level",
        "code",
        "reason_codes",
        "evidence_refs",
    }
    evidence = [window_evidence("primary_risk_assessment", "w", "primary").model_dump()]
    base = {"window_id": "w", "level": "PRIMARY", "evidence_refs": evidence}
    ok = Recommendation.model_validate(
        {**base, "code": "REVIEW_REQUIRED", "reason_codes": ["G1_FLAGGED"]}
    )
    assert ok.code == "REVIEW_REQUIRED"
    for text in ("Bitte prüfen", "review required"):
        with pytest.raises(ValidationError):
            Recommendation.model_validate({**base, "code": text, "reason_codes": ["RSN_X"]})
    with pytest.raises(ValidationError):
        Recommendation.model_validate({**base, "code": "REVIEW", "reason_codes": []})
    # Empfehlungsnachweis nur aus Risk-Ergebnissen des eigenen Windows (P4-03C).
    for foreign in (
        {"kind": "artifact", "ref_id": "a1"},
        window_evidence("primary_risk_assessment", "other", "primary").model_dump(),
    ):
        with pytest.raises(ValidationError):
            Recommendation.model_validate(
                {**base, "evidence_refs": [foreign], "code": "REVIEW", "reason_codes": ["RSN_X"]}
            )


# --- Deep Immutability ----------------------------------------------------------------------


def test_nested_scientific_config_is_immutable(calibrated_config: RunConfig) -> None:
    signal = calibrated_config.scientific.monitoring.signals[1]
    params = signal.params
    assert isinstance(params, FrozenMapping)
    with pytest.raises(TypeError):
        params["k"] = 99  # type: ignore[index]
    nested = params["nested"]
    assert isinstance(nested, FrozenMapping)
    with pytest.raises(TypeError):
        nested["values"] = [3]  # type: ignore[index]
    values = nested["values"]
    assert isinstance(values, tuple)
    with pytest.raises(AttributeError):
        params._data = {}
    with pytest.raises(TypeError):
        calibrated_config.scientific.preprocessing.params["x"] = 1  # type: ignore[index]


def test_frozen_params_do_not_share_state_with_input() -> None:
    source: dict[str, Any] = {"values": [1, 2]}
    signal = SignalConfig.model_validate(
        {"signal_id": "s", "group": "G1", "method": "m", "params": source, "threshold": None}
    )
    source["values"].append(3)
    source["new"] = 1
    assert signal.params == FrozenMapping({"values": [1, 2]})


def _reachable_containers(value: object, seen: set[int] | None = None) -> list[str]:
    """Veränderbare Container, die über Feldwerte, FrozenMapping-Slots oder Tupel erreichbar sind.

    Bei Pydantic-Modellen werden nur Feldwerte betrachtet, nicht die Pydantic-Interna.
    """
    seen = set() if seen is None else seen
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, dict | list | set | bytearray):
        return [type(value).__name__]
    found: list[str] = []
    if isinstance(value, FrozenModel):
        for name in type(value).model_fields:
            found += _reachable_containers(getattr(value, name), seen)
    elif isinstance(value, tuple):
        for item in value:
            found += _reachable_containers(item, seen)
    elif isinstance(value, FrozenMapping):
        for cls in type(value).__mro__:
            for slot in getattr(cls, "__slots__", ()):
                if hasattr(value, slot):
                    found += _reachable_containers(getattr(value, slot), seen)
        assert not hasattr(value, "__dict__")
    return found


def test_frozen_mapping_internals_are_not_mutable(calibrated_config: RunConfig) -> None:
    params = calibrated_config.scientific.monitoring.signals[1].params
    assert not hasattr(params, "_data")
    assert not hasattr(params, "__dict__")
    with pytest.raises(AttributeError):
        params._data["k"] = 999  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        params._items[0] = ("k", 999)  # type: ignore[index]
    with pytest.raises(AttributeError):
        del params._items
    assert params["k"] == 1
    assert _reachable_containers(params) == []


def test_nested_frozen_structures_are_not_mutable(calibrated_config: RunConfig) -> None:
    params = calibrated_config.scientific.monitoring.signals[1].params
    nested = params["nested"]
    assert isinstance(nested, FrozenMapping)
    with pytest.raises(AttributeError):
        nested._data["values"] = [9]  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        nested._items[0] = ("values", (9,))  # type: ignore[index]
    values = nested["values"]
    assert values == (1, 2)
    with pytest.raises(TypeError):
        values[0] = 9  # type: ignore[index]
    with pytest.raises(AttributeError):
        values.append(3)  # type: ignore[attr-defined]
    assert _reachable_containers(calibrated_config) == []


def test_frozen_mapping_copies_preserve_identity_and_content(calibrated_config: RunConfig) -> None:
    params = calibrated_config.scientific.monitoring.signals[1].params
    assert copy.copy(params) is params
    assert copy.deepcopy(params) is params
    restored = pickle.loads(pickle.dumps(params))
    assert restored == params
    assert isinstance(restored, FrozenMapping)
    assert calibrated_config.model_copy(deep=True) == calibrated_config
    assert pickle.loads(pickle.dumps(calibrated_config)) == calibrated_config


def test_frozen_params_round_trip_to_plain_json(calibrated_config: RunConfig) -> None:
    dumped = calibrated_config.model_dump(mode="json")
    params_json = dumped["scientific"]["monitoring"]["signals"][1]["params"]
    assert params_json == {"k": 1, "nested": {"values": [1, 2]}}
    assert type(params_json) is dict
    assert type(params_json["nested"]["values"]) is list
    assert RunConfig.model_validate(dumped) == calibrated_config
    assert RunConfig.model_validate_json(calibrated_config.model_dump_json()) == calibrated_config


def test_evaluation_metrics_are_immutable() -> None:
    result = EvaluationResult(
        evaluation_id="e",
        freeze_id="0" * 64,
        method_id="m",
        metrics=(MetricValue(metric=EvaluationMetric.F1, value=0.5),),
    )
    assert isinstance(result.metrics, tuple)
    with pytest.raises(ValidationError):
        result.metrics[0].value = 1.0  # type: ignore[misc]


def test_no_mutable_container_fields_in_contracts() -> None:
    offending = [
        f"{m.__name__}.{name}"
        for m in _all_models()
        for name, info in m.model_fields.items()
        if re.search(r"\b(dict|list|set)\[", str(info.annotation))
    ]
    assert offending == []


# --- Ports -------------------------------------------------------------------------------------


def test_model_port_is_inference_only() -> None:
    members = {
        name
        for name, value in vars(ModelPort).items()
        if not name.startswith("_") and inspect.isfunction(value)
    }
    assert members == {"snapshot_ref", "predict_proba", "embed"}
    forbidden = re.compile(r"train|fit|update|retrain|mutate|set_|load_state|step|backward|optim")
    assert not [m for m in members if forbidden.search(m)]


@pytest.mark.parametrize(
    ("port", "method"), [(DatasetPort, "load_inputs"), (LabelPort, "labels_for")]
)
def test_partitioned_data_ports_require_access_context(port: type, method: str) -> None:
    parameters = inspect.signature(getattr(port, method)).parameters
    assert "access" in parameters
    assert parameters["access"].annotation is DataAccessContext
    assert inspect.isabstract(DataAccessContext)


# --- Defaults ----------------------------------------------------------------------------------


def test_no_model_field_has_a_hidden_default() -> None:
    models = _all_models()
    assert len(models) > 20
    with_defaults: dict[str, Any] = {
        f"{m.__module__}.{m.__name__}.{name}": info.default
        for m in models
        for name, info in m.model_fields.items()
        if not info.is_required()
    }
    assert with_defaults == {}


# --- Paketgrenze Core ↔ Experiment -----------------------------------------------------------


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Auch importierte Namen prüfen (`from vigiscoutx import evaluation`, `from .. import
            # experiment`); relative Importe ohne Modul liefern nur die Namen (P4-03AG).
            base = node.module or ""
            if base:
                names.add(base)
            names.update(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return names


NON_CORE_PACKAGES = frozenset({"experiment", "evaluation"})
"""Experimentumgebung und experimentelle Evaluationsschicht, kein Core (DEC-086)."""


def _non_core_imports(path: Path) -> list[str]:
    """Importierte Module, deren Pfad ein Nicht-Core-Paket nennt (absolut oder relativ)."""
    return sorted(m for m in _imported_modules(path) if NON_CORE_PACKAGES & set(m.split(".")))


def test_core_never_imports_experiment_package() -> None:
    core_files = [
        p for p in SRC.rglob("*.py") if p.relative_to(SRC).parts[0] not in NON_CORE_PACKAGES
    ]
    assert core_files
    violations = {str(p.relative_to(SRC)): _non_core_imports(p) for p in core_files}
    assert {k: v for k, v in violations.items() if v} == {}


@pytest.mark.parametrize(
    ("source", "detected"),
    [
        ("import vigiscoutx.experiment.freeze\n", True),
        ("from vigiscoutx.experiment.held_out_guard import HeldOutGuard\n", True),
        ("from vigiscoutx.evaluation.reference_sources import derive_reference\n", True),
        ("from ..evaluation import reference_reconstruction\n", True),
        ("from .experiment import split\n", True),
        ("from vigiscoutx import evaluation\n", True),
        ("from vigiscoutx import experiment\n", True),
        ("from .. import evaluation\n", True),
        ("from .. import experiment\n", True),
        ("from vigiscoutx import domain, experiment as exp\n", True),
        ("from vigiscoutx.signals.config import G2QualityConfig\n", False),
        ("from vigiscoutx import domain, signals\n", False),
        ("from .. import ports\n", False),
        ("from vigiscoutx.experiment_notes import x\n", False),
        ("from vigiscoutx.domain.types import evaluation_metrics\n", False),
    ],
)
def test_core_import_check_detects_real_violations(
    tmp_path: Path, source: str, detected: bool
) -> None:
    path = tmp_path / "module.py"
    path.write_text(source, encoding="utf-8")
    assert bool(_non_core_imports(path)) is detected


def test_experiment_package_uses_core_contracts() -> None:
    experiment_imports = set().union(
        *(_imported_modules(p) for p in (SRC / "experiment").rglob("*.py"))
    )
    assert "vigiscoutx.domain.types" in experiment_imports
    assert "vigiscoutx.ports" in experiment_imports
