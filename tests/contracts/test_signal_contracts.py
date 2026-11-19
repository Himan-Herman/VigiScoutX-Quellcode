"""Signal-Contracts, Methodenkonfiguration und Primary-/Label-aware-Grenze (P3-03A)."""

import ast
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import vigiscoutx
from signal_factories import (
    g1_config,
    g2_config,
    g3_config,
    g4_config,
    group_assessment,
    group_data,
    signal_data,
    signal_result,
)
from vigiscoutx.config.scientific_config import SignalConfig
from vigiscoutx.domain.assessments import (
    ComponentResult,
    G3Substatus,
    ReasonCode,
    SignalStatus,
)
from vigiscoutx.domain.frozen_json import FrozenMapping
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.types import SignalGroup
from vigiscoutx.signals.config import (
    METHOD_VERSION,
    G1MmdConfig,
    G2QualityConfig,
    G3PredictiveConfig,
    method_config_from_signal_config,
)
from vigiscoutx.signals.reference import ReferenceCacheKey, reference_sample_ids_sha256

SRC = Path(vigiscoutx.__file__).parent
H = "0" * 64


def test_signal_status_vocabulary_is_exact() -> None:
    assert [s.value for s in SignalStatus] == ["VALID", "NOT_EVALUABLE", "ERROR"]
    assert [s.value for s in G3Substatus] == ["NONE", "OUTPUT_ONLY", "CONFIDENCE_ONLY", "BOTH"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "VALID", "alarm": None},
        {"status": "VALID", "alarm": False, "reason_code": "KS_EXACT_FALLBACK"},
        {"status": "ERROR", "alarm": False, "reason_code": "KS_EXACT_FALLBACK"},
        {"status": "ERROR", "alarm": None, "reason_code": None},
        {"status": "NOT_EVALUABLE", "alarm": True, "reason_code": "INSUFFICIENT_SAMPLE"},
        {"status": "NOT_EVALUABLE", "alarm": None, "reason_code": None},
        {"reason_code": "free text"},
        {"adjusted_p_value": 0.2, "p_value": None},
    ],
)
def test_status_alarm_consistency_is_enforced(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        signal_result(**overrides)


def test_error_and_not_evaluable_are_distinct_valid_contracts() -> None:
    error = signal_result(status="ERROR", alarm=None, reason_code="KS_EXACT_FALLBACK")
    missing = signal_result(status="NOT_EVALUABLE", alarm=None, reason_code="INSUFFICIENT_SAMPLE")
    assert error.status is SignalStatus.ERROR and error.alarm is None
    assert missing.status is SignalStatus.NOT_EVALUABLE
    assert error != missing
    with pytest.raises(ValidationError):
        error.model_copy(update={"alarm": False})


def test_signal_result_is_deeply_immutable() -> None:
    component = ComponentResult(
        component_id="MEL",
        status=SignalStatus.VALID,
        statistics=FrozenMapping({"ks_d": 0.1, "nested": {"values": [1, 2]}}),
        p_value=0.3,
        adjusted_p_value=1.0,
        alarm=False,
        reason_code=None,
    )
    result = signal_result(components=(component,))
    with pytest.raises(ValidationError):
        result.alarm = True  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.statistics["d"] = 1.0  # type: ignore[index]
    nested = result.components[0].statistics["nested"]
    assert isinstance(nested, FrozenMapping)
    assert isinstance(nested["values"], tuple)
    with pytest.raises(ValidationError):
        result.components[0].alarm = True  # type: ignore[misc]


def test_signal_result_hash_is_deterministic() -> None:
    assert canonical_sha256(signal_result()) == canonical_sha256(signal_result())
    assert canonical_sha256(signal_result()) != canonical_sha256(signal_result(p_value=0.4))


def test_components_must_be_unique() -> None:
    component = ComponentResult(
        component_id="MEL",
        status=SignalStatus.ERROR,
        statistics=FrozenMapping({}),
        p_value=None,
        adjusted_p_value=None,
        alarm=None,
        reason_code=ReasonCode.KS_EXACT_FALLBACK,
    )
    with pytest.raises(ValidationError):
        signal_result(components=(component, component))


@pytest.mark.parametrize(
    "overrides",
    [
        {"group": "G1", "substatus": "NONE", "alarm": False},
        {"group": "G3", "substatus": None},
        {"group": "G3", "substatus": "NONE", "alarm": True},
        {"group": "G3", "substatus": "BOTH", "alarm": False},
        {
            "group": "G3",
            "status": "ERROR",
            "alarm": None,
            "reason_code": "SUBSIGNAL_ERROR",
            "substatus": "OUTPUT_ONLY",
        },
    ],
)
def test_group_substatus_rules(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        group_assessment(**overrides)


def test_valid_g3_group_with_substatus() -> None:
    group = group_assessment(group="G3", substatus="OUTPUT_ONLY", alarm=True)
    assert group.substatus is G3Substatus.OUTPUT_ONLY


def test_performance_evidence_rejected_in_primary_signal_contracts() -> None:
    perf = [{"kind": "performance_assessment", "ref_id": "p"}]
    with pytest.raises(ValidationError):
        signal_result(evidence_refs=perf)
    with pytest.raises(ValidationError):
        group_assessment(**{**group_data(), "evidence_refs": perf})
    with pytest.raises(ValidationError):
        signal_result(**{**signal_data(), "group": "G4"})


# --- Methodenkonfiguration ---------------------------------------------------------------------


def test_method_version_constant_matches_configs() -> None:
    assert METHOD_VERSION == "monitoring-methods-baseline:1.2"
    for config in (g1_config(), g2_config(), g3_config(), g4_config()):
        assert config.method_version == METHOD_VERSION


@pytest.mark.parametrize(
    ("factory", "update"),
    [
        (g1_config, {"alpha": 0.01}),
        (g1_config, {"kernel": "LAPLACE"}),
        (g2_config, {"alpha": 0.1}),
        (g2_config, {"luma_coefficients": (0.299, 0.587, 0.114)}),
        (g2_config, {"laplacian_kernel": ((1, 1, 1), (1, -8, 1), (1, 1, 1))}),
        (g2_config, {"immerkaer_kernel": ((1, -2, 1), (-2, 4, -2), (1, -2, 2))}),
        (g2_config, {"bonferroni_factor": 5}),
        (g2_config, {"indicators": ("mean_luminance",)}),
        (g3_config, {"output_bonferroni_factor": 6}),
        (g3_config, {"class_order": ("NV", "MEL", "BCC", "AKIEC", "BKL", "DF", "VASC")}),
        (g3_config, {"alpha": 0.049}),
        (g4_config, {"zero_division": "ZERO"}),
    ],
)
def test_frozen_method_values_cannot_be_changed(factory: Any, update: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        factory().model_copy(update=update)


def test_ks_spec_forbids_auto_and_asymptotic() -> None:
    for method in ("auto", "asymp"):
        with pytest.raises(ValidationError):
            g2_config().ks.model_copy(update={"method": method})


def test_config_hash_identifies_parameters() -> None:
    assert g1_config(2.0).parameters_ref() == g1_config(2.0).parameters_ref()
    assert g1_config(2.0).parameters_ref().sha256 != g1_config(3.0).parameters_ref().sha256
    assert g2_config().parameters_ref().config_id == "G2"


def test_method_config_from_scientific_signal_config() -> None:
    config = g1_config(4.0)
    signal = SignalConfig(
        signal_id="G1_MMD_EMBEDDING",
        group=SignalGroup.G1_INPUT,
        method="MMD_U2_RBF_PERMUTATION",
        params=FrozenMapping(config.model_dump(mode="json")),
        threshold=None,
    )
    assert method_config_from_signal_config(signal, G1MmdConfig) == config
    with pytest.raises(ValueError):
        method_config_from_signal_config(signal, G3PredictiveConfig)
    g2 = SignalConfig(
        signal_id="G2",
        group=SignalGroup.G2_DATA_QUALITY,
        method="G2_QUALITY_KS_BONFERRONI",
        params=FrozenMapping(g2_config().model_dump(mode="json")),
        threshold=None,
    )
    assert method_config_from_signal_config(g2, G2QualityConfig) == g2_config()


# --- Reference-Cache-Key -----------------------------------------------------------------------


def _key(**overrides: Any) -> ReferenceCacheKey:
    data: dict[str, Any] = {
        "group": "G1",
        "dataset_snapshot_sha256": H,
        "split_manifest_sha256": H,
        "reference_sample_ids_sha256": reference_sample_ids_sha256(["b", "a"]),
        "preprocessing_sha256": H,
        "model_weights_sha256": "a" * 64,
        "signal_config_sha256": H,
    }
    data.update(overrides)
    return ReferenceCacheKey.model_validate(data)


def test_reference_cache_key_depends_on_every_component() -> None:
    base = _key()
    assert reference_sample_ids_sha256(["a", "b"]) == reference_sample_ids_sha256(["b", "a"])
    for field in ("dataset_snapshot_sha256", "preprocessing_sha256", "signal_config_sha256"):
        assert _key(**{field: "f" * 64}).key_sha256() != base.key_sha256()
    assert _key(model_weights_sha256="b" * 64).key_sha256() != base.key_sha256()
    with pytest.raises(ValueError):
        reference_sample_ids_sha256(["a", "a"])


def test_model_weights_relevance_per_group() -> None:
    assert _key(group="G2", model_weights_sha256=None)
    with pytest.raises(ValidationError):
        _key(group="G2")
    with pytest.raises(ValidationError):
        _key(group="G3", model_weights_sha256=None)


# --- Primary-/Label-aware-Grenze ---------------------------------------------------------------


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


PRIMARY_FILES = [
    *(SRC / "signals" / "input").rglob("*.py"),
    *(SRC / "signals" / "data_quality").rglob("*.py"),
    *(SRC / "signals" / "predictive").rglob("*.py"),
    *(SRC / "grouping").rglob("*.py"),
    SRC / "signals" / "statistics.py",
    SRC / "signals" / "config.py",
    SRC / "signals" / "reference.py",
]


def test_primary_path_never_imports_labels_or_g4() -> None:
    assert len(PRIMARY_FILES) >= 8
    forbidden = ("signals.performance", "LabelPort", "PerformanceAssessment", "experiment")
    violations = {
        str(p.relative_to(SRC)): sorted(m for m in _imports(p) if any(f in m for f in forbidden))
        for p in PRIMARY_FILES
    }
    assert {k: v for k, v in violations.items() if v} == {}


def test_primary_signal_functions_take_no_labels() -> None:
    for path in PRIMARY_FILES:
        text = path.read_text(encoding="utf-8")
        assert "y_true" not in text and "labels_for" not in text, path


def test_g4_stays_label_aware_and_outside_primary_contracts() -> None:
    g4_imports = _imports(SRC / "signals" / "performance" / "g4.py")
    assert "vigiscoutx.domain.assessments.SignalResult" not in g4_imports
    assert "vigiscoutx.domain.assessments.PerformanceAssessment" in g4_imports


def test_signal_code_needs_no_partition_access() -> None:
    signal_files = [*(SRC / "signals").rglob("*.py"), *(SRC / "grouping").rglob("*.py")]
    forbidden = {"HeldOutGuard", "Partition", "PartitionAccessPermit", "DataAccessContext"}
    for path in signal_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not names & forbidden, path
        assert not any("experiment" in m for m in _imports(path)), path
