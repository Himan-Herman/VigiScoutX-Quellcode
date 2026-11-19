import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from vigiscoutx.config.loader import (
    dump_resolved_snapshot,
    parse_yaml,
    resolved_snapshot,
    write_resolved_snapshot,
)
from vigiscoutx.config.scientific_config import calibration_gaps, scientific_config_sha256
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.experiment.run_config import RunConfig, load_run_config, parse_run_config


def _copy(data: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(data)


def test_yaml_round_trip_to_resolved_snapshot(
    tmp_path: Path, calibration_config_data: dict[str, Any]
) -> None:
    source = tmp_path / "run.yaml"
    source.write_text(yaml.safe_dump(calibration_config_data), encoding="utf-8")

    config = load_run_config(source)
    snapshot = resolved_snapshot(config)

    assert snapshot == calibration_config_data
    assert parse_run_config(dump_resolved_snapshot(config)) == config

    target = tmp_path / "resolved.yaml"
    write_resolved_snapshot(config, target)
    assert load_run_config(target) == config


def test_resolved_snapshot_is_never_overwritten(
    tmp_path: Path, calibration_config_data: dict[str, Any]
) -> None:
    config = RunConfig.model_validate(calibration_config_data)
    target = tmp_path / "resolved.yaml"
    write_resolved_snapshot(config, target)
    with pytest.raises(FileExistsError):
        write_resolved_snapshot(config, target)


@pytest.mark.parametrize(
    "path",
    [
        ("scientific", "monitoring", "window_size"),
        ("scientific", "relevance", "delta_rel"),
        ("scientific", "relevance", "clean_quantile"),
        ("scientific", "fusion", "rule"),
        ("scientific", "preprocessing"),
        ("scientific", "split"),
        ("repetitions",),
        ("seeds",),
        ("evaluation_metrics",),
    ],
)
def test_missing_scientific_parameter_is_rejected(
    calibration_config_data: dict[str, Any], path: tuple[str, ...]
) -> None:
    data = _copy(calibration_config_data)
    parent = data
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_missing_signal_threshold_key_is_rejected(calibration_config_data: dict[str, Any]) -> None:
    data = _copy(calibration_config_data)
    del data["scientific"]["monitoring"]["signals"][0]["threshold"]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_open_calibration_parameters_are_reported(calibration_config_data: dict[str, Any]) -> None:
    gaps = calibration_gaps(RunConfig.model_validate(calibration_config_data).scientific)
    assert gaps == [
        "monitoring.signals[g1-test].threshold",
        "monitoring.signals[g3-test].threshold",
        "fusion.rule",
        "relevance.delta_rel",
    ]


def test_held_out_config_requires_all_calibration_values(
    calibration_config_data: dict[str, Any],
) -> None:
    data = {**calibration_config_data, "phase": "HELD_OUT_EVALUATION", "freeze_id": "f" * 64}
    with pytest.raises(ValidationError, match="HELD_OUT_EVALUATION erfordert"):
        RunConfig.model_validate(data)


def test_development_config_must_not_carry_freeze_id(
    calibration_config_data: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**calibration_config_data, "freeze_id": "f" * 64})


def test_frozen_is_not_a_run_phase(calibration_config_data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**calibration_config_data, "phase": "FROZEN"})


def test_fusion_and_signals_reject_g4(calibration_config_data: dict[str, Any]) -> None:
    data = _copy(calibration_config_data)
    data["scientific"]["fusion"] = {"rule": None, "groups": ["G1", "G4"]}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)

    data = _copy(calibration_config_data)
    data["scientific"]["monitoring"]["signals"][0]["group"] = "G4"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_unknown_config_keys_are_rejected(calibration_config_data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**calibration_config_data, "shift_severity": 3})


def test_scientific_hash_ignores_run_identity_but_not_content(
    calibration_config_data: dict[str, Any],
) -> None:
    base = RunConfig.model_validate(calibration_config_data)
    other_run = RunConfig.model_validate(
        {
            **calibration_config_data,
            "run_id": "run-dev-002",
            "seeds": {**calibration_config_data["seeds"], "repeat_id": 7},
        }
    )
    changed_data = _copy(calibration_config_data)
    changed_data["scientific"]["monitoring"]["window_size"] = 5
    changed = RunConfig.model_validate(changed_data)

    assert scientific_config_sha256(base.scientific) == scientific_config_sha256(
        other_run.scientific
    )
    assert canonical_sha256(base) != canonical_sha256(other_run)
    assert scientific_config_sha256(base.scientific) != scientific_config_sha256(changed.scientific)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("scientific", "monitoring", "window_size"), "4"),
        (("scientific", "monitoring", "window_size"), 4.0),
        (("scientific", "monitoring", "window_size"), True),
        (("scientific", "relevance", "clean_quantile"), "0.5"),
        (("scientific", "relevance", "clean_quantile"), 0.0),
        (("scientific", "relevance", "clean_quantile"), 1.0),
        (("scientific", "preprocessing", "input_height"), "8"),
        (("repetitions",), "2"),
        (("seeds", "split_seed"), "1"),
    ],
)
def test_strict_scalars_reject_silent_coercion(
    calibration_config_data: dict[str, Any], path: tuple[str, ...], value: object
) -> None:
    data = _copy(calibration_config_data)
    parent = data
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


@pytest.mark.parametrize("value", ["3.5", "5e-2", "yes", "off", "true", "null", "nan", " 12 "])
def test_params_reject_ambiguous_strings(
    calibration_config_data: dict[str, Any], value: str
) -> None:
    data = _copy(calibration_config_data)
    data["scientific"]["monitoring"]["signals"][1]["params"] = {"x": {"y": [value]}}
    with pytest.raises(ValidationError, match="typisiert"):
        RunConfig.model_validate(data)


def test_yaml_loader_uses_strict_scalar_rules() -> None:
    parsed = parse_yaml("a: 5e-2\nb: yes\nc: 010\nd: 1:30\ne: true\nf: 1.0e-5\ng: 7\n")
    assert parsed == {
        "a": 0.05,
        "b": "yes",
        "c": "010",
        "d": "1:30",
        "e": True,
        "f": 1.0e-5,
        "g": 7,
    }


def test_yaml_yes_is_not_accepted_as_bool(calibration_config_data: dict[str, Any]) -> None:
    text = yaml.safe_dump(calibration_config_data).replace("repetitions: 2", "repetitions: yes")
    assert "repetitions: yes" in text
    with pytest.raises(ValidationError):
        parse_run_config(text)


def test_yaml_exponent_param_loads_as_float(calibration_config_data: dict[str, Any]) -> None:
    data = _copy(calibration_config_data)
    data["scientific"]["monitoring"]["signals"][1]["params"] = {"alpha": 0.05}
    text = yaml.safe_dump(data).replace("alpha: 0.05", "alpha: 5e-2")
    assert "alpha: 5e-2" in text
    config = parse_run_config(text)
    assert config.scientific.monitoring.signals[1].params["alpha"] == 0.05
