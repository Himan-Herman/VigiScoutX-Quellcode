"""FreezeManifest-Versionierung: v1 historisch unverändert, v2 für den P4-Calibration-Freeze."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from policy_fixtures import with_frozen_policies
from vigiscoutx.domain.refs import SeedBundle
from vigiscoutx.experiment.freeze import (
    FreezeManifest,
    FreezeManifestV2,
    build_freeze_manifest_v2,
    compute_freeze_id,
    parse_freeze_manifest,
    verify_freeze_manifest,
)
from vigiscoutx.experiment.run_config import RunConfig

HISTORICAL_V1_FREEZE_ID = "47a63a5e5ebb5ab227d917c40253b7ad630122c2176533b86a4287c616076fa1"
"""Referenz-freeze_id des conftest-Manifests (Stand 9427ce0), vor Einführung von Schema 2."""


def test_historical_v1_payload_parses_and_reproduces_hash(freeze_manifest: FreezeManifest) -> None:
    payload = json.loads(freeze_manifest.model_dump_json())
    assert "calibration_record_sha256" not in payload
    assert payload["schema_version"] == 1
    parsed = parse_freeze_manifest(payload)
    assert isinstance(parsed, FreezeManifest)
    assert parsed.freeze_id == HISTORICAL_V1_FREEZE_ID == compute_freeze_id(parsed)
    verify_freeze_manifest(parsed)


def test_v1_rejects_calibration_binding(freeze_manifest: FreezeManifest) -> None:
    payload = {
        **json.loads(freeze_manifest.model_dump_json()),
        "calibration_record_sha256": "c" * 64,
    }
    with pytest.raises(ValidationError):
        parse_freeze_manifest(payload)


def _v2(
    config: RunConfig, seeds: tuple[SeedBundle, ...], record: str = "c" * 64
) -> FreezeManifestV2:
    frozen = RunConfig.model_validate(with_frozen_policies(config.model_dump(mode="json")))
    return build_freeze_manifest_v2(
        frozen,
        seed_list=seeds,
        git_commit="a" * 40,
        environment_lock_sha256="e" * 64,
        calibration_record_sha256=record,
    )


def test_v2_requires_binding_and_hashes_deterministically(
    calibrated_config: RunConfig, seed_list: tuple[SeedBundle, ...]
) -> None:
    manifest = _v2(calibrated_config, seed_list)
    assert manifest.schema_version == 2
    assert manifest == _v2(calibrated_config, seed_list)
    assert manifest.freeze_id != HISTORICAL_V1_FREEZE_ID
    assert manifest.freeze_id != _v2(calibrated_config, seed_list, "d" * 64).freeze_id
    verify_freeze_manifest(manifest)
    payload: dict[str, Any] = json.loads(manifest.model_dump_json())
    assert isinstance(parse_freeze_manifest(payload), FreezeManifestV2)
    payload.pop("calibration_record_sha256")
    with pytest.raises(ValidationError):
        parse_freeze_manifest(payload)


def test_unknown_schema_version_rejected(freeze_manifest: FreezeManifest) -> None:
    payload = {**json.loads(freeze_manifest.model_dump_json()), "schema_version": 3}
    with pytest.raises(ValueError, match="schema_version"):
        parse_freeze_manifest(payload)
