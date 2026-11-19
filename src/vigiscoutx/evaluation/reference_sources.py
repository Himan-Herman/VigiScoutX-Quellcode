"""Nicht-zirkuläre strukturelle Ableitung der eingefrorenen Reference (DEC-084/085, Protokoll §2).

Gelesen werden ausschließlich die gehashten Freeze-Artefakte eines Calibration-Laufs
(Freeze-Record, FreezeManifestV2, Run-Konfiguration, ArtifactManifest, öffentliches
Pool-Manifest und Canonical-Mapping). Die Reference wird mit der unveränderten
`select_reference()`-Regel aus den `CAL_REF`-Poolblöcken, der kanonischen Sample-Zuordnung und dem
daraus folgenden Klassenvektor neu abgeleitet; die im Freeze-Record vorgegebenen Reference-IDs
dienen ausschließlich dem anschließenden Vergleich. Kein Bild-, Label- oder Metadatenzugriff.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from vigiscoutx.artifacts.manifests import ArtifactManifest, canonical_artifact_path
from vigiscoutx.config.scientific_config import scientific_config_sha256
from vigiscoutx.domain.hashing import canonical_json_bytes, canonical_sha256, sha256_hex
from vigiscoutx.domain.types import FrozenModel, Identifier, NonNegativeInt, Sha256Hex
from vigiscoutx.errors import IntegrityError
from vigiscoutx.experiment.calibration.config import CalibrationPool, baseline_protocol_config
from vigiscoutx.experiment.calibration.freeze_record import (
    CalibrationFreezeRecord,
    verify_record_against_manifest,
)
from vigiscoutx.experiment.calibration.structure import (
    CalibrationLesion,
    PoolClassBlock,
    WindowSelection,
    canonical_sample,
    class_vector,
    reference_sha256,
    select_reference,
)
from vigiscoutx.experiment.freeze import FreezeManifestV2, verify_freeze_manifest
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.split import SplitManifest
from vigiscoutx.experiment.split_integrity import split_manifest_sha256, validate_split_manifest
from vigiscoutx.experiment.types import Partition

RECORD_FILE: Final = "freeze/calibration_freeze_record.json"
MANIFEST_FILE: Final = "freeze/freeze_manifest_v2.json"
RUN_CONFIG_FILE: Final = "freeze/run_config.json"
ARTIFACT_MANIFEST_FILE: Final = "freeze/artifact_manifest.json"
POOL_MANIFEST_ROLE: Final = "calibration.pool_manifest"
CANONICAL_SAMPLES_ROLE: Final = "calibration.canonical_samples"


class ReferenceReconstructionStop(IntegrityError):
    """Terminaler STOP der Reference-Rekonstruktion; kein Wiederanlauf oder Ersatz."""


def _stop(condition: bool, message: str) -> None:
    if not condition:
        raise ReferenceReconstructionStop(message)


def _read_canonical(path: Path) -> tuple[bytes, Any]:
    """Liest eine reguläre Datei (kein Symlink), deren Inhalt kanonisches JSON sein muss."""
    _stop(not path.is_symlink() and path.is_file(), f"Freeze-Artefakt fehlt: {path.name}")
    data = path.read_bytes()
    value = json.loads(data)
    _stop(canonical_json_bytes(value) == data, f"Freeze-Artefakt nicht kanonisch: {path.name}")
    return data, value


@dataclass(frozen=True, slots=True)
class FrozenCalibration:
    """Gegeneinander geprüfte Freeze-Artefakte genau eines Calibration-Laufs."""

    record: CalibrationFreezeRecord
    manifest: FreezeManifestV2
    run_config: RunConfig
    artifact_manifest: ArtifactManifest
    pool_manifest_bytes: bytes
    canonical_samples_bytes: bytes


def _role_bytes(run_dir: Path, manifest: ArtifactManifest, role: str) -> bytes:
    entries = [e for e in manifest.entries if e.role == role]
    _stop(len(entries) == 1, f"ArtifactManifest benötigt genau einen Eintrag {role}")
    data, _ = _read_canonical(run_dir / canonical_artifact_path(entries[0].path))
    _stop(sha256_hex(data) == entries[0].sha256, f"{role}: Inhalt weicht vom ArtifactManifest ab")
    return data


def load_frozen_calibration(run_dir: Path) -> FrozenCalibration:
    """Lädt und verknüpft Record, FreezeManifestV2, Run-Konfiguration und ArtifactManifest."""
    _, record_value = _read_canonical(run_dir / RECORD_FILE)
    _, manifest_value = _read_canonical(run_dir / MANIFEST_FILE)
    _, config_value = _read_canonical(run_dir / RUN_CONFIG_FILE)
    artifact_bytes, artifact_value = _read_canonical(run_dir / ARTIFACT_MANIFEST_FILE)
    record = CalibrationFreezeRecord.model_validate(record_value)
    manifest = FreezeManifestV2.model_validate(manifest_value)
    verify_freeze_manifest(manifest)
    verify_record_against_manifest(record, manifest)
    run_config = RunConfig.model_validate(config_value)
    _stop(
        scientific_config_sha256(run_config.scientific) == manifest.config_sha256,
        "Run-Konfiguration weicht vom FreezeManifest ab",
    )
    artifact_manifest = ArtifactManifest.model_validate(artifact_value)
    _stop(
        sha256_hex(artifact_bytes) == record.artifact_manifest_sha256,
        "ArtifactManifest weicht vom Freeze-Record ab",
    )
    return FrozenCalibration(
        record=record,
        manifest=manifest,
        run_config=run_config,
        artifact_manifest=artifact_manifest,
        pool_manifest_bytes=_role_bytes(run_dir, artifact_manifest, POOL_MANIFEST_ROLE),
        canonical_samples_bytes=_role_bytes(run_dir, artifact_manifest, CANONICAL_SAMPLES_ROLE),
    )


class DerivedReference(FrozenModel):
    """Unabhängig abgeleitete Reference samt der Hashes ihrer strukturellen Quellen."""

    context: str
    window_id: Sha256Hex
    lesion_ids: tuple[Identifier, ...]
    sample_ids: tuple[Identifier, ...]
    reference_sha256: Sha256Hex
    class_counts: tuple[tuple[Identifier, NonNegativeInt], ...]
    split_manifest_sha256: Sha256Hex
    artifact_manifest_sha256: Sha256Hex
    pool_manifest_artifact_sha256: Sha256Hex
    cal_ref_blocks_sha256: Sha256Hex
    canonical_samples_sha256: Sha256Hex


def _cal_ref_blocks(
    pool_manifest: Mapping[str, Any], record: CalibrationFreezeRecord
) -> tuple[PoolClassBlock, ...]:
    config = baseline_protocol_config()
    _stop(
        pool_manifest.get("split_hash") == record.split_manifest_sha256,
        "Pool-Manifest gehört zu einem anderen Split",
    )
    _stop(
        pool_manifest.get("pool_manifest_sha256") == record.pool_manifest_sha256,
        "Pool-Manifest-Hash weicht vom Freeze-Record ab",
    )
    _stop(
        pool_manifest.get("pool_manifests")
        == [m.model_dump(mode="json") for m in record.pool_manifests],
        "Pool-Manifeste weichen vom Freeze-Record ab",
    )
    blocks = tuple(
        PoolClassBlock.model_validate(b)
        for b in pool_manifest.get("ref_tune_blocks", ())
        if isinstance(b, Mapping) and b.get("pool") == CalibrationPool.CAL_REF.value
    )
    _stop(
        tuple(b.class_id for b in blocks) == config.class_order,
        "CAL_REF benötigt genau einen Klassenblock je Diagnoseklasse",
    )
    ref = next(m for m in record.pool_manifests if m.pool is CalibrationPool.CAL_REF)
    _stop(
        canonical_sha256([b.model_dump(mode="json") for b in blocks]) == ref.sha256,
        "CAL_REF-Poolblöcke weichen vom eingefrorenen Pool-Hash ab",
    )
    _stop(
        sum(len(b.lesion_ids) for b in blocks) == ref.n_lesions,
        "CAL_REF-Läsionszahl weicht ab",
    )
    return blocks


def _canonical_mapping(data: bytes, record: CalibrationFreezeRecord) -> dict[str, str]:
    _stop(
        sha256_hex(data) == record.canonical_samples_sha256,
        "Canonical-Mapping weicht vom Freeze-Record ab",
    )
    pairs = json.loads(data)
    _stop(
        isinstance(pairs, list)
        and all(
            isinstance(p, list) and len(p) == 2 and all(isinstance(v, str) for v in p)
            for p in pairs
        ),
        "Canonical-Mapping muss eine Liste von (lesion_id, sample_id)-Paaren sein",
    )
    lesions = [p[0] for p in pairs]
    _stop(lesions == sorted(set(lesions)), "Canonical-Mapping: lesion_id eindeutig und sortiert")
    return {p[0]: p[1] for p in pairs}


def _verify_canonical_rule(
    blocks: Sequence[PoolClassBlock], canonical: Mapping[str, str], split: SplitManifest
) -> None:
    """Canonical Sample jeder CAL_REF-Läsion erneut aus dem Split nach der Hash-Regel."""
    config = baseline_protocol_config()
    split_hash = split_manifest_sha256(split)
    samples: dict[str, list[str]] = {}
    for a in split.assignments:
        if a.partition is Partition.CALIBRATION:
            samples.setdefault(a.group_id, []).append(a.sample_id)
    for block in blocks:
        for lesion_id in block.lesion_ids:
            _stop(lesion_id in samples, f"CAL_REF-Läsion {lesion_id} liegt nicht in CALIBRATION")
            lesion = CalibrationLesion(
                lesion_id=lesion_id,
                class_id=block.class_id,
                sample_ids=tuple(sorted(samples[lesion_id])),
            )
            _stop(
                canonical.get(lesion_id) == canonical_sample(lesion, split_hash, config),
                f"{lesion_id}: Canonical Sample folgt nicht der Hash-Regel",
            )


def derive_reference(frozen: FrozenCalibration, split_manifest: SplitManifest) -> DerivedReference:
    """Leitet die Reference unabhängig ab und vergleicht sie danach exakt mit dem Freeze-Record.

    STOP bei jeder Abweichung in IDs, Reihenfolge, `window_id`-Kontext, `window_id` oder
    `reference_sha256`. Der Vergleich belegt die Auswahlidentität, nicht die numerische Identität
    der Reference-Merkmale.
    """
    record = frozen.record
    config = baseline_protocol_config()
    validate_split_manifest(split_manifest)
    split_hash = split_manifest_sha256(split_manifest)
    _stop(split_hash == record.split_manifest_sha256, "Split-Manifest weicht vom Freeze ab")
    blocks = _cal_ref_blocks(json.loads(frozen.pool_manifest_bytes), record)
    canonical = _canonical_mapping(frozen.canonical_samples_bytes, record)
    _verify_canonical_rule(blocks, canonical, split_manifest)
    by_class = {b.class_id: b.lesion_ids for b in blocks}
    vector = class_vector(
        record.window_size, {c: len(by_class[c]) for c in config.class_order}, config
    )
    selection: WindowSelection = select_reference(by_class, vector, canonical, split_hash, config)
    checks = {
        "context": (selection.context, f"{config.reference_namespace}{record.window_size}"),
        "lesion_ids": (selection.lesion_ids, record.reference_lesion_ids),
        "sample_ids": (selection.sample_ids, record.reference_sample_ids),
        "window_id": (selection.window_id, record.reference_window_id),
        "reference_sha256": (reference_sha256(selection), record.reference_sha256),
    }
    mismatches = sorted(k for k, (have, want) in checks.items() if have != want)
    _stop(not mismatches, f"abgeleitete Reference weicht vom Freeze-Record ab: {mismatches}")
    return DerivedReference(
        context=selection.context,
        window_id=selection.window_id,
        lesion_ids=selection.lesion_ids,
        sample_ids=selection.sample_ids,
        reference_sha256=reference_sha256(selection),
        class_counts=tuple(vector.as_dict().items()),
        split_manifest_sha256=split_hash,
        artifact_manifest_sha256=record.artifact_manifest_sha256,
        pool_manifest_artifact_sha256=sha256_hex(frozen.pool_manifest_bytes),
        cal_ref_blocks_sha256=canonical_sha256([b.model_dump(mode="json") for b in blocks]),
        canonical_samples_sha256=record.canonical_samples_sha256,
    )
