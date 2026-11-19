"""Reproduzierbare Split-Erzeugung auf lesion_id-Ebene (DEC-017).

- Die Leakage-Gruppe (`lesion_id`) ist unteilbar; die Gruppierung hat Vorrang vor perfekter
  Stratifikation.
- Stratifiziert wird je Klasse über Läsionen (Klasse einer Läsion = häufigstes Label ihrer
  Bilder); die Anteile 60/20/20 beziehen sich damit auf Läsionen je Klasse.
- Alle Klassen müssen in jeder Hauptpartition sowie in TRAIN und MODEL_VALIDATION vorkommen,
  sonst schlägt die Erzeugung fehl.
- Die Trennung gilt auf Läsionsebene, nicht auf Patientenebene.
"""

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Protocol

from pydantic import Field, StrictFloat

from vigiscoutx.domain.hashing import canonical_json_bytes
from vigiscoutx.domain.refs import SampleRef, SplitManifestRef
from vigiscoutx.domain.types import FrozenModel, Identifier, NonNegativeInt, Sha256Hex
from vigiscoutx.errors import IntegrityError, SplitGenerationError
from vigiscoutx.experiment.held_out_guard import HeldOutGuard
from vigiscoutx.experiment.split import SplitAssignment, SplitManifest
from vigiscoutx.experiment.split_integrity import split_manifest_sha256, validate_split_manifest
from vigiscoutx.experiment.types import ModelDevelopmentRole, Partition
from vigiscoutx.ports import DataAccessContext

MAIN_PARTITION_FRACTIONS: tuple[tuple[Partition, Fraction], ...] = (
    (Partition.MODEL_DEVELOPMENT, Fraction(3, 5)),
    (Partition.CALIBRATION, Fraction(1, 5)),
    (Partition.HELD_OUT, Fraction(1, 5)),
)
"""Eingefroren durch DEC-017 (60 % / 20 % / 20 %); nicht konfigurierbar."""


class SplitGenerationConfig(FrozenModel):
    manifest_id: Identifier
    split_seed: NonNegativeInt
    train_fraction: Annotated[StrictFloat, Field(gt=0.0, lt=1.0)]
    """Anteil TRAIN innerhalb MODEL_DEVELOPMENT; Pflichtparameter ohne Default."""


class LabeledSample(FrozenModel):
    sample_id: Identifier
    group_id: Identifier
    label: Identifier


def _apportion(total: int, fractions: Sequence[Fraction]) -> list[int]:
    """Largest-Remainder-Verteilung; Gleichstände werden in fester Reihenfolge aufgelöst."""
    raw = [total * f for f in fractions]
    counts = [int(r) for r in raw]
    remainders = sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))
    for index in remainders[: total - sum(counts)]:
        counts[index] += 1
    return counts


def _lesion_labels(
    samples: Sequence[LabeledSample], class_order: tuple[str, ...]
) -> dict[str, str]:
    labels_by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        if sample.label not in class_order:
            raise SplitGenerationError(f"{sample.sample_id}: unbekanntes Label {sample.label}")
        labels_by_group[sample.group_id][sample.label] += 1
    return {
        group: min(counts, key=lambda label: (-counts[label], class_order.index(label)))
        for group, counts in labels_by_group.items()
    }


def generate_split(
    samples: Sequence[LabeledSample],
    config: SplitGenerationConfig,
    class_order: tuple[str, ...],
) -> SplitManifest:
    ids = [s.sample_id for s in samples]
    if len(ids) != len(set(ids)):
        raise SplitGenerationError("sample_id muss eindeutig sein")
    lesion_label = _lesion_labels(samples, class_order)
    rng = random.Random(config.split_seed)
    train_fraction = Fraction(config.train_fraction).limit_denominator(10**6)

    role_by_group: dict[str, tuple[Partition, ModelDevelopmentRole | None]] = {}
    for label in class_order:
        groups = sorted(g for g, lab in lesion_label.items() if lab == label)
        rng.shuffle(groups)
        sizes = _apportion(len(groups), [f for _, f in MAIN_PARTITION_FRACTIONS])
        start = 0
        for (partition, _), size in zip(MAIN_PARTITION_FRACTIONS, sizes, strict=True):
            chunk = groups[start : start + size]
            start += size
            if partition is Partition.MODEL_DEVELOPMENT:
                n_train, _ = _apportion(len(chunk), [train_fraction, 1 - train_fraction])
                for position, group in enumerate(chunk):
                    role = (
                        ModelDevelopmentRole.TRAIN
                        if position < n_train
                        else ModelDevelopmentRole.MODEL_VALIDATION
                    )
                    role_by_group[group] = (partition, role)
            else:
                for group in chunk:
                    role_by_group[group] = (partition, None)

    manifest = SplitManifest(
        manifest_id=config.manifest_id,
        assignments=tuple(
            SplitAssignment(
                sample_id=s.sample_id,
                group_id=s.group_id,
                partition=role_by_group[s.group_id][0],
                model_development_role=role_by_group[s.group_id][1],
            )
            for s in sorted(samples, key=lambda s: s.sample_id)
        ),
    )
    validate_split_manifest(manifest)
    require_all_classes(manifest, {s.sample_id: s.label for s in samples}, class_order)
    return manifest


class StratificationLabelSource(Protocol):
    def sample_refs(self) -> tuple[SampleRef, ...]: ...

    def stratification_labels_for(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> Mapping[str, int]: ...


def generate_split_from_source(
    source: StratificationLabelSource,
    guard: HeldOutGuard,
    config: SplitGenerationConfig,
    class_order: tuple[str, ...],
) -> SplitManifest:
    """Liest Labels einmalig über eine Split-Erzeugungsberechtigung und verbraucht sie danach.

    Die Berechtigung wird auch bei einem Fehler verbraucht; ein zweiter Label-Vollzugriff über
    denselben Guard ist damit ausgeschlossen.
    """
    refs = source.sample_refs()
    ids = [r.sample_id for r in refs]
    permit = guard.issue_split_generation_permit(ids)
    try:
        labels = source.stratification_labels_for(permit, ids)
        samples = [
            LabeledSample(
                sample_id=r.sample_id, group_id=r.group_id, label=class_order[labels[r.sample_id]]
            )
            for r in refs
        ]
        return generate_split(samples, config, class_order)
    finally:
        permit.consume()


def require_all_classes(
    manifest: SplitManifest, labels: Mapping[str, str], class_order: tuple[str, ...]
) -> None:
    """Jede Klasse in jeder Hauptpartition sowie in TRAIN und MODEL_VALIDATION."""
    present: dict[str, set[str]] = defaultdict(set)
    for a in manifest.assignments:
        present[a.partition.value].add(labels[a.sample_id])
        if a.model_development_role is not None:
            present[a.model_development_role.value].add(labels[a.sample_id])
    scopes = [p.value for p, _ in MAIN_PARTITION_FRACTIONS] + [
        r.value for r in ModelDevelopmentRole
    ]
    missing = {
        scope: sorted(set(class_order) - present[scope])
        for scope in scopes
        if set(class_order) - present[scope]
    }
    if missing:
        raise SplitGenerationError(f"Klassen fehlen in Partitionen/Rollen: {missing}")


def split_manifest_ref(manifest: SplitManifest) -> SplitManifestRef:
    return SplitManifestRef(
        manifest_id=manifest.manifest_id,
        uri=f"local:{manifest.manifest_id}",
        sha256=split_manifest_sha256(manifest),
    )


def held_out_sample_ids(manifest: SplitManifest) -> tuple[str, ...]:
    return tuple(
        sorted(a.sample_id for a in manifest.assignments if a.partition is Partition.HELD_OUT)
    )


class HeldOutReference(FrozenModel):
    """Unveränderliche Referenz auf die Held-out-IDs eines Split-Manifests."""

    split_manifest_sha256: Sha256Hex
    sample_ids: tuple[Identifier, ...]


def held_out_reference(manifest: SplitManifest) -> HeldOutReference:
    return HeldOutReference(
        split_manifest_sha256=split_manifest_sha256(manifest),
        sample_ids=held_out_sample_ids(manifest),
    )


def write_split_manifest(manifest: SplitManifest, path: Path) -> SplitManifestRef:
    """Schreibt das Manifest kanonisch; eine vorhandene Datei wird nie überschrieben."""
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(manifest))
    return split_manifest_ref(manifest)


def read_split_manifest(path: Path, expected: SplitManifestRef) -> SplitManifest:
    manifest = SplitManifest.model_validate_json(path.read_bytes())
    if split_manifest_sha256(manifest) != expected.sha256:
        raise IntegrityError("Split-Manifest weicht von der erwarteten Referenz ab")
    return manifest
