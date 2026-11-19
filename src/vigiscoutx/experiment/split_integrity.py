"""SplitIntegrityCheck (ARC-017): reine Validierungslogik ohne Datensatzbezug.

`group_id` ist die Leakage-Gruppe; im Demonstrator entspricht sie der `lesion_id`
(DEC-013, DEC-017). Die Prüfung garantiert Trennung auf Läsionsebene, nicht auf Patientenebene.
"""

from collections import defaultdict
from collections.abc import Sequence

from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.refs import SampleRef, SplitManifestRef
from vigiscoutx.domain.windows import MonitoringWindow, ReferenceWindow
from vigiscoutx.errors import IntegrityError, LeakageError
from vigiscoutx.experiment.split import SplitAssignment, SplitManifest
from vigiscoutx.experiment.types import ModelDevelopmentRole, Partition

RoleKey = tuple[Partition, ModelDevelopmentRole | None]


def split_manifest_sha256(manifest: SplitManifest) -> str:
    return canonical_sha256(manifest)


def verify_split_manifest_hash(manifest: SplitManifest, ref: SplitManifestRef) -> None:
    actual = split_manifest_sha256(manifest)
    if actual != ref.sha256:
        raise IntegrityError(f"Split-Manifest verändert: erwartet {ref.sha256}, ist {actual}")


def _role_key(assignment: SplitAssignment) -> RoleKey:
    return (assignment.partition, assignment.model_development_role)


def validate_split_manifest(manifest: SplitManifest) -> None:
    """Jede sample_id genau einmal; jede Leakage-Gruppe in genau einer Rolle.

    Rollen sind TRAIN, MODEL_VALIDATION (beide in MODEL_DEVELOPMENT), CALIBRATION und HELD_OUT.
    """
    seen: set[str] = set()
    conflicting_samples: set[str] = set()
    for assignment in manifest.assignments:
        if assignment.sample_id in seen:
            conflicting_samples.add(assignment.sample_id)
        seen.add(assignment.sample_id)
    if conflicting_samples:
        raise LeakageError(f"sample_id in mehreren Rollen: {sorted(conflicting_samples)}")

    roles_by_group: dict[str, set[RoleKey]] = defaultdict(set)
    for assignment in manifest.assignments:
        roles_by_group[assignment.group_id].add(_role_key(assignment))
    leaking = sorted(g for g, roles in roles_by_group.items() if len(roles) > 1)
    if leaking:
        raise LeakageError(f"group_id (lesion_id) rollen-/partitionsübergreifend: {leaking}")


def partition_sample_ids(manifest: SplitManifest, partition: Partition) -> frozenset[str]:
    return frozenset(a.sample_id for a in manifest.assignments if a.partition is partition)


def _check_membership(
    samples: tuple[SampleRef, ...],
    index: dict[str, SplitAssignment],
    partition: Partition,
    role: str,
) -> None:
    for sample in samples:
        assignment = index.get(sample.sample_id)
        if assignment is None:
            raise LeakageError(f"{role}: sample_id {sample.sample_id} fehlt im Split-Manifest")
        if assignment.partition is not partition:
            raise LeakageError(
                f"{role}: sample_id {sample.sample_id} gehört zu {assignment.partition.value}, "
                f"nicht zu {partition.value}"
            )
        if assignment.group_id != sample.group_id:
            raise LeakageError(f"{role}: group_id von {sample.sample_id} weicht vom Manifest ab")


def validate_run_windows(
    manifest: SplitManifest,
    partition: Partition,
    reference: ReferenceWindow,
    monitoring: MonitoringWindow,
) -> None:
    """Reference und Monitoring stammen aus derselben Partition und sind gruppen-disjunkt."""
    validate_split_manifest(manifest)
    index = {a.sample_id: a for a in manifest.assignments}
    _check_membership(reference.samples, index, partition, "Reference")
    _check_membership(monitoring.samples, index, partition, "Monitoring")

    shared_samples = {s.sample_id for s in reference.samples} & {
        s.sample_id for s in monitoring.samples
    }
    if shared_samples:
        raise LeakageError(f"sample_id in Reference und Monitoring: {sorted(shared_samples)}")
    reference_groups = {s.group_id for s in reference.samples}
    shared_groups = reference_groups & {s.group_id for s in monitoring.samples}
    if shared_groups:
        raise LeakageError(f"group_id in Reference und Monitoring: {sorted(shared_groups)}")


def validate_split_matches_samples(manifest: SplitManifest, samples: Sequence[SampleRef]) -> None:
    """Ein vorhandenes Split-Manifest muss exakt die Samples des Datensatzes abdecken.

    Prüft Vollständigkeit, unbekannte sample_ids und die Übereinstimmung der group_id
    (lesion_id). Es wird kein Split erzeugt.
    """
    validate_split_manifest(manifest)
    dataset = {s.sample_id: s.group_id for s in samples}
    assigned = {a.sample_id: a.group_id for a in manifest.assignments}
    unknown = sorted(set(assigned) - set(dataset))
    if unknown:
        raise LeakageError(f"Split-Manifest enthält unbekannte sample_ids: {unknown}")
    unassigned = sorted(set(dataset) - set(assigned))
    if unassigned:
        raise LeakageError(f"Samples ohne Split-Zuordnung: {unassigned}")
    mismatched = sorted(s for s, g in assigned.items() if dataset[s] != g)
    if mismatched:
        raise LeakageError(f"group_id weicht von der lesion_id des Datensatzes ab: {mismatched}")
