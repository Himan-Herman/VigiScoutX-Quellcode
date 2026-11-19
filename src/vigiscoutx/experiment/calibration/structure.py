"""Strukturelle, rein ID-/Hash-basierte Protokolllogik (DEC-050, DEC-051, DEC-052, DEC-057).

Keine Funktion dieses Moduls öffnet Bilder oder liest Pixel; Eingaben sind ausschließlich IDs,
Diagnoseklassen und Counts.
"""

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from vigiscoutx.domain.hashing import canonical_json_bytes, canonical_sha256
from vigiscoutx.domain.types import FrozenModel, Identifier, NonNegativeInt, Sha256Hex
from vigiscoutx.experiment.calibration.config import (
    CalibrationPool,
    CalibrationProtocolConfig,
    SamplingNamespace,
)
from vigiscoutx.experiment.calibration.hashing import (
    first64bits,
    rank_ids,
    sha256_concat,
)


class WindowNotConstructibleError(ValueError):
    """Ein Window lässt sich mit den verfügbaren Läsionen nicht ohne Replacement bilden."""


def largest_remainder(total: int, weights: Sequence[Fraction]) -> tuple[int, ...]:
    """Largest-Remainder-Apportionment; Gleichstand der Reste → Reihenfolge der `weights`."""
    if total < 0 or not weights or any(w < 0 for w in weights) or sum(weights) != 1:
        raise ValueError("Gewichte müssen nichtnegativ sein und sich zu 1 summieren")
    quotas = [total * w for w in weights]
    counts = [math.floor(q) for q in quotas]
    remaining = total - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: (-(quotas[i] - counts[i]), i))
    for index in order[:remaining]:
        counts[index] += 1
    return tuple(counts)


def round_half_up(value: Fraction) -> int:
    return math.floor(value + Fraction(1, 2))


# --- Läsionen, Pools, Canonical Samples -------------------------------------------------------


class CalibrationLesion(FrozenModel):
    """Strukturelle Sicht auf eine CALIBRATION-Läsion (IDs und Diagnoseklasse)."""

    lesion_id: Identifier
    class_id: Identifier
    sample_ids: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids einer Läsion müssen eindeutig sein")
        return self


class PoolClassBlock(FrozenModel):
    pool: CalibrationPool
    class_id: Identifier
    lesion_ids: tuple[Identifier, ...]


class PoolAssignment(FrozenModel):
    """Ergebnis des Pool-Splits; `lesion_ids` je Block in Ranking-Reihenfolge."""

    split_hash: Sha256Hex
    blocks: tuple[PoolClassBlock, ...]

    def lesion_ids(self, pool: CalibrationPool) -> frozenset[str]:
        return frozenset(i for b in self.blocks if b.pool is pool for i in b.lesion_ids)

    def by_class(self, pool: CalibrationPool) -> dict[str, tuple[str, ...]]:
        return {b.class_id: b.lesion_ids for b in self.blocks if b.pool is pool}

    def manifest_sha256(self, pool: CalibrationPool) -> str:
        return canonical_sha256([b.model_dump(mode="json") for b in self.blocks if b.pool is pool])


def _validate_lesions(
    lesions: Sequence[CalibrationLesion], config: CalibrationProtocolConfig
) -> None:
    ids = [lesion.lesion_id for lesion in lesions]
    if len(ids) != len(set(ids)):
        raise ValueError("lesion_id muss eindeutig sein")
    samples = [s for lesion in lesions for s in lesion.sample_ids]
    if len(samples) != len(set(samples)):
        raise ValueError("sample_id darf nur zu genau einer Läsion gehören")
    unknown = {lesion.class_id for lesion in lesions} - set(config.class_order)
    if unknown:
        raise ValueError(f"unbekannte Diagnoseklassen: {sorted(unknown)}")


def pool_sizes(n_lesions: int, config: CalibrationProtocolConfig) -> dict[CalibrationPool, int]:
    sizes = largest_remainder(n_lesions, config.pool_fractions())
    return dict(zip(config.pool_order, sizes, strict=True))


def assign_pools(
    lesions: Sequence[CalibrationLesion], split_hash: str, config: CalibrationProtocolConfig
) -> PoolAssignment:
    """30/50/20 je Diagnoseklasse; Ranking nach `SHA256(split_hash ‖ "P4-01|POOL" ‖ lesion_id)`."""
    _validate_lesions(lesions, config)
    blocks: list[PoolClassBlock] = []
    for class_id in config.class_order:
        members = [lesion.lesion_id for lesion in lesions if lesion.class_id == class_id]
        ranked = rank_ids(split_hash, config.pool_rank_namespace, members)
        start = 0
        for pool, size in pool_sizes(len(ranked), config).items():
            blocks.append(
                PoolClassBlock(
                    pool=pool, class_id=class_id, lesion_ids=ranked[start : start + size]
                )
            )
            start += size
    return PoolAssignment(split_hash=split_hash, blocks=tuple(blocks))


def canonical_sample(
    lesion: CalibrationLesion, split_hash: str, config: CalibrationProtocolConfig
) -> str:
    """Kleinster `SHA256(split_hash ‖ "P4-01|CANONICAL" ‖ sample_id)`; Tie lexikographisch."""
    return rank_ids(split_hash, config.canonical_namespace, lesion.sample_ids)[0]


def canonical_samples(
    lesions: Sequence[CalibrationLesion], split_hash: str, config: CalibrationProtocolConfig
) -> dict[str, str]:
    _validate_lesions(lesions, config)
    return {lesion.lesion_id: canonical_sample(lesion, split_hash, config) for lesion in lesions}


# --- Klassenvektor und Reference ----------------------------------------------------------------


class ClassVector(FrozenModel):
    window_size: int
    counts: tuple[NonNegativeInt, ...]
    class_order: tuple[Identifier, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(self.counts) != len(self.class_order) or sum(self.counts) != self.window_size:
            raise ValueError("Klassenvektor: je Klasse ein Count, Summe = W")
        return self

    @property
    def valid(self) -> bool:
        """`n_c(W) ≥ 1` für alle Klassen."""
        return all(c >= 1 for c in self.counts)

    def as_dict(self) -> dict[str, int]:
        return dict(zip(self.class_order, self.counts, strict=True))


def class_vector(
    window_size: int, ref_class_counts: Mapping[str, int], config: CalibrationProtocolConfig
) -> ClassVector:
    """`n_c(W)` per Largest-Remainder aus der lesion-level Klassenverteilung von `CAL_REF`."""
    if window_size not in config.window_size_candidates:
        raise ValueError(f"W muss in {config.window_size_candidates} liegen")
    if set(ref_class_counts) != set(config.class_order):
        raise ValueError("Klassencounts für exakt die sieben Klassen erforderlich")
    total = sum(ref_class_counts.values())
    if total <= 0 or any(v < 0 for v in ref_class_counts.values()):
        raise ValueError("CAL_REF-Klassencounts müssen nichtnegativ und nicht leer sein")
    weights = [Fraction(ref_class_counts[c], total) for c in config.class_order]
    return ClassVector(
        window_size=window_size,
        counts=largest_remainder(window_size, weights),
        class_order=config.class_order,
    )


class WindowSelection(FrozenModel):
    """Deterministisch ausgewähltes Window: Läsionen je Stratum und Canonical Samples."""

    window_id: Sha256Hex
    context: str
    lesion_ids: tuple[Identifier, ...]
    sample_ids: tuple[Identifier, ...]


def window_id(split_hash: str, context: str, sample_ids: Sequence[str]) -> str:
    """`SHA256(split_hash ‖ context ‖ sorted(sample_ids))`; Liste als kanonisches JSON."""
    return sha256_concat(split_hash, context, canonical_json_bytes(sorted(sample_ids)).decode())


def _select(
    split_hash: str,
    prefix_for: Mapping[str, str],
    available: Mapping[str, Sequence[str]],
    counts: Mapping[str, int],
) -> tuple[str, ...]:
    chosen: list[str] = []
    for stratum, needed in counts.items():
        pool = available.get(stratum, ())
        if needed < 0 or needed > len(pool):
            raise WindowNotConstructibleError(
                f"Stratum {stratum}: benötigt {needed}, verfügbar {len(pool)}"
            )
        chosen.extend(rank_ids(split_hash, prefix_for[stratum], pool)[:needed])
    if len(set(chosen)) != len(chosen):
        raise WindowNotConstructibleError("eine Läsion wurde mehrfach ausgewählt")
    return tuple(chosen)


def _canonical_ids(lesion_ids: Sequence[str], canonical: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(canonical[lesion] for lesion in lesion_ids)


def select_reference(
    cal_ref_by_class: Mapping[str, Sequence[str]],
    vector: ClassVector,
    canonical: Mapping[str, str],
    split_hash: str,
    config: CalibrationProtocolConfig,
) -> WindowSelection:
    """Je Klasse die ersten `n_c(W)` nach `SHA256(split_hash ‖ "P4-01|OFFICIAL_REFERENCE|" ‖
    class_id ‖ lesion_id)`; gilt für Kandidaten-W und offizielle Reference gleichermaßen."""
    if not vector.valid:
        raise WindowNotConstructibleError("Klassenvektor ist nicht zulässig (n_c(W) < 1)")
    prefixes = {c: f"{config.reference_namespace}{c}" for c in config.class_order}
    lesions = _select(split_hash, prefixes, cal_ref_by_class, vector.as_dict())
    samples = _canonical_ids(lesions, canonical)
    context = f"{config.reference_namespace}{vector.window_size}"
    return WindowSelection(
        window_id=window_id(split_hash, context, samples),
        context=context,
        lesion_ids=lesions,
        sample_ids=samples,
    )


def reference_sha256(reference: WindowSelection) -> str:
    """Hash der sortierten ID-Listen des Reference Windows."""
    return canonical_sha256(
        {"lesion_ids": sorted(reference.lesion_ids), "sample_ids": sorted(reference.sample_ids)}
    )


# --- Monitoring-Window-Sampling ---------------------------------------------------------------

_SCENARIO_KEY_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def sampling_context(
    namespace: SamplingNamespace, window_size: int, scenario_key: str, repeat_id: int
) -> str:
    """`<namespace>|<window_size>|<scenario_key>|<repeat_id>`; `scenario_key` nur strukturell."""
    if not scenario_key or not set(scenario_key) <= _SCENARIO_KEY_CHARS:
        raise ValueError("scenario_key muss ein struktureller Code (A–Z, 0–9, _) sein")
    if window_size <= 0 or repeat_id < 0:
        raise ValueError("window_size > 0 und repeat_id ≥ 0 erforderlich")
    return f"{namespace.value}|{window_size}|{scenario_key}|{repeat_id}"


def sample_window(
    split_hash: str,
    context: str,
    available: Mapping[str, Sequence[str]],
    counts: Mapping[str, int],
    canonical: Mapping[str, str],
) -> WindowSelection:
    """Je Stratum (Klasse bzw. Klasse×Subgruppe) die ersten benötigten Läsionen nach
    `SHA256(split_hash ‖ context ‖ lesion_id)`; ohne Replacement innerhalb des Windows."""
    lesions = _select(split_hash, dict.fromkeys(counts, context), available, counts)
    samples = _canonical_ids(lesions, canonical)
    return WindowSelection(
        window_id=window_id(split_hash, context, samples),
        context=context,
        lesion_ids=lesions,
        sample_ids=samples,
    )


def repeat_seed(
    split_hash: str,
    namespace: SamplingNamespace,
    window_size: int,
    scenario_key: str,
    repeat_id: int,
) -> int:
    """G1-Permutations-Seed: `first64bits(SHA256(split_hash ‖ purpose ‖ repeat_id))` mit
    `purpose = "<namespace>|<window_size>|<scenario_key>"`."""
    sampling_context(namespace, window_size, scenario_key, repeat_id)
    purpose = f"{namespace.value}|{window_size}|{scenario_key}"
    return first64bits(sha256_concat(split_hash, purpose, str(repeat_id)))


# --- Autoritatives Strukturinventar und Integritätsprüfung (P4-02B) ---------------------------


class PoolIntegrityError(ValueError):
    """Pool-Zuordnung oder Canonical-Mapping weicht vom autoritativen Inventar ab."""


class CalibrationInventory(FrozenModel):
    """Strukturelle CALIBRATION-Sicht, gebunden an die exakte Split-Identität.

    Enthält ausschließlich IDs, `lesion_id`-Zuordnung, Diagnoseklasse und Sample-IDs je Läsion –
    keine Pixel und keine sonstigen Metadaten. Wird ausschließlich vom Calibration-Prozess über
    Split-Manifest und strukturelle Berechtigung erzeugt.
    """

    dataset_snapshot_id: Identifier
    dataset_snapshot_sha256: Sha256Hex
    split_id: Identifier
    split_manifest_sha256: Sha256Hex
    lesions: tuple[CalibrationLesion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [lesion.lesion_id for lesion in self.lesions]
        if ids != sorted(set(ids)):
            raise ValueError("Läsionen müssen eindeutig und nach lesion_id sortiert sein")
        samples = [s for lesion in self.lesions for s in lesion.sample_ids]
        if len(samples) != len(set(samples)):
            raise ValueError("sample_id darf nur zu genau einer Läsion gehören")
        return self

    def by_id(self) -> dict[str, CalibrationLesion]:
        return {lesion.lesion_id: lesion for lesion in self.lesions}

    def inventory_sha256(self) -> str:
        return canonical_sha256(self)


def pool_manifest_sha256(assignment: PoolAssignment) -> str:
    """Hash der vollständigen, akzeptierten Pool-Zuordnung inklusive Split-Hash."""
    return canonical_sha256(assignment)


def verify_pool_assignment(
    assignment: PoolAssignment, inventory: CalibrationInventory, config: CalibrationProtocolConfig
) -> None:
    """Prüft eine Pool-Zuordnung vollständig gegen das autoritative Inventar (A–J).

    Jede Abweichung ist ein harter Fehler; es wird keine Berechtigung ausgestellt.
    """
    if assignment.split_hash != inventory.split_manifest_sha256:
        raise PoolIntegrityError("split_hash weicht vom autoritativen Split ab")
    lesions = inventory.by_id()
    members = [
        (block.pool, block.class_id, lesion_id)
        for block in assignment.blocks
        for lesion_id in block.lesion_ids
    ]
    ids = [lesion_id for _, _, lesion_id in members]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise PoolIntegrityError(f"Läsion in mehreren Pools/Blöcken: {duplicates}")
    foreign = sorted(set(ids) - set(lesions))
    if foreign:
        raise PoolIntegrityError(f"Läsionen außerhalb von CALIBRATION: {foreign}")
    missing = sorted(set(lesions) - set(ids))
    if missing:
        raise PoolIntegrityError(f"CALIBRATION-Läsionen fehlen: {missing}")
    wrong_class = sorted(i for _, c, i in members if lesions[i].class_id != c)
    if wrong_class:
        raise PoolIntegrityError(f"Diagnoseklasse weicht ab: {wrong_class}")
    expected = assign_pools(list(inventory.lesions), inventory.split_manifest_sha256, config)
    for block in expected.blocks:
        actual = [
            b.lesion_ids
            for b in assignment.blocks
            if b.pool is block.pool and b.class_id == block.class_id
        ]
        if len(actual) != 1 or len(actual[0]) != len(block.lesion_ids):
            raise PoolIntegrityError(
                f"{block.pool.value}/{block.class_id}: Anzahl weicht vom 30/50/20-Ergebnis ab"
            )
        if actual[0] != block.lesion_ids:
            raise PoolIntegrityError(
                f"{block.pool.value}/{block.class_id}: Mitglieder weichen vom SHA-256-Ranking ab"
            )
    if assignment != expected:
        raise PoolIntegrityError("Pool-Zuordnung weicht vom eingefrorenen Algorithmus ab")


def verify_canonical_mapping(
    canonical: Mapping[str, str],
    inventory: CalibrationInventory,
    config: CalibrationProtocolConfig,
) -> None:
    """Canonical Sample gehört zur eigenen Läsion, zu CALIBRATION und folgt der Hash-Regel."""
    lesions = inventory.by_id()
    if set(canonical) != set(lesions):
        raise PoolIntegrityError("Canonical-Mapping muss exakt alle CALIBRATION-Läsionen abdecken")
    for lesion_id, sample_id in canonical.items():
        lesion = lesions[lesion_id]
        if sample_id not in lesion.sample_ids:
            raise PoolIntegrityError(f"{sample_id} gehört nicht zu Läsion {lesion_id}")
        if sample_id != canonical_sample(lesion, inventory.split_manifest_sha256, config):
            raise PoolIntegrityError(f"{lesion_id}: Canonical Sample folgt nicht der Hash-Regel")
