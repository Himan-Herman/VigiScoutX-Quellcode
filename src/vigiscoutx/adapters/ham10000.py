"""HAM10000 / ISIC 2018 Task 3 Training – lokaler Dataset-Adapter (ARC-018, ARC-021; DEC-012).

Es wird nichts heruntergeladen. Der öffentliche Produktionspfad `Ham10000Dataset.from_directory`
validiert gegen die fest verdrahtete kanonische Baseline (10.015 Bilder, 7.470 Läsionen,
exakt die 7 Demonstratorklassen); sie kann vom Aufrufer nicht überschrieben werden.
Bilder, Labels und Metadaten werden nur mit gültigem `DataAccessContext` herausgegeben.
`lesion_id` ist die Leakage-Gruppe. Eine Patient-ID wird weder angenommen noch erzeugt.

Die fachliche Snapshot-Identität hängt nur von Inhalten ab (IDs, Labels, Metadatenwerte,
Bild-Inhaltshashes), nicht von lokalen Pfaden, Verzeichnisnamen oder Zeilen-/Spaltenreihenfolge.
"""

import csv
import io
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from PIL import Image
from pydantic import Field, StrictFloat, field_validator, model_validator

from vigiscoutx.adapters.demonstrator import (
    DEMONSTRATOR_CLASS_ORDER,
    require_canonical_class_order,
)
from vigiscoutx.artifacts.files import sha256_file
from vigiscoutx.artifacts.manifests import canonical_artifact_path
from vigiscoutx.domain.hashing import canonical_sha256, sha256_hex
from vigiscoutx.domain.refs import DatasetSnapshotRef, SampleRef
from vigiscoutx.domain.types import FrozenModel, Identifier, NonNegativeInt, PositiveInt, Sha256Hex
from vigiscoutx.errors import DatasetValidationError, PhaseViolationError
from vigiscoutx.ports import AccessPurpose, DataAccessContext

DATASET_ID = "ISIC2018_TASK3_TRAINING_HAM10000"
COLLECTION = "ISIC Challenge 2018: Task 3: Training"

DX_TO_CLASS: Mapping[str, str] = {
    "mel": "MEL",
    "nv": "NV",
    "bcc": "BCC",
    "akiec": "AKIEC",
    "bkl": "BKL",
    "df": "DF",
    "vasc": "VASC",
}
if set(DX_TO_CLASS.values()) != set(DEMONSTRATOR_CLASS_ORDER):
    raise RuntimeError("DX_TO_CLASS muss exakt die kanonischen Demonstratorklassen abbilden")

REQUIRED_COLUMNS = ("image_id", "lesion_id", "dx")


class CanonicalBaseline(FrozenModel):
    n_samples: PositiveInt
    n_lesions: PositiveInt
    class_order: tuple[Identifier, ...]


ISIC2018_TASK3_TRAINING_BASELINE = CanonicalBaseline(
    n_samples=10015, n_lesions=7470, class_order=DEMONSTRATOR_CLASS_ORDER
)
"""Feste Baseline gemäß DEC-012 / CLM-017; nicht vom Aufrufer überschreibbar."""


class Ham10000Layout(FrozenModel):
    """Lokale Dateistruktur (relativ zum Dataset-Wurzelverzeichnis); nur Laufzeitinformation."""

    metadata_file: str = Field(min_length=1, strict=True)
    image_dirs: tuple[str, ...] = Field(min_length=1)
    image_suffix: Literal[".jpg", ".jpeg", ".png"]

    @field_validator("metadata_file")
    @classmethod
    def _relative_file(cls, value: str) -> str:
        return canonical_artifact_path(value)

    @field_validator("image_dirs")
    @classmethod
    def _relative_dirs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        canonical = tuple(canonical_artifact_path(v) for v in values)
        if len(set(canonical)) != len(canonical):
            raise ValueError("image_dirs müssen eindeutig sein")
        return canonical


class SampleMetadata(FrozenModel):
    """Optionale Metadaten, nur soweit im Metadatensatz vorhanden; Rohwerte ohne Deutung."""

    dx_type: str | None
    age: StrictFloat | None
    sex: str | None
    localization: str | None


class ClassCount(FrozenModel):
    label: Identifier
    count: NonNegativeInt


class DatasetSnapshotInfo(FrozenModel):
    """Fachliche, pfadunabhängige Snapshot-Identität."""

    dataset_id: Literal["ISIC2018_TASK3_TRAINING_HAM10000"]
    collection: Literal["ISIC Challenge 2018: Task 3: Training"]
    metadata_content_sha256: Sha256Hex
    metadata_columns: tuple[str, ...]
    n_samples: PositiveInt
    n_lesions: PositiveInt
    class_order: tuple[Identifier, ...]
    class_counts: tuple[ClassCount, ...]
    records_sha256: Sha256Hex

    @field_validator("class_order")
    @classmethod
    def _canonical_order(cls, order: tuple[str, ...]) -> tuple[str, ...]:
        return require_canonical_class_order(order)

    @model_validator(mode="after")
    def _counts_consistent(self) -> Self:
        if tuple(c.label for c in self.class_counts) != self.class_order:
            raise ValueError("class_counts müssen der kanonischen Klassenreihenfolge folgen")
        if sum(c.count for c in self.class_counts) != self.n_samples:
            raise ValueError("Summe der Klassenhäufigkeiten muss n_samples entsprechen")
        if tuple(sorted(self.metadata_columns)) != self.metadata_columns:
            raise ValueError("metadata_columns müssen sortiert sein")
        return self


@dataclass(frozen=True, slots=True)
class _Record:
    sample_id: str
    lesion_id: str
    label: str
    image_sha256: str
    metadata: SampleMetadata

    def canonical(self) -> dict[str, object]:
        """Pfadunabhängige Darstellung für den fachlichen Hash."""
        return {
            "sample_id": self.sample_id,
            "lesion_id": self.lesion_id,
            "label": self.label,
            "image_sha256": self.image_sha256,
            "metadata": self.metadata.model_dump(mode="json"),
        }


@dataclass(frozen=True, slots=True)
class _ParsedDirectory:
    records: Mapping[str, _Record]
    image_paths: Mapping[str, str]
    metadata_columns: tuple[str, ...]
    metadata_content_sha256: str


def _optional_text(row: Mapping[str, str | None], column: str) -> str | None:
    value = row.get(column)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_age(raw: str | None, sample_id: str) -> float | None:
    if raw is None:
        return None
    try:
        age = float(raw)
    except ValueError as exc:
        raise DatasetValidationError(f"{sample_id}: Alter {raw!r} ist keine Zahl") from exc
    if not math.isfinite(age):
        return None
    return age


def _discover_images(root: Path, layout: Ham10000Layout) -> dict[str, str]:
    images: dict[str, str] = {}
    duplicates: set[str] = set()
    for directory in layout.image_dirs:
        path = root / directory
        if not path.is_dir():
            raise DatasetValidationError(f"Bildverzeichnis fehlt: {directory}")
        for file in sorted(path.iterdir()):
            if not file.is_file() or file.suffix.lower() != layout.image_suffix:
                continue
            if file.stem in images:
                duplicates.add(file.stem)
            images[file.stem] = f"{directory}/{file.name}"
    if duplicates:
        raise DatasetValidationError(
            f"Bild in mehreren Verzeichnissen (inkonsistente Zuordnung): {sorted(duplicates)}"
        )
    return images


def _parse_directory(root: Path, layout: Ham10000Layout) -> _ParsedDirectory:
    """Interne Strukturvalidierung ohne Baseline-Prüfung und ohne Dataset-Identität."""
    metadata_path = root / layout.metadata_file
    if not metadata_path.is_file():
        raise DatasetValidationError(f"Metadatendatei fehlt: {layout.metadata_file}")
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise DatasetValidationError(f"Pflichtspalten fehlen: {missing}")
        rows = list(reader)

    images = _discover_images(root, layout)
    records: dict[str, _Record] = {}
    raw_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = (row.get("image_id") or "").strip()
        lesion_id = (row.get("lesion_id") or "").strip()
        dx = (row.get("dx") or "").strip()
        if not sample_id:
            raise DatasetValidationError("Zeile ohne image_id")
        if not lesion_id:
            raise DatasetValidationError(f"{sample_id}: lesion_id fehlt")
        if sample_id in records:
            raise DatasetValidationError(f"doppelte sample_id: {sample_id}")
        if dx not in DX_TO_CLASS:
            raise DatasetValidationError(f"{sample_id}: unbekanntes Label {dx!r}")
        if sample_id not in images:
            raise DatasetValidationError(f"{sample_id}: Bilddatei fehlt")
        metadata = SampleMetadata(
            dx_type=_optional_text(row, "dx_type"),
            age=_parse_age(_optional_text(row, "age"), sample_id),
            sex=_optional_text(row, "sex"),
            localization=_optional_text(row, "localization"),
        )
        records[sample_id] = _Record(
            sample_id=sample_id,
            lesion_id=lesion_id,
            label=DX_TO_CLASS[dx],
            image_sha256=sha256_file(root / images[sample_id]),
            metadata=metadata,
        )
        raw_rows[sample_id] = {c: (row.get(c) or "").strip() for c in columns}
    orphans = sorted(set(images) - set(records))
    if orphans:
        raise DatasetValidationError(f"Bilder ohne Metadaten: {orphans}")
    return _ParsedDirectory(
        records=records,
        image_paths={s: images[s] for s in records},
        metadata_columns=tuple(sorted(columns)),
        metadata_content_sha256=canonical_sha256([raw_rows[s] for s in sorted(raw_rows)]),
    )


def _baseline_deviations(parsed: _ParsedDirectory, baseline: CanonicalBaseline) -> list[str]:
    records = parsed.records.values()
    found_classes = {r.label for r in records}
    actual = {
        "n_samples": len(parsed.records),
        "n_lesions": len({r.lesion_id for r in records}),
    }
    expected = {"n_samples": baseline.n_samples, "n_lesions": baseline.n_lesions}
    deviations = [
        f"{k}: erwartet {expected[k]}, gefunden {actual[k]}"
        for k in expected
        if expected[k] != actual[k]
    ]
    if found_classes != set(baseline.class_order):
        deviations.append(
            f"Klassen: erwartet {sorted(baseline.class_order)}, gefunden {sorted(found_classes)}"
        )
    return deviations


def _snapshot_info(parsed: _ParsedDirectory) -> DatasetSnapshotInfo:
    records = parsed.records
    counts = Counter(r.label for r in records.values())
    return DatasetSnapshotInfo(
        dataset_id="ISIC2018_TASK3_TRAINING_HAM10000",
        collection="ISIC Challenge 2018: Task 3: Training",
        metadata_content_sha256=parsed.metadata_content_sha256,
        metadata_columns=parsed.metadata_columns,
        n_samples=len(records),
        n_lesions=len({r.lesion_id for r in records.values()}),
        class_order=DEMONSTRATOR_CLASS_ORDER,
        class_counts=tuple(
            ClassCount(label=label, count=counts.get(label, 0))
            for label in DEMONSTRATOR_CLASS_ORDER
        ),
        records_sha256=canonical_sha256([records[key].canonical() for key in sorted(records)]),
    )


def _content_sha256(parsed: _ParsedDirectory) -> str:
    """Fachlicher Snapshot-Hash; identisch für identische Inhalte an beliebigen Orten."""
    return canonical_sha256(_snapshot_info(parsed))


class _AccessGatedImageStore:
    """Interner, zugriffsgebundener Bild-/Labelspeicher ohne offizielle Dataset-Identität."""

    def __init__(self, root: Path, parsed: _ParsedDirectory) -> None:
        self._root = root
        self._parsed = parsed

    def class_order(self) -> tuple[str, ...]:
        return DEMONSTRATOR_CLASS_ORDER

    def sample_refs(self) -> tuple[SampleRef, ...]:
        """Nur sample_id und lesion_id (für die Split-Prüfung), keine Labels."""
        records = self._parsed.records
        return tuple(
            SampleRef(sample_id=key, group_id=records[key].lesion_id) for key in sorted(records)
        )

    def _lookup(
        self, access: DataAccessContext, sample_ids: Sequence[str], purpose: AccessPurpose
    ) -> list[_Record]:
        access.require_access(sample_ids, purpose)
        unknown = sorted(set(sample_ids) - set(self._parsed.records))
        if unknown:
            raise DatasetValidationError(f"unbekannte sample_ids: {unknown}")
        return [self._parsed.records[s] for s in sample_ids]

    def load_inputs(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> list[Image.Image]:
        images: list[Image.Image] = []
        for record in self._lookup(access, sample_ids, AccessPurpose.INPUTS):
            data = (self._root / self._parsed.image_paths[record.sample_id]).read_bytes()
            if sha256_hex(data) != record.image_sha256:
                raise DatasetValidationError(
                    f"{record.sample_id}: Bilddatei seit Snapshot verändert"
                )
            with Image.open(io.BytesIO(data)) as image:
                images.append(image.convert("RGB"))
        return images

    def labels_for(self, access: DataAccessContext, sample_ids: Sequence[str]) -> Mapping[str, int]:
        return {
            r.sample_id: DEMONSTRATOR_CLASS_ORDER.index(r.label)
            for r in self._lookup(access, sample_ids, AccessPurpose.LABELS)
        }

    def stratification_labels_for(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> Mapping[str, int]:
        """Labels ausschließlich für die stratifizierte Split-Erzeugung."""
        return {
            r.sample_id: DEMONSTRATOR_CLASS_ORDER.index(r.label)
            for r in self._lookup(access, sample_ids, AccessPurpose.STRATIFICATION_LABELS)
        }

    def metadata_for(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> Mapping[str, SampleMetadata]:
        return {
            r.sample_id: r.metadata
            for r in self._lookup(access, sample_ids, AccessPurpose.METADATA)
        }


_FACTORY_TOKEN = object()


class Ham10000Dataset(_AccessGatedImageStore):
    """Offizieller DatasetPort-/LabelPort-Adapter; nur über `from_directory` erzeugbar."""

    def __init__(
        self,
        root: Path,
        parsed: _ParsedDirectory,
        info: DatasetSnapshotInfo,
        *,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _FACTORY_TOKEN:
            raise PhaseViolationError("Ham10000Dataset nur über from_directory erzeugbar")
        super().__init__(root, parsed)
        self._info = info

    @classmethod
    def from_directory(cls, root: Path, *, layout: Ham10000Layout) -> "Ham10000Dataset":
        """Lädt den lokalen Snapshot und erzwingt die kanonische ISIC-2018-Task-3-Baseline."""
        parsed = _parse_directory(root, layout)
        deviations = _baseline_deviations(parsed, ISIC2018_TASK3_TRAINING_BASELINE)
        if deviations:
            raise DatasetValidationError(
                f"Kein offizieller {DATASET_ID}-Snapshot ({'; '.join(deviations)})"
            )
        return cls(root, parsed, _snapshot_info(parsed), _factory_token=_FACTORY_TOKEN)

    def snapshot_info(self) -> DatasetSnapshotInfo:
        return self._info

    def snapshot_ref(self) -> DatasetSnapshotRef:
        return DatasetSnapshotRef(
            snapshot_id=DATASET_ID, uri=f"local:{DATASET_ID}", sha256=canonical_sha256(self._info)
        )
