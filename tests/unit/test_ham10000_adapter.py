"""HAM10000-/ISIC-2018-Task-3-Adapter (P3-02B)."""

import inspect
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from vigiscoutx.adapters.demonstrator import DEMONSTRATOR_CLASS_ORDER
from vigiscoutx.adapters.ham10000 import (
    DATASET_ID,
    DX_TO_CLASS,
    ISIC2018_TASK3_TRAINING_BASELINE,
    DatasetSnapshotInfo,
    Ham10000Dataset,
    Ham10000Layout,
    SampleMetadata,
    _AccessGatedImageStore,
    _content_sha256,
    _parse_directory,
    _snapshot_info,
)
from vigiscoutx.errors import (
    DatasetValidationError,
    LeakageError,
    PhaseViolationError,
)
from vigiscoutx.experiment.held_out_guard import HeldOutGuard, PartitionAccessPermit
from vigiscoutx.experiment.split import SplitAssignment, SplitManifest
from vigiscoutx.experiment.split_integrity import validate_split_matches_samples
from vigiscoutx.experiment.types import Partition

WriteFixture = Callable[..., Path]
CanonicalWriter = Callable[..., Path]


def store(root: Path, layout: Ham10000Layout) -> _AccessGatedImageStore:
    """Interner, nicht offizieller Speicher für kleine Fixtures (keine Dataset-Identität)."""
    return _AccessGatedImageStore(root, _parse_directory(root, layout))


def split_for(dataset: _AccessGatedImageStore, held_out_lesion: str = "HAM_L7") -> SplitManifest:
    return SplitManifest(
        manifest_id="split-fixture",
        assignments=tuple(
            SplitAssignment(
                sample_id=ref.sample_id,
                group_id=ref.group_id,
                partition=Partition.HELD_OUT
                if ref.group_id == held_out_lesion
                else Partition.CALIBRATION,
                model_development_role=None,
            )
            for ref in dataset.sample_refs()
        ),
    )


def calibration_permit(dataset: _AccessGatedImageStore) -> PartitionAccessPermit:
    return HeldOutGuard().issue_permit(Partition.CALIBRATION, split_for(dataset))


def image_path(root: Path, image_id: str) -> Path:
    (path,) = root.rglob(f"{image_id}.jpg")
    return path


def replace_row(rows: list[dict[str, str]], index: int, **changes: str) -> list[dict[str, str]]:
    changed = [dict(r) for r in rows]
    changed[index].update(changes)
    return changed


# --- gültiges Fixture (interne Strukturvalidierung) ----------------------------------------


def test_valid_fixture_parses(
    ham_rows: list[dict[str, str]], write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    parsed = _parse_directory(write_ham_fixture(), ham_layout)
    info = _snapshot_info(parsed)
    assert (info.n_samples, info.n_lesions) == (9, 7)
    assert info.class_order == DEMONSTRATOR_CLASS_ORDER
    assert {c.label: c.count for c in info.class_counts} == {
        "MEL": 1,
        "NV": 2,
        "BCC": 2,
        "AKIEC": 1,
        "BKL": 1,
        "DF": 1,
        "VASC": 1,
    }
    refs = store(write_ham_fixture(), ham_layout).sample_refs()
    assert [r.sample_id for r in refs] == sorted(r["image_id"] for r in ham_rows)
    assert {r.sample_id: r.group_id for r in refs}["ISIC_T002"] == "HAM_L1"


def test_labels_and_images_follow_canonical_class_order(
    write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    dataset = store(write_ham_fixture(), ham_layout)
    permit = calibration_permit(dataset)
    ids = ["ISIC_T003", "ISIC_T001", "ISIC_T008"]
    assert dataset.labels_for(permit, ids) == {"ISIC_T003": 0, "ISIC_T001": 1, "ISIC_T008": 5}
    images = dataset.load_inputs(permit, ids)
    assert [image.mode for image in images] == ["RGB"] * 3
    assert images[0].size == (60, 45)
    assert set(DX_TO_CLASS.values()) == set(DEMONSTRATOR_CLASS_ORDER)
    assert dataset.class_order() is DEMONSTRATOR_CLASS_ORDER


def test_optional_metadata_is_mapped_without_interpretation(
    write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    dataset = store(write_ham_fixture(), ham_layout)
    metadata = dataset.metadata_for(
        calibration_permit(dataset), ["ISIC_T003", "ISIC_T004", "ISIC_T007"]
    )
    assert metadata["ISIC_T003"] == SampleMetadata(
        dx_type="histo", age=None, sex="male", localization="face"
    )
    assert metadata["ISIC_T004"].sex == "unknown"
    assert metadata["ISIC_T007"].localization is None


def test_missing_optional_columns_become_none(
    write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    root = write_ham_fixture(columns=("lesion_id", "image_id", "dx"))
    parsed = _parse_directory(root, ham_layout)
    dataset = _AccessGatedImageStore(root, parsed)
    metadata = dataset.metadata_for(calibration_permit(dataset), ["ISIC_T001"])["ISIC_T001"]
    assert metadata == SampleMetadata(dx_type=None, age=None, sex=None, localization=None)
    assert parsed.metadata_columns == ("dx", "image_id", "lesion_id")


def test_no_patient_id_is_assumed_or_exposed(
    ham_rows: list[dict[str, str]],
    ham_columns: tuple[str, ...],
    write_ham_fixture: WriteFixture,
    ham_layout: Ham10000Layout,
) -> None:
    root = write_ham_fixture(
        rows=[{**r, "patient_id": "P1"} for r in ham_rows], columns=(*ham_columns, "patient_id")
    )
    dataset = store(root, ham_layout)
    assert set(SampleMetadata.model_fields) == {"dx_type", "age", "sex", "localization"}
    exposed = str(_snapshot_info(_parse_directory(root, ham_layout)).model_dump())
    assert "P1" not in exposed + str(dataset.sample_refs())
    assert "patient" not in " ".join(SampleMetadata.model_fields)
    metadata = dataset.metadata_for(calibration_permit(dataset), ["ISIC_T001"])["ISIC_T001"]
    assert metadata.sex == "female"


# --- Strukturvalidierung ----------------------------------------------------------------------


def test_duplicate_sample_id_fails(
    ham_rows: list[dict[str, str]], write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    with pytest.raises(DatasetValidationError, match="doppelte sample_id"):
        _parse_directory(write_ham_fixture(rows=[*ham_rows, dict(ham_rows[0])]), ham_layout)


def test_missing_lesion_id_fails(
    ham_rows: list[dict[str, str]], write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    root = write_ham_fixture(rows=replace_row(ham_rows, 2, lesion_id=""))
    with pytest.raises(DatasetValidationError, match="lesion_id fehlt"):
        _parse_directory(root, ham_layout)


def test_unknown_label_fails(
    ham_rows: list[dict[str, str]], write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    root = write_ham_fixture(rows=replace_row(ham_rows, 2, dx="scc"))
    with pytest.raises(DatasetValidationError, match="unbekanntes Label"):
        _parse_directory(root, ham_layout)


def test_missing_image_fails(write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout) -> None:
    with pytest.raises(DatasetValidationError, match="Bilddatei fehlt"):
        _parse_directory(write_ham_fixture(skip_images=("ISIC_T005",)), ham_layout)


def test_image_without_metadata_fails(
    write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    root = write_ham_fixture(extra_images=(("images_a", "ISIC_ORPHAN"),))
    with pytest.raises(DatasetValidationError, match="ohne Metadaten"):
        _parse_directory(root, ham_layout)


def test_image_in_two_directories_fails(
    write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    root = write_ham_fixture()
    existing = image_path(root, "ISIC_T001")
    other = "images_a" if existing.parent.name == "images_b" else "images_b"
    shutil.copy(existing, root / other / existing.name)
    with pytest.raises(DatasetValidationError, match="mehreren Verzeichnissen"):
        _parse_directory(root, ham_layout)


def test_missing_required_column_fails(
    write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    with pytest.raises(DatasetValidationError, match="Pflichtspalten"):
        _parse_directory(write_ham_fixture(columns=("image_id", "dx")), ham_layout)


def test_non_numeric_age_fails(
    ham_rows: list[dict[str, str]], write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    root = write_ham_fixture(rows=replace_row(ham_rows, 0, age="forty"))
    with pytest.raises(DatasetValidationError, match="keine Zahl"):
        _parse_directory(root, ham_layout)


# --- Kanonische Baseline (offizieller Produktionspfad) ---------------------------------------


def test_canonical_baseline_is_fixed() -> None:
    assert ISIC2018_TASK3_TRAINING_BASELINE.n_samples == 10015
    assert ISIC2018_TASK3_TRAINING_BASELINE.n_lesions == 7470
    assert ISIC2018_TASK3_TRAINING_BASELINE.class_order == DEMONSTRATOR_CLASS_ORDER
    with pytest.raises(ValidationError):
        ISIC2018_TASK3_TRAINING_BASELINE.n_samples = 9  # type: ignore[misc]
    parameters = set(inspect.signature(Ham10000Dataset.from_directory).parameters)
    assert parameters == {"root", "layout"}


def test_small_fixture_cannot_be_loaded_as_official_snapshot(
    write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    with pytest.raises(DatasetValidationError, match="n_samples: erwartet 10015, gefunden 9"):
        Ham10000Dataset.from_directory(write_ham_fixture(), layout=ham_layout)
    with pytest.raises(TypeError):
        Ham10000Dataset.from_directory(  # type: ignore[call-arg]
            write_ham_fixture(), layout=ham_layout, expectation=None
        )


def test_official_snapshot_loads_canonical_dataset(
    canonical_dataset_root: Path, canonical_layout: Ham10000Layout
) -> None:
    dataset = Ham10000Dataset.from_directory(canonical_dataset_root, layout=canonical_layout)
    info = dataset.snapshot_info()
    assert (info.n_samples, info.n_lesions) == (10015, 7470)
    assert info.dataset_id == DATASET_ID
    assert info.collection == "ISIC Challenge 2018: Task 3: Training"
    assert dataset.snapshot_ref().snapshot_id == DATASET_ID
    assert dataset.snapshot_ref().sha256 == _content_sha256(
        _parse_directory(canonical_dataset_root, canonical_layout)
    )


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({"n_samples": 10014}, "n_samples: erwartet 10015, gefunden 10014"),
        ({"n_lesions": 7469}, "n_lesions: erwartet 7470, gefunden 7469"),
        ({"n_classes": 6}, "Klassen"),
    ],
)
def test_official_path_rejects_baseline_deviation(
    tmp_path: Path,
    canonical_writer: CanonicalWriter,
    canonical_layout: Ham10000Layout,
    counts: dict[str, int],
    message: str,
) -> None:
    root = canonical_writer(tmp_path / "variant", **counts)
    with pytest.raises(DatasetValidationError, match=message):
        Ham10000Dataset.from_directory(root, layout=canonical_layout)


def test_dataset_identity_cannot_be_relabelled(
    canonical_dataset_root: Path, canonical_layout: Ham10000Layout
) -> None:
    info = Ham10000Dataset.from_directory(
        canonical_dataset_root, layout=canonical_layout
    ).snapshot_info()
    for update in (
        {"dataset_id": "OTHER_COLLECTION"},
        {"collection": "ISIC Archive (dynamisch)"},
        {"class_order": tuple(sorted(DEMONSTRATOR_CLASS_ORDER))},
    ):
        with pytest.raises(ValidationError):
            info.model_copy(update=update)
    with pytest.raises(ValidationError):
        DatasetSnapshotInfo.model_validate({**info.model_dump(), "dataset_id": "FIXTURE"})


def test_direct_constructor_is_blocked(
    write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    root = write_ham_fixture()
    parsed = _parse_directory(root, ham_layout)
    with pytest.raises(PhaseViolationError, match="from_directory"):
        Ham10000Dataset(root, parsed, _snapshot_info(parsed), _factory_token=object())


# --- Pfadunabhängiger fachlicher Hash ---------------------------------------------------------


def test_snapshot_hash_is_independent_of_local_paths(
    tmp_path: Path, canonical_dataset_root: Path, canonical_layout: Ham10000Layout
) -> None:
    moved = tmp_path / "ganz" / "anderer_ort"
    shutil.copytree(canonical_dataset_root, moved)
    shutil.move(moved / "part_1", moved / "bilder_x")
    shutil.move(moved / "part_2", moved / "bilder_y")
    shutil.move(moved / "HAM10000_metadata.csv", moved / "meta.csv")
    other_layout = Ham10000Layout(
        metadata_file="meta.csv", image_dirs=("bilder_y", "bilder_x"), image_suffix=".png"
    )
    original = Ham10000Dataset.from_directory(canonical_dataset_root, layout=canonical_layout)
    relocated = Ham10000Dataset.from_directory(moved, layout=other_layout)
    assert relocated.snapshot_ref().sha256 == original.snapshot_ref().sha256
    assert relocated.snapshot_info() == original.snapshot_info()


def test_snapshot_hash_is_independent_of_row_and_column_order(
    ham_rows: list[dict[str, str]],
    ham_columns: tuple[str, ...],
    write_ham_fixture: WriteFixture,
    ham_layout: Ham10000Layout,
) -> None:
    base = _content_sha256(_parse_directory(write_ham_fixture(), ham_layout))
    reordered_root = write_ham_fixture(rows=ham_rows[::-1], columns=ham_columns[::-1])
    assert _content_sha256(_parse_directory(reordered_root, ham_layout)) == base


def test_snapshot_hash_changes_with_image_content(
    write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    root = write_ham_fixture()
    base = _content_sha256(_parse_directory(root, ham_layout))
    image = image_path(root, "ISIC_T001")
    image.write_bytes(image.read_bytes() + b"\0")
    assert _content_sha256(_parse_directory(root, ham_layout)) != base


@pytest.mark.parametrize(
    "change",
    [{"localization": "chest"}, {"age": "46.0"}, {"lesion_id": "HAM_L99"}, {"dx": "mel"}],
)
def test_snapshot_hash_changes_with_metadata(
    ham_rows: list[dict[str, str]],
    write_ham_fixture: WriteFixture,
    ham_layout: Ham10000Layout,
    change: dict[str, str],
) -> None:
    base = _content_sha256(_parse_directory(write_ham_fixture(), ham_layout))
    changed_root = write_ham_fixture(rows=replace_row(ham_rows, 0, **change))
    assert _content_sha256(_parse_directory(changed_root, ham_layout)) != base


def test_image_modified_after_snapshot_fails(
    write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    root = write_ham_fixture()
    dataset = store(root, ham_layout)
    image_path(root, "ISIC_T001").write_bytes(b"corrupted")
    with pytest.raises(DatasetValidationError, match="verändert"):
        dataset.load_inputs(calibration_permit(dataset), ["ISIC_T001"])


# --- Zugriffskontrolle ------------------------------------------------------------------------


def test_dataset_access_requires_valid_context(
    write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    dataset = store(write_ham_fixture(), ham_layout)
    permit = calibration_permit(dataset)
    with pytest.raises(TypeError):
        dataset.load_inputs(["ISIC_T001"])  # type: ignore[arg-type, call-arg]
    for call in (dataset.load_inputs, dataset.labels_for, dataset.metadata_for):
        with pytest.raises(PhaseViolationError, match="außerhalb"):
            call(permit, ["ISIC_T009"])  # HAM_L7 liegt in HELD_OUT
    with pytest.raises(PhaseViolationError, match="HELD_OUT"):
        HeldOutGuard().issue_permit(Partition.HELD_OUT, split_for(dataset))


def test_official_dataset_access_requires_valid_context(
    canonical_dataset_root: Path, canonical_layout: Ham10000Layout
) -> None:
    dataset = Ham10000Dataset.from_directory(canonical_dataset_root, layout=canonical_layout)
    split = split_for(dataset, held_out_lesion="HAM_L0000000")
    permit = HeldOutGuard().issue_permit(Partition.CALIBRATION, split)
    assert len(dataset.load_inputs(permit, ["ISIC_S0000001"])) == 1
    with pytest.raises(PhaseViolationError, match="außerhalb"):
        dataset.labels_for(permit, ["ISIC_S0000000"])


@pytest.mark.parametrize("cls", [_AccessGatedImageStore, Ham10000Dataset])
def test_every_public_per_sample_accessor_requires_access_context(cls: type) -> None:
    open_api = {"from_directory", "snapshot_info", "snapshot_ref", "class_order", "sample_refs"}
    for name, member in inspect.getmembers(cls, inspect.isfunction):
        if name.startswith("_") or name in open_api:
            continue
        assert "access" in inspect.signature(member).parameters, name


# --- Split-Manifeste konsumieren --------------------------------------------------------------


def test_existing_split_manifest_is_validated_against_dataset(
    write_ham_fixture: WriteFixture, ham_layout: Ham10000Layout
) -> None:
    dataset = store(write_ham_fixture(), ham_layout)
    split = split_for(dataset)
    validate_split_matches_samples(split, dataset.sample_refs())

    incomplete = SplitManifest(manifest_id="x", assignments=split.assignments[:-1])
    with pytest.raises(LeakageError, match="ohne Split-Zuordnung"):
        validate_split_matches_samples(incomplete, dataset.sample_refs())

    wrong_group = SplitManifest(
        manifest_id="x",
        assignments=(
            split.assignments[0].model_copy(update={"group_id": "HAM_OTHER"}),
            *split.assignments[1:],
        ),
    )
    with pytest.raises(LeakageError, match="lesion_id"):
        validate_split_matches_samples(wrong_group, dataset.sample_refs())
