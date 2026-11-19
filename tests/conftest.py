"""Gemeinsame, deterministische Test-Fixtures.

Alle Zahlenwerte sind reine Testwerte und keine wissenschaftlichen Parameter.
"""

import copy
import csv
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torchvision.models import resnet18

from provenance_repo import (
    GUARD_PATH,
    HELD_OUT,
    ProvenanceRepo,
    approve_and_verify,
    create_provenance_repo,
)
from vigiscoutx.adapters.demonstrator import DEMONSTRATOR_CLASS_ORDER
from vigiscoutx.adapters.ham10000 import (
    DX_TO_CLASS,
    Ham10000Layout,
    _AccessGatedImageStore,
    _content_sha256,
    _parse_directory,
)
from vigiscoutx.adapters.preprocessing import (
    ImagePreprocessingSpec,
    Interpolation,
    Normalization,
    ResizeMode,
)
from vigiscoutx.adapters.resnet18 import MANIFEST_FILE, write_initialization_snapshot
from vigiscoutx.artifacts.files import sha256_file
from vigiscoutx.domain.refs import (
    DatasetSnapshotRef,
    ModelSnapshotRef,
    SeedBundle,
    SplitManifestRef,
)
from vigiscoutx.experiment import code_provenance
from vigiscoutx.experiment.code_provenance import EvaluationProvenance
from vigiscoutx.experiment.freeze import FreezeManifest, build_freeze_manifest
from vigiscoutx.experiment.held_out_guard import HeldOutGuard, ObservedHashes, PartitionAccessPermit
from vigiscoutx.experiment.model_development import pretrained as pretrained_module
from vigiscoutx.experiment.model_development.config import ModelDevelopmentConfig
from vigiscoutx.experiment.model_development.trainer import ModelDevelopmentTrainer, TrainedModel
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.split import SplitManifest
from vigiscoutx.experiment.splitting import (
    SplitGenerationConfig,
    generate_split_from_source,
    split_manifest_ref,
)
from vigiscoutx.experiment.types import ModelDevelopmentRole

GIT_COMMIT = "a" * 40
ENV_LOCK = "e" * 64


def hex64(char: str) -> str:
    return char * 64


@pytest.fixture
def calibration_config_data() -> dict[str, Any]:
    """Development-Konfiguration mit explizit offenen Calibration-Parametern (null)."""
    return {
        "schema_version": 1,
        "run_id": "run-dev-001",
        "phase": "DEVELOPMENT_CALIBRATION",
        "freeze_id": None,
        "scientific": {
            "dataset": {"snapshot_id": "ds-test", "uri": "memory://dataset", "sha256": hex64("1")},
            "split": {"manifest_id": "split-test", "uri": "memory://split", "sha256": hex64("2")},
            "model": {
                "model_id": "model-test",
                "architecture": "test-arch",
                "uri": "memory://model",
                "sha256": hex64("3"),
            },
            "preprocessing": {
                "pipeline_id": "prep-test",
                "input_height": 8,
                "input_width": 8,
                "color_mode": "RGB",
                "params": {},
            },
            "monitoring": {
                "window_size": 4,
                "signals": [
                    {
                        "signal_id": "g1-test",
                        "group": "G1",
                        "method": "test-method",
                        "params": {},
                        "threshold": None,
                    },
                    {
                        "signal_id": "g3-test",
                        "group": "G3",
                        "method": "test-method",
                        "params": {"k": 1, "nested": {"values": [1, 2]}},
                        "threshold": None,
                    },
                ],
            },
            "fusion": {"rule": None, "groups": ["G1", "G2", "G3"]},
            "relevance": {
                "performance_metric": "BALANCED_ACCURACY",
                "clean_quantile": 0.5,
                "delta_rel": None,
            },
            "risk_policy": {"policy_id": "risk-test", "version": "1"},
            "recommendation_policy": {"policy_id": "rec-test", "version": "1"},
        },
        "repetitions": 2,
        "seeds": {
            "split_seed": 1,
            "model_development_seed": 2,
            "sampling_seed": 3,
            "shift_seed": 4,
            "statistical_seed": 5,
            "repeat_id": 0,
        },
        "evaluation_metrics": ["SENSITIVITY", "FALSE_ALARM_RATE"],
    }


@pytest.fixture
def calibrated_config_data(calibration_config_data: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(calibration_config_data)
    scientific = data["scientific"]
    for signal in scientific["monitoring"]["signals"]:
        signal["threshold"] = {"kind": "test-kind", "value": 0.1}
    scientific["fusion"]["rule"] = "B_AT_LEAST_TWO_PRIMARY_GROUPS"
    scientific["relevance"]["delta_rel"] = 0.2
    return data


@pytest.fixture
def calibrated_config(calibrated_config_data: dict[str, Any]) -> RunConfig:
    return RunConfig.model_validate(calibrated_config_data)


@pytest.fixture
def seed_list(calibrated_config: RunConfig) -> tuple[SeedBundle, ...]:
    second = calibrated_config.seeds.model_copy(update={"sampling_seed": 30, "repeat_id": 1})
    return (calibrated_config.seeds, second)


@pytest.fixture
def freeze_manifest(
    calibrated_config: RunConfig, seed_list: tuple[SeedBundle, ...]
) -> FreezeManifest:
    return build_freeze_manifest(
        calibrated_config,
        seed_list=seed_list,
        git_commit=GIT_COMMIT,
        environment_lock_sha256=ENV_LOCK,
    )


@pytest.fixture
def held_out_config(
    calibrated_config_data: dict[str, Any], freeze_manifest: FreezeManifest
) -> RunConfig:
    return RunConfig.model_validate(
        {
            **calibrated_config_data,
            "run_id": "run-eval-001",
            "phase": "HELD_OUT_EVALUATION",
            "freeze_id": freeze_manifest.freeze_id,
        }
    )


@pytest.fixture
def observed(freeze_manifest: FreezeManifest) -> ObservedHashes:
    return ObservedHashes(
        dataset_snapshot_sha256=freeze_manifest.dataset_snapshot_sha256,
        split_manifest_sha256=freeze_manifest.split_manifest_sha256,
        model_weights_sha256=freeze_manifest.model_weights_sha256,
        environment_lock_sha256=ENV_LOCK,
    )


# --- Post-Freeze-Code-Provenienz (P4-03Q, DEC-082) ----------------------------------------


@pytest.fixture
def provenance_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProvenanceRepo:
    """Synthetisches Git-Repository; dessen erster Commit ist der Calibration-Commit.

    Das synthetische Repository gilt in diesem Test als ausgeführtes Repository.
    """
    repo = create_provenance_repo(tmp_path / "provenance-repo")
    monkeypatch.setattr(code_provenance, "_EXECUTING_FILE", repo.root / GUARD_PATH)
    return repo


@pytest.fixture
def evaluation_provenance(
    provenance_repo: ProvenanceRepo, evaluation_manifest: FreezeManifest, tmp_path: Path
) -> EvaluationProvenance:
    provenance = approve_and_verify(
        provenance_repo, evaluation_manifest.freeze_id, tmp_path / "approvals", purpose=HELD_OUT
    )
    assert isinstance(provenance, EvaluationProvenance)
    return provenance


@pytest.fixture
def evaluation_manifest(
    calibrated_config: RunConfig,
    seed_list: tuple[SeedBundle, ...],
    provenance_repo: ProvenanceRepo,
) -> FreezeManifest:
    """Freeze, dessen Calibration-Commit im synthetischen Repository existiert."""
    return build_freeze_manifest(
        calibrated_config,
        seed_list=seed_list,
        git_commit=provenance_repo.calibration_commit,
        environment_lock_sha256=ENV_LOCK,
    )


@pytest.fixture
def evaluation_held_out_config(
    calibrated_config_data: dict[str, Any], evaluation_manifest: FreezeManifest
) -> RunConfig:
    return RunConfig.model_validate(
        {
            **calibrated_config_data,
            "run_id": "run-eval-001",
            "phase": "HELD_OUT_EVALUATION",
            "freeze_id": evaluation_manifest.freeze_id,
        }
    )


# --- HAM10000-/ResNet-18-Fixtures (P3-02B) -------------------------------------------------

HAM_COLUMNS = ("lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization")
HAM_ROWS: tuple[dict[str, str], ...] = (
    {
        "lesion_id": "HAM_L1",
        "image_id": "ISIC_T001",
        "dx": "nv",
        "dx_type": "follow_up",
        "age": "45.0",
        "sex": "female",
        "localization": "back",
    },
    {
        "lesion_id": "HAM_L1",
        "image_id": "ISIC_T002",
        "dx": "nv",
        "dx_type": "follow_up",
        "age": "45.0",
        "sex": "female",
        "localization": "back",
    },
    {
        "lesion_id": "HAM_L2",
        "image_id": "ISIC_T003",
        "dx": "mel",
        "dx_type": "histo",
        "age": "",
        "sex": "male",
        "localization": "face",
    },
    {
        "lesion_id": "HAM_L3",
        "image_id": "ISIC_T004",
        "dx": "bcc",
        "dx_type": "histo",
        "age": "70.0",
        "sex": "unknown",
        "localization": "trunk",
    },
    {
        "lesion_id": "HAM_L3",
        "image_id": "ISIC_T005",
        "dx": "bcc",
        "dx_type": "histo",
        "age": "70.0",
        "sex": "unknown",
        "localization": "trunk",
    },
    {
        "lesion_id": "HAM_L4",
        "image_id": "ISIC_T006",
        "dx": "akiec",
        "dx_type": "histo",
        "age": "80.0",
        "sex": "male",
        "localization": "scalp",
    },
    {
        "lesion_id": "HAM_L5",
        "image_id": "ISIC_T007",
        "dx": "bkl",
        "dx_type": "consensus",
        "age": "60.0",
        "sex": "female",
        "localization": "",
    },
    {
        "lesion_id": "HAM_L6",
        "image_id": "ISIC_T008",
        "dx": "df",
        "dx_type": "confocal",
        "age": "35.0",
        "sex": "male",
        "localization": "lower extremity",
    },
    {
        "lesion_id": "HAM_L7",
        "image_id": "ISIC_T009",
        "dx": "vasc",
        "dx_type": "consensus",
        "age": "50.0",
        "sex": "female",
        "localization": "abdomen",
    },
)


@pytest.fixture
def ham_rows() -> list[dict[str, str]]:
    return [dict(row) for row in HAM_ROWS]


@pytest.fixture
def ham_columns() -> tuple[str, ...]:
    return HAM_COLUMNS


@pytest.fixture
def ham_layout() -> Ham10000Layout:
    return Ham10000Layout(
        metadata_file="metadata.csv", image_dirs=("images_a", "images_b"), image_suffix=".jpg"
    )


def _write_image(path: Path, image_id: str) -> None:
    """Bildinhalt hängt nur von der image_id ab (unabhängig von Zeilen-/Verzeichnisposition)."""
    seed = zlib.crc32(image_id.encode())
    image = Image.new("RGB", (60, 45), color=(seed % 256, (seed >> 8) % 256, (seed >> 16) % 256))
    for x in range(0, 60, 7):
        image.putpixel((x, (x + seed) % 45), (255, 255, 255))
    image.save(path, format="JPEG", quality=95)


@pytest.fixture
def write_ham_fixture(tmp_path: Path, ham_layout: Ham10000Layout) -> Callable[..., Path]:
    """Erzeugt je Aufruf einen neuen kleinen HAM10000-artigen Datensatz."""
    calls = iter(range(1_000_000))

    def _write(
        rows: Sequence[Mapping[str, str]] = HAM_ROWS,
        columns: Sequence[str] = HAM_COLUMNS,
        skip_images: Sequence[str] = (),
        extra_images: Sequence[tuple[str, str]] = (),
    ) -> Path:
        root = tmp_path / f"ham{next(calls)}"
        for directory in ham_layout.image_dirs:
            (root / directory).mkdir(parents=True, exist_ok=True)
        with (root / ham_layout.metadata_file).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        written: set[str] = set()
        for row in rows:
            image_id = row.get("image_id", "")
            if image_id and image_id not in skip_images and image_id not in written:
                directory = ham_layout.image_dirs[zlib.crc32(image_id.encode()) % 2]
                _write_image(root / directory / f"{image_id}{ham_layout.image_suffix}", image_id)
                written.add(image_id)
        for directory, image_id in extra_images:
            _write_image(root / directory / f"{image_id}{ham_layout.image_suffix}", image_id)
        return root

    return _write


@pytest.fixture(scope="session")
def preprocessing_spec() -> ImagePreprocessingSpec:
    return ImagePreprocessingSpec(
        pipeline_id="prep-test",
        output_height=224,
        output_width=224,
        color_mode="RGB",
        resize_mode=ResizeMode.EXACT,
        resize_shorter_side=None,
        interpolation=Interpolation.BILINEAR,
        antialias=True,
        normalization=Normalization.TORCHVISION_IMAGENET1K_V1,
    )


@pytest.fixture(scope="session")
def model_snapshot(
    tmp_path_factory: pytest.TempPathFactory, preprocessing_spec: ImagePreprocessingSpec
) -> tuple[Path, ModelSnapshotRef]:
    directory = tmp_path_factory.mktemp("model")
    ref = write_initialization_snapshot(
        directory,
        model_id="resnet18-init-test",
        preprocessing=preprocessing_spec,
        seed=7,
    )
    return directory / MANIFEST_FILE, ref


# --- Synthetischer Datensatz in voller Baseline-Größe ------------------------------------------

CANONICAL_LAYOUT = Ham10000Layout(
    metadata_file="HAM10000_metadata.csv", image_dirs=("part_1", "part_2"), image_suffix=".png"
)
_DX_CODES = tuple(DX_TO_CLASS)


def write_canonical_dataset(
    root: Path,
    *,
    n_samples: int = 10015,
    n_lesions: int = 7470,
    n_classes: int = 7,
    image_dirs: tuple[str, str] = ("part_1", "part_2"),
    reverse_rows: bool = False,
    reverse_columns: bool = False,
) -> Path:
    """Synthetischer Datensatz; Inhalt hängt nur von den Zählparametern ab, nicht vom Ort."""
    for directory in image_dirs:
        (root / directory).mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(n_samples):
        lesion = index if index < n_lesions else index % n_lesions
        image_id = f"ISIC_S{index:07d}"
        rows.append(
            {
                "lesion_id": f"HAM_L{lesion:07d}",
                "image_id": image_id,
                "dx": _DX_CODES[lesion % n_classes],
                "dx_type": "histo",
                "age": str(float(20 + lesion % 60)),
                "sex": ("female", "male")[lesion % 2],
                "localization": "back",
            }
        )
        color = (index % 256, (index // 256) % 256, (index // 65536) % 256)
        Image.new("RGB", (2, 2), color=color).save(
            root / image_dirs[index % 2] / f"{image_id}.png", format="PNG"
        )
    columns = list(HAM_COLUMNS)[::-1] if reverse_columns else list(HAM_COLUMNS)
    with (root / CANONICAL_LAYOUT.metadata_file).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows[::-1] if reverse_rows else rows)
    return root


@pytest.fixture(scope="session")
def canonical_dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_canonical_dataset(tmp_path_factory.mktemp("canonical"))


@pytest.fixture
def canonical_layout() -> Ham10000Layout:
    return CANONICAL_LAYOUT


@pytest.fixture
def canonical_writer() -> Callable[..., Path]:
    return write_canonical_dataset


# --- Model-Development-Fixtures (P3-02C) ---------------------------------------------------

TRAINING_LESIONS_PER_CLASS = {"mel": 5, "nv": 8, "bcc": 5, "akiec": 5, "bkl": 5, "df": 5, "vasc": 5}
TRAINING_LAYOUT = Ham10000Layout(
    metadata_file="metadata.csv", image_dirs=("images",), image_suffix=".png"
)


def write_training_fixture(root: Path) -> Path:
    """Kleiner, bewusst unbalancierter Datensatz (1 Bild je Läsion, Farbe je Klasse)."""
    (root / "images").mkdir(parents=True)
    rows = []
    for class_index, (dx, n_lesions) in enumerate(TRAINING_LESIONS_PER_CLASS.items()):
        for lesion in range(n_lesions):
            image_id = f"TR_{dx}_{lesion}"
            rows.append({"lesion_id": f"L_{dx}_{lesion}", "image_id": image_id, "dx": dx})
            color = (30 * class_index, 255 - 30 * class_index, 10 * lesion)
            Image.new("RGB", (32, 24), color=color).save(root / "images" / f"{image_id}.png")
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["lesion_id", "image_id", "dx"])
        writer.writeheader()
        writer.writerows(rows)
    return root


@pytest.fixture(scope="session")
def fake_pretrained_weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Test-Ersatz für die offiziellen Gewichte (ResNet-18 mit 1000 Klassen, zufällig)."""
    path = tmp_path_factory.mktemp("pretrained") / "resnet18-test.pth"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        network = resnet18(weights=None)
    torch.save(network.state_dict(), path)
    return path


class RecordingSource:
    """Datenquelle, die jede angefragte sample_id protokolliert (Zugriffsnachweis).

    `snapshot_ref` ist eine Test-Identität (kein offizieller HAM10000-Snapshot).
    """

    def __init__(self, store: Any, dataset_ref: DatasetSnapshotRef) -> None:
        self.store = store
        self.dataset_ref = dataset_ref
        self.accessed: set[str] = set()

    def snapshot_ref(self) -> DatasetSnapshotRef:
        return self.dataset_ref

    def load_inputs(self, access: Any, sample_ids: Sequence[str]) -> Sequence[Image.Image]:
        self.accessed.update(sample_ids)
        return self.store.load_inputs(access, sample_ids)  # type: ignore[no-any-return]

    def labels_for(self, access: Any, sample_ids: Sequence[str]) -> Mapping[str, int]:
        self.accessed.update(sample_ids)
        return self.store.labels_for(access, sample_ids)  # type: ignore[no-any-return]


def make_training_config(
    split_ref: SplitManifestRef,
    weights: Path,
    dataset_ref: DatasetSnapshotRef,
    **overrides: Any,
) -> ModelDevelopmentConfig:
    data: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "md-run-test",
        "model_id": "resnet18-md-test",
        "dataset": dataset_ref.model_dump(),
        "split": split_ref.model_dump(),
        "split_generation": {"manifest_id": "md-split", "split_seed": 5, "train_fraction": 0.5},
        "model_seed": 13,
        "resize": {
            "strategy": "DIRECT_RESIZE_224",
            "interpolation": "BILINEAR",
            "antialias": True,
        },
        "batch_size": 4,
        "learning_rate": 0.01,
        "optimizer": {"kind": "SGD", "weight_decay": 0.0, "momentum": 0.9, "betas": None},
        "lr_scheduler": {"kind": "STEP_LR", "step_size": 7, "gamma": 0.1},
        "max_epochs": 2,
        "early_stopping": {
            "monitor": "VALIDATION_BALANCED_ACCURACY",
            "patience": 5,
            "min_delta": 0.0,
        },
        "augmentation": {
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.5,
            "random_rotate_90": True,
        },
        "device": "cpu",
        "pretrained_weights": {"path": str(weights), "sha256": sha256_file(weights)},
        "output_dir": "outputs/md-run-test",
    }
    data.update(overrides)
    return ModelDevelopmentConfig.model_validate(data)


@dataclass
class TrainingRun:
    root: Path
    store: Any
    source: RecordingSource
    manifest: SplitManifest
    labels: dict[str, str]
    config: ModelDevelopmentConfig
    guard: HeldOutGuard
    train_permit: PartitionAccessPermit
    validation_permit: PartitionAccessPermit
    trained: TrainedModel


@pytest.fixture(scope="session")
def training_run(
    tmp_path_factory: pytest.TempPathFactory, fake_pretrained_weights: Path
) -> Iterator[TrainingRun]:
    root = write_training_fixture(tmp_path_factory.mktemp("training"))
    parsed = _parse_directory(root, TRAINING_LAYOUT)
    store = _AccessGatedImageStore(root, parsed)
    guard = HeldOutGuard()
    manifest = generate_split_from_source(
        store,
        guard,
        SplitGenerationConfig(manifest_id="md-split", split_seed=5, train_fraction=0.5),
        DEMONSTRATOR_CLASS_ORDER,
    )
    labels = {  # unabhängig aus der Fixture-Definition (TR_<dx>_<n>)
        r.sample_id: DX_TO_CLASS[r.sample_id.split("_")[1]] for r in store.sample_refs()
    }
    dataset_ref = DatasetSnapshotRef(
        snapshot_id="TEST_FIXTURE", uri="local:TEST_FIXTURE", sha256=_content_sha256(parsed)
    )
    config = make_training_config(
        split_manifest_ref(manifest), fake_pretrained_weights, dataset_ref
    )
    train_permit = guard.issue_model_development_permit(ModelDevelopmentRole.TRAIN, manifest)
    validation_permit = guard.issue_model_development_permit(
        ModelDevelopmentRole.MODEL_VALIDATION, manifest
    )
    source = RecordingSource(store, dataset_ref)
    prefix = sha256_file(fake_pretrained_weights)[:8]
    with pytest.MonkeyPatch.context() as patch:
        # Nur während des Trainings gilt die Test-Datei als "offizielle" Gewichtsdatei.
        patch.setattr(pretrained_module, "_official_sha256_prefix", lambda: prefix)
        trained = ModelDevelopmentTrainer(config, source, manifest).train(
            train_permit, validation_permit
        )
    yield TrainingRun(
        root=root,
        store=store,
        source=source,
        manifest=manifest,
        labels=labels,
        config=config,
        guard=guard,
        train_permit=train_permit,
        validation_permit=validation_permit,
        trained=trained,
    )
