"""Synthetische End-to-End-Welt für den Calibration-Runner (P4-02E; keine echten Daten).

- Datensatz: synthetischer Snapshot in kanonischer Baseline-Größe (10.015 Bilder, 7.470 Läsionen),
  geladen über den echten `Ham10000Dataset.from_directory`. Nur die CALIBRATION-Läsionen tragen
  224 × 224-Texturbilder; alle übrigen Bilder sind 2 × 2-Platzhalter.
- Jede Klasse besitzt eine eigene Schachbrett-Textur (Periode, Amplitude); jede fünfte Läsion
  trägt die Textur der Folgeklasse („schwierige“ Läsion → systematische Fehlklassifikation).
- Modell: deterministisches Texturmodell (Standardabweichung und Gradientenenergie der Luma) mit
  Nearest-Template-Softmax und linearer 512-d-Projektion; implementiert den Modell-Port des Runners
  auf `PreprocessedImage`-Zuständen. Kein ResNet, kein Training.

Alle Zahlen sind reine Testwerte ohne wissenschaftliche Bedeutung.
"""

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from calibration_runner.authorization import SYNTHETIC_FIXTURE_COLUMN, SyntheticExecution
from calibration_runner.freeze_assembly import config_snapshot_hashes, method_refs, template_methods
from signal_factories import g1_config, g2_config, g3_config
from vigiscoutx.adapters.ham10000 import (
    DX_TO_CLASS,
    Ham10000Dataset,
    Ham10000Layout,
    SampleMetadata,
)
from vigiscoutx.adapters.preprocessing import (
    ImagePreprocessingSpec,
    ImagePreprocessor,
    PreprocessedImage,
)
from vigiscoutx.domain.hashing import canonical_sha256
from vigiscoutx.domain.refs import DatasetSnapshotRef, ModelSnapshotRef, SampleRef
from vigiscoutx.domain.windows import ModelOutputBatch, RepresentationBatch
from vigiscoutx.experiment.calibration.config import (
    CalibrationPool,
    CalibrationProtocolConfig,
    baseline_protocol_config,
)
from vigiscoutx.experiment.calibration.process import CalibrationRunProvenance
from vigiscoutx.experiment.calibration.structure import CalibrationLesion, assign_pools
from vigiscoutx.experiment.run_config import RunConfig
from vigiscoutx.experiment.split import SplitAssignment, SplitManifest
from vigiscoutx.experiment.split_integrity import split_manifest_sha256
from vigiscoutx.experiment.splitting import split_manifest_ref
from vigiscoutx.experiment.types import ModelDevelopmentRole, Partition
from vigiscoutx.ports import AccessPurpose, DataAccessContext

PROTOCOL = baseline_protocol_config()
CLASSES = PROTOCOL.class_order
LAYOUT = Ham10000Layout(
    metadata_file="HAM10000_metadata.csv", image_dirs=("part_1", "part_2"), image_suffix=".png"
)
N_SAMPLES, N_LESIONS = 10015, 7470
CALIBRATION_LESIONS: Mapping[str, int] = {
    "MEL": 100,
    "NV": 600,
    "BCC": 50,
    "AKIEC": 30,
    "BKL": 100,
    "DF": 12,
    "VASC": 15,
}
PERIODS = (2, 3, 4, 6, 8, 12, 16)
AMPLITUDES = (0.10, 0.14, 0.18, 0.22, 0.12, 0.16, 0.20)
HARD_EVERY = 5
DOUBLE_IMAGE_LESIONS = 300
HELD_OUT_LESIONS = 1500
_CODE = {cls: dx for dx, cls in DX_TO_CLASS.items()}
GIT_COMMIT = "c" * 40
ENV_LOCK = "d" * 64


def texture(class_index: int, brightness: float = 0.0) -> Image.Image:
    """224 × 224-RGB-Schachbrett der Klasse (Periode und Amplitude klassenspezifisch)."""
    period, amplitude = PERIODS[class_index], AMPLITUDES[class_index]
    y, x = np.mgrid[0:224, 0:224]
    board = ((x // period + y // period) % 2) * 2 - 1
    luma = 0.5 + amplitude * board + 0.01 * class_index + brightness
    rgb = np.stack([luma * 0.95 + 0.02, luma, luma * 0.9 + 0.04], axis=-1)
    return Image.fromarray(np.clip(rgb * 255, 0, 255).round().astype(np.uint8))


@dataclass(frozen=True, slots=True)
class LesionSpec:
    lesion_id: str
    class_id: str
    partition: Partition
    sample_ids: tuple[str, ...]
    texture_class: int | None
    sex: str


def _lesions(
    *, calibration: Mapping[str, int], held_out_textures: bool = False
) -> list[LesionSpec]:
    lesions: list[LesionSpec] = []
    image = 0
    total_cal = sum(calibration.values())
    for class_id, count in calibration.items():
        k = CLASSES.index(class_id)
        for i in range(count):
            double = len(lesions) < DOUBLE_IMAGE_LESIONS
            ids = tuple(f"ISIC_C{image + j:07d}" for j in range(2 if double else 1))
            image += len(ids)
            hard = i % HARD_EVERY == 0
            lesions.append(
                LesionSpec(
                    lesion_id=f"CAL_{len(lesions):05d}",  # klassenneutral
                    class_id=class_id,
                    partition=Partition.CALIBRATION,
                    sample_ids=ids,
                    texture_class=(k + 1) % len(CLASSES) if hard else k,
                    sex=("male", "female")[(i // 2) % 2],
                )
            )
    others = N_LESIONS - total_cal
    extra = N_SAMPLES - image - others
    for n in range(others):
        count = 2 if n < extra else 1
        ids = tuple(f"ISIC_O{image + j:07d}" for j in range(count))
        image += count
        partition = Partition.HELD_OUT if n < HELD_OUT_LESIONS else Partition.MODEL_DEVELOPMENT
        k = n % len(CLASSES)
        textured = held_out_textures and partition is Partition.HELD_OUT
        lesions.append(
            LesionSpec(
                lesion_id=f"OTH_{n:07d}",
                class_id=CLASSES[k],
                partition=partition,
                sample_ids=ids,
                # Optional (P4-03AI): HELD_OUT mit Klassentextur, jede fünfte Läsion „schwierig“.
                texture_class=((k + 1) % len(CLASSES) if n % HARD_EVERY == 0 else k)
                if textured
                else None,
                sex=("male", "female")[n % 2],
            )
        )
    if image != N_SAMPLES or len(lesions) != N_LESIONS:
        raise AssertionError("synthetische Welt verletzt die Baseline-Größe")
    return lesions


@dataclass(frozen=True, slots=True)
class World:
    root: Path
    lesions: tuple[LesionSpec, ...]
    split: SplitManifest

    def partition_samples(self, partition: Partition) -> frozenset[str]:
        return frozenset(
            s for lesion in self.lesions if lesion.partition is partition for s in lesion.sample_ids
        )


def _split(lesions: Sequence[LesionSpec]) -> SplitManifest:
    assignments = []
    for n, lesion in enumerate(lesions):
        role = None
        if lesion.partition is Partition.MODEL_DEVELOPMENT:
            role = (ModelDevelopmentRole.TRAIN, ModelDevelopmentRole.MODEL_VALIDATION)[n % 2]
        assignments += [
            SplitAssignment(
                sample_id=s,
                group_id=lesion.lesion_id,
                partition=lesion.partition,
                model_development_role=role,
            )
            for s in lesion.sample_ids
        ]
    return SplitManifest(manifest_id="synthetic-e2e-split", assignments=tuple(assignments))


def audit_lesions(lesions: Sequence[LesionSpec], split: SplitManifest) -> frozenset[str]:
    """CAL_AUDIT-Läsionen nach dem eingefrorenen Pool-Algorithmus (nur zum Präparieren der
    synthetischen Testwelt, nicht im Runner)."""
    inventory = [
        CalibrationLesion(lesion_id=x.lesion_id, class_id=x.class_id, sample_ids=x.sample_ids)
        for x in lesions
        if x.partition is Partition.CALIBRATION
    ]
    pools = assign_pools(inventory, split_manifest_sha256(split), PROTOCOL)
    return pools.lesion_ids(CalibrationPool.CAL_AUDIT)


def write_world(
    root: Path,
    *,
    calibration: Mapping[str, int] = CALIBRATION_LESIONS,
    audit_brightness: float = 0.0,
    synthetic_marker: bool = True,
    held_out_textures: bool = False,
) -> World:
    """`audit_brightness ≠ 0` verschiebt ausschließlich die CAL_AUDIT-Bilder (Audit-FAR-Test)."""
    lesions = _lesions(calibration=calibration, held_out_textures=held_out_textures)
    split = _split(lesions)
    shifted = audit_lesions(lesions, split) if audit_brightness else frozenset()
    for directory in LAYOUT.image_dirs:
        (root / directory).mkdir(parents=True, exist_ok=True)
    textures = [texture(k) for k in range(len(CLASSES))]
    audit_textures = [texture(k, audit_brightness) for k in range(len(CLASSES))]
    placeholder = Image.new("RGB", (2, 2), color=(128, 128, 128))
    rows: list[dict[str, str]] = []
    for index, lesion in enumerate(lesions):
        for sample_id in lesion.sample_ids:
            rows.append(
                {
                    "lesion_id": lesion.lesion_id,
                    "image_id": sample_id,
                    "dx": _CODE[lesion.class_id],
                    "dx_type": "histo",
                    "age": "50.0",
                    "sex": lesion.sex,
                    "localization": "back",
                    **({SYNTHETIC_FIXTURE_COLUMN: "1"} if synthetic_marker else {}),
                }
            )
            palette = audit_textures if lesion.lesion_id in shifted else textures
            image = placeholder if lesion.texture_class is None else palette[lesion.texture_class]
            directory = LAYOUT.image_dirs[index % 2]
            image.save(root / directory / f"{sample_id}{LAYOUT.image_suffix}", format="PNG")
    with (root / LAYOUT.metadata_file).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return World(root=root, lesions=tuple(lesions), split=split)


class RecordingDataset:
    """Leitet an `Ham10000Dataset` weiter und protokolliert jeden Zugriff (Sample, Zweck, Stufe)."""

    def __init__(self, dataset: Ham10000Dataset) -> None:
        self._dataset = dataset
        self.accesses: list[tuple[str, AccessPurpose, int]] = []
        self.step = 0

    def _log(self, sample_ids: Sequence[str], purpose: AccessPurpose) -> None:
        self.accesses += [(s, purpose, self.step) for s in sample_ids]

    def snapshot_ref(self) -> DatasetSnapshotRef:
        return self._dataset.snapshot_ref()

    def sample_refs(self) -> tuple[SampleRef, ...]:
        return self._dataset.sample_refs()

    def stratification_labels_for(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> Mapping[str, int]:
        result = self._dataset.stratification_labels_for(access, sample_ids)
        self._log(sample_ids, AccessPurpose.STRATIFICATION_LABELS)
        return result

    def load_inputs(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> list[Image.Image]:
        result = self._dataset.load_inputs(access, sample_ids)
        self._log(sample_ids, AccessPurpose.INPUTS)
        return result

    def labels_for(self, access: DataAccessContext, sample_ids: Sequence[str]) -> Mapping[str, int]:
        result = self._dataset.labels_for(access, sample_ids)
        self._log(sample_ids, AccessPurpose.LABELS)
        return result

    def metadata_for(
        self, access: DataAccessContext, sample_ids: Sequence[str]
    ) -> Mapping[str, SampleMetadata]:
        result = self._dataset.metadata_for(access, sample_ids)
        self._log(sample_ids, AccessPurpose.METADATA)
        return result


def preprocessing_spec() -> ImagePreprocessingSpec:
    return ImagePreprocessingSpec.from_config(_preprocessing_config())


def _preprocessing_config() -> Any:
    from vigiscoutx.config.scientific_config import PreprocessingConfig
    from vigiscoutx.domain.frozen_json import FrozenMapping

    return PreprocessingConfig(
        pipeline_id="synthetic-direct-224",
        input_height=224,
        input_width=224,
        color_mode="RGB",
        params=FrozenMapping(
            {
                "resize_mode": "EXACT",
                "resize_shorter_side": None,
                "interpolation": "BILINEAR",
                "antialias": True,
                "normalization": "TORCHVISION_IMAGENET1K_V1",
            }
        ),
    )


def _texture_features(state: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    luma = 0.2126 * state[0] + 0.7152 * state[1] + 0.0722 * state[2]
    gradient = np.abs(np.diff(luma, axis=1)).mean() + np.abs(np.diff(luma, axis=0)).mean()
    return np.array([luma.std(), gradient], dtype=np.float64)


class TextureModel:
    """Deterministisches Testmodell auf `PreprocessedImage`-Zuständen."""

    def __init__(self, preprocessor: ImagePreprocessor, embedding_dim: int = 512) -> None:
        templates = np.array(
            [
                _texture_features(preprocessor.unnormalized(texture(k)).numpy())
                for k in range(len(CLASSES))
            ]
        )
        self._templates = templates
        self._scale = templates.std(axis=0)
        rng = np.random.default_rng(20261016)
        self._projection = rng.standard_normal((embedding_dim, 2))
        self._ref = ModelSnapshotRef(
            model_id="resnet18-pilot-P1",
            architecture="synthetic-texture",
            uri="memory://synthetic-texture-model",
            sha256=canonical_sha256(
                {"templates": templates.tolist(), "embedding_dim": embedding_dim}
            ),
        )
        self.calls = 0
        self._last: tuple[Sequence[PreprocessedImage], npt.NDArray[np.float64]] | None = None

    def snapshot_ref(self) -> ModelSnapshotRef:
        return self._ref

    def _features(self, inputs: Sequence[PreprocessedImage]) -> npt.NDArray[np.float64]:
        """Merkmale je Eingabeliste; `embed` und `predict_proba` derselben Liste teilen sie."""
        if self._last is not None and self._last[0] is inputs:
            return self._last[1]
        self.calls += 1
        features = np.array([_texture_features(i.tensor.numpy()) for i in inputs])
        self._last = (inputs, features)
        return features

    def predict_proba(
        self, window_id: str, sample_ids: Sequence[str], inputs: Sequence[PreprocessedImage]
    ) -> ModelOutputBatch:
        features = self._features(inputs)
        distance = (((features[:, None, :] - self._templates[None]) / self._scale) ** 2).sum(-1)
        logits = -distance / 0.5
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        return ModelOutputBatch(
            window_id=window_id,
            sample_ids=tuple(sample_ids),
            probabilities=tuple(tuple(float(p) for p in row) for row in probabilities),
        )

    def embed(
        self, window_id: str, sample_ids: Sequence[str], inputs: Sequence[PreprocessedImage]
    ) -> RepresentationBatch:
        standardized = (self._features(inputs) - self._templates.mean(axis=0)) / self._scale
        embeddings = standardized @ self._projection.T
        return RepresentationBatch(
            window_id=window_id,
            sample_ids=tuple(sample_ids),
            embeddings=tuple(tuple(float(v) for v in row) for row in embeddings),
        )


def _signal(signal_id: str, group: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "signal_id": signal_id,
        "group": group,
        "method": method,
        "params": params,
        "threshold": None,
    }


def template_data(world: World, dataset: Ham10000Dataset, model: TextureModel) -> dict[str, Any]:
    g1 = g1_config().model_dump(mode="json")
    g1["sigma"] = None
    return {
        "schema_version": 1,
        "run_id": "synthetic-calibration-e2e",
        "phase": "DEVELOPMENT_CALIBRATION",
        "freeze_id": None,
        "scientific": {
            "dataset": dataset.snapshot_ref().model_dump(mode="json"),
            "split": split_manifest_ref(world.split).model_dump(mode="json"),
            "model": {
                "model_id": "resnet18-pilot-P1",
                "architecture": "synthetic-texture",
                "uri": "memory://synthetic-texture-model",
                "sha256": "b" * 64,
            },
            "preprocessing": _preprocessing_config().model_dump(mode="json"),
            "monitoring": {
                "window_size": 128,
                "signals": [
                    _signal("G1_MMD_EMBEDDING", "G1", "MMD_U2_RBF_PERMUTATION", g1),
                    _signal(
                        "G2", "G2", "G2_QUALITY_KS_BONFERRONI", g2_config().model_dump(mode="json")
                    ),
                    _signal(
                        "G3", "G3", "G3_MUKS_CONFIDENCE_KS", g3_config().model_dump(mode="json")
                    ),
                ],
            },
            "fusion": {"rule": None, "groups": ["G1", "G2", "G3"]},
            "relevance": {
                "performance_metric": "BALANCED_ACCURACY",
                "clean_quantile": 0.95,
                "delta_rel": None,
            },
            # Eingefrorene v1.0.0-Policies (P4-03C); Datensatz und Modell bleiben synthetisch.
            "risk_policy": {"policy_id": "vsx-risk-r0-r3", "version": "1.0.0"},
            "recommendation_policy": {"policy_id": "vsx-recommendation-review", "version": "1.0.0"},
        },
        "repetitions": 1,
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


def provenance_for(
    template: RunConfig,
    dataset: Ham10000Dataset,
    model: TextureModel,
    protocol: CalibrationProtocolConfig = PROTOCOL,
    **overrides: Any,
) -> CalibrationRunProvenance:
    data: dict[str, Any] = {
        "run_id": template.run_id,
        "dataset": dataset.snapshot_ref(),
        "split": template.scientific.split,
        "model_id": "resnet18-pilot-P1",
        "model_weights_sha256": template.scientific.model.sha256,
        "model_snapshot_sha256": model.snapshot_ref().sha256,
        "preprocessing_sha256": canonical_sha256(template.scientific.preprocessing),
        "method_refs": method_refs(template_methods(template)),
        "config_snapshot_sha256s": config_snapshot_hashes(template, protocol),
        "git_commit": GIT_COMMIT,
        "environment_lock_sha256": ENV_LOCK,
    }
    data.update(overrides)
    return CalibrationRunProvenance.model_validate(data)


def synthetic_execution(dataset: Ham10000Dataset, model: Any) -> SyntheticExecution:
    """Gebundene synthetische Provenienz genau dieses Fixture-Datensatzes und Testmodells."""
    return SyntheticExecution(
        dataset_snapshot_sha256=dataset.snapshot_ref().sha256,
        model_snapshot_sha256=model.snapshot_ref().sha256,
    )
