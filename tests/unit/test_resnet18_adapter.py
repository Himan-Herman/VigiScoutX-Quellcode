"""ResNet-18-ModelPort-Adapter (P3-02B)."""

import inspect
import json
import math
import re
import shutil
from pathlib import Path

import pytest
import torch
from PIL import Image

from vigiscoutx.adapters import resnet18 as resnet18_module
from vigiscoutx.adapters.demonstrator import DEMONSTRATOR_CLASS_ORDER
from vigiscoutx.adapters.preprocessing import ImagePreprocessingSpec, ImagePreprocessor
from vigiscoutx.adapters.resnet18 import (
    EMBEDDING_DIM,
    MANIFEST_FILE,
    NUM_CLASSES,
    ModelSnapshotManifest,
    ResNet18ModelAdapter,
    TrainingStatus,
    write_initialization_snapshot,
)
from vigiscoutx.domain.hashing import canonical_json_bytes, canonical_sha256
from vigiscoutx.domain.refs import ModelSnapshotRef
from vigiscoutx.errors import ModelSnapshotError, PhaseViolationError
from vigiscoutx.ports import ModelPort

REPRODUCIBILITY_ATOL = 1e-6
Snapshot = tuple[Path, ModelSnapshotRef]


def images(count: int = 3) -> list[Image.Image]:
    return [Image.new("RGB", (60 + i, 45), color=(40 * i, 120, 200 - 30 * i)) for i in range(count)]


def adapter(
    snapshot: Snapshot, spec: ImagePreprocessingSpec, **overrides: object
) -> ResNet18ModelAdapter:
    path, ref = snapshot
    options: dict[str, object] = {
        "expected": ref,
        "class_order": DEMONSTRATOR_CLASS_ORDER,
        "preprocessor": ImagePreprocessor(spec),
        "batch_size": 2,
        "allow_untrained": True,
    }
    options.update(overrides)
    return ResNet18ModelAdapter.from_snapshot(path, **options)  # type: ignore[arg-type]


def copy_snapshot(snapshot: Snapshot, tmp_path: Path) -> Path:
    target = tmp_path / "copy"
    shutil.copytree(snapshot[0].parent, target)
    return target / MANIFEST_FILE


def test_predict_proba_returns_seven_normalised_probabilities(
    model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    model: ModelPort[Image.Image] = adapter(model_snapshot, preprocessing_spec)
    batch = model.predict_proba("w1", ["a", "b", "c"], images())
    assert batch.sample_ids == ("a", "b", "c")
    assert all(len(row) == NUM_CLASSES == 7 for row in batch.probabilities)
    assert all(math.isclose(sum(row), 1.0, abs_tol=1e-9) for row in batch.probabilities)


def test_embed_returns_penultimate_representation(
    model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    batch = adapter(model_snapshot, preprocessing_spec).embed("w1", ["a", "b", "c"], images())
    assert len(batch.embeddings) == 3
    assert {len(row) for row in batch.embeddings} == {EMBEDDING_DIM}
    assert all(value >= 0.0 for row in batch.embeddings for value in row)  # nach ReLU/AvgPool


def test_model_is_in_eval_mode_without_gradients(
    model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    loaded = adapter(model_snapshot, preprocessing_spec)
    assert loaded.in_eval_mode
    assert not any(p.requires_grad for p in loaded._network.parameters())


def test_same_inputs_and_snapshot_give_same_outputs(
    model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    first = adapter(model_snapshot, preprocessing_spec)
    second = adapter(model_snapshot, preprocessing_spec, batch_size=3)
    a = torch.tensor(first.predict_proba("w", ["a", "b", "c"], images()).probabilities)
    b = torch.tensor(second.predict_proba("w", ["a", "b", "c"], images()).probabilities)
    assert torch.allclose(a, b, atol=REPRODUCIBILITY_ATOL, rtol=0.0)
    e1 = torch.tensor(first.embed("w", ["a", "b", "c"], images()).embeddings)
    e2 = torch.tensor(second.embed("w", ["a", "b", "c"], images()).embeddings)
    assert torch.allclose(e1, e2, atol=REPRODUCIBILITY_ATOL, rtol=0.0)


def test_initialization_snapshot_is_reproducible(
    tmp_path: Path, model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    ref = write_initialization_snapshot(
        tmp_path / "again",
        model_id="resnet18-init-test",
        preprocessing=preprocessing_spec,
        seed=7,
    )
    original = torch.load(model_snapshot[0].parent / "weights.pt", weights_only=True)
    again = torch.load(tmp_path / "again" / "weights.pt", weights_only=True)
    assert original.keys() == again.keys()
    assert all(torch.equal(original[k], again[k]) for k in original)
    assert ref.model_id == model_snapshot[1].model_id


def test_model_adapter_is_inference_only() -> None:
    public = {
        name for name, _ in inspect.getmembers(ResNet18ModelAdapter) if not name.startswith("_")
    }
    forbidden = re.compile(r"train|fit|update|retrain|mutate|optim|backward|step|load_state|set_")
    assert not [name for name in public if forbidden.search(name)]
    assert {"predict_proba", "embed", "snapshot_ref"} <= public


def test_untrained_snapshot_requires_explicit_permission(
    model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    with pytest.raises(ModelSnapshotError, match="kein finales Modell"):
        adapter(model_snapshot, preprocessing_spec, allow_untrained=False)


def test_wrong_class_order_fails(
    model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    reordered = tuple(sorted(DEMONSTRATOR_CLASS_ORDER))
    with pytest.raises(ModelSnapshotError, match="class_order"):
        adapter(model_snapshot, preprocessing_spec, class_order=reordered)


def test_wrong_expected_reference_fails(
    model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    ref = model_snapshot[1]
    for changed in (
        ref.model_copy(update={"sha256": "0" * 64}),
        ref.model_copy(update={"model_id": "other-model"}),
    ):
        with pytest.raises(ModelSnapshotError):
            adapter(model_snapshot, preprocessing_spec, expected=changed)


def test_other_preprocessing_fails(
    model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    other = preprocessing_spec.model_copy(update={"pipeline_id": "prep-other"})
    with pytest.raises(ModelSnapshotError, match="preprocessing_sha256"):
        adapter(model_snapshot, preprocessing_spec, preprocessor=ImagePreprocessor(other))


def test_tampered_weights_fail(
    tmp_path: Path, model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    manifest_path = copy_snapshot(model_snapshot, tmp_path)
    weights = manifest_path.parent / "weights.pt"
    weights.write_bytes(weights.read_bytes() + b"\0")
    with pytest.raises(ModelSnapshotError, match="Gewichtsdatei"):
        adapter((manifest_path, model_snapshot[1]), preprocessing_spec)


def test_tampered_manifest_metadata_fails(
    tmp_path: Path, model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    manifest_path = copy_snapshot(model_snapshot, tmp_path)
    manifest = ModelSnapshotManifest.model_validate(json.loads(manifest_path.read_bytes()))
    changed = manifest.model_copy(update={"torch_version": "0.0.0"})
    manifest_path.write_bytes(canonical_json_bytes(changed))
    ref = model_snapshot[1].model_copy(update={"sha256": canonical_sha256(changed)})
    with pytest.raises(ModelSnapshotError, match="torch_version"):
        adapter((manifest_path, ref), preprocessing_spec, expected=ref)

    manifest_path.write_bytes(json.dumps(json.loads(manifest_path.read_bytes()), indent=2).encode())
    with pytest.raises(ModelSnapshotError, match="kanonisch"):
        adapter((manifest_path, ref), preprocessing_spec, expected=ref)


def test_manifest_requires_seven_classes_and_resnet18(model_snapshot: Snapshot) -> None:
    manifest = ModelSnapshotManifest.model_validate(json.loads(model_snapshot[0].read_bytes()))
    for update in (
        {"class_order": DEMONSTRATOR_CLASS_ORDER[:6]},
        {"num_classes": 8},
        {"architecture": "resnet50"},
    ):
        with pytest.raises(ValueError):
            manifest.model_copy(update=update)


def test_no_pretrained_download_happens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    torch_home = tmp_path / "torch_home"
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    ref = write_initialization_snapshot(
        tmp_path / "snap",
        model_id="resnet18-no-download",
        preprocessing=preprocessing_spec,
        seed=1,
    )
    adapter((tmp_path / "snap" / MANIFEST_FILE, ref), preprocessing_spec, expected=ref)
    assert not torch_home.exists() or not any(torch_home.rglob("*"))


def test_input_alignment_is_checked(
    model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    loaded = adapter(model_snapshot, preprocessing_spec)
    with pytest.raises(ValueError):
        loaded.predict_proba("w", ["a", "b"], images(3))
    with pytest.raises(ValueError):
        loaded.embed("w", [], [])


def _write_manifest(directory: Path, payload: dict[str, object]) -> ModelSnapshotRef:
    (directory / MANIFEST_FILE).write_bytes(canonical_json_bytes(payload))
    return ModelSnapshotRef(
        model_id=str(payload["model_id"]),
        architecture="resnet18",
        uri=f"local:{payload['model_id']}",
        sha256=canonical_sha256(payload),
    )


def test_consistent_but_wrong_class_order_is_rejected(
    tmp_path: Path, model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    manifest_path = copy_snapshot(model_snapshot, tmp_path)
    payload = json.loads(manifest_path.read_bytes())
    alphabetical = sorted(DEMONSTRATOR_CLASS_ORDER)
    payload["class_order"] = alphabetical
    ref = _write_manifest(manifest_path.parent, payload)
    for class_order in (tuple(alphabetical), DEMONSTRATOR_CLASS_ORDER):
        with pytest.raises(ModelSnapshotError, match=r"class_order|Klassenreihenfolge"):
            adapter((manifest_path, ref), preprocessing_spec, expected=ref, class_order=class_order)


def test_direct_constructor_is_blocked(
    model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    loaded = adapter(model_snapshot, preprocessing_spec)
    with pytest.raises(PhaseViolationError, match="from_snapshot"):
        ResNet18ModelAdapter(
            loaded.manifest,
            loaded._network,
            ImagePreprocessor(preprocessing_spec),
            2,
            _factory_token=object(),
        )
    with pytest.raises(TypeError):
        ResNet18ModelAdapter(  # type: ignore[call-arg]
            loaded.manifest, loaded._network, ImagePreprocessor(preprocessing_spec), 2
        )


def test_final_fine_tuned_snapshot_loads_without_untrained_permission(
    tmp_path: Path, model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    manifest_path = copy_snapshot(model_snapshot, tmp_path)
    history = manifest_path.parent / "training_result.json"
    history.write_bytes(canonical_json_bytes({"history": "test"}))
    payload = json.loads(manifest_path.read_bytes())
    payload.update(
        {
            "model_id": "resnet18-final-test",
            "training_status": TrainingStatus.FINE_TUNED.value,
            "initialization_seed": None,
            "provenance": {
                "training_config_sha256": "1" * 64,
                "training_result_sha256": canonical_sha256({"history": "test"}),
                "split_manifest_sha256": "3" * 64,
                "dataset_snapshot_sha256": "4" * 64,
                "pretrained_weights_sha256": "5" * 64,
                "resize_strategy": "DIRECT_RESIZE_224",
                "model_seed": 11,
                "git_commit": "a" * 40,
            },
        }
    )
    ref = _write_manifest(manifest_path.parent, payload)
    loaded = adapter((manifest_path, ref), preprocessing_spec, expected=ref, allow_untrained=False)
    assert loaded.manifest.training_status is TrainingStatus.FINE_TUNED
    assert loaded.snapshot_ref() == ref


def test_untrained_snapshot_loads_only_with_explicit_permission(
    model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    loaded = adapter(model_snapshot, preprocessing_spec, allow_untrained=True)
    assert loaded.manifest.training_status is TrainingStatus.UNTRAINED_INITIALIZATION
    with pytest.raises(ModelSnapshotError):
        adapter(model_snapshot, preprocessing_spec, allow_untrained=False)


def test_no_alternative_public_factory_creates_adapters() -> None:
    factories = [
        name
        for name, member in inspect.getmembers(ResNet18ModelAdapter)
        if not name.startswith("_")
        and isinstance(
            inspect.getattr_static(ResNet18ModelAdapter, name), classmethod | staticmethod
        )
    ]
    assert factories == ["from_snapshot"]
    module_functions = [
        name
        for name, member in inspect.getmembers(resnet18_module, inspect.isfunction)
        if member.__module__ == resnet18_module.__name__ and not name.startswith("_")
    ]
    assert module_functions == ["snapshot_ref_for_manifest", "write_initialization_snapshot"]
    for name in module_functions:
        annotation = inspect.signature(getattr(resnet18_module, name)).return_annotation
        assert annotation in (ModelSnapshotRef, "ModelSnapshotRef")


# --- Bildzustand vor der Normalisierung (Shift-Pfad, P4-02E) -----------------------------------


def test_preprocessed_state_gives_identical_model_input_and_outputs(
    model_snapshot: Snapshot, preprocessing_spec: ImagePreprocessingSpec
) -> None:
    from vigiscoutx.adapters.preprocessing import PreprocessedImage

    preprocessor = ImagePreprocessor(preprocessing_spec)
    pil = images()
    states = [PreprocessedImage(preprocessor.unnormalized(image)) for image in pil]
    assert torch.equal(preprocessor.batch(pil), preprocessor.batch(states))
    loaded = adapter(model_snapshot, preprocessing_spec)
    ids = ["a", "b", "c"]
    assert loaded.predict_proba("w", ids, pil) == loaded.predict_proba("w", ids, states)
    assert loaded.embed("w", ids, pil) == loaded.embed("w", ids, states)


def test_preprocessed_state_is_validated_and_immutable() -> None:
    from vigiscoutx.adapters.preprocessing import PreprocessedImage

    with pytest.raises(ValueError, match="float32"):
        PreprocessedImage(torch.zeros((3, 224, 224), dtype=torch.float64))
    with pytest.raises(ValueError, match="float32"):
        PreprocessedImage(torch.zeros((3, 32, 32), dtype=torch.float32))
    with pytest.raises(ValueError, match="endliche"):
        PreprocessedImage(torch.full((3, 224, 224), float("nan"), dtype=torch.float32))
    source = torch.zeros((3, 224, 224), dtype=torch.float32)
    state = PreprocessedImage(source)
    source[0, 0, 0] = 1.0
    state.tensor[0, 0, 1] = 1.0
    assert float(state.tensor.abs().sum()) == 0.0
    with pytest.raises(AttributeError):
        state._tensor = source
