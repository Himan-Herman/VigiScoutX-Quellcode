"""Offizielle TorchVision-ImageNet-Gewichte aus lokaler Datei (kein Download).

Geprüft wird wie beim TorchVision-Download: Der SHA-256 der Datei muss mit dem Hash-Präfix im
offiziellen Dateinamen (`ResNet18_Weights.IMAGENET1K_V1.url`) beginnen. Zusätzlich muss er dem in
der Konfiguration festgehaltenen vollständigen SHA-256 entsprechen.
"""

import re
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from vigiscoutx.adapters.resnet18 import EMBEDDING_DIM, NUM_CLASSES
from vigiscoutx.artifacts.files import sha256_file
from vigiscoutx.errors import TrainingError
from vigiscoutx.experiment.model_development.config import PretrainedWeightsRef

_HASH_IN_FILENAME = re.compile(r"-([a-f0-9]+)\.")


def official_imagenet_weights_filename() -> str:
    return PurePosixPath(urlparse(str(ResNet18_Weights.IMAGENET1K_V1.url)).path).name


def _official_sha256_prefix() -> str:
    match = _HASH_IN_FILENAME.search(official_imagenet_weights_filename())
    if match is None:
        raise TrainingError("Offizieller Gewichtsdateiname enthält keinen Hash")
    return match.group(1)


def load_pretrained_resnet18(ref: PretrainedWeightsRef, *, model_seed: int) -> nn.Module:
    """ResNet-18 mit offiziellen ImageNet-Gewichten und neu initialisiertem 7-Klassen-Kopf."""
    path = Path(ref.path)
    if not path.is_file():
        raise TrainingError(
            f"Vortrainierte Gewichte fehlen lokal ({official_imagenet_weights_filename()} "
            "bereitstellen); es wird nichts heruntergeladen"
        )
    actual = sha256_file(path)
    if actual != ref.sha256:
        raise TrainingError("SHA-256 der Gewichtsdatei weicht von der Konfiguration ab")
    if not actual.startswith(_official_sha256_prefix()):
        raise TrainingError(
            "Gewichtsdatei entspricht nicht den offiziellen IMAGENET1K_V1-Gewichten"
        )
    network = cast(nn.Module, resnet18(weights=None))
    state = torch.load(path, map_location="cpu", weights_only=True)
    network.load_state_dict(state, strict=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        head = nn.Linear(EMBEDDING_DIM, NUM_CLASSES)
    network.fc = head
    return network
