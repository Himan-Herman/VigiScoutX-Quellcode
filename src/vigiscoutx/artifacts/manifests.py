"""ArtifactManifest (ARC-045, DEC-032): Index der Run-Artefakte mit SHA-256."""

from pathlib import PurePosixPath
from typing import Self

from pydantic import Field, field_validator, model_validator

from vigiscoutx.domain.types import FrozenModel, Identifier, Sha256Hex
from vigiscoutx.errors import IntegrityError


def canonical_artifact_path(path: str) -> str:
    """Relativer, kanonischer POSIX-Pfad; `.`-Segmente und doppelte `/` werden normalisiert."""
    if "\\" in path:
        raise ValueError("Artefaktpfade verwenden '/' als Trenner")
    pure = PurePosixPath(path)
    if pure.is_absolute():
        raise ValueError("Artefaktpfade müssen relativ sein")
    if ".." in pure.parts:
        raise ValueError("Artefaktpfade dürfen kein '..' enthalten")
    canonical = pure.as_posix()
    if canonical in {"", "."}:
        raise ValueError("Artefaktpfad ist leer")
    return canonical


class ArtifactEntry(FrozenModel):
    path: str = Field(min_length=1, strict=True)
    sha256: Sha256Hex
    role: Identifier

    @field_validator("path")
    @classmethod
    def _canonical(cls, path: str) -> str:
        return canonical_artifact_path(path)


class ArtifactManifest(FrozenModel):
    run_id: Identifier
    entries: tuple[ArtifactEntry, ...]

    @model_validator(mode="after")
    def _unique_paths(self) -> Self:
        paths = [e.path for e in self.entries]
        if len(paths) != len(set(paths)):
            raise ValueError("Artefaktpfade müssen eindeutig sein")
        return self

    def with_entry(self, entry: ArtifactEntry) -> "ArtifactManifest":
        """Neues Manifest mit zusätzlichem Eintrag; bestehende Einträge werden nie ersetzt."""
        if any(e.path == entry.path for e in self.entries):
            raise IntegrityError(f"Artefakt existiert bereits, kein Überschreiben: {entry.path}")
        return ArtifactManifest(run_id=self.run_id, entries=(*self.entries, entry))
