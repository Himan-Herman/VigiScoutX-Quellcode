"""YAML-Laden und vollständig aufgelöster Config Snapshot.

Der Loader folgt für Skalare den YAML-1.2-Core-Regeln: Nur `true`/`false` sind Bools
(`yes`/`no`/`on`/`off` bleiben Strings), Ganzzahlen sind nur dezimal (keine Oktal- oder
Sexagesimalwerte) und `5e-2` ist eine Gleitkommazahl.
"""

import re
from pathlib import Path
from typing import Any

import yaml

from vigiscoutx.domain.types import FrozenModel

_BOOL_TAG = "tag:yaml.org,2002:bool"
_INT_TAG = "tag:yaml.org,2002:int"
_FLOAT_TAG = "tag:yaml.org,2002:float"


class _StrictSafeLoader(yaml.SafeLoader):
    pass


_StrictSafeLoader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag not in {_BOOL_TAG, _INT_TAG, _FLOAT_TAG}]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_StrictSafeLoader.add_implicit_resolver(
    _BOOL_TAG, re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)
_StrictSafeLoader.add_implicit_resolver(
    _INT_TAG, re.compile(r"^[-+]?(?:0|[1-9][0-9]*)$"), list("-+0123456789")
)
_StrictSafeLoader.add_implicit_resolver(
    _FLOAT_TAG,
    re.compile(
        r"^[-+]?(?:[0-9]+\.[0-9]*(?:[eE][-+]?[0-9]+)?"
        r"|\.[0-9]+(?:[eE][-+]?[0-9]+)?"
        r"|[0-9]+[eE][-+]?[0-9]+)$"
    ),
    list("-+0123456789."),
)


def parse_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_StrictSafeLoader)


def parse_model[M: FrozenModel](text: str, model: type[M]) -> M:
    data = parse_yaml(text)
    if not isinstance(data, dict):
        raise ValueError("Konfigurations-YAML muss ein Mapping sein")
    return model.model_validate(data)


def load_model[M: FrozenModel](path: Path, model: type[M]) -> M:
    return parse_model(path.read_text(encoding="utf-8"), model)


def resolved_snapshot(config: FrozenModel) -> dict[str, Any]:
    """Vollständig aufgelöste, JSON-kompatible Darstellung aller Felder."""
    return config.model_dump(mode="json")


def dump_resolved_snapshot(config: FrozenModel) -> str:
    return yaml.safe_dump(resolved_snapshot(config), sort_keys=True, allow_unicode=True)


def write_resolved_snapshot(config: FrozenModel, path: Path) -> None:
    """Schreibt den Snapshot; eine vorhandene Datei wird nie überschrieben."""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(dump_resolved_snapshot(config))
