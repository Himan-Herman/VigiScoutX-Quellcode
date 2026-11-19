"""Rekursiv unveränderliche JSON-Struktur für wissenschaftlich wirksame Parameter.

Objekte werden zu `FrozenMapping`, Listen zu Tupeln. Serialisiert wird wieder zu normalem
JSON (dict/list), damit Snapshots und kanonische Hashes unverändert bleiben.
"""

import math
import re
from collections.abc import Iterator, Mapping
from typing import Annotated, Any

from pydantic import PlainSerializer, PlainValidator

type FrozenJson = bool | int | float | str | tuple[FrozenJson, ...] | FrozenMapping | None


class FrozenMapping(Mapping[str, FrozenJson]):
    """Unveränderliches, hashbares JSON-Objekt mit sortierten String-Schlüsseln.

    Intern ausschließlich ein Tupel sortierter `(schlüssel, wert)`-Paare; es existiert kein
    veränderbares dict, list oder set. Werte sind selbst `FrozenMapping`, Tupel oder Skalare.
    """

    __slots__ = ("_items",)
    _items: tuple[tuple[str, FrozenJson], ...]

    def __init__(self, data: Mapping[Any, object]) -> None:
        items: list[tuple[str, FrozenJson]] = []
        for key in sorted(data, key=str):
            if not isinstance(key, str):
                raise ValueError(f"JSON-Schlüssel müssen Strings sein, nicht {key!r}")
            items.append((key, freeze_json(data[key])))
        object.__setattr__(self, "_items", tuple(items))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("FrozenMapping ist unveränderlich")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("FrozenMapping ist unveränderlich")

    def __getitem__(self, key: str) -> FrozenJson:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._items)!r})"

    def __reduce__(self) -> tuple[type["FrozenMapping"], tuple[dict[str, FrozenJson]]]:
        return (FrozenMapping, (dict(self._items),))

    def __copy__(self) -> "FrozenMapping":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "FrozenMapping":
        return self


def freeze_json(value: object) -> FrozenJson:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NaN/Inf sind in wissenschaftlichen Parametern unzulässig")
        return value
    if isinstance(value, Mapping):
        return value if isinstance(value, FrozenMapping) else FrozenMapping(value)
    if isinstance(value, list | tuple):
        return tuple(freeze_json(item) for item in value)
    raise ValueError(f"kein JSON-Wert: {type(value).__name__}")


def thaw_json(value: object) -> Any:
    """Rückwandlung in normales JSON (dict/list) für Serialisierung und Hashing."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [thaw_json(item) for item in value]
    return value


_AMBIGUOUS_STRING = re.compile(
    r"^\s*(?:[-+]?(?:\d[\d_]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    r"|[-+]?\.?(?:inf|nan)|yes|no|on|off|true|false|null|~)\s*$",
    re.IGNORECASE,
)


def _reject_ambiguous_strings(value: FrozenJson, path: str) -> None:
    if isinstance(value, str) and _AMBIGUOUS_STRING.match(value):
        raise ValueError(
            f"{path}: String {value!r} sieht wie eine Zahl/ein Bool/null aus; "
            "wissenschaftliche Parameter bitte typisiert angeben"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_ambiguous_strings(item, f"{path}.{key}")
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            _reject_ambiguous_strings(item, f"{path}[{index}]")


def _validate_params(value: object) -> FrozenMapping:
    if not isinstance(value, Mapping):
        raise ValueError("Parameter müssen ein JSON-Objekt sein")
    frozen = value if isinstance(value, FrozenMapping) else FrozenMapping(value)
    _reject_ambiguous_strings(frozen, "params")
    return frozen


ScientificParams = Annotated[
    FrozenMapping,
    PlainValidator(_validate_params),
    PlainSerializer(thaw_json, return_type=dict[str, Any]),
]
"""Tief unveränderliche, typisierte (nicht stringifizierte) Parameterstruktur."""
