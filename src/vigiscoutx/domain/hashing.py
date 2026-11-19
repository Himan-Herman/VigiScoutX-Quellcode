"""Kanonische Serialisierung und SHA-256 für hash-relevante Strukturen."""

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from vigiscoutx.domain.frozen_json import FrozenMapping, thaw_json


def _json_default(value: object) -> Any:
    if isinstance(value, FrozenMapping):
        return thaw_json(value)
    raise TypeError(f"nicht kanonisch serialisierbar: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Deterministische JSON-Kodierung: sortierte Schlüssel, keine Leerzeichen, UTF-8, kein NaN."""
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))
