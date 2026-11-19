"""Namespaced SHA-256-Rankings und Seed-Ableitung des Calibration-Protokolls (DEC-057).

`‖` der Baseline ist die direkte Verkettung der UTF-8-Zeichenketten ohne Trennzeichen;
Trennzeichen sind ausschließlich die in den Namespaces bzw. Kontexten enthaltenen `|`.
"""

import hashlib
from collections.abc import Iterable


def sha256_concat(*parts: str) -> str:
    """`SHA256(part_1 ‖ part_2 ‖ …)` als Hex-String."""
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def first64bits(hex_digest: str) -> int:
    """Erste 64 Bit (8 Byte, Big Endian) eines SHA-256-Hex-Digests als nichtnegative Ganzzahl."""
    if len(hex_digest) != 64:
        raise ValueError("SHA-256-Hex-Digest mit 64 Zeichen erwartet")
    return int(hex_digest[:16], 16)


def rank_ids(split_hash: str, prefix: str, ids: Iterable[str]) -> tuple[str, ...]:
    """Aufsteigend nach `SHA256(split_hash ‖ prefix ‖ id)`, Gleichstand lexikographisch nach id."""
    unique = list(ids)
    if len(unique) != len(set(unique)):
        raise ValueError("IDs für das Ranking müssen eindeutig sein")
    return tuple(sorted(unique, key=lambda i: (sha256_concat(split_hash, prefix, i), i)))
