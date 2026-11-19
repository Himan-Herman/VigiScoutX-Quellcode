"""Suche nach CAL_AUDIT-Diagnoseverknüpfungen in öffentlich erzeugten JSON-Strukturen (P4-02E)."""

from collections.abc import Iterable, Mapping
from typing import Any

ALLOWED_AUDIT_PATHS = frozenset(
    {
        # lesion_id → canonical sample_id (ohne Klasse)
        "calibration.canonical_samples/*/*",
        "view/canonical_samples/*/*",
        # sortierte CAL_AUDIT-lesion_ids ohne Klassenzuordnung
        "calibration.pool_manifest/audit_lesion_ids/*",
        "view/pools/audit_lesion_ids/*",
    }
)


def audit_id_paths(name: str, value: Any, audit_ids: frozenset[str]) -> set[str]:
    """Pfade (Listenindizes als `*`), an denen eine CAL_AUDIT-ID vorkommt."""
    found: set[str] = set()

    def walk(node: Any, path: str) -> None:
        if isinstance(node, str):
            if node in audit_ids:
                found.add(path)
        elif isinstance(node, Mapping):
            for key, child in node.items():
                if key in audit_ids:
                    found.add(f"{path}/<key>")
                walk(child, f"{path}/{key}")
        elif isinstance(node, Iterable):
            for child in node:
                walk(child, f"{path}/*")

    walk(value, name)
    return found


def leaking_paths(objects: Mapping[str, Any], audit_ids: Iterable[str]) -> set[str]:
    ids = frozenset(audit_ids)
    paths: set[str] = set()
    for name, value in objects.items():
        paths |= audit_id_paths(name, value, ids)
    return paths - ALLOWED_AUDIT_PATHS


def conflict_reconstruction(
    split_hash: str,
    windows: Iterable[tuple[str, frozenset[str]]],
    audit_lesions: Iterable[str],
) -> dict[str, set[str]]:
    """Prüfung: Aus Window-Zugehörigkeit und öffentlichem Ranking
    `SHA256(split_hash ‖ context ‖ lesion_id)` folgt ein Konfliktgraph – Läsionen derselben Klasse
    verletzen die Top-`n_c`-Ordnung nie. Liefert je Läsion die Menge sicher klassenfremder Läsionen.
    """
    import hashlib

    lesions = sorted(audit_lesions)
    conflict: dict[str, set[str]] = {lesion: set() for lesion in lesions}
    for context, members in windows:
        rank = {
            lesion: (hashlib.sha256(f"{split_hash}{context}{lesion}".encode()).hexdigest(), lesion)
            for lesion in lesions
        }
        for outside in lesions:
            if outside in members:
                continue
            for inside in members:
                if rank[outside] < rank[inside]:
                    conflict[outside].add(inside)
                    conflict[inside].add(outside)
    return conflict


def identifiable_share(conflict: Mapping[str, set[str]], truth: Mapping[str, str]) -> float:
    """Anteil der Läsionen, deren Klasse bei sonst bekannter Partition eindeutig folgt."""
    classes = set(truth.values())
    unique = 0
    for lesion in conflict:
        consistent = {
            c
            for c in classes
            if all(m == lesion or m not in conflict[lesion] for m in truth if truth[m] == c)
        }
        unique += len(consistent) == 1
    return unique / len(conflict) if conflict else 0.0
