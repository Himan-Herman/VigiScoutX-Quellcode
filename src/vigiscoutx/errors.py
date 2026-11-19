"""Fehlerklassen des Cores. Integritäts- und Leakage-Verstöße sind immer harte Fehler."""


class VigiScoutXError(Exception):
    """Basisklasse aller fachlichen Fehler."""


class IntegrityError(VigiScoutXError):
    """Hash- oder Manifestabweichung (FAIL, keine Warnung)."""


class PhaseViolationError(VigiScoutXError):
    """Operation ist in der aktuellen Phase des Held-out Guards unzulässig."""


class LeakageError(VigiScoutXError):
    """Verletzung der Split- bzw. Rollentrennung."""


class DatasetValidationError(VigiScoutXError):
    """Lokaler Datensatz entspricht nicht der erwarteten Struktur oder Baseline."""


class ModelSnapshotError(VigiScoutXError):
    """Modell-Snapshot, Gewichte oder Metadaten sind inkonsistent."""


class SplitGenerationError(VigiScoutXError):
    """Split lässt sich unter den Gruppierungs- und Klassenregeln nicht erzeugen."""


class TrainingError(VigiScoutXError):
    """Model-Development-Lauf ist unzulässig konfiguriert oder inkonsistent."""


class CalibrationHelperError(VigiScoutXError):
    """Ein Calibration-Hilfsverfahren (z. B. Bandbreitenregel) ist auf den Daten undefiniert."""


class CalibrationStopError(VigiScoutXError):
    """Vorab definierter STOP des Calibration-Protokolls; Entscheidung erfolgt separat (P4-01)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class CalibrationProtocolError(VigiScoutXError):
    """Unzulässige Reihenfolge, unzulässiger Zugriff oder nicht-valides Pflichtsignal (P4-01)."""


class PolicyResolutionError(VigiScoutXError):
    """Unbekannte oder abweichende Risk-/Recommendation-`PolicyRef` (DEC-074, DEC-075)."""
