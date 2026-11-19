"""Ausführbarer Calibration-End-to-End-Runner (P4-02E).

Orchestriert ausschließlich die vorhandenen Komponenten gemäß `CALIBRATION_PROTOCOL_BASELINE.md`
v1.0 und `CALIBRATION_ORCHESTRATION_BASELINE.md` v1.0 (`DEC-061` – `DEC-066`):
`CalibrationProcess` (autoritative 28-Schritt-Zustandsmaschine), `HeldOutGuard` mit
scope-gebundenen Berechtigungen, Dataset-/Modell-Ports, die Signalmodule G1–G4, die Gruppen-
aggregation, die Shift-Operatoren und den Freeze (ArtifactManifest, Calibration-Freeze-Record,
FreezeManifestV2). Wissenschaftliche Regeln werden nicht dupliziert, sondern aufgerufen.

HELD_OUT wird nie geöffnet; `open_held_out()` ist nicht Teil dieses Pakets.
"""
