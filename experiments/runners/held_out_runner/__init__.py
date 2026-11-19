"""HELD_OUT-Evaluationsrunner (Protokoll v1.1, DEC-082 – DEC-086), bisher nur synthetisch geprüft.

Orchestriert ausschließlich bestehende Komponenten: `HeldOutGuard.open_held_out()` mit dem
bestehenden Release-Gate, die partitionsgebundene HELD_OUT-Berechtigung, die hashgeprüfte
Wiederverwendung der rekonstruierten Reference, das evaluationslokale Sampling und Structural Gate
(`vigiscoutx.evaluation.held_out_sampling`), die unveränderten Shift-Operatoren und Signalmodule
G1–G4, die Gruppenaggregation, RiskMapper, label-aware Analyse, RecommendationEngine und den
`MonitoringRunResult`. Ausführung nur mit einer eigenen, nicht versionierten
`HELD_OUT_EVALUATION`-Freigabe (Schema 2, `DEC-087`) für denselben Evaluation-Commit wie die
Reference-Phase, die den SHA-256 des Reference-Evidence-Manifests bindet; sonst terminaler STOP vor
jedem Zugriff (fail-closed).
"""
