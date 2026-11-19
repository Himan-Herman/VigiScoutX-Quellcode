"""Offizieller Monitoring-Bewertungsnachweis `MonitoringRunResult` (ARC-042, P4-03C).

Liegt oberhalb von `domain`, `risk` und `recommendations`, damit die Run-Assembly-Validierung die
eingefrorenen Policy-Definitionen selbst wiederverwendet, ohne zyklische Domain-/Policy-Imports.
"""

from vigiscoutx.monitoring.run_result import (
    MonitoringRunResult,
    verify_run_assembly,
    verify_signal_to_group,
)

__all__ = ["MonitoringRunResult", "verify_run_assembly", "verify_signal_to_group"]
