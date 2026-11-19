"""Eingefrorene Policy-Referenzen für synthetische FreezeManifestV2-Testwelten (P4-03C).

Schema-2-Manifeste binden ausschließlich `vsx-risk-r0-r3`/`vsx-recommendation-review` 1.0.0.
Synthetische Datensätze und Modelle bleiben synthetisch; gültige `PolicyRef`s machen keinen Lauf
zu einem Real-Run. Historische Schema-1-Fixtures behalten ihre ursprünglichen Referenzen.
"""

import copy
from typing import Any

from vigiscoutx.config.scientific_config import RECOMMENDATION_POLICY_REF, RISK_POLICY_REF


def with_frozen_policies(data: dict[str, Any]) -> dict[str, Any]:
    """Kopie einer RunConfig-Payload mit den eingefrorenen v1.0.0-Policies."""
    result = copy.deepcopy(data)
    result["scientific"]["risk_policy"] = RISK_POLICY_REF.model_dump(mode="json")
    result["scientific"]["recommendation_policy"] = RECOMMENDATION_POLICY_REF.model_dump(
        mode="json"
    )
    return result
