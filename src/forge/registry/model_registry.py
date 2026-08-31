"""Model registry.

A model version is not weights. It is weights plus config plus dataset_version plus
code commit plus metrics plus environment. Anything less cannot be reproduced or
audited, and a detector that accuses people needs to be auditable.
"""

from __future__ import annotations

REQUIRED_FIELDS = (
    "version",
    "weights_uri",
    "model_config",
    "dataset_version",
    "code_commit",
    "metrics",
    "environment",
)


def validate_entry(entry: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS if f not in entry]
