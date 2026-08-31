"""Phase 8. The FORGE flywheel as an Airflow DAG.

This is the recurring pipeline worth orchestrating: it has real inter-task
dependencies, tasks that fail for different reasons and need different retry policies,
and a gate at the end that can stop the whole thing. Written now as the target shape;
the callables land in Phase 8.
"""

from __future__ import annotations

# Imported lazily so the repo does not require Airflow to be installed.
TASKS = [
    "scan_reserve_pool",       # Spark job over 5M documents, see orchestration/spark/
    "collect_failures",
    "embed_and_cluster",       # Failure Atlas
    "select_targets",          # proportional across clusters, not global top-k
    "generate_targeted_mirrors",
    "build_dataset_version",
    "train_candidate",
    "evaluate_all_regimes",
    "release_gate",            # terminal: blocks promotion on any failure
    "canary",
    "promote",
]

SCHEDULE = "@weekly"
