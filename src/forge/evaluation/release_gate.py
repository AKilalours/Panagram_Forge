"""Phase 8. The release gate.

A model does not ship because validation accuracy went up. It ships when it passes
every threshold below and regresses on nothing. This function is pure so the policy
itself is testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateResult:
    passed: bool
    failures: list[str]


def evaluate(metrics: dict, gate: dict, baseline: dict | None = None) -> GateResult:
    failures: list[str] = []

    def _upper(key: str, gate_key: str) -> None:
        if gate_key in gate and key in metrics and metrics[key] > gate[gate_key]:
            failures.append(f"{key}={metrics[key]:.5g} exceeds {gate_key}={gate[gate_key]:.5g}")

    _upper("fpr", "max_fpr")
    _upper("fnr", "max_fnr")
    _upper("ece", "max_ece")
    _upper("p95_latency_ms", "max_p95_latency_ms")

    if "min_ood_auroc" in gate and "ood_auroc" in metrics:
        if metrics["ood_auroc"] < gate["min_ood_auroc"]:
            failures.append(
                f"ood_auroc={metrics['ood_auroc']:.5g} below min_ood_auroc={gate['min_ood_auroc']:.5g}"
            )

    for regime in gate.get("no_regression_on", []):
        if baseline and regime in baseline and regime in metrics:
            if metrics[regime] < baseline[regime]:
                failures.append(f"regression on {regime}: {metrics[regime]:.5g} < {baseline[regime]:.5g}")

    return GateResult(passed=not failures, failures=failures)
