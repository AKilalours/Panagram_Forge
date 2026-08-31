"""Temperature scaling. A confidence score that is not calibrated is a liability here.

If the API returns 0.97 and is wrong 20 percent of the time, every downstream policy
built on that number is wrong too.
"""

from __future__ import annotations


def fit_temperature(logits, labels) -> float:  # noqa: ANN001
    raise NotImplementedError("Phase 3")


def apply_temperature(logits, temperature: float):  # noqa: ANN001, ANN201
    raise NotImplementedError("Phase 3")
