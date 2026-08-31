"""Phase 8. Drift detection.

Population Stability Index and KL divergence on the score distribution, plus embedding
drift on the inputs. A rising AI-probability distribution in production usually means
one of two things: the world changed (new model everyone is using), or the detector
did. Both go to the Failure Atlas, neither goes straight to training.
"""

from __future__ import annotations

import numpy as np


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index. Above ~0.25 is conventionally 'significant shift'."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    e, _ = np.histogram(expected, bins=edges)
    a, _ = np.histogram(actual, bins=edges)
    e = np.clip(e / max(e.sum(), 1), 1e-6, None)
    a = np.clip(a / max(a.sum(), 1), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
