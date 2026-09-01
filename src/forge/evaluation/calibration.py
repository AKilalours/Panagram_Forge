"""Temperature scaling.

Why this is not optional here. The release gate has an ECE threshold and the operating
threshold is derived from an FPR budget, both of which assume the score is a probability.
If the model says 0.97 and is wrong 20 percent of the time, every policy built on that
number is wrong too, and a user told "97 percent confident this is AI" is being misled
about something that affects them.

One parameter, fitted on the validation split only. Fitting on test would be a leak, and
fitting on train would fit the temperature to the model's overconfidence on data it
memorised.

Implemented in numpy with a bisection on the NLL derivative, so calibration runs without
torch or scipy.
"""

from __future__ import annotations

import numpy as np


def _nll(logits: np.ndarray, labels: np.ndarray, t: float) -> float:
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(labels)), labels].mean())


def fit_temperature(
    logits, labels, lo: float = 0.05, hi: float = 10.0, iters: int = 60,
    return_boundary_flag: bool = False,
):
    """Golden-section search on NLL. Convex in log t for softmax, so this is safe.

    Returns the temperature, or (temperature, at_boundary) when return_boundary_flag.

    The boundary flag matters. If the search converges to `lo` or `hi`, the true optimum
    lies outside the range and the returned value is the edge of the box, not a fitted
    parameter. It looks exactly like a real temperature in a report. Observed in the very
    first smoke run: an under-confident model drove the search to the 0.05 floor, which
    would have been reported as a calibrated temperature.
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if logits.ndim != 2:
        raise ValueError("logits must be (n, n_classes)")
    if len(logits) != len(labels):
        raise ValueError("logits and labels length mismatch")

    phi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = _nll(logits, labels, c), _nll(logits, labels, d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _nll(logits, labels, c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _nll(logits, labels, d)
    t = float((a + b) / 2)
    at_boundary = (t <= lo * 1.02) or (t >= hi * 0.98)
    return (t, at_boundary) if return_boundary_flag else t


def apply_temperature(logits, temperature: float) -> np.ndarray:
    """Return calibrated probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ai_probability(logits, temperature: float = 1.0, ai_index: int = 1) -> np.ndarray:
    return apply_temperature(logits, temperature)[:, ai_index]
