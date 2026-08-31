"""External benchmark loaders. EVAL-ONLY. These never enter a training split.

MAGE label polarity is asserted at load time rather than trusted. The HF card says 1
means machine-generated, but the original DeepfakeTextDetect release used the opposite
convention in places. A silently inverted label turns an AUROC of 0.05 into a reported
0.95 and nothing crashes, which is exactly the kind of bug that survives to a writeup.
"""

from __future__ import annotations


def assert_label_polarity(known_human_scores, known_machine_scores) -> None:
    """Fail loudly if the benchmark's labels are inverted relative to our convention."""
    import numpy as np

    h = float(np.mean(known_human_scores))
    m = float(np.mean(known_machine_scores))
    if not m > h:
        raise AssertionError(
            f"label polarity looks inverted: mean machine score {m:.4g} <= mean human score {h:.4g}"
        )


def load_raid(split: str = "RAID-extra", excluded_domains=("code", "czech", "german")):  # noqa: ANN001, ANN201
    raise NotImplementedError("Phase 5")


def load_mage(split: str = "test"):  # noqa: ANN201
    raise NotImplementedError("Phase 5")


def load_hc3():  # noqa: ANN201
    raise NotImplementedError("Phase 5")


def contamination_check(train_hashes: set[str], eval_hashes: set[str]) -> set[str]:
    """Overlap must be empty. Any overlap invalidates the external result."""
    return train_hashes & eval_hashes
