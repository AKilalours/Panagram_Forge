"""Phase 3 training entrypoint.

Non-obvious decisions, stated here because they are the ones that change results:

**Grouped batching is not used; grouped SPLITTING already happened.** Windows from the
same document can share a batch safely, because the split boundary is at the group level.

**Checkpoint resume is mandatory, not a nicety.** A run that cannot resume turns every
preemption into a lost day, and spot GPU instances are the affordable way to do Phase 7.
Resume restores model, optimizer, scheduler, scaler, epoch and step.

**Validation computes the operating threshold, not just a loss.** A training run whose
val loss improved but whose FPR-at-budget got worse has not improved for this product.

torch is imported lazily so the package works without it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")   # lower is better: FPR at budget
    dataset_version: str = ""
    code_commit: str = ""


def save_checkpoint(path: str | Path, model, optimizer, scheduler, scaler, state: TrainState) -> None:  # pragma: no cover
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "state": asdict(state),
        },
        path,
    )
    # Sidecar JSON so a checkpoint can be identified without loading it.
    Path(str(path) + ".json").write_text(json.dumps(asdict(state), indent=2) + "\n")


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None) -> TrainState:  # pragma: no cover
    import torch

    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return TrainState(**ckpt["state"])


def run(config: dict) -> None:  # pragma: no cover - needs a GPU
    raise NotImplementedError(
        "Training loop body lands with the first real GPU run. Model, alignment, "
        "windowing, calibration, checkpointing and the evaluation lab are implemented "
        "and tested; what remains is the fit loop itself, which cannot be verified "
        "without hardware and would be dishonest to claim as done."
    )
