"""Phase 3 training entrypoint. The fit loop.

Decisions here that change results, stated because they are the ones worth arguing about:

**Validation reports FPR at the budget, not just loss.** A run whose val loss improved but
whose FPR-at-budget got worse has not improved for this product. `best_metric` is FPR,
lower is better, and checkpoint selection uses it.

**Checkpoint resume is mandatory, not a nicety.** Spot GPU instances are the affordable
way to run this, and a run that cannot resume turns every preemption into a lost day.
Resume restores model, optimizer, scheduler, scaler, epoch and step.

**Grouped batching is not needed; grouped SPLITTING already happened.** Windows from one
document can share a batch safely because the split boundary is at the group level.

**--smoke runs 20 steps on 200 examples.** Find the bugs in two minutes, not two hours
into a paid run. Every real run should be preceded by a smoke run on the same config.

torch and transformers are imported lazily so the package imports without them.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from forge.common.config import REPO_ROOT, load
from forge.evaluation import metrics as M
from forge.evaluation.calibration import fit_temperature


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")     # FPR at budget, lower is better
    best_step: int = 0
    dataset_version: str = ""
    code_commit: str = ""
    temperature: float = 1.0
    threshold: float = 0.5


def _code_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def save_checkpoint(path, model, optimizer, scheduler, scaler, state: TrainState) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "state": asdict(state),
    }, path)
    # Sidecar so a checkpoint can be identified without loading it.
    Path(str(path) + ".json").write_text(json.dumps(asdict(state), indent=2) + "\n")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None) -> TrainState:
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    if scaler and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    return TrainState(**ck["state"])


@dataclass
class ValResult:
    loss: float
    fpr_at_budget: float
    threshold: float
    fnr: float
    auroc: float
    ece: float
    temperature: float
    n_human: int
    n_ai: int

    def as_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def evaluate_split(model, loader, device, fpr_budget: float, calibrate: bool = True) -> ValResult:
    import torch
    from torch import nn

    model.eval()
    logits_all, labels_all, losses = [], [], []
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"])
            dl = out["doc_logits"].float()
            losses.append(ce(dl, batch["doc_labels"]).item())
            logits_all.append(dl.cpu().numpy())
            labels_all.append(batch["doc_labels"].cpu().numpy())

    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)

    temp = fit_temperature(logits, labels) if calibrate and len(labels) > 50 else 1.0
    z = logits / temp
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    scores = (p / p.sum(axis=1, keepdims=True))[:, 1]

    n_human = int((labels == 0).sum())
    # A threshold cannot be fitted without human examples in the split.
    thr = M.threshold_at_fpr(labels, scores, fpr_budget) if n_human else 0.5
    return ValResult(
        loss=float(np.mean(losses)),
        fpr_at_budget=M.false_positive_rate(labels, scores, thr),
        threshold=float(thr),
        fnr=M.false_negative_rate(labels, scores, thr),
        auroc=M.auroc(labels, scores),
        ece=M.expected_calibration_error(labels, scores),
        temperature=float(temp),
        n_human=n_human,
        n_ai=int((labels == 1).sum()),
    )


def validate_config(cfg: dict) -> str:
    """Config checks that must not require torch.

    Config errors should surface in a second on a laptop, not after a GPU stack has
    loaded on a paid pod.
    """
    arm = cfg.get("data", {}).get("arm")
    if arm is None:
        raise RuntimeError(
            "training config must declare data.arm (random | mirror | hard_negative). "
            "Without it the loader cannot verify the arm is reading its own data, and "
            "two arms pointed at the same directory produce a false finding silently."
        )
    paths = cfg.get("paths", {})
    if not (paths.get("ai") or paths.get("mirror")):
        raise RuntimeError("training config must declare paths.ai, this arm's AI source")
    return arm


def run(config: dict | str, smoke: bool = False, resume: str | None = None) -> dict:
    cfg_early = load(config) if isinstance(config, str) else config
    validate_config(cfg_early)

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from forge.modeling.encoder import ForgeConfig, build_model
    from forge.training.data import Collator, build_dataset, load_examples

    cfg = load(config) if isinstance(config, str) else config
    mcfg = load(cfg["model_config"])
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    paths = cfg.get("paths", {})

    exp_id = cfg["experiment"]["id"]
    out_dir = Path(paths.get("out", REPO_ROOT / "outputs")) / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tcfg.get("seed", 42))

    fpr_budget = cfg.get("fpr_budget", 0.001)

    tok = AutoTokenizer.from_pretrained(mcfg["backbone"])
    limit = 200 if smoke else dcfg.get("limit")
    arm = validate_config(cfg)
    # ai_cap holds the AI budget equal across arms. Each arm's validator rejects at its
    # own rate, so generation yields different counts; without the cap the comparison
    # measures data volume as much as data strategy. Set the SAME value in every arm's
    # config, equal to the smallest arm's accepted count.
    examples = load_examples(
        human_root=paths.get("human"), ai_root=paths.get("ai") or paths.get("mirror"),
        mixed_root=paths.get("mixed"), limit=limit, expect_arm=arm,
        ai_cap=dcfg.get("ai_cap"),
    )
    if not examples:
        raise RuntimeError(
            f"no examples loaded from {paths}. Run `forge ingest` and `forge mirror` first."
        )

    feats = build_dataset(examples, tok, max_length=mcfg["max_length"],
                          stride=mcfg["window"]["stride"])
    by_split = {s: [f for f in feats if f["split"] == s] for s in ("train", "val", "test")}

    # Hard invariant, re-checked here rather than trusted: no group may span two splits.
    seen: dict[str, str] = {}
    for f in feats:
        prior = seen.setdefault(f["source_group_id"], f["split"])
        if prior != f["split"]:
            raise RuntimeError(
                f"group {f['source_group_id']} appears in both {prior} and {f['split']}. "
                "Training on this data would produce inflated metrics."
            )

    if not by_split["train"]:
        raise RuntimeError("no training windows after windowing")

    coll = Collator(tok, max_length=mcfg["max_length"])
    bs = tcfg["batch_size"]
    dl_train = DataLoader(by_split["train"], batch_size=bs, shuffle=True, collate_fn=coll,
                          num_workers=tcfg.get("num_workers", 2), drop_last=False)
    dl_val = DataLoader(by_split["val"] or by_split["train"], batch_size=bs * 2,
                        shuffle=False, collate_fn=coll)

    model = build_model(ForgeConfig(
        backbone=mcfg["backbone"], max_length=mcfg["max_length"],
        stride=mcfg["window"]["stride"],
        token_loss_weight=tcfg.get("token_loss_weight", 0.5),
    )).to(device)

    accum = tcfg.get("grad_accum", 1)
    epochs = 1 if smoke else tcfg["epochs"]
    steps_per_epoch = math.ceil(len(dl_train) / accum)
    total_steps = 20 if smoke else steps_per_epoch * epochs

    decay = [p for n, p in model.named_parameters() if p.requires_grad and "bias" not in n and "LayerNorm" not in n]
    nodecay = [p for n, p in model.named_parameters() if p.requires_grad and ("bias" in n or "LayerNorm" in n)]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg.get("weight_decay", 0.01)},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=float(tcfg["learning_rate"]),
    )
    warmup = int(total_steps * tcfg.get("warmup_ratio", 0.06))
    sched = get_linear_schedule_with_warmup(optim, warmup, total_steps)

    precision = tcfg.get("precision", "bf16")
    use_amp = device == "cuda" and precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    if precision == "bf16" and device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "config requests bf16 but this GPU does not support it (T4 and older). "
            "DeBERTa-v3 in fp16 is known to produce NaN losses, so switch to fp32 or use "
            "an Ampere-or-newer GPU rather than silently downgrading to fp16."
        )
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and device == "cuda"))

    if tcfg.get("gradient_checkpointing") and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()

    state = TrainState(dataset_version=dcfg.get("dataset_version", ""), code_commit=_code_commit())
    if resume:
        state = load_checkpoint(resume, model, optim, sched, scaler)

    wb = None
    if cfg.get("tracking", {}).get("backend") == "wandb" and not smoke and os.getenv("WANDB_API_KEY"):
        import wandb

        wb = wandb.init(project=cfg["tracking"].get("project", "forge"), name=exp_id,
                        config={"model": mcfg, "training": tcfg, "data": dcfg})

    history = []
    t0 = time.time()
    stop = False
    for epoch in range(state.epoch, epochs):
        model.train()
        optim.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl_train):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(batch["input_ids"], batch["attention_mask"],
                            batch["doc_labels"], batch["token_labels"])
                loss = out["loss"] / accum
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at step {state.global_step}. With DeBERTa-v3 this is "
                    "usually fp16 overflow in disentangled attention; use bf16 or fp32."
                )
            scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()

            if (i + 1) % accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optim); scaler.update()
                else:
                    optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                state.global_step += 1

                if state.global_step % tcfg.get("log_every", 50) == 0:
                    rec = {"step": state.global_step, "loss": float(loss.item() * accum),
                           "lr": sched.get_last_lr()[0],
                           "examples_per_second": state.global_step * bs * accum / (time.time() - t0)}
                    history.append(rec)
                    print(f"step {rec['step']:>6}  loss {rec['loss']:.4f}  "
                          f"lr {rec['lr']:.2e}  {rec['examples_per_second']:.0f} ex/s", flush=True)
                    if wb:
                        wb.log(rec, step=state.global_step)

                if smoke and state.global_step >= total_steps:
                    stop = True
                    break
        state.epoch = epoch + 1

        val = evaluate_split(model, dl_val, device, fpr_budget)
        print(f"[epoch {state.epoch}] {json.dumps(val.as_dict())}", flush=True)
        if wb:
            wb.log({f"val/{k}": v for k, v in val.as_dict().items()}, step=state.global_step)

        # Selection on FPR at the budget, NOT on loss.
        if val.fpr_at_budget <= state.best_metric:
            state.best_metric = val.fpr_at_budget
            state.best_step = state.global_step
            state.temperature = val.temperature
            state.threshold = val.threshold
            save_checkpoint(out_dir / "best.pt", model, optim, sched, scaler, state)
        save_checkpoint(out_dir / "last.pt", model, optim, sched, scaler, state)
        if stop:
            break

    val = evaluate_split(model, dl_val, device, fpr_budget)
    summary = {
        "experiment": exp_id,
        "smoke": smoke,
        "code_commit": state.code_commit,
        "dataset_version": state.dataset_version,
        "backbone": mcfg["backbone"],
        "device": device,
        "precision": precision,
        "arm": arm,
        "ai_source": paths.get("ai") or paths.get("mirror"),
        "n_windows": {k: len(v) for k, v in by_split.items()},
        "steps": state.global_step,
        "wall_seconds": round(time.time() - t0, 1),
        "fpr_budget": fpr_budget,
        "val": val.as_dict(),
        "history": history[-20:],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if wb:
        wb.finish()
    return summary
