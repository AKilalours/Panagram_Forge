"""Config loading plus the spec-check that keeps configs honest.

CI runs `forge spec-check`. It fails the build when a config drifts from the frozen
Data Spec v1. The specific things it catches are the ones that would silently
invalidate results rather than crash: an eval-only corpus wandering into training, a
held-out generator becoming held-in, split ratios changing under an unchanged
dataset version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SPEC_VERSION = "data_spec_v1"

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"

EVAL_ONLY_SOURCES = {"hc3", "raid", "mage"}
HELD_OUT_FAMILIES = {"gemma", "deepseek", "api"}
EXPECTED_SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
EXPECTED_SPLIT_SALT = "forge-v1"


def load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open() as fh:
        return yaml.safe_load(fh)


def spec_check() -> list[str]:
    """Return a list of violations. Empty list means the configs match the spec."""
    problems: list[str] = []

    human = load("configs/data/human.yaml")
    evals = load("configs/data/eval.yaml")
    gens = load("configs/generation/generators.yaml")
    regimes = load("configs/eval/regimes.yaml")

    for name, cfg in [
        ("human.yaml", human),
        ("eval.yaml", evals),
        ("generators.yaml", gens),
        ("regimes.yaml", regimes),
    ]:
        if cfg.get("spec_version") != SPEC_VERSION:
            problems.append(f"{name}: spec_version is {cfg.get('spec_version')!r}, expected {SPEC_VERSION!r}")

    # 1. No eval-only corpus may appear as a training source.
    train_ids = {s["id"] for s in human.get("sources", [])}
    leaked = train_ids & EVAL_ONLY_SOURCES
    if leaked:
        problems.append(f"human.yaml: EVAL-ONLY source(s) present in training sources: {sorted(leaked)}")

    # 2. Training shares must sum to 1.
    shares = [s.get("train_share", 0.0) for s in human.get("sources", [])]
    total = round(sum(shares), 6)
    if total != 1.0:
        problems.append(f"human.yaml: train_share values sum to {total}, expected 1.0")

    # 3. Split salt and ratios must match the spec, or old datasets stop being comparable.
    splits = human.get("splits", {})
    if splits.get("salt") != EXPECTED_SPLIT_SALT:
        problems.append(f"human.yaml: split salt is {splits.get('salt')!r}, expected {EXPECTED_SPLIT_SALT!r}")
    if splits.get("ratios") != EXPECTED_SPLIT_RATIOS:
        problems.append(f"human.yaml: split ratios {splits.get('ratios')} != {EXPECTED_SPLIT_RATIOS}")
    if splits.get("group_key") != "source_group_id":
        problems.append("human.yaml: group_key must be source_group_id")

    # 4. Held-out generator families must actually be held out.
    roles = {f["family"]: f["role"] for f in gens.get("families", [])}
    for fam in HELD_OUT_FAMILIES:
        if roles.get(fam) != "held_out":
            problems.append(f"generators.yaml: family {fam!r} has role {roles.get(fam)!r}, expected 'held_out'")

    # 5. R3 must test on exactly the held-out families.
    r3 = regimes.get("regimes", {}).get("R3_unseen_generator", {})
    if set(r3.get("test_families", [])) != HELD_OUT_FAMILIES:
        problems.append(
            f"regimes.yaml: R3 test_families {r3.get('test_families')} != held-out set {sorted(HELD_OUT_FAMILIES)}"
        )
    overlap = set(r3.get("train_families", [])) & HELD_OUT_FAMILIES
    if overlap:
        problems.append(f"regimes.yaml: R3 trains on held-out families {sorted(overlap)}")

    # 6. Every external benchmark must require a contamination check.
    for ext in regimes.get("external", []):
        if ext.get("contamination_check") != "required":
            problems.append(f"regimes.yaml: external {ext.get('id')} missing contamination_check: required")

    # 7. FPR must be the primary metric and the gate must enforce a budget.
    if regimes.get("primary_metric") != "fpr":
        problems.append("regimes.yaml: primary_metric must be 'fpr'")
    gate = regimes.get("release_gate", {})
    if "max_fpr" not in gate:
        problems.append("regimes.yaml: release_gate must define max_fpr")

    return problems
