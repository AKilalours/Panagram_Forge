#!/usr/bin/env bash
# Run the rest of the experiment without a human present.
#
# WHY THIS EXISTS. Generation has no resume: killing it loses the hours. Training does have
# resume, but only if something restarts it. Between the two arms there are four steps that
# each take one to two hours and each need the previous one to have finished. Sitting awake
# to type six commands is not a plan.
#
# WHAT IT WILL NOT DO. It refuses to train if the length confound is still present. The
# whole reason the first corpora were thrown away is that a detector could score 0.84 on
# them without reading a word; training on a second corpus with the same defect would burn
# two GPU-hours to produce the same uninterpretable number. The gate is measured, not
# assumed, and if it trips the script stops and says so rather than pressing on.
#
# Every step appends to /workspace/overnight.log with a timestamp, so the morning's first
# question, "what actually happened", is answered by one file.

set -u
cd /workspace/forge

LOG=/workspace/overnight.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- 1. wait for Arm B
say "waiting for the running mirror generation to finish"
while pgrep -f "forge.cli mirror" > /dev/null; do sleep 60; done
say "mirror generation is no longer running"

count() {
  python - "$1" <<'PY'
import sys
import pyarrow.dataset as ds
try:
    print(ds.dataset(sys.argv[1], format="parquet", partitioning="hive").count_rows())
except Exception:
    print(0)
PY
}

MIRRORS=$(count data/silver/mirrors)
say "mirror arm wrote $MIRRORS documents"
if [ "$MIRRORS" -lt 1000 ]; then
  say "ABORT: the mirror arm produced almost nothing. Something failed; read gen2.log."
  exit 1
fi

# Safe to pull now: no python process is holding the code.
git pull --ff-only >> "$LOG" 2>&1 && say "pulled latest code"

# ---------------------------------------------------------------- 2. Arm A
say "starting the control arm at n=$MIRRORS, matched to the mirror arm's count"
python -m forge.cli generate-random --n "$MIRRORS" \
  --humans data/silver --out data/silver/random --backend vllm >> /workspace/gen2.log 2>&1
RANDOMS=$(count data/silver/random)
say "control arm wrote $RANDOMS documents"
if [ "$RANDOMS" -lt 1000 ]; then
  say "ABORT: the control arm produced almost nothing. Read gen2.log."
  exit 1
fi

# ---------------------------------------------------------------- 3. the gate
say "measuring the length confound on both arms"
python scripts/diagnose_arm.py configs/training/mirror_minimal.yaml   >> "$LOG" 2>&1
python scripts/diagnose_arm.py configs/training/baseline_minimal.yaml >> "$LOG" 2>&1

WORST=$(python - <<'PY'
# Largest deviation from chance across both arms, on the FULL pools rather than a
# 200-document sample, because that is the number the decision rests on.
import numpy as np
from forge.common.config import load
from forge.evaluation import metrics as M
from forge.training.data import load_examples
from forge.training.train import validate_config

worst = 0.0
for path in ("configs/training/mirror_minimal.yaml", "configs/training/baseline_minimal.yaml"):
    cfg = load(path); paths = cfg.get("paths", {})
    rows = load_examples(human_root=paths.get("human"),
                         ai_root=paths.get("ai") or paths.get("mirror"),
                         limit=None, expect_arm=validate_config(cfg))
    y = np.array([int(r.label) for r in rows])
    n = np.array([len(r.text.split()) for r in rows], dtype=float)
    worst = max(worst, abs(float(M.auroc(y, n)) - 0.5))
print(f"{worst:.4f}")
PY
)
say "worst length-only deviation from chance: $WORST (was 0.3408 before the fix)"

if python -c "import sys; sys.exit(0 if float('$WORST') > 0.15 else 1)"; then
  say "STOPPING BEFORE TRAINING. Length still separates the classes by more than 0.15"
  say "from chance, so a trained detector could reach that score without reading a word."
  say "Training now would spend two GPU-hours on an uninterpretable result. Read the two"
  say "diagnose_arm blocks above: the per-class medians say which direction it went."
  exit 2
fi
say "gate passed: length carries little signal, the arms are worth training"

# ---------------------------------------------------------------- 4. train both arms
for arm in baseline mirror; do
  say "smoke run for $arm"
  python -m forge.cli train --config "configs/training/${arm}_minimal.yaml" --smoke \
    >> "$LOG" 2>&1 || { say "ABORT: the $arm smoke run failed; not starting the real run"; exit 3; }
  say "training $arm"
  python -m forge.cli train --config "configs/training/${arm}_minimal.yaml" \
    >> /workspace/train_${arm}.log 2>&1 || { say "ABORT: $arm training failed"; exit 4; }
  say "$arm finished"
  tail -5 /workspace/train_${arm}.log >> "$LOG"
done

say "DONE. Both arms trained. Read /workspace/train_baseline.log and /workspace/train_mirror.log"
say "The pod can be stopped now."
