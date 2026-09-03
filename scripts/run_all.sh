#!/usr/bin/env bash
# The whole experiment, from an empty pod to two trained arms, in one detached command.
#
# WHY THIS REPLACES overnight.sh. That script assumed a corpus already existed and a
# generation was already running. A RunPod GPU migration took the pod's volume with it and
# both assumptions became false at once, which cost an ingest and ninety minutes of
# generation. This one starts from nothing.
#
# IDEMPOTENT BY DESIGN. Every stage checks whether its OUTPUT already exists and skips if
# so. That is not tidiness: a rented spot GPU can be reclaimed at any moment, and a run
# that must start from zero after every interruption will never finish. Re-running this
# script after a restart resumes where it stopped.
#
# The check is on the output, not on a "done" marker file, because a marker can be written
# by a run that produced nothing. Counting rows in the parquet asks the only question that
# matters: is the data there.
#
# THE GATE. It refuses to train if document length still separates the classes. The first
# pair of corpora were thrown away because a detector could score 0.84 on them without
# reading a word; training on a second corpus with the same defect would burn GPU-hours to
# produce the same uninterpretable number.

set -u
cd "$(dirname "$0")/.."

LOG=/workspace/run_all.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

count() {
  # THE BUG THIS FIXES, and it aborted two runs whose data was already on disk.
  #
  # This called ds.dataset(<dir>) and let discovery walk the directory. data/silver holds
  # MANIFEST.json next to the parquet partitions, pyarrow tried to read the JSON as
  # parquet, threw, and the except returned 0. The log then said "human corpus: 0
  # documents" about a corpus of exactly 60,000 documents, and the pipeline aborted.
  #
  # Globbing for *.parquet and handing pyarrow an explicit FILE LIST removes discovery
  # from the picture entirely: a non-parquet file in the directory can no longer decide
  # whether the stage runs. An empty list is a real zero, not a swallowed exception.
  python - "$1" <<'PY' 2>/dev/null || echo 0
import glob
import sys

import pyarrow.dataset as ds

files = sorted(glob.glob(f"{sys.argv[1]}/**/*.parquet", recursive=True))
print(ds.dataset(files, format="parquet").count_rows() if files else 0)
PY
}

say "=== starting. stages already complete will be skipped."

# ------------------------------------------------------------------ 1. human corpus
HUMANS=$(count data/silver)
if [ "$HUMANS" -lt 1000 ]; then
  say "ingesting the human corpus"
  python -m forge.cli ingest --config configs/data/human_minimal.yaml --out data/silver \
    >> /workspace/ingest.log 2>&1 || { say "ABORT: ingest failed, see ingest.log"; exit 1; }
  HUMANS=$(count data/silver)
fi
say "human corpus: $HUMANS documents"
[ "$HUMANS" -lt 1000 ] && { say "ABORT: no human corpus"; exit 1; }

# ------------------------------------------------------------------ 2. mirrors, arm B
MIRRORS=$(count data/silver/mirrors)
if [ "$MIRRORS" -lt 1000 ]; then
  say "generating mirrors"
  python -m forge.cli mirror --config configs/generation/mirror_minimal.yaml \
    --humans data/silver --out data/silver/mirrors --backend vllm \
    >> /workspace/gen_mirror.log 2>&1 || { say "ABORT: mirror generation failed"; exit 1; }
  MIRRORS=$(count data/silver/mirrors)
fi
say "mirror arm: $MIRRORS documents"
[ "$MIRRORS" -lt 1000 ] && { say "ABORT: mirror arm is empty"; exit 1; }

# ------------------------------------------------------------------ 3. control, arm A
RANDOMS=$(count data/silver/random)
if [ "$RANDOMS" -lt 1000 ]; then
  # Matched to the mirror arm's count, so neither arm is bigger by accident.
  say "generating the control arm at n=$MIRRORS"
  python -m forge.cli generate-random --n "$MIRRORS" --humans data/silver \
    --out data/silver/random --backend vllm \
    >> /workspace/gen_random.log 2>&1 || { say "ABORT: control generation failed"; exit 1; }
  RANDOMS=$(count data/silver/random)
fi
say "control arm: $RANDOMS documents"
[ "$RANDOMS" -lt 1000 ] && { say "ABORT: control arm is empty"; exit 1; }

# ------------------------------------------------------------------ 4. the length gate
#
# Measured on the ASSEMBLED training set, not the raw arm directories. The first version of
# this gate loaded the pools with no ai_cap, no human_cap and no length matching, so it
# refused to train because of a confound the loader was about to remove. A check that
# reports a real number about the wrong data is the failure this project keeps finding.
say "measuring the length confound on the assembled training set"
python scripts/length_gate.py 0.15 >> "$LOG" 2>&1
GATE=$?
if [ "$GATE" -ne 0 ]; then
  say "STOPPING BEFORE TRAINING. See the length_gate block above: it prints each arm's"
  say "human and AI medians and the length-only AUROC on the data that would be trained."
  exit 2
fi
say "gate passed"

# ------------------------------------------------------------------ 5. both arms
for arm in baseline mirror; do
  if [ -f "outputs/forge_min_${arm}/summary.json" ]; then
    say "$arm already trained, skipping"
    continue
  fi
  say "smoke run for $arm"
  python -m forge.cli train --config "configs/training/${arm}_minimal.yaml" --smoke \
    >> "$LOG" 2>&1 || { say "ABORT: $arm smoke run failed"; exit 3; }
  say "training $arm"
  python -m forge.cli train --config "configs/training/${arm}_minimal.yaml" \
    >> "/workspace/train_${arm}.log" 2>&1 || { say "ABORT: $arm training failed"; exit 4; }
  say "$arm finished"
  tail -5 "/workspace/train_${arm}.log" >> "$LOG"
done

say "DONE. Both arms trained. The pod can be stopped now."
