#!/usr/bin/env bash
# The out-of-distribution run: both arms against three benchmarks they have never seen.
#
# WHY THIS IS THE RUN THAT SETTLES THE QUESTION. The in-distribution comparison finished
# saturated and underpowered: AUROC 0.99997 on both arms, and a two-proportion test on the
# FNR gap gave p = 0.217. Nine misses against fifteen cannot separate two models. RAID,
# MAGE and HC3 carry generators neither arm has seen, which is the question the project
# actually asks: does matched generation transfer.
#
# Idempotent, like run_all.sh. Each (arm, benchmark) writes its own JSON and is skipped if
# that file exists, so a reclaimed GPU costs one cell rather than the whole grid.
#
# Six cells, roughly 20 minutes each at 4,000 documents.

set -u
cd "$(dirname "$0")/.."

LOG=/workspace/eval_all.log
LIMIT="${LIMIT:-4000}"
# Which benchmarks to run. Overridable so a benchmark that needs investigation does not
# hold up the ones that work: BENCHMARKS="hc3 mage" bash scripts/eval_all.sh
BENCHMARKS="${BENCHMARKS:-hc3 mage raid}"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== out-of-distribution evaluation, limit=$LIMIT per benchmark"

for arm in baseline mirror; do
  if [ ! -f "outputs/forge_min_${arm}/best.pt" ]; then
    say "ABORT: no checkpoint for $arm. Train it first."
    exit 1
  fi
done

FAILED=0
for benchmark in $BENCHMARKS; do
  for arm in baseline mirror; do
    target="reports/experiments/ood_${arm}_${benchmark}.json"
    if [ -f "$target" ]; then
      say "$arm on $benchmark already done, skipping"
      continue
    fi
    say "scoring $arm on $benchmark"
    if python scripts/eval_ood.py --arm "$arm" --benchmark "$benchmark" --limit "$LIMIT" \
        >> "$LOG" 2>&1; then
      say "  done"
    else
      # A benchmark that will not download must not stop the other five cells.
      say "  FAILED on $benchmark for $arm; see the traceback above. Continuing."
      FAILED=$((FAILED + 1))
    fi
  done
done

say "writing the comparison table"
python scripts/ood_table.py >> "$LOG" 2>&1 || say "  table generation failed"

if [ "$FAILED" -gt 0 ]; then
  say "DONE with $FAILED failed cells. The rest are in reports/experiments/."
else
  say "DONE. All six cells written. The pod can be stopped now."
fi
