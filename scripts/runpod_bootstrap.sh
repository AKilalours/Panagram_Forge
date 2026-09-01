#!/usr/bin/env bash
# One-shot setup on a fresh RunPod pod. Fails fast and loudly.
#
#   Template: RunPod PyTorch 2.x (CUDA 12.x)
#   Volume:   mount a 50 GB network volume at /workspace so data survives pod restarts
#
# Usage:  bash scripts/runpod_bootstrap.sh
set -euo pipefail

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device visible"
print("torch", torch.__version__, "| bf16:", torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported():
    raise SystemExit(
        "This GPU has no bf16 (T4 or older). DeBERTa-v3 in fp16 is known to produce NaN "
        "losses, so pick an Ampere-or-newer GPU (A40, A100, 4090, L40S) or switch the "
        "config to fp32."
    )
PY

echo "== install =="
pip install -q -U pip
pip install -q -e ".[data,train,mining,serve]"
pip install -q vllm

echo "== verify =="
python -m forge.cli spec-check
pytest -q

echo
echo "Bootstrap OK. Next:"
echo "  forge pin-revisions --config configs/generation/generators_minimal.yaml --write"
echo "  make -f Makefile min-ingest"
