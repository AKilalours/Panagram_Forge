#!/usr/bin/env bash
# One-shot setup on a fresh RunPod pod. Fails fast and loudly.
#
#   Template: RunPod PyTorch 2.x (CUDA 12.x)
#   Volume:   mount a network volume at /workspace so data survives pod restarts
#
# Usage:  bash scripts/runpod_bootstrap.sh 2>&1 | tee /workspace/bootstrap.log
set -euo pipefail

log() { printf '\n== %s ==\n' "$1"; }

log "GPU"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

log "existing torch"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
assert torch.cuda.is_available(), "no CUDA device visible"
if not torch.cuda.is_bf16_supported():
    raise SystemExit(
        "This GPU has no bf16 (T4 or older). DeBERTa-v3 in fp16 is known to produce NaN "
        "losses, so pick an Ampere-or-newer GPU (A40, A100, 4090, L40S) or switch the "
        "config to fp32."
    )
PY

# The RunPod image already ships a torch built against the pod's CUDA. Installing our
# extras would let pip resolve torch from PyPI and replace that build with a generic
# wheel, which silently costs CUDA. So torch is pinned to whatever is already installed
# and pip is forbidden from moving it.
TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"
log "protecting torch $TORCH_VER from being replaced"
echo "torch==${TORCH_VER}" > /tmp/forge-constraints.txt

log "install"
pip install -q -U pip
pip install -q -c /tmp/forge-constraints.txt -e ".[data,train,mining,serve]"

log "install vllm"
# vllm pins torch tightly. If it cannot satisfy the constraint, install it WITHOUT
# letting it move torch and report that generation may need a matching vllm build.
pip install -q -c /tmp/forge-constraints.txt vllm || {
    echo "WARNING: vllm could not install against torch ${TORCH_VER}."
    echo "Generation with --backend vllm will not work. Use --backend transformers,"
    echo "which is slower but correct, or pick a pod image whose torch matches vllm."
}

log "verify torch survived the install"
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda", ok)
assert ok, (
    "CUDA is gone after installing dependencies. pip replaced the image's torch with a "
    "CPU wheel. Reinstall the pod image's torch build before continuing."
)
PY

log "verify the package and tests"
python -m forge.cli spec-check
pytest -q

log "ready"
cat <<'MSG'
Next:
  forge pin-revisions --config configs/generation/generators_minimal.yaml --write
  make min-ingest
MSG
