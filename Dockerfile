# FORGE on Google Cloud Run. CPU only.
#
# WHY THE WEIGHTS ARE BAKED IN, NOT FETCHED AT BOOT. Cloud Run scales to zero, so a cold
# start happens whenever nobody has visited recently, which for a portfolio demo is almost
# every visit. Downloading 1.5 GB of checkpoints on each cold start would put a minute of
# network transfer in front of a reviewer's first click and would fail entirely if the Hub
# were slow. Baking them into the image makes the image large and the container start fast,
# which is the correct trade when starts are rare and first impressions are not.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hfcache \
    TOKENIZERS_PARALLELISM=false \
    # Cloud Run gives 2 vCPU on the tier this targets. torch will otherwise oversubscribe
    # and get slower, not faster.
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

# Dependencies first, so a code change does not re-download torch.
COPY space/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "huggingface_hub[cli]>=0.36,<1.0"

# The two trained arms, from the public weights repo. Overridable at build time:
#   --build-arg WEIGHTS_REPO=<owner>/<repo>
ARG WEIGHTS_REPO=Akilalourdes/forge-detect-weights
RUN hf download "$WEIGHTS_REPO" --local-dir /app/outputs \
 && ls -la /app/outputs/forge_min_baseline /app/outputs/forge_min_mirror

COPY . /app

# Warm the image detector into the image's HF cache too. Without this the FIRST image
# upload on every cold start pays a ~350 MB download, which reads as a broken page rather
# than a slow one. Never fatal: a build should not fail because a hub was briefly down.
RUN python -c "\
import os;\
from transformers import AutoImageProcessor, AutoModelForImageClassification as M;\
mid='umm-maybe/AI-image-detector';\
AutoImageProcessor.from_pretrained(mid); M.from_pretrained(mid);\
print('image detector cached')" || echo "image detector not pre-cached; it will download on first use"

# Cloud Run injects PORT; space/serve.py already honours it and defaults to 7860.
ENV PORT=8080
EXPOSE 8080
CMD ["python", "space/serve.py"]
