# CPU only. No GPU is required or used anywhere in this image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Hugging Face Spaces runs as a non-root user with $HOME=/home/user; the model cache
    # has to live somewhere writable or every cold start re-downloads several hundred MB.
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    # Threads are capped because the free tier has two vCPUs and torch will otherwise
    # oversubscribe them and get slower, not faster.
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

RUN useradd -m -u 1000 user
WORKDIR /app

COPY space/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app
RUN mkdir -p /app/outputs /home/user/.cache/huggingface && chown -R user /app /home/user

USER user
ENV PYTHONPATH=/app/src:/app

EXPOSE 7860
CMD ["python", "space/serve.py"]
