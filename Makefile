.PHONY: help setup install-data install-train lint fmt test spec smoke smoke-mirror min-ingest min-mirror min-smoke min-train ingest mirror train eval mine serve up down clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:            ## Create venv and install the package in editable mode with dev extras
	python3 -m venv .venv
	./.venv/bin/pip install -U pip
	./.venv/bin/pip install -e ".[dev]"

install-data:     ## Install Phase 1 ingestion dependencies
	./.venv/bin/pip install -e ".[data]"

install-train:    ## Install Phase 3 training dependencies
	./.venv/bin/pip install -e ".[train,mining]"

lint:             ## Static checks
	./.venv/bin/ruff check src tests
	./.venv/bin/mypy src || true

fmt:              ## Auto-format
	./.venv/bin/ruff check --fix src tests
	./.venv/bin/ruff format src tests

test:             ## Run the test suite
	./.venv/bin/pytest

spec:             ## Validate that configs match the frozen Data Spec v1
	./.venv/bin/python -m forge.cli spec-check

smoke:            ## Offline end-to-end run of the Phase 1 pipeline on a fixture corpus
	./.venv/bin/python scripts/make_fixture_corpus.py --n 300
	./.venv/bin/python -m forge.cli ingest --config configs/data/local_smoke.yaml --out data/silver/smoke --total 5000

min-ingest:       ## Minimal scale: 60k train + 500k reserve, short docs
	./.venv/bin/python -m forge.cli ingest --config configs/data/human_minimal.yaml --out data/silver

min-mirror:       ## Minimal scale: 60k mirrors from 4 small held-in families
	./.venv/bin/python -m forge.cli mirror --config configs/generation/mirror_minimal.yaml \
		--humans data/silver --out data/silver/mirrors --backend vllm

min-smoke:        ## 20 training steps on 200 examples. ALWAYS run this before a paid run
	./.venv/bin/python -m forge.cli train --config configs/training/baseline_minimal.yaml --smoke

min-train:        ## Minimal scale: baseline arm
	./.venv/bin/python -m forge.cli train --config configs/training/baseline_minimal.yaml

ingest:           ## Phase 1: build FORGE-HUMAN from configured sources
	./.venv/bin/python -m forge.cli ingest --config configs/data/human.yaml

smoke-mirror:     ## Offline end-to-end run of the Phase 2 mirror engine (fake generator)
	./.venv/bin/python -m forge.cli mirror --config configs/generation/mirror.yaml \
		--humans data/silver/smoke --out data/silver/smoke-mirrors --backend fake

mirror:           ## Phase 2: generate synthetic mirrors (needs a GPU and pinned revisions)
	./.venv/bin/python -m forge.cli mirror --config configs/generation/mirror.yaml \
		--humans data/silver --out data/silver/mirrors --backend vllm

train:            ## Phase 3: train a detector
	./.venv/bin/python -m forge.cli train --config configs/training/baseline.yaml

eval:             ## Run the evaluation lab across all five regimes
	./.venv/bin/python -m forge.cli evaluate --config configs/eval/regimes.yaml

mine:             ## Phase 4: hard negative mining pass
	./.venv/bin/python -m forge.cli mine --config configs/training/hard_negative.yaml

serve:            ## Run the inference API locally
	./.venv/bin/uvicorn api.main:app --reload --port 8000

up:               ## Start local infra (MinIO, Prometheus, Grafana)
	docker compose -f infra/docker/docker-compose.yml up -d

down:
	docker compose -f infra/docker/docker-compose.yml down

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
