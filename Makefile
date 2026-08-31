.PHONY: help setup install-data install-train lint fmt test spec ingest mirror train eval mine serve up down clean

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

ingest:           ## Phase 1: build FORGE-HUMAN from configured sources
	./.venv/bin/python -m forge.cli ingest --config configs/data/human.yaml

mirror:           ## Phase 2: generate synthetic mirrors
	./.venv/bin/python -m forge.cli mirror --config configs/generation/mirror.yaml

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
