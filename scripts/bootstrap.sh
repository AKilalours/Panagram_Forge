#!/usr/bin/env bash
# One-command setup for a fresh clone.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install -e ".[dev]"
./.venv/bin/pytest
./.venv/bin/python -m forge.cli spec-check
echo "FORGE scaffold verified."
