"""FORGE command line entrypoint.

Commands that belong to phases which have not started fail loudly with the phase
they need, rather than pretending to run. A stub that silently returns success is
worse than no stub, because it hides that the work has not been done.
"""

from __future__ import annotations

import sys

import typer

from forge.common import config as cfg

app = typer.Typer(add_completion=False, help="FORGE: failure-driven AI-content detection")


class PhaseNotImplemented(RuntimeError):
    pass


def _not_yet(phase: str, what: str) -> None:
    raise PhaseNotImplemented(
        f"{what} is Phase {phase} and is not implemented yet. "
        f"See docs/roadmap.md for what has to land first."
    )


@app.command("spec-check")
def spec_check() -> None:
    """Fail if any config has drifted from the frozen Data Spec v1."""
    problems = cfg.spec_check()
    if problems:
        typer.secho(f"spec-check FAILED ({len(problems)} violation(s)):", fg=typer.colors.RED, bold=True)
        for p in problems:
            typer.echo(f"  - {p}")
        sys.exit(1)
    typer.secho("spec-check OK: configs match data_spec_v1", fg=typer.colors.GREEN)


@app.command()
def ingest(config: str = typer.Option(..., "--config")) -> None:
    """Phase 1: build FORGE-HUMAN from the configured sources."""
    cfg.load(config)
    _not_yet("1", "ingestion")


@app.command()
def mirror(config: str = typer.Option(..., "--config")) -> None:
    """Phase 2: generate synthetic mirrors."""
    cfg.load(config)
    _not_yet("2", "the mirror engine")


@app.command()
def train(config: str = typer.Option(..., "--config")) -> None:
    """Phase 3: train a detector."""
    cfg.load(config)
    _not_yet("3", "training")


@app.command()
def evaluate(config: str = typer.Option(..., "--config")) -> None:
    """Run the evaluation lab across all five regimes plus external benchmarks."""
    cfg.load(config)
    _not_yet("3", "the evaluation lab")


@app.command()
def mine(config: str = typer.Option(..., "--config")) -> None:
    """Phase 4: hard negative mining pass over the human reserve pool."""
    cfg.load(config)
    _not_yet("4", "hard negative mining")


@app.command()
def gate(model_version: str = typer.Option(..., "--model-version")) -> None:
    """Phase 8: run the release gate against a candidate model."""
    _not_yet("8", "the release gate")


if __name__ == "__main__":
    app()
