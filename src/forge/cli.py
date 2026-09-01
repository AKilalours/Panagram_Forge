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
def ingest(
    config: str = typer.Option(..., "--config"),
    out: str = typer.Option("data/silver", "--out", help="output root for Parquet"),
    total: int = typer.Option(None, "--total", help="override the document target (smoke runs)"),
) -> None:
    """Phase 1: build FORGE-HUMAN from the configured sources."""
    from forge.ingestion.run import ingest as _ingest

    result = _ingest(config, out_root=out, total=total)
    typer.secho(f"kept {len(result.docs)} documents", fg=typer.colors.GREEN)
    for sid, st in result.stats_by_source.items():
        typer.echo(f"  {sid}: seen={st['seen']} kept={st['kept']} rejected={st['rejected']}")
    for part, n in sorted(result.partitions.items()):
        typer.echo(f"  parquet {part}: {n}")
    typer.echo(f"  manifest: {result.manifest_path}")


@app.command()
def mirror(
    config: str = typer.Option(..., "--config"),
    humans: str = typer.Option("data/silver", "--humans", help="human parquet root"),
    out: str = typer.Option("data/silver/mirrors", "--out"),
    backend: str = typer.Option("fake", "--backend", help="fake | vllm | transformers"),
    limit: int = typer.Option(None, "--limit", help="cap human documents (smoke runs)"),
) -> None:
    """Phase 2: generate synthetic mirrors."""
    from forge.generation.run import run as _run

    if backend == "fake":
        typer.secho(
            "backend=fake: output is a pipeline test, NOT training data.",
            fg=typer.colors.YELLOW,
        )
    result = _run(config, humans_root=humans, out_root=out, backend=backend, limit=limit)
    typer.secho(f"generated {len(result.docs)} mirrors", fg=typer.colors.GREEN)
    for k, v in result.stats.items():
        typer.echo(f"  {k}: {v}")
    for split, n in sorted(result.partitions.items()):
        typer.echo(f"  parquet split={split}: {n}")


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
def mine(
    config: str = typer.Option(..., "--config"),
    reserve: str = typer.Option("data/reserve", "--reserve", help="reserve pool parquet root"),
    out: str = typer.Option("reports/experiments", "--out"),
    model: str = typer.Option(None, "--model", help="path to a trained detector"),
) -> None:
    """Phase 4: hard negative mining pass over the human reserve pool."""
    cfg.load(config)
    if model is None:
        raise PhaseNotImplemented(
            "mining needs a trained detector to score the reserve pool, and Phase 3 "
            "training has not run. The mining, atlas and selection code is implemented "
            "and tested against synthetic failure structure; pass --model once a "
            "checkpoint exists. See docs/roadmap.md."
        )


@app.command()
def gate(
    model_version: str = typer.Option(..., "--model-version"),
    eval_report: str = typer.Option(None, "--eval-report", help="JSON from the evaluation lab"),
    config: str = typer.Option("configs/eval/regimes.yaml", "--config"),
) -> None:
    """Run the release gate against a candidate model's evaluation report."""
    import json
    from pathlib import Path

    from forge.evaluation.release_gate import evaluate

    gate_cfg = cfg.load(config)["release_gate"]
    if eval_report is None:
        raise PhaseNotImplemented(
            "the release gate needs an evaluation report to judge. The gate policy is "
            "implemented and tested; produce a report with `forge evaluate` once a model "
            "exists, then pass it with --eval-report."
        )
    report = json.loads(Path(eval_report).read_text())
    metrics = report.get("headline") or report
    result = evaluate(metrics, gate_cfg)
    if result.passed:
        typer.secho(f"GATE PASSED: {model_version} may be promoted to canary", fg=typer.colors.GREEN)
        return
    typer.secho(f"GATE FAILED: {model_version} must not ship", fg=typer.colors.RED, bold=True)
    for f in result.failures:
        typer.echo(f"  - {f}")
    sys.exit(1)


if __name__ == "__main__":
    app()
