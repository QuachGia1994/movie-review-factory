from pathlib import Path
from typing import Optional

import typer

from .models import JobConfig
from .pipeline import create_job, validate_job, run_job, job_status, manifest_path

app = typer.Typer(no_args_is_help=True)


@app.command()
def init_job(
    path: Path,
    language: str = "vi",
    target_minutes: float = 10,
    aspect_ratio: str = "16:9",
    source_video: Optional[Path] = None,
):
    config = JobConfig(
        job_id=path.name,
        language=language,
        target_minutes=target_minutes,
        aspect_ratio=aspect_ratio,
        source_video=source_video,
    )
    typer.echo(f"created {create_job(path, config)}")


@app.command("validate-job")
def validate_job_cmd(path: Path):
    errors = validate_job(path)
    if errors:
        for error in errors:
            typer.echo(f"ERROR: {error}")
        raise typer.Exit(1)
    typer.echo("PASS: job manifest valid")


@app.command()
def run(path: Path, force: bool = False, until: Optional[str] = None):
    """Run the pipeline for a job, resuming from where it left off."""
    if not manifest_path(path).exists():
        typer.echo("ERROR: manifest.json is missing (run init-job first)")
        raise typer.Exit(1)
    manifest = run_job(path, force=force, until=until)
    for s in manifest.stages:
        line = f"{s.stage:<12} {s.status}"
        if s.message:
            line += f" - {s.message}"
        typer.echo(line)
    if any(s.status == "failed" for s in manifest.stages):
        raise typer.Exit(1)


@app.command()
def status(path: Path):
    """Show per-stage status for a job."""
    if not manifest_path(path).exists():
        typer.echo("ERROR: manifest.json is missing")
        raise typer.Exit(1)
    info = job_status(path)
    typer.echo(f"job: {info['job_id']}  complete: {info['complete']}")
    typer.echo(f"counts: {info['counts']}")
    for s in info["stages"]:
        line = f"  {s['stage']:<12} {s['status']}"
        if s["message"]:
            line += f" - {s['message']}"
        typer.echo(line)
