from pathlib import Path
from typing import Optional

import typer

from .models import JobConfig
from .pipeline import (
    approve_metadata,
    create_job,
    job_status,
    manifest_path,
    run_job,
    validate_job,
)

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


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    jobs_root: Path = Path("jobs"),
):
    """Launch the local web UI for creating and reviewing jobs.

    Serves a single-page dashboard (job creation, progress, artifact review,
    final-render preview, metadata approval, and the publish gate) with a
    stdlib-only server. Binds to localhost by default.
    """
    from .webapp import run_server

    run_server(host=host, port=port, jobs_root=jobs_root)


@app.command("approve-metadata")
def approve_metadata_cmd(
    path: Path,
    confirm: bool = typer.Option(
        False, "--confirm", help="Required: approval is a deliberate act."
    ),
):
    """Explicitly approve a job's YouTube metadata (required before publish).

    Refuses to act without --confirm so approval is never accidental. Never
    uploads anything and never runs the publish stage.
    """
    if not confirm:
        typer.echo("Refusing to approve metadata without --confirm (approval is deliberate).")
        raise typer.Exit(2)
    meta = approve_metadata(path)
    typer.echo(f"approved metadata for {meta.get('job_id', path.name)}")


@app.command()
def publish(
    path: Path,
    confirm: bool = typer.Option(
        False, "--confirm", help="Required: publishing is a deliberate act."
    ),
):
    """Run the publish gate. Writes publish_record.json only when approvals are
    complete; never uploads. Refuses without --confirm."""
    if not manifest_path(path).exists():
        typer.echo("ERROR: manifest.json is missing")
        raise typer.Exit(1)
    if not confirm:
        typer.echo(
            "Refusing to publish without --confirm "
            "(publishing is deliberate; this only writes a handoff record, never uploads)."
        )
        raise typer.Exit(2)
    manifest = run_job(path, until="publish")
    stage = manifest.stage("publish")
    status_text = stage.status if stage else "missing"
    message = stage.message if stage else "publish stage not found"
    typer.echo(f"publish: {status_text} - {message}")
    if status_text != "ready":
        raise typer.Exit(1)
