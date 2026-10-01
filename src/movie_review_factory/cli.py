import sys
from pathlib import Path
from typing import Optional

import typer

from .models import WATERMARK_METHODS, JobConfig, WatermarkDetect, WatermarkRemoval
from .pipeline import (
    approve_metadata,
    approve_script,
    create_job,
    job_status,
    manifest_path,
    run_index,
    run_job,
    validate_job,
)

app = typer.Typer(no_args_is_help=True)


@app.callback()
def _utf8_output() -> None:
    # Windows pipes default to cp1252, which mangles Vietnamese text and "—".
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@app.command()
def init_job(
    path: Path,
    language: str = "vi",
    target_minutes: float = 10,
    aspect_ratio: str = "16:9",
    source_video: Optional[Path] = None,
    movie_title: Optional[str] = None,
    content_agent: str = typer.Option(
        "scaffold",
        help="Content generator for research/outline/script: scaffold, claude, or agy.",
    ),
    watermark_detect: Optional[str] = typer.Option(
        None,
        "--watermark-detect",
        help="Auto-detect & remove a full-frame watermark: color, temporal, or external.",
    ),
    watermark_method: str = typer.Option(
        "propainter",
        "--watermark-method",
        help="Removal method for --watermark-detect: propainter (AI, needs GPU), "
             "delogo (FFmpeg, fast), or blur (FFmpeg, fastest, hides only).",
    ),
    detector_cmd: Optional[str] = typer.Option(
        None,
        "--detector-cmd",
        help="External detector command (with {video}/{out}) for --watermark-detect external; "
             "falls back to MRF_MASK_DETECTOR_CMD when omitted.",
    ),
):
    removal_method = watermark_method.strip().lower()
    if removal_method not in WATERMARK_METHODS:
        raise typer.BadParameter(f"watermark-method must be one of: {', '.join(WATERMARK_METHODS)}")
    watermark_removal = WatermarkRemoval(method=removal_method)
    if watermark_detect is not None:
        method = watermark_detect.strip().lower()
        if method not in ("color", "temporal", "external"):
            raise typer.BadParameter("watermark-detect must be color, temporal, or external")
        watermark_removal = WatermarkRemoval(
            enabled=True,
            method=removal_method,
            detect=WatermarkDetect(method=method, external_cmd=(detector_cmd or "").strip()),
        )
    config = JobConfig(
        job_id=path.name,
        language=language,
        target_minutes=target_minutes,
        aspect_ratio=aspect_ratio,
        source_video=source_video,
        movie_title=movie_title,
        content_agent=content_agent,
        watermark_removal=watermark_removal,
    )
    typer.echo(f"created {create_job(path, config)}")


@app.command("probe-detector")
def probe_detector_cmd(
    video: Path,
    detector_cmd: Optional[str] = typer.Option(
        None,
        "--detector-cmd",
        help="External detector command (with {video}/{out}); falls back to MRF_MASK_DETECTOR_CMD.",
    ),
    timeout: float = typer.Option(120, "--timeout", help="Seconds to wait for the detector."),
):
    """Run the external watermark detector on a single frame to validate it."""
    from .mask_detection import DetectSettings, probe_external_detector

    settings = DetectSettings(method="external", external_cmd=(detector_cmd or "").strip())
    probe = probe_external_detector(video, settings, timeout=timeout)
    typer.echo(probe.message)
    if not probe.ok:
        raise typer.Exit(1)


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
def index(path: Path, force: bool = False):
    """Index a job's source (transcript, scenes, scene-memory) without content.

    Runs only ingest -> transcript -> scenes plus visual/story/embedding
    scene-memory (for claude/agy), building media_index.sqlite3. This is the
    same work the background indexing queue performs after import (roadmap #14);
    a later ``run`` reuses the cached index.
    """
    if not manifest_path(path).exists():
        typer.echo("ERROR: manifest.json is missing (run init-job first)")
        raise typer.Exit(1)
    summary = run_index(path, force=force)
    for stage, status in summary["stages"].items():
        typer.echo(f"{stage:<12} {status}")
    memory = summary.get("scene_memory") or {}
    if memory:
        typer.echo("scene_memory  " + ", ".join(f"{key}={value}" for key, value in memory.items()))
    if any(status == "failed" for status in summary["stages"].values()):
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


@app.command("validate-scenes")
def validate_scenes_cmd(
    path: Path,
    truth: Optional[Path] = typer.Option(
        None,
        "--truth",
        help="Optional JSON ground truth with expected_scene_indexes per script section.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Optional output path; defaults to <job>/scene_validation.json.",
    ),
):
    """Compare lexical/visual scoring against semantic+identity scoring without rendering."""
    from .scene_validation import write_validation_report

    try:
        report = write_validation_report(path, truth_path=truth, output=output)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"ERROR: {exc}")
        raise typer.Exit(1) from exc
    typer.echo(f"scene validation report: {report}")


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
    from . import licensing

    status = licensing.current_status()
    if not status.ok:
        typer.echo(f"[license] Chưa kích hoạt: {status.reason}")
        typer.echo(f"[license] Mã máy (gửi cho nhà cung cấp để lấy key): {status.machine}")
        typer.echo(f"[license] Mở http://{host}:{port} và dán license key để kích hoạt.")

    run_server(host=host, port=port, jobs_root=jobs_root, require_license=True)


@app.command("approve-script")
def approve_script_cmd(
    path: Path,
    confirm: bool = typer.Option(
        False, "--confirm", help="Required: script approval is a deliberate act."
    ),
):
    """Explicitly approve the current script so TTS may consume it."""
    if not confirm:
        typer.echo("Refusing to approve script without --confirm (approval is deliberate).")
        raise typer.Exit(2)
    script = approve_script(path)
    typer.echo(f"approved script for {script.get('job_id', path.name)}")


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
