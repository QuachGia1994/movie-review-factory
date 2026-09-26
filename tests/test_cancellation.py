import threading
import time
from pathlib import Path
from types import SimpleNamespace

from movie_review_factory import cancellation, content_agent, pipeline
from movie_review_factory.models import JobConfig
from movie_review_factory.webapp import JobsService


def wait_done(service: JobsService, job_id: str) -> dict:
    deadline = time.time() + 5
    while time.time() < deadline:
        status = service.status(job_id)
        if not status["running"]:
            return status
        time.sleep(0.01)
    raise AssertionError("run did not stop")


def test_stop_is_idempotent_and_cancelled_run_resumes(tmp_path: Path, monkeypatch) -> None:
    service = JobsService(tmp_path)
    service.create_job({"job_id": "demo"})
    entered = threading.Event()

    def blocking(root, manifest):
        entered.set()
        while True:
            cancellation.cancellable_sleep(0.01)

    monkeypatch.setitem(pipeline.STAGE_HANDLERS, "research", blocking)
    service.start_run("demo")
    assert entered.wait(2)
    assert service.stop_run("demo")["stopping"] is True
    assert service.stop_run("demo")["stopping"] is True
    status = wait_done(service, "demo")
    assert status["cancelled"] is True
    assert next(s for s in status["stages"] if s["stage"] == "research")["status"] == "cancelled"

    monkeypatch.setitem(pipeline.STAGE_HANDLERS, "research", lambda root, manifest: ([], "resumed"))
    service.start_run("demo", until="research")
    status = wait_done(service, "demo")
    assert next(s for s in status["stages"] if s["stage"] == "research")["status"] == "ready"


def test_service_recovers_stale_running_stage(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    pipeline.create_job(root, JobConfig(job_id="demo"))
    manifest = pipeline.load_manifest(root)
    manifest.stage("research").mark("running")
    pipeline.save_manifest(root, manifest)
    service = JobsService(tmp_path)
    stage = next(s for s in service.status("demo")["stages"] if s["stage"] == "research")
    assert stage["status"] == "cancelled"
    assert "restart" in stage["message"]


def test_registered_process_is_terminated_outside_lock(tmp_path: Path, monkeypatch) -> None:
    service = JobsService(tmp_path)
    service.create_job({"job_id": "demo"})
    process = SimpleNamespace(pid=123, poll=lambda: None)
    terminated = []

    def fake_run(root, until=None):
        context = cancellation.current_context()
        context.register_process(process)
        context.event.wait(2)
        raise cancellation.RunCancelled("stopped")

    monkeypatch.setattr(pipeline, "run_job", fake_run)
    monkeypatch.setattr("movie_review_factory.webapp._terminate_process_tree", lambda item: terminated.append(item))
    service.start_run("demo")
    deadline = time.time() + 2
    while service._runs["demo"].get("process") is None and time.time() < deadline:
        time.sleep(0.01)
    service.stop_run("demo")
    assert terminated == [process]


def test_dashboard_exposes_vietnamese_stop_control_and_route() -> None:
    source = Path("src/movie_review_factory/webapp.py").read_text(encoding="utf-8")
    assert 'id="stopBtn"' in source
    assert "Dừng project" in source
    assert 'parts[2] == "stop"' in source
