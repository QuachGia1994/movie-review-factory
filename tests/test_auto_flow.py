"""MRF_AUTO_RUN chaining: import -> script (auto-approved) -> video + cover -> metadata -> handoff."""
from __future__ import annotations

import io
import time
from pathlib import Path

import pytest

import movie_review_factory.pipeline as pipeline
import movie_review_factory.webapp as webapp_mod
from movie_review_factory.webapp import JobsService


@pytest.fixture()
def auto_svc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JobsService:
    monkeypatch.setenv("MRF_AUTO_RUN", "1")
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "demo"})
    return svc


def _record_runs(svc: JobsService, monkeypatch: pytest.MonkeyPatch) -> list:
    runs: list = []
    monkeypatch.setattr(svc, "start_run", lambda job_id, **kw: runs.append((job_id, kw)) or {"started": True})
    return runs


def _wait_idle(svc: JobsService, job_id: str = "demo") -> None:
    deadline = time.time() + 10
    while time.time() < deadline and svc.status(job_id)["running"]:
        time.sleep(0.05)


def test_import_starts_run_to_script_gate_in_auto_mode(auto_svc: JobsService, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _record_runs(auto_svc, monkeypatch)
    data = b"\x00" * 64
    auto_svc.import_video("demo", "source.mp4", len(data), io.BytesIO(data))
    assert runs == [("demo", {})]  # until=None -> script gate while unapproved


def test_manual_mode_import_never_starts_a_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = JobsService(tmp_path)  # conftest sets MRF_AUTO_RUN=0
    svc.create_job({"job_id": "demo"})
    runs = _record_runs(svc, monkeypatch)
    data = b"\x00" * 64
    svc.import_video("demo", "source.mp4", len(data), io.BytesIO(data))
    assert runs == []


def test_approve_script_auto_continues_the_run(auto_svc: JobsService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline.run_job(tmp_path / "demo", until="script")
    runs = _record_runs(auto_svc, monkeypatch)
    result = auto_svc.approve_script("demo")
    assert result["approved"] is True and result["auto_run"] is True
    assert runs == [("demo", {})]


def test_full_run_preselects_cover_only_in_auto_mode(auto_svc: JobsService, monkeypatch: pytest.MonkeyPatch) -> None:
    covers: list = []
    monkeypatch.setattr(webapp_mod.pipeline, "run_job", lambda root, until=None: ("manifest", until))
    monkeypatch.setattr(auto_svc, "_auto_cover", lambda job_id, root, manifest: covers.append(manifest))

    auto_svc.start_run("demo", until="script")
    _wait_idle(auto_svc)
    auto_svc.start_run("demo", until=webapp_mod.RUN_UNTIL_STAGE)
    _wait_idle(auto_svc)

    assert covers == [("manifest", webapp_mod.RUN_UNTIL_STAGE)]


class _Manifest:
    def __init__(self, until):
        self.until = until
        self.stages = [type("Stage", (), {"stage": "script", "status": "ready"})()]


def _fake_one_pass(svc: JobsService, monkeypatch: pytest.MonkeyPatch, approve=None) -> tuple[list, list, list]:
    calls: list = []
    approvals: list = []
    covers: list = []
    monkeypatch.setattr(webapp_mod.pipeline, "run_job", lambda root, until=None: calls.append(until) or _Manifest(until))
    monkeypatch.setattr(webapp_mod.pipeline, "approve_script", approve or (lambda root: approvals.append(root.name) or {}))
    monkeypatch.setattr(svc, "_auto_cover", lambda job_id, root, manifest: covers.append(manifest.until))
    return calls, approvals, covers


def test_auto_run_goes_from_script_to_video_in_one_pass(auto_svc: JobsService, monkeypatch: pytest.MonkeyPatch) -> None:
    calls, approvals, covers = _fake_one_pass(auto_svc, monkeypatch)
    assert auto_svc.start_run("demo") == {"started": True, "until": "script"}
    _wait_idle(auto_svc)
    assert calls == ["script", webapp_mod.RUN_UNTIL_STAGE]
    assert approvals == ["demo"]
    assert covers == [webapp_mod.RUN_UNTIL_STAGE]
    assert auto_svc._runs["demo"]["error"] is None
    assert auto_svc.status("demo")["approvals"]["auto_flow"] is True


def test_auto_run_stops_at_script_when_auto_approve_is_rejected(auto_svc: JobsService, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(root):
        raise ValueError("script evidence tags need review: [S9]")
    calls, _, covers = _fake_one_pass(auto_svc, monkeypatch, approve=reject)
    auto_svc.start_run("demo")
    _wait_idle(auto_svc)
    assert calls == ["script"] and covers == []
    assert "Tự duyệt kịch bản dừng lại" in str(auto_svc._runs["demo"]["error"])


def test_script_review_env_keeps_the_script_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRF_AUTO_RUN", "1")
    monkeypatch.setenv("MRF_SCRIPT_REVIEW", "1")
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "demo"})
    calls, approvals, covers = _fake_one_pass(svc, monkeypatch)
    svc.start_run("demo")
    _wait_idle(svc)
    assert calls == ["script"] and approvals == [] and covers == []
    assert svc.status("demo")["approvals"]["auto_flow"] is False


def test_approve_metadata_builds_handoff_but_never_publishes(auto_svc: JobsService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "demo"
    pipeline.run_job(root, until="metadata")
    pipeline.approve_script(root)
    handoffs: list = []
    monkeypatch.setattr(auto_svc, "build_handoff", lambda job_id: handoffs.append(job_id) or {"ready": True})

    result = auto_svc.approve_metadata("demo")

    assert result["approved"] is True and result["handoff"] == {"ready": True}
    assert handoffs == ["demo"]
    assert not (root / "publish_record.json").exists()
    assert pipeline.load_manifest(root).stage("publish").status != "ready"
