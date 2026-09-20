"""Tests for the local web UI/API service, HTTP layer, and new CLI commands.

Every test operates in tmp_path - none of them touch the repository's
jobs/smoke-test job, whose approval gate must stay untouched.
"""

import importlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from typer.testing import CliRunner

import movie_review_factory.pipeline as pipeline
import movie_review_factory.webapp as webapp_mod
from movie_review_factory.cli import app
from movie_review_factory.models import JobConfig
from movie_review_factory.webapp import JobsService, create_server

runner = CliRunner()


def _job_with_metadata(jobs_root: Path, job_id: str = "demo") -> Path:
    """Create a media-less job and run it up to the metadata draft.

    With no source video the media stages honestly skip, but research/outline/
    script/scene_plan/metadata still run, so youtube_metadata.json exists with
    approved=false - exactly the state the UI reviews.
    """
    root = jobs_root / job_id
    pipeline.create_job(root, JobConfig(job_id=job_id))
    pipeline.run_job(root, until="metadata")
    return root


# --- JobsService ------------------------------------------------------------


def test_create_job_status_is_localized(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    status = svc.create_job({"job_id": "demo", "language": "vi"})
    assert status["job_id"] == "demo"
    by_stage = {s["stage"]: s for s in status["stages"]}
    assert by_stage["publish"]["stage_label"] == "Xuất bản"
    assert by_stage["ingest"]["status_label"] == "Chờ xử lý"
    assert status["approvals"]["metadata_present"] is False


def test_list_jobs(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "alpha"})
    svc.create_job({"job_id": "beta"})
    ids = {j["job_id"] for j in svc.list_jobs()}
    assert ids == {"alpha", "beta"}


def test_duplicate_job_rejected(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "demo"})
    with pytest.raises(FileExistsError):
        svc.create_job({"job_id": "demo"})


def test_invalid_job_id_rejected(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    for bad in ("../evil", "..", ".", "a/b", "space name"):
        with pytest.raises(ValueError):
            svc.create_job({"job_id": bad})


def test_artifact_path_blocks_traversal(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    _job_with_metadata(tmp_path, "demo")
    # A legitimate artifact resolves.
    assert svc.artifact_path("demo", "research.json").name == "research.json"
    # Traversal / bad names are refused.
    for bad in ("..", "../manifest.json", "a/b", "..%2f"):
        with pytest.raises((ValueError, FileNotFoundError)):
            svc.artifact_path("demo", bad)


def test_status_remains_readable_while_run_persists_manifest(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "demo"})
    svc.start_run("demo")

    deadline = time.time() + 10
    while time.time() < deadline:
        # Every concurrent read must succeed; discard a potentially stale
        # snapshot once the worker has finished.
        if not svc.status("demo")["running"]:
            break
        time.sleep(0.01)
    else:
        pytest.fail("background run did not finish")

    status = svc.status("demo")
    assert status["running"] is False
    assert status["approvals"]["metadata_present"] is True


def test_start_run_reaches_metadata_and_stops(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "demo"})
    svc.start_run("demo")
    # run_job persists per-stage; poll until the background run finishes.
    deadline = time.time() + 10
    while time.time() < deadline and svc.status("demo")["running"]:
        time.sleep(0.1)
    status = svc.status("demo")
    assert status["running"] is False
    assert status["approvals"]["metadata_present"] is True
    # The UI run must never advance the publish stage on its own.
    publish = next(s for s in status["stages"] if s["stage"] == "publish")
    assert publish["status"] == "pending"
    assert not (tmp_path / "demo" / "publish_record.json").exists()


# --- approval + publishing gate ---------------------------------------------


def test_publish_gate_requires_confirm(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    _job_with_metadata(tmp_path, "demo")
    result = svc.prepare_publish("demo", confirm=False)
    assert result["confirmed"] is False
    assert result["published"] is False
    assert not (tmp_path / "demo" / "publish_record.json").exists()


def test_publish_blocked_until_approved_then_succeeds(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    root = _job_with_metadata(tmp_path, "demo")

    # Nothing approved yet: confirmed publish stays blocked, writes no record.
    blocked = svc.prepare_publish("demo", confirm=True)
    assert blocked["published"] is False
    assert "Chưa thể xuất bản" in blocked["message_vi"]
    assert not (root / "publish_record.json").exists()

    # Approve the script (human step) and the metadata (explicit endpoint).
    script = json.loads((root / "script.json").read_text(encoding="utf-8"))
    script["approved"] = True
    (root / "script.json").write_text(json.dumps(script), encoding="utf-8")
    approved = svc.approve_metadata("demo")
    assert approved["metadata"]["approved"] is True

    ok = svc.prepare_publish("demo", confirm=True)
    assert ok["published"] is True
    assert (root / "publish_record.json").exists()


def test_update_metadata_clears_approval(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    _job_with_metadata(tmp_path, "demo")
    svc.approve_metadata("demo")
    assert svc.status("demo")["approvals"]["metadata_approved"] is True
    svc.update_metadata("demo", {"title": "Tiêu đề mới"})
    status = svc.status("demo")
    assert status["approvals"]["metadata_approved"] is False
    assert svc.get_metadata("demo")["metadata"]["title"] == "Tiêu đề mới"


def test_script_get_update_approve(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    root = _job_with_metadata(tmp_path, "demo")
    result = svc.get_script("demo")
    assert result["present"] is True
    assert result["script"]["approved"] is False

    updated = svc.update_script("demo", {"notes": "test note"})
    assert updated["approved"] is False
    assert updated["script"]["notes"] == "test note"

    approved = svc.approve_script("demo")
    assert approved["approved"] is True
    persisted = json.loads((root / "script.json").read_text(encoding="utf-8"))
    assert persisted["approved"] is True

    svc.update_script("demo", {"notes": "changed"})
    assert svc.status("demo")["approvals"]["script_approved"] is False


# --- HTTP layer -------------------------------------------------------------


def _serve(jobs_root: Path):
    server = create_server("127.0.0.1", 0, jobs_root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    return server, base


def _get_json(url: str, headers: dict | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        body = response.read().decode("utf-8")
        return response.status, (json.loads(body) if body else {})


def test_http_index_and_job_lifecycle(tmp_path: Path) -> None:
    server, base = _serve(tmp_path)
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            html = response.read().decode("utf-8")
        assert "Xưởng Review Phim" in html

        status_code, created = _post_json(base + "/api/jobs", {"job_id": "demo"})
        assert status_code == 201
        assert created["job_id"] == "demo"

        _, listing = _get_json(base + "/api/jobs")
        assert any(j["job_id"] == "demo" for j in listing["jobs"])

        _, status = _get_json(base + "/api/jobs/demo")
        assert any(s["stage_label"] == "Xuất bản" for s in status["stages"])
    finally:
        server.shutdown()
        server.server_close()


def test_http_missing_job_returns_404(tmp_path: Path) -> None:
    server, base = _serve(tmp_path)
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _get_json(base + "/api/jobs/nope")
        assert excinfo.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_http_artifact_supports_range(tmp_path: Path) -> None:
    _job_with_metadata(tmp_path, "demo")
    server, base = _serve(tmp_path)
    try:
        request = urllib.request.Request(
            base + "/api/jobs/demo/artifacts/research.json",
            headers={"Range": "bytes=0-3"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 206
            assert response.headers["Accept-Ranges"] == "bytes"
            assert len(response.read()) == 4
    finally:
        server.shutdown()
        server.server_close()


def test_http_script_routes(tmp_path: Path) -> None:
    _job_with_metadata(tmp_path, "demo")
    server, base = _serve(tmp_path)
    try:
        _, result = _get_json(base + "/api/jobs/demo/script")
        assert result["present"] is True

        _, updated = _post_json(base + "/api/jobs/demo/script", {"notes": "http note"})
        assert updated["approved"] is False

        _, approved = _post_json(base + "/api/jobs/demo/script/approve", {})
        assert approved["approved"] is True
        persisted = json.loads((tmp_path / "demo" / "script.json").read_text(encoding="utf-8"))
        assert persisted["approved"] is True
    finally:
        server.shutdown()
        server.server_close()


# --- new CLI commands -------------------------------------------------------


def test_cli_approve_metadata_requires_confirm(tmp_path: Path) -> None:
    root = _job_with_metadata(tmp_path, "demo")
    meta_path = root / "youtube_metadata.json"

    result = runner.invoke(app, ["approve-metadata", str(root)])
    assert result.exit_code == 2
    assert json.loads(meta_path.read_text(encoding="utf-8"))["approved"] is False

    result = runner.invoke(app, ["approve-metadata", str(root), "--confirm"])
    assert result.exit_code == 0
    assert json.loads(meta_path.read_text(encoding="utf-8"))["approved"] is True


def test_cli_publish_requires_confirm_and_respects_gate(tmp_path: Path) -> None:
    root = _job_with_metadata(tmp_path, "demo")

    # No --confirm: refuses.
    result = runner.invoke(app, ["publish", str(root)])
    assert result.exit_code == 2
    assert not (root / "publish_record.json").exists()

    # Confirmed but unapproved: gate blocks, exit 1, still no record.
    result = runner.invoke(app, ["publish", str(root), "--confirm"])
    assert result.exit_code == 1
    assert not (root / "publish_record.json").exists()


# --- security: auth, headers, oversized body --------------------------------


def _serve_with_token(jobs_root: Path, token: str):
    """Start a server with DASHBOARD_TOKEN set via module-level reload."""
    orig = webapp_mod._DASHBOARD_TOKEN
    webapp_mod._DASHBOARD_TOKEN = token
    server = create_server("127.0.0.1", 0, jobs_root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    return server, base, orig


def _restore_token(server, orig):
    server.shutdown()
    server.server_close()
    webapp_mod._DASHBOARD_TOKEN = orig


def test_security_headers_present_no_auth(tmp_path: Path) -> None:
    """Security headers appear on every response even without auth."""
    server, base = _serve(tmp_path)
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as resp:
            assert resp.headers["X-Content-Type-Options"] == "nosniff"
            assert resp.headers["X-Frame-Options"] == "DENY"
            assert "strict-origin" in resp.headers["Referrer-Policy"]
            assert resp.headers["Content-Security-Policy"]
    finally:
        server.shutdown()
        server.server_close()


def test_security_headers_present_on_json(tmp_path: Path) -> None:
    server, base = _serve(tmp_path)
    try:
        _, _ = _post_json(base + "/api/jobs", {"job_id": "sec-test"})
        with urllib.request.urlopen(base + "/api/jobs", timeout=5) as resp:
            assert resp.headers["X-Content-Type-Options"] == "nosniff"
    finally:
        server.shutdown()
        server.server_close()


def test_auth_required_when_token_set(tmp_path: Path) -> None:
    """401 is returned when DASHBOARD_TOKEN is set and no header provided."""
    server, base, orig = _serve_with_token(tmp_path, "s3cr3t")
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/api/jobs", timeout=5)
        assert exc.value.code == 401
        assert "Bearer" in exc.value.headers.get("WWW-Authenticate", "")
    finally:
        _restore_token(server, orig)


def test_auth_passes_with_correct_token(tmp_path: Path) -> None:
    server, base, orig = _serve_with_token(tmp_path, "s3cr3t")
    try:
        req = urllib.request.Request(
            base + "/api/jobs",
            headers={"Authorization": "Bearer s3cr3t"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
    finally:
        _restore_token(server, orig)


def test_auth_fails_with_wrong_token(tmp_path: Path) -> None:
    server, base, orig = _serve_with_token(tmp_path, "s3cr3t")
    try:
        req = urllib.request.Request(
            base + "/api/jobs",
            headers={"Authorization": "Bearer wrongtoken"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 401
    finally:
        _restore_token(server, orig)


def test_no_auth_when_token_absent(tmp_path: Path) -> None:
    """When DASHBOARD_TOKEN is empty every request succeeds without a header."""
    orig = webapp_mod._DASHBOARD_TOKEN
    webapp_mod._DASHBOARD_TOKEN = ""
    server, base = _serve(tmp_path)
    try:
        with urllib.request.urlopen(base + "/api/jobs", timeout=5) as resp:
            assert resp.status == 200
    finally:
        server.shutdown()
        server.server_close()
        webapp_mod._DASHBOARD_TOKEN = orig


def test_oversized_body_returns_413(tmp_path: Path) -> None:
    server, base = _serve(tmp_path)
    try:
        big = json.dumps({"job_id": "x", "pad": "A" * (1_048_576 + 1)}).encode()
        req = urllib.request.Request(
            base + "/api/jobs",
            data=big,
            headers={"Content-Type": "application/json", "Content-Length": str(len(big))},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 413
    finally:
        server.shutdown()
        server.server_close()


def test_unexpected_exception_hides_details(tmp_path: Path) -> None:
    """When auth is on, unhandled exceptions return generic 500 without stack text."""
    server, base, orig = _serve_with_token(tmp_path, "tok")
    try:
        # Trigger a 500 by pointing at a job that exists structurally but
        # whose jobs_root we corrupt right after creation.
        req_create = urllib.request.Request(
            base + "/api/jobs",
            data=json.dumps({"job_id": "boom"}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer tok"},
            method="POST",
        )
        urllib.request.urlopen(req_create, timeout=5)
        # Delete the manifest so status() raises unexpectedly deep inside.
        manifest = tmp_path / "boom" / pipeline.MANIFEST_NAME
        manifest.unlink()
        req_status = urllib.request.Request(
            base + "/api/jobs/boom",
            headers={"Authorization": "Bearer tok"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req_status, timeout=5)
        assert exc.value.code in (404, 500)
        body = json.loads(exc.value.read().decode())
        assert "Traceback" not in body.get("error", "")
        assert "Traceback" not in body.get("error_vi", "")
    finally:
        _restore_token(server, orig)


def test_auth_header_not_required_on_html_page_without_token(tmp_path: Path) -> None:
    """The HTML page is accessible without auth when no token is configured."""
    orig = webapp_mod._DASHBOARD_TOKEN
    webapp_mod._DASHBOARD_TOKEN = ""
    server, base = _serve(tmp_path)
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as resp:
            assert resp.status == 200
            assert "Xưởng Review Phim" in resp.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        webapp_mod._DASHBOARD_TOKEN = orig
