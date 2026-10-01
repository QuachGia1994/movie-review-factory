from pathlib import Path

import pytest

import movie_review_factory.webapp as webapp


def test_create_server_rejects_non_loopback_without_token(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(webapp, "_DASHBOARD_TOKEN", "")
    with pytest.raises(ValueError, match="DASHBOARD_TOKEN"):
        webapp.create_server("0.0.0.0", 0, tmp_path)


def test_loopback_variants_allowed_without_token(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(webapp, "_DASHBOARD_TOKEN", "")
    assert all(webapp._is_loopback_host(host) for host in ("localhost", "127.0.0.1", "::1"))
    server = webapp.create_server("127.0.0.1", 0, tmp_path)
    server.server_close()


def test_jobs_service_close_is_idempotent_and_stops_owned_worker(tmp_path: Path) -> None:
    service = webapp.JobsService(tmp_path)
    assert service._index_worker_thread.is_alive()
    service.close(timeout=1)
    service.close(timeout=1)
    assert not service._index_worker_thread.is_alive()
    with pytest.raises(RuntimeError, match="shutting down"):
        service.enqueue_index("missing")
