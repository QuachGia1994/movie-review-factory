"""Offline tests for the 1-click library installer (POST /api/system/install-package).

The pip runner is injected so these never touch the network or real pip.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from movie_review_factory.webapp import JobsService


def test_install_package_rejects_non_whitelisted(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    with pytest.raises(ValueError):
        svc.install_package("requests")  # not whitelisted
    with pytest.raises(ValueError):
        svc.install_package("vieneu; rm -rf /")  # injection attempt


def test_install_package_runs_whitelisted_in_background(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    seen = {}
    ran = threading.Event()

    def fake_runner(package: str) -> None:
        seen["package"] = package
        ran.set()

    state = svc.install_package("vieneu", runner=fake_runner)
    assert state["package"] == "vieneu"
    # An injected runner may finish before install_package returns; a real pip install reads 'running'.
    assert state["status"] in ("running", "done")
    assert ran.wait(5)
    assert seen["package"] == "vieneu"

    for _ in range(100):
        if svc.install_status("vieneu")["status"] == "done":
            break
        time.sleep(0.02)
    assert svc.install_status("vieneu")["status"] == "done"


def test_install_package_reports_error_detail_on_failure(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)

    def boom(package: str) -> None:
        raise RuntimeError("pip exploded")

    svc.install_package("edge-tts", runner=boom)
    for _ in range(100):
        st = svc.install_status("edge-tts")
        if st["status"] in ("error", "done"):
            break
        time.sleep(0.02)
    st = svc.install_status("edge-tts")
    assert st["status"] == "error"
    assert "pip exploded" in st["detail"]


def test_install_status_rejects_non_whitelisted(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    with pytest.raises(ValueError):
        svc.install_status("requests")


def test_install_status_absent_before_any_install(tmp_path: Path) -> None:
    # vieneu is not installed in the test environment -> absent (never raises).
    st = JobsService(tmp_path).install_status("vieneu")
    assert st["package"] == "vieneu"
    assert st["status"] in ("absent", "installed")


def test_propainter_installer_is_whitelisted(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    seen = threading.Event()

    state = svc.install_package("propainter", runner=lambda package: seen.set())
    assert state["package"] == "propainter"
    assert seen.wait(5)
