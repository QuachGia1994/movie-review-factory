"""Shared pytest fixtures.

The background indexing queue (roadmap #14) auto-enqueues real transcript/scene
indexing after every import. That is the production default, but running actual
faster-whisper transcription in the background of every import test would make
the suite slow and non-deterministic, so tests opt out of auto-indexing by
default. Queue tests re-enable it explicitly (``MRF_AUTO_INDEX=1``) with a fake
``pipeline.run_index`` so the mechanism is still covered deterministically.
"""

import pytest


@pytest.fixture(autouse=True)
def _disable_auto_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRF_AUTO_INDEX", "0")
    # Scout localization calls the AGY pool; tests opt in with a fake runner.
    monkeypatch.setenv("MRF_SCOUT_LOCALIZE", "0")
