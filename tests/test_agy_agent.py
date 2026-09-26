import io
import json
import urllib.error
from pathlib import Path

import pytest

from movie_review_factory import agy_agent, pool_scheduler
from movie_review_factory.agy_vision import VisionUnavailable
from movie_review_factory.content_agent import ContentAgentError

SCHEMA = {
    "type": "object",
    "properties": {"brief": {"type": "string"}},
    "required": ["brief"],
}

WORKERS = [
    ("advisor", "http://127.0.0.1:7411/run", Path("C:/pool/agy-workspaces"), "token-a"),
    ("executor", "http://127.0.0.1:7412/run", Path("C:/pool/agy-workspaces"), "token-b"),
]


@pytest.fixture(autouse=True)
def _isolated_pool_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep AGY tests off the shared %LOCALAPPDATA% pool health file."""
    monkeypatch.setenv("MRF_POOL_STATE_FILE", str(tmp_path / "pool_health.json"))
    for name in (
        "MRF_AGY_RETRIES",
        "MRF_AGY_COOLDOWN_SECONDS",
        "MRF_AGY_MODEL",
        "MRF_AGY_EFFORT",
        "MRF_AGY_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _http_error(url: str, code: int, reason: str, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, reason, {}, io.BytesIO(body))


def test_run_agy_json_extracts_object_behind_cli_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[urllib.request.Request] = []

    def open_ok(request: urllib.request.Request, timeout: float) -> _Response:
        captured.append(request)
        return _Response(json.dumps({"ok": True, "text": '[RULES] agent (always_on) + coding (viewed)\n{"brief":"done"}'}).encode())

    monkeypatch.setattr(agy_agent, "pool_workers", lambda pool_dir: WORKERS)
    monkeypatch.setattr(agy_agent.urllib.request, "urlopen", open_ok)

    result = agy_agent.run_agy_json(stage="research", prompt="do it", schema=SCHEMA)

    assert result == {"brief": "done"}
    payload = json.loads(captured[0].data)
    assert payload["mode"] == "plan"
    assert payload["model"] == agy_agent.DEFAULT_MODEL
    assert payload["effort"] == "medium"
    assert payload["cwd"] == str(WORKERS[0][2])
    assert payload["prompt"].startswith("do it")
    assert '"brief"' in payload["prompt"]
    assert captured[0].get_header("Authorization") == "Bearer token-a"
    assert captured[0].get_header("Content-type") == "application/json"


def test_run_agy_json_fails_over_from_quota_to_next_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def open_quota(request: urllib.request.Request, timeout: float) -> _Response:
        calls.append(request.full_url)
        raise _http_error(request.full_url, 429, "Too Many Requests", b"pool cooling down")

    def open_ok(request: urllib.request.Request, timeout: float) -> _Response:
        calls.append(request.full_url)
        return _Response(json.dumps({"ok": True, "text": '{"brief":"recovered"}'}).encode())

    responses = [open_quota, open_ok]

    def open_next(request: urllib.request.Request, timeout: float) -> _Response:
        return responses.pop(0)(request, timeout)

    monkeypatch.setattr(agy_agent, "pool_workers", lambda pool_dir: WORKERS)
    monkeypatch.setattr(agy_agent.urllib.request, "urlopen", open_next)

    assert agy_agent.run_agy_json(stage="research", prompt="p", schema=SCHEMA) == {"brief": "recovered"}
    assert calls == [WORKERS[0][1], WORKERS[1][1]]


def test_run_agy_json_moves_past_busy_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def open_busy(request: urllib.request.Request, timeout: float) -> _Response:
        raise _http_error(request.full_url, 409, "Conflict", b'{"error":"worker \\"advisor\\" is busy"}')

    def open_ok(request: urllib.request.Request, timeout: float) -> _Response:
        return _Response(json.dumps({"ok": True, "text": '{"brief":"ok"}'}).encode())

    responses = [open_busy, open_ok]

    def open_next(request: urllib.request.Request, timeout: float) -> _Response:
        return responses.pop(0)(request, timeout)

    monkeypatch.setattr(agy_agent, "pool_workers", lambda pool_dir: WORKERS)
    monkeypatch.setattr(agy_agent.urllib.request, "urlopen", open_next)

    assert agy_agent.run_agy_json(stage="script", prompt="p", schema=SCHEMA) == {"brief": "ok"}
    assert not responses


def test_run_agy_json_reports_every_failed_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def open_conflict(request: urllib.request.Request, timeout: float) -> _Response:
        raise _http_error(request.full_url, 409, "Conflict", b'{"error":"worker is busy"}')

    monkeypatch.setattr(agy_agent, "pool_workers", lambda pool_dir: WORKERS)
    monkeypatch.setattr(agy_agent.urllib.request, "urlopen", open_conflict)

    with pytest.raises(ContentAgentError, match="advisor busy; executor busy"):
        agy_agent.run_agy_json(stage="scene_plan", prompt="p", schema=SCHEMA)


def test_run_agy_json_rejects_output_missing_required_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def open_ok(request: urllib.request.Request, timeout: float) -> _Response:
        return _Response(json.dumps({"ok": True, "text": '{"other": 1}'}).encode())

    monkeypatch.setattr(agy_agent, "pool_workers", lambda pool_dir: WORKERS)
    monkeypatch.setattr(agy_agent.urllib.request, "urlopen", open_ok)

    with pytest.raises(ContentAgentError, match="missing required keys brief"):
        agy_agent.run_agy_json(stage="outline", prompt="p", schema=SCHEMA)


def test_run_agy_json_reports_unconfigured_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(pool_dir: object) -> object:
        raise VisionUnavailable("AGY pool is not configured")

    monkeypatch.setattr(agy_agent, "pool_workers", unavailable)

    with pytest.raises(ContentAgentError, match="unavailable during research"):
        agy_agent.run_agy_json(stage="research", prompt="p", schema=SCHEMA)


def test_run_agy_json_honours_model_and_effort_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def open_ok(request: urllib.request.Request, timeout: float) -> _Response:
        captured.update(json.loads(request.data))
        return _Response(json.dumps({"ok": True, "text": '{"brief":"done"}'}).encode())

    monkeypatch.setattr(agy_agent, "pool_workers", lambda pool_dir: WORKERS)
    monkeypatch.setattr(agy_agent.urllib.request, "urlopen", open_ok)
    monkeypatch.setenv("MRF_AGY_MODEL", "gemini-3.7-pro")
    monkeypatch.setenv("MRF_AGY_EFFORT", "high")

    agy_agent.run_agy_json(stage="research", prompt="p", schema=SCHEMA)

    assert captured["model"] == "gemini-3.7-pro"
    assert captured["effort"] == "high"


def test_run_agy_json_rejects_invalid_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agy_agent, "pool_workers", lambda pool_dir: WORKERS)
    monkeypatch.setenv("MRF_AGY_EFFORT", "xhigh")

    with pytest.raises(ContentAgentError, match="MRF_AGY_EFFORT"):
        agy_agent.run_agy_json(stage="research", prompt="p", schema=SCHEMA)


def test_run_agy_json_cools_down_a_quota_role_for_the_next_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def open_next(request: urllib.request.Request, timeout: float) -> _Response:
        calls.append(request.full_url)
        if request.full_url == WORKERS[0][1]:
            raise _http_error(request.full_url, 429, "Too Many Requests", b"quota")
        return _Response(json.dumps({"ok": True, "text": '{"brief":"recovered"}'}).encode())

    clock = {"now": 3_000_000_000.0}
    monkeypatch.setattr(agy_agent, "pool_workers", lambda pool_dir: WORKERS)
    monkeypatch.setattr(agy_agent.urllib.request, "urlopen", open_next)
    monkeypatch.setattr(pool_scheduler, "_now", lambda: clock["now"])

    assert agy_agent.run_agy_json(stage="research", prompt="p", schema=SCHEMA) == {"brief": "recovered"}
    assert calls == [WORKERS[0][1], WORKERS[1][1]]

    calls.clear()
    assert agy_agent.run_agy_json(stage="outline", prompt="p", schema=SCHEMA) == {"brief": "recovered"}
    assert calls == [WORKERS[1][1]]  # advisor is cooling down: no second 429

    clock["now"] += 400
    calls.clear()
    assert agy_agent.run_agy_json(stage="script", prompt="p", schema=SCHEMA) == {"brief": "recovered"}
    assert calls == [WORKERS[0][1], WORKERS[1][1]]  # cooldown expired
