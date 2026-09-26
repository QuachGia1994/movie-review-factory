import json
import os
from pathlib import Path

import pytest

from movie_review_factory import content_agent


def test_extract_structured_output_prefers_schema_payload() -> None:
    payload = {"brief": "ok"}
    envelope = json.dumps({"type": "result", "structured_output": payload})
    assert content_agent._extract_structured_output(envelope) == payload


def test_extract_structured_output_accepts_json_result_string() -> None:
    envelope = json.dumps({"type": "result", "result": '{"brief":"ok"}'})
    assert content_agent._extract_structured_output(envelope) == {"brief": "ok"}


def test_extract_structured_output_rejects_missing_payload() -> None:
    with pytest.raises(content_agent.ContentAgentError):
        content_agent._extract_structured_output(
            json.dumps({"type": "result", "result": "not-json"})
        )


def test_run_claude_json_uses_noninteractive_plan_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    monkeypatch.setattr(content_agent.shutil, "which", lambda name: "claude.exe")

    class FakeProcess:
        pid = 987654
        returncode = 0

        def __init__(self, args: list[str], **kwargs: object) -> None:
            captured["args"] = args
            captured.update(kwargs)

        def communicate(self, input=None, timeout=None):
            captured["input"] = input
            return (
                json.dumps({
                    "type": "result",
                    "structured_output": {"brief": "done"},
                }),
                "",
            )

        def poll(self):
            return self.returncode

    monkeypatch.setattr(content_agent.subprocess, "Popen", FakeProcess)

    result = content_agent.run_claude_json(
        root=tmp_path,
        stage="research",
        prompt="prompt body",
        schema={
            "type": "object",
            "properties": {"brief": {"type": "string"}},
            "required": ["brief"],
        },
        allowed_tools=["WebSearch", "WebFetch"],
    )

    assert result == {"brief": "done"}
    args = captured["args"]
    assert args[0] == "claude.exe"
    assert "-p" in args
    assert "--restricted" in args
    assert "--strict-mcp-config" in args
    assert "--json-schema" in args
    assert "--permission-mode" in args
    assert args[args.index("--permission-mode") + 1] == "plan"
    assert "--permission-prompts" in args
    assert args[args.index("--permission-prompts") + 1] == "none"
    assert "--mcp-config" in args
    assert args[args.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert args.count("--tools") == 1
    assert args[args.index("--tools") + 1] == "WebSearch,WebFetch"
    assert "--allowed-tools" not in args
    assert captured["input"] == "prompt body"
    assert captured["cwd"] == tmp_path


def test_run_claude_json_fails_when_cli_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(content_agent.shutil, "which", lambda name: None)
    with pytest.raises(content_agent.ContentAgentError, match="not found via PATH"):
        content_agent.run_claude_json(
            root=tmp_path,
            stage="script",
            prompt="x",
            schema={"type": "object"},
        )


@pytest.mark.parametrize("stage", ["outline", "script", "scene_plan"])
def test_run_claude_json_disables_tools_for_non_research_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    captured: dict = {}
    monkeypatch.setattr(content_agent.shutil, "which", lambda name: "claude.exe")

    class FakeProcess:
        pid = 987655
        returncode = 0

        def __init__(self, args: list[str], **kwargs: object) -> None:
            captured["args"] = args

        def communicate(self, input=None, timeout=None):
            return json.dumps({"structured_output": {"notes": "ok"}}), ""

        def poll(self):
            return self.returncode

    monkeypatch.setattr(content_agent.subprocess, "Popen", FakeProcess)
    content_agent.run_claude_json(
        root=tmp_path,
        stage=stage,
        prompt="x",
        schema={"type": "object"},
    )

    args = captured["args"]
    assert "--tools=" in args
    assert "--tools" not in args
    assert "" not in args
    assert "--allowed-tools" not in args


def _error_envelope(status: int, result: str) -> str:
    """Error envelope whose actionable fields sit past the first 400 chars."""
    envelope = json.dumps({
        "type": "result",
        "subtype": "error",
        "is_error": True,
        "duration_ms": 4200,
        "num_turns": 3,
        "session_id": "01J" + "x" * 64,
        "model_usage": {
            "claude-sonnet-4": {
                "input_tokens": 120000,
                "output_tokens": 4200,
                "cache_read_input_tokens": 900000,
                "cache_creation_input_tokens": 5000,
            }
        },
        "usage": {"input_tokens": 120000, "output_tokens": 4200},
        "result": result,
        "api_error_status": status,
    })
    assert len(envelope) > 400
    return envelope


def test_failure_detail_surfaces_trailing_api_fields() -> None:
    envelope = _error_envelope(429, "rate limited by provider")
    detail = content_agent._failure_detail(envelope, "")
    assert detail == "API 429: rate limited by provider"


def test_run_claude_json_reports_trailing_api_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(content_agent.shutil, "which", lambda name: "claude.exe")
    calls = 0

    class FakeProcess:
        pid = 987700
        returncode = 1

        def __init__(self, args: list[str], **kwargs: object) -> None:
            nonlocal calls
            calls += 1

        def communicate(self, input=None, timeout=None):
            return _error_envelope(400, "invalid request payload"), "usage dump"

        def poll(self):
            return self.returncode

    monkeypatch.setattr(content_agent.subprocess, "Popen", FakeProcess)
    with pytest.raises(content_agent.ContentAgentError) as excinfo:
        content_agent.run_claude_json(
            root=tmp_path,
            stage="script",
            prompt="x",
            schema={"type": "object"},
        )
    message = str(excinfo.value)
    assert "API 400: invalid request payload" in message
    assert "usage dump" not in message
    assert calls == 1


def test_run_claude_json_retries_transient_api_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(content_agent.shutil, "which", lambda name: "claude.exe")
    sleeps: list[float] = []
    monkeypatch.setattr(content_agent.time, "sleep", lambda seconds: sleeps.append(seconds))
    calls = 0

    class FakeProcess:
        pid = 987701
        returncode = 0

        def __init__(self, args: list[str], **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            self.returncode = 1 if calls == 1 else 0

        def communicate(self, input=None, timeout=None):
            if calls == 1:
                return 'claude: overloaded "api_error_status":429 retry later', "usage dump"
            return json.dumps({"type": "result", "structured_output": {"ok": True}}), ""

        def poll(self):
            return self.returncode

    monkeypatch.setattr(content_agent.subprocess, "Popen", FakeProcess)
    result = content_agent.run_claude_json(
        root=tmp_path,
        stage="script",
        prompt="x",
        schema={"type": "object"},
    )
    assert result == {"ok": True}
    assert calls == 2
    assert sleeps == [5]


def test_run_claude_json_raises_immediately_on_permanent_api_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(content_agent.shutil, "which", lambda name: "claude.exe")
    calls = 0

    class FakeProcess:
        pid = 987702
        returncode = 1

        def __init__(self, args: list[str], **kwargs: object) -> None:
            nonlocal calls
            calls += 1

        def communicate(self, input=None, timeout=None):
            return 'claude: bad request "api_error_status":400', ""

        def poll(self):
            return self.returncode

    monkeypatch.setattr(content_agent.subprocess, "Popen", FakeProcess)
    with pytest.raises(content_agent.ContentAgentError, match="api_error_status\":400"):
        content_agent.run_claude_json(
            root=tmp_path,
            stage="script",
            prompt="x",
            schema={"type": "object"},
        )
    assert calls == 1


def test_failure_detail_truncates_long_messages_keeping_head_and_tail() -> None:
    raw = "A" * 300 + "\n" + "B" * 300
    detail = content_agent._failure_detail(raw, "")
    assert " ... " in detail
    assert detail.startswith("A" * 200)
    assert detail.endswith("B" * 180)
    assert len(detail) <= 400


def test_run_claude_json_long_failure_message_keeps_head_and_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(content_agent.shutil, "which", lambda name: "claude.exe")
    stdout = "A" * 300 + "\n" + "B" * 300

    class FakeProcess:
        pid = 987703
        returncode = 1

        def __init__(self, args: list[str], **kwargs: object) -> None:
            pass

        def communicate(self, input=None, timeout=None):
            return stdout, ""

        def poll(self):
            return self.returncode

    monkeypatch.setattr(content_agent.subprocess, "Popen", FakeProcess)
    with pytest.raises(content_agent.ContentAgentError) as excinfo:
        content_agent.run_claude_json(
            root=tmp_path,
            stage="long_stage",
            prompt="x",
            schema={"type": "object"},
        )
    message = str(excinfo.value)
    assert " ... " in message
    assert "A" * 200 in message
    assert "B" * 180 in message
    assert len(message) > 400


def test_api_error_status_detection() -> None:
    assert content_agent._api_error_status('{"api_error_status": 429}') == 429
    assert content_agent._api_error_status('raw text "api_error_status":503 tail') == 503
    assert content_agent._api_error_status('{"result": "boom"}') is None
    assert content_agent._is_transient_status(429) is True
    assert content_agent._is_transient_status(503) is True
    assert content_agent._is_transient_status(400) is False
    assert content_agent._is_transient_status(401) is False
    assert content_agent._is_transient_status(None) is False


def test_run_claude_json_retries_one_timeout_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(content_agent.shutil, "which", lambda name: "claude.exe")
    monkeypatch.setattr(content_agent, "_timeout_seconds", lambda: 0)
    monkeypatch.setattr(content_agent.cancellation, "cancellable_sleep", lambda seconds: None)
    calls = 0
    inputs: list[str | None] = []

    class FakeProcess:
        returncode = 0

        def __init__(self, args: list[str], **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            self.pid = 987655 + calls
            self._timed_out = calls == 1

        def communicate(self, input=None, timeout=None):
            inputs.append(input)
            if self._timed_out:
                raise content_agent.subprocess.TimeoutExpired("claude", timeout)
            return json.dumps({"structured_output": {"ok": True}}), ""

        def poll(self):
            return self.returncode if not self._timed_out else None

        def wait(self, timeout=None):
            self.returncode = -1
            return self.returncode

        def terminate(self):
            self.returncode = -1

        def kill(self):
            self.returncode = -1

    monkeypatch.setattr(content_agent.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(content_agent.subprocess, "run", lambda *args, **kwargs: None)
    result = content_agent.run_claude_json(
        root=tmp_path,
        stage="retry",
        prompt="x",
        schema={"type": "object"},
    )
    assert result == {"ok": True}
    assert calls == 2
    # Each attempt is a fresh Popen: the retried attempt must re-send the prompt.
    assert inputs == ["x", "x"]


@pytest.mark.parametrize("pid_source", ["self", "parent", "missing", "invalid", "unregistered"])
def test_terminate_process_tree_rejects_unsafe_targets(
    monkeypatch: pytest.MonkeyPatch,
    pid_source: str,
) -> None:
    pid = {
        "self": os.getpid(),
        "parent": os.getppid(),
        "missing": None,
        "invalid": "123",
        "unregistered": 987999,
    }[pid_source]
    calls: list[object] = []

    class UnsafeProcess:
        def __init__(self) -> None:
            if pid is not None:
                self.pid = pid

        def poll(self):
            calls.append("poll")
            return None

        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")

        def wait(self, timeout=None):
            calls.append("wait")

    monkeypatch.setattr(content_agent.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    content_agent._terminate_process_tree(UnsafeProcess())
    assert calls == []


def test_retry_count_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRF_CLAUDE_RETRIES", "4")
    with pytest.raises(content_agent.ContentAgentError, match="between 0 and 3"):
        content_agent._retry_count()


def test_resolve_executable_honors_mrf_claude_bin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom = tmp_path / "custom-claude.exe"
    custom.write_bytes(b"stub")
    monkeypatch.setenv("MRF_CLAUDE_BIN", str(custom))
    monkeypatch.setattr(content_agent.shutil, "which", lambda name: None)
    assert content_agent._resolve_executable() == str(custom)


def test_invalid_effort_fails_before_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRF_CLAUDE_EFFORT", "turbo")
    with pytest.raises(content_agent.ContentAgentError, match="must be low"):
        content_agent._effort()


@pytest.mark.skipif(
    os.environ.get("MRF_RUN_LIVE_CLAUDE_TESTS") != "1",
    reason="live Claude network test is opt-in",
)
def test_live_claude_json_schema_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if content_agent.shutil.which("claude") is None:
        pytest.skip("Claude Code is not installed")
    monkeypatch.setenv("MRF_CLAUDE_TIMEOUT_SECONDS", "60")
    result = content_agent.run_claude_json(
        root=tmp_path,
        stage="live_probe",
        prompt='Return the JSON object {"ok": true}. Do not add commentary.',
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean", "const": True}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    )
    assert result == {"ok": True}
