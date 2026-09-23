import json
import os
import types
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

    def fake_run(args: list[str], **kwargs: object) -> object:
        captured["args"] = args
        captured.update(kwargs)
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "type": "result",
                "structured_output": {"brief": "done"},
            }),
            stderr="",
        )

    monkeypatch.setattr(content_agent.subprocess, "run", fake_run)

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
    assert "--tools" in args
    assert args[args.index("--tools") + 1] == "WebSearch,WebFetch"
    assert "--allowed-tools" in args
    assert args[args.index("--allowed-tools") + 1] == "WebSearch,WebFetch"
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


def test_run_claude_json_disables_tools_when_none_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    monkeypatch.setattr(content_agent.shutil, "which", lambda name: "claude.exe")

    def fake_run(args: list[str], **kwargs: object) -> object:
        captured["args"] = args
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"structured_output": {"notes": "ok"}}),
            stderr="",
        )

    monkeypatch.setattr(content_agent.subprocess, "run", fake_run)
    content_agent.run_claude_json(
        root=tmp_path,
        stage="outline",
        prompt="x",
        schema={"type": "object"},
    )

    args = captured["args"]
    assert args[args.index("--tools") + 1] == ""
    assert "--allowed-tools" not in args


def test_run_claude_json_retries_one_timeout_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(content_agent.shutil, "which", lambda name: "claude.exe")
    monkeypatch.setattr(content_agent.time, "sleep", lambda seconds: None)
    calls = 0

    def flaky_run(args: list[str], **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise content_agent.subprocess.TimeoutExpired(args, 1)
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"structured_output": {"ok": True}}),
            stderr="",
        )

    monkeypatch.setattr(content_agent.subprocess, "run", flaky_run)
    result = content_agent.run_claude_json(
        root=tmp_path,
        stage="retry",
        prompt="x",
        schema={"type": "object"},
    )
    assert result == {"ok": True}
    assert calls == 2


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
