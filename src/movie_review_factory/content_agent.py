from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


class ContentAgentError(RuntimeError):
    """Raised when an explicitly requested content agent cannot return valid output."""


def _timeout_seconds() -> float:
    raw = os.environ.get("MRF_CLAUDE_TIMEOUT_SECONDS", "300")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ContentAgentError("MRF_CLAUDE_TIMEOUT_SECONDS must be numeric") from exc
    if value <= 0:
        raise ContentAgentError("MRF_CLAUDE_TIMEOUT_SECONDS must be greater than zero")
    return value


def _retry_count() -> int:
    raw = os.environ.get("MRF_CLAUDE_RETRIES", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ContentAgentError("MRF_CLAUDE_RETRIES must be an integer") from exc
    if value < 0 or value > 3:
        raise ContentAgentError("MRF_CLAUDE_RETRIES must be between 0 and 3")
    return value


def _extract_structured_output(stdout: str) -> dict[str, Any]:
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ContentAgentError("Claude returned invalid JSON output") from exc

    if not isinstance(envelope, dict):
        raise ContentAgentError("Claude JSON output must be an object")

    structured = envelope.get("structured_output")
    if isinstance(structured, dict):
        return structured

    result = envelope.get("result")
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        candidate = result.strip()
        fence = chr(96) * 3
        if candidate.startswith(fence):
            lines = candidate.splitlines()
            if lines and lines[0].startswith(fence):
                lines = lines[1:]
            if lines and lines[-1].strip() == fence:
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ContentAgentError("Claude result did not contain structured JSON") from exc
        if isinstance(parsed, dict):
            return parsed

    # Some compatible launchers may emit the schema object directly.
    if "type" not in envelope and "subtype" not in envelope:
        return envelope

    raise ContentAgentError("Claude output did not contain structured_output")


def _resolve_executable() -> str:
    configured = os.environ.get("MRF_CLAUDE_BIN", "").strip()
    requested = configured or "claude"
    executable = shutil.which(requested)
    if executable:
        return executable
    candidate = Path(requested).expanduser()
    if candidate.is_file():
        return str(candidate)
    hint = "MRF_CLAUDE_BIN" if configured else "PATH"
    raise ContentAgentError(
        f"content_agent=claude requested but Claude Code was not found via {hint}"
    )


def _effort() -> str:
    value = os.environ.get("MRF_CLAUDE_EFFORT", "medium").strip().lower()
    if value not in {"low", "medium", "high", "xhigh", "max"}:
        raise ContentAgentError(
            "MRF_CLAUDE_EFFORT must be low, medium, high, xhigh, or max"
        )
    return value


def run_claude_json(
    *,
    root: Path,
    stage: str,
    prompt: str,
    schema: dict[str, Any],
    allowed_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Run Claude Code non-interactively and return schema-constrained JSON.

    The caller supplies all job context in the prompt. The subprocess runs in
    restricted + plan mode, loads no MCP servers, exposes only the explicitly
    requested built-in tools, and cannot mutate the job directory.
    """
    executable = _resolve_executable()
    tools = list(dict.fromkeys(allowed_tools or []))
    tool_spec = ",".join(tools)

    args = [
        executable,
        "-p",
        "--restricted",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--tools",
        tool_spec,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
        "--permission-mode",
        "plan",
        "--permission-prompts",
        "none",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--effort",
        _effort(),
        "--name",
        f"mrf-{stage}",
    ]
    model = os.environ.get("MRF_CLAUDE_MODEL", "").strip()
    if model:
        args.extend(["--model", model])
    if tools:
        args.extend(["--allowed-tools", tool_spec])

    retries = _retry_count()
    proc = None
    for attempt in range(retries + 1):
        try:
            proc = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=root,
                timeout=_timeout_seconds(),
            )
            break
        except subprocess.TimeoutExpired as exc:
            if attempt >= retries:
                raise ContentAgentError(
                    f"Claude content agent timed out during {stage}"
                ) from exc
            time.sleep(min(1.0 + attempt, 2.0))
        except OSError as exc:
            raise ContentAgentError(
                f"could not launch Claude content agent during {stage}: {exc}"
            ) from exc

    assert proc is not None
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().replace("\n", " ")
        raise ContentAgentError(
            f"Claude content agent failed during {stage}: {detail[:400]}"
        )

    return _extract_structured_output(proc.stdout)
