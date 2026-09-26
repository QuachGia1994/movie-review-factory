from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from . import cancellation


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


_API_STATUS_RE = re.compile(r'"api_error_status"\s*:\s*(\d{3})')


def _coerce_status(raw: Any) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return None


def _parse_envelope(stdout: str) -> dict[str, Any] | None:
    try:
        envelope = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(envelope, dict):
        return None
    return envelope


def _api_error_status(stdout: str) -> int | None:
    """Return the API status from a Claude error envelope, JSON first, regex fallback."""
    envelope = _parse_envelope(stdout)
    if envelope is not None:
        status = _coerce_status(envelope.get("api_error_status"))
        if status is not None:
            return status
    match = _API_STATUS_RE.search(stdout or "")
    if match:
        return int(match.group(1))
    return None


def _is_transient_status(status: int | None) -> bool:
    return status is not None and (status == 429 or 500 <= status < 600)


def _api_failure_message(stdout: str) -> str | None:
    """Surface only the actionable API fields, ignoring usage/telemetry noise."""
    envelope = _parse_envelope(stdout)
    if envelope is None:
        return None
    status = _coerce_status(envelope.get("api_error_status"))
    result = envelope.get("result")
    if status is None and not isinstance(result, str):
        return None
    if status is None:
        return result
    if isinstance(result, str) and result.strip():
        return f"API {status}: {result}"
    return f"API {status}"


def _failure_detail(stdout: str, stderr: str) -> str:
    detail = _api_failure_message(stdout)
    if detail is None:
        detail = stderr or stdout
    detail = detail.replace("\n", " ").strip()
    if len(detail) > 400:
        detail = detail[:200] + " ... " + detail[-180:]
    return detail


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


_OWNED_PROCESSES: dict[int, subprocess.Popen[str]] = {}


def _track_process(process: subprocess.Popen[str]) -> None:
    """Mark a process created here as an eligible Claude-tree target."""
    pid = getattr(process, "pid", None)
    if isinstance(pid, int) and pid > 0:
        _OWNED_PROCESSES[pid] = process


def _untrack_process(process: subprocess.Popen[str]) -> None:
    pid = getattr(process, "pid", None)
    if isinstance(pid, int) and _OWNED_PROCESSES.get(pid) is process:
        _OWNED_PROCESSES.pop(pid, None)


def _safe_owned_pid(process: subprocess.Popen[str]) -> int | None:
    """Return the PID only when it is the exact child object created here."""
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    if pid in {os.getpid(), os.getppid()}:
        return None
    if _OWNED_PROCESSES.get(pid) is not process:
        return None
    return pid


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate only a tracked Claude child tree; safe to call repeatedly."""
    pid = _safe_owned_pid(process)
    if pid is None or process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
            timeout=10,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Revalidate before the direct fallback in case the object was untracked.
        if _safe_owned_pid(process) is None:
            return
        process.kill()
        process.wait(timeout=5)


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
    tool_args = ["--tools", ",".join(tools)] if tools else ["--tools="]

    args = [
        executable,
        "-p",
        "--restricted",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        *tool_args,
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

    retries = _retry_count()
    completed: subprocess.CompletedProcess[str] | None = None
    for attempt in range(retries + 1):
        cancellation.checkpoint()
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        process: subprocess.Popen[str] | None = None
        context = cancellation.current_context()
        attempt_prompt: str | None = prompt
        try:
            process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=root,
                creationflags=creationflags,
            )
            _track_process(process)
            if context and context.register_process:
                context.register_process(process)
            deadline = time.monotonic() + _timeout_seconds()
            while True:
                cancellation.checkpoint()
                try:
                    stdout, stderr = process.communicate(input=attempt_prompt, timeout=0.1)
                    completed = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
                    break
                except subprocess.TimeoutExpired:
                    # Same Popen: stdin may already hold the prompt; never re-send.
                    attempt_prompt = None
                    if time.monotonic() >= deadline:
                        _terminate_process_tree(process)
                        if attempt >= retries:
                            raise ContentAgentError(
                                f"Claude content agent timed out during {stage}"
                            )
                        cancellation.cancellable_sleep(min(1.0 + attempt, 2.0))
                        break
            if completed is not None:
                if (
                    completed.returncode != 0
                    and attempt < retries
                    and _is_transient_status(_api_error_status(completed.stdout))
                ):
                    completed = None
                    time.sleep(5 * (attempt + 1))
                    continue
                break
        except cancellation.RunCancelled:
            if process is not None:
                _terminate_process_tree(process)
            raise
        except OSError as exc:
            raise ContentAgentError(
                f"could not launch Claude content agent during {stage}: {exc}"
            ) from exc
        finally:
            if process is not None:
                if context and context.unregister_process:
                    context.unregister_process(process)
                _untrack_process(process)

    assert completed is not None
    if completed.returncode != 0:
        raise ContentAgentError(
            f"Claude content agent failed during {stage}: "
            f"{_failure_detail(completed.stdout, completed.stderr)}"
        )

    return _extract_structured_output(completed.stdout)
