from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import pool_scheduler
from .agy_vision import VisionUnavailable, pool_workers
from .content_agent import ContentAgentError
from .pool_scheduler import BUSY_ERROR, QUOTA_ERROR, BusySignal, QuotaSignal, RotateSignal

DEFAULT_MODEL = "gemini-3.7-flash-medium"


def _model() -> str:
    return os.environ.get("MRF_AGY_MODEL", "").strip() or DEFAULT_MODEL


def _effort() -> str:
    value = os.environ.get("MRF_AGY_EFFORT", "medium").strip().lower()
    if value not in {"low", "medium", "high"}:
        raise ContentAgentError("MRF_AGY_EFFORT must be low, medium, or high")
    return value


def _timeout_seconds() -> float:
    # Worker runs cap server-side at 120s; stay above it so the worker's own error response arrives instead of a client-side timeout.
    raw = os.environ.get("MRF_AGY_TIMEOUT_SECONDS", "130")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ContentAgentError("MRF_AGY_TIMEOUT_SECONDS must be numeric") from exc
    if value <= 0:
        raise ContentAgentError("MRF_AGY_TIMEOUT_SECONDS must be greater than zero")
    return value


def _extract_json_object(text: str) -> dict[str, Any]:
    # The AGY CLI may prefix its answer (e.g. a rule receipt), so decode the first brace-delimited object that parses instead of trusting the whole string.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ContentAgentError("AGY returned no JSON object")


def _failure_detail(result: object) -> str:
    if isinstance(result, dict):
        error = result.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
    return "invalid worker response"


def run_agy_json(
    *,
    stage: str,
    prompt: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Run one structured prompt through the AGY pool and return schema-shaped JSON.

    Every configured plan worker is tried in order: quota exhaustion, a busy
    worker, or unusable output moves the request to the next role.  The shared
    :class:`movie_review_factory.pool_scheduler.PoolScheduler` owns that
    failover loop together with quota telemetry, cooldown, bounded retries,
    and role-order guarantees; results are not cached across calls.
    """
    try:
        workers = pool_workers(None)
    except VisionUnavailable as exc:
        raise ContentAgentError(f"AGY content agent unavailable during {stage}: {exc}") from exc

    full_prompt = (
        prompt
        + "\nReturn ONLY a single JSON object with no markdown fence and no commentary. "
        "It must satisfy this JSON schema:\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )
    required = [key for key in schema.get("required", []) if isinstance(key, str)]
    timeout = _timeout_seconds()
    model, effort = _model(), _effort()

    def attempt(worker: tuple, unit: object) -> dict[str, Any]:
        _, url, root, token = worker
        request = urllib.request.Request(
            url,
            data=json.dumps(
                {
                    "prompt": full_prompt,
                    "mode": "plan",
                    "model": model,
                    "effort": effort,
                    "cwd": str(root),
                }
            ).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            reason = f"{exc.reason} {detail}"
            if exc.code == 429 or QUOTA_ERROR.search(reason):
                raise QuotaSignal("quota exhausted") from exc
            if exc.code == 409 or BUSY_ERROR.search(reason):
                raise BusySignal("busy") from exc
            raise RotateSignal(f"HTTP {exc.code}: {detail[:120]}") from exc
        except (OSError, TimeoutError) as exc:
            raise RotateSignal(f"unreachable: {exc}") from exc

        if not isinstance(result, dict) or result.get("ok") is not True or not isinstance(result.get("text"), str):
            detail = _failure_detail(result)
            if QUOTA_ERROR.search(detail):
                raise QuotaSignal("quota exhausted")
            raise RotateSignal(detail[:160])

        try:
            data = _extract_json_object(result["text"])
        except ContentAgentError as exc:
            raise RotateSignal(str(exc)) from exc

        missing = [key for key in required if key not in data]
        if missing:
            raise RotateSignal(f"missing required keys {', '.join(missing)}")
        return data

    outcome = pool_scheduler.PoolScheduler(workers).run(
        f"agy-json:{stage}", attempt, resume=False
    )
    if outcome.data:
        return next(iter(outcome.data.values()))

    detail = "; ".join(outcome.failures) or "no configured plan workers"
    raise ContentAgentError(f"AGY content agent failed during {stage}: {detail[:400]}")
