"""Client-side scheduler for the AGY pool (roadmap #13).

``PoolScheduler`` walks the configured pool roles in their canonical order
(``advisor -> executor -> experiment -> reviewer``) and gives every worker
call quota/health telemetry, a quota cooldown, bounded retries, and partial
resume.  It is the shared failover loop behind ``agy_agent.run_agy_json`` and
the ``agy_vision`` worker loops; both keep their public contracts.

State lives in one small JSON file (never a database):

* ``MRF_POOL_STATE_FILE`` overrides the path, otherwise the file is
  ``%LOCALAPPDATA%\\MovieReviewFactory\\runtime\\pool_health.json``; when
  ``LOCALAPPDATA`` is unset (non-Windows) it falls back to
  ``~/.movie-review-factory/runtime/pool_health.json``.

The file holds ``roles`` (per call: ``status`` in ``ok``/``quota``/``busy``/
``error``, ``latency_ms``, ``checked_at`` ISO-8601 UTC, ``cooldown_until``
epoch seconds) and ``progress`` (units a key already completed, so a resumed
run skips the work that already succeeded).  Writes are atomic and best
effort: a telemetry failure never fails a worker call, and at most
:data:`MAX_PROGRESS_KEYS` keys are kept so the file stays small.

Environment overrides:

* ``MRF_AGY_COOLDOWN_SECONDS`` -- seconds a quota role stays out (default
  ``300``).  Roles in cooldown are skipped while the other roles still run.
* ``MRF_AGY_RETRIES`` -- attempts per role per run (default ``1``, integer
  ``0..3`` validated like ``MRF_CLAUDE_RETRIES``; ``0`` and ``1`` both mean a
  single attempt, larger values add retries).  Between attempts the scheduler
  sleeps ``0.5s * attempt``.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QUOTA_ERROR = re.compile(
    r"quota|resource.?exhausted|rate.?limit|usage.?limit|limit.?reached|too many requests",
    re.I,
)
BUSY_ERROR = re.compile(r"\bbusy\b", re.I)

DEFAULT_COOLDOWN_SECONDS = 300.0
DEFAULT_RETRIES = 1
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 0.5
MAX_PROGRESS_KEYS = 16
MAX_PROGRESS_BYTES = 64 * 1024
DEFAULT_UNIT = "job"

Worker = tuple[str, str, Path, str]
Attempt = Callable[[Worker, Any], Any]


class PoolConfigError(ValueError):
    """A scheduler environment override is invalid."""


class RoleSignal(Exception):
    """Raised by an attempt callback to steer the scheduler.

    ``status`` is the telemetry status, ``rotate`` decides whether the next
    role is tried when the callback raises this signal.
    """

    status = "error"
    rotate = True


class QuotaSignal(RoleSignal):
    """Quota/429: cool the role down, rotate, and allow bounded retries."""

    status = "quota"


class BusySignal(RoleSignal):
    """Worker busy (HTTP 409): rotate immediately, never cool down."""

    status = "busy"


class RotateSignal(RoleSignal):
    """A plain role failure that must move on to the next role."""

    status = "error"


@dataclass
class ScheduleResult:
    """``(data, roles_used, skipped_roles, failures)`` returned by ``run``."""

    data: dict[Any, Any]
    roles_used: list[str]
    skipped_roles: list[str]
    failures: list[str]


def retry_count() -> int:
    raw = os.environ.get("MRF_AGY_RETRIES", str(DEFAULT_RETRIES))
    try:
        value = int(raw)
    except ValueError as exc:
        raise PoolConfigError("MRF_AGY_RETRIES must be an integer") from exc
    if value < 0 or value > MAX_RETRIES:
        raise PoolConfigError(f"MRF_AGY_RETRIES must be between 0 and {MAX_RETRIES}")
    return value


def cooldown_seconds() -> float:
    raw = os.environ.get("MRF_AGY_COOLDOWN_SECONDS", str(int(DEFAULT_COOLDOWN_SECONDS)))
    try:
        value = float(raw)
    except ValueError as exc:
        raise PoolConfigError("MRF_AGY_COOLDOWN_SECONDS must be numeric") from exc
    if value <= 0:
        raise PoolConfigError("MRF_AGY_COOLDOWN_SECONDS must be greater than zero")
    return value


def default_state_file() -> Path:
    override = os.environ.get("MRF_POOL_STATE_FILE", "").strip()
    if override:
        return Path(override)
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data) / "MovieReviewFactory" / "runtime" / "pool_health.json"
    return Path.home() / ".movie-review-factory" / "runtime" / "pool_health.json"


def classify_error(exc: BaseException) -> str:
    if isinstance(exc, RoleSignal):
        return exc.status
    code = getattr(exc, "code", None)
    if code == 429:
        return "quota"
    if code == 409:
        return "busy"
    text = f"{getattr(exc, 'reason', '')} {exc}"
    if QUOTA_ERROR.search(text):
        return "quota"
    if BUSY_ERROR.search(text):
        return "busy"
    return "error"


def _now() -> float:
    return time.time()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _slot(unit: Any) -> str:
    return str(unit)


def _load_state(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        raw = None
    roles = raw.get("roles") if isinstance(raw, dict) else None
    progress = raw.get("progress") if isinstance(raw, dict) else None
    return {
        "version": 1,
        "roles": roles if isinstance(roles, dict) else {},
        "progress": progress if isinstance(progress, dict) else {},
    }


def _save_state(path: Path, state: dict[str, Any]) -> None:
    progress = state.get("progress")
    if isinstance(progress, dict) and len(progress) > MAX_PROGRESS_KEYS:
        def touched(item: tuple[str, Any]) -> float:
            entry = item[1]
            value = entry.get("updated_at") if isinstance(entry, dict) else None
            return float(value) if isinstance(value, (int, float)) else 0.0

        state["progress"] = dict(
            sorted(progress.items(), key=touched, reverse=True)[:MAX_PROGRESS_KEYS]
        )
    try:
        text = json.dumps(state, ensure_ascii=False)
    except (TypeError, ValueError):
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass  # Telemetry is best effort and must never fail a worker call.


def _record_call(
    roles: dict[str, Any],
    role: str,
    status: str,
    latency_ms: int,
    now: float,
    cooldown: float,
) -> None:
    # A cooldown this run already opened for the role survives a later successful retry: one lucky answer does not restore the quota.
    retained = 0.0
    previous = roles.get(role)
    if isinstance(previous, dict):
        value = previous.get("cooldown_until")
        if isinstance(value, (int, float)) and float(value) > now:
            retained = float(value)
    roles[role] = {
        "status": status,
        "latency_ms": int(max(0, latency_ms)),
        "checked_at": _iso(now),
        "cooldown_until": round(now + cooldown, 3) if status == "quota" else retained,
    }


def _record_progress(
    progress: dict[str, Any], unit: Any, role: str, value: Any, now: float
) -> None:
    try:
        payload = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return  # Values that cannot round-trip through JSON cannot be resumed.
    if len(payload.encode("utf-8", errors="ignore")) > MAX_PROGRESS_BYTES:
        return
    progress[_slot(unit)] = {"role": role, "data": value, "updated_at": now}


def _failure_message(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text[:200]


class PoolScheduler:
    """Quota/health aware failover over the ordered AGY pool roles."""

    def __init__(self, workers: Sequence[Worker], *, state_file: Path | None = None) -> None:
        self.workers = list(workers)
        self.state_file = Path(state_file) if state_file is not None else default_state_file()

    def completed(self, key: str, unit: Any) -> bool:
        """Whether ``unit`` already holds data for ``key`` in pool health state.

        Only meaningful for the default ``resume=True`` runs: a caller uses it
        to skip local work (frame extraction) that a resumed run will replay.
        """
        state = _load_state(self.state_file)
        entry = state["progress"].get(key)
        if not isinstance(entry, dict):
            return False
        units = entry.get("units")
        item = units.get(_slot(unit)) if isinstance(units, dict) else None
        return isinstance(item, dict) and "data" in item

    def run(
        self,
        key: str,
        attempt: Attempt,
        *,
        units: Sequence[Any] | None = None,
        rotate_on_error: bool = True,
        resume: bool = True,
    ) -> ScheduleResult:
        """Run ``attempt(worker, unit)`` for every pending unit of ``key``.

        Units already completed for ``key`` (when ``resume``) are returned
        from state without touching a worker, so a resumed run only retries
        the units that did not succeed.  Roles in quota cooldown are skipped
        while the remaining roles still run.  ``rotate_on_error=False`` makes
        a non-quota/non-busy failure re-raise its original exception instead
        of moving to the next role.
        """
        unit_list = list(units) if units is not None else [DEFAULT_UNIT]
        retries = max(1, retry_count())
        cooldown = cooldown_seconds()
        state = _load_state(self.state_file)
        roles_state: dict[str, Any] = state["roles"]
        progress_key = state["progress"].get(key)
        if not isinstance(progress_key, dict) or not isinstance(progress_key.get("units"), dict):
            progress_key = {"updated_at": _now(), "units": {}}
            if resume:
                state["progress"][key] = progress_key
        progress: dict[str, Any] = progress_key["units"]

        data: dict[Any, Any] = {}
        pending: list[Any] = []
        for unit in unit_list:
            entry = progress.get(_slot(unit)) if resume else None
            if isinstance(entry, dict) and "data" in entry:
                data[unit] = entry["data"]
            else:
                pending.append(unit)

        roles_used: list[str] = []
        skipped_roles: list[str] = []
        failures: list[str] = []
        cooldowns = {
            role: float(info.get("cooldown_until") or 0.0)
            for role, info in roles_state.items()
            if isinstance(info, dict)
        }

        for unit in pending:
            served = False
            for worker in self.workers:
                role = worker[0]
                if cooldowns.get(role, 0.0) > _now():
                    if role not in skipped_roles:
                        skipped_roles.append(role)
                    if f"{role} in cooldown" not in failures:
                        failures.append(f"{role} in cooldown")
                    continue
                for attempt_no in range(1, retries + 1):
                    started = _now()
                    try:
                        value = attempt(worker, unit)
                    except Exception as exc:
                        status = classify_error(exc)
                        now = _now()
                        _record_call(roles_state, role, status, int((now - started) * 1000), now, cooldown)
                        if status == "quota":
                            cooldowns[role] = float(roles_state[role]["cooldown_until"])
                            if attempt_no < retries:
                                _sleep(RETRY_BACKOFF_SECONDS * attempt_no)
                                continue
                        failures.append(f"{role} {_failure_message(exc)}")
                        if isinstance(exc, RoleSignal):
                            rotate = exc.rotate
                        else:
                            rotate = status in ("quota", "busy") or rotate_on_error
                        if not rotate:
                            _save_state(self.state_file, state)
                            raise
                        break  # next role
                    now = _now()
                    _record_call(roles_state, role, "ok", int((now - started) * 1000), now, cooldown)
                    if resume:
                        _record_progress(progress, unit, role, value, now)
                        progress_key["updated_at"] = now
                    data[unit] = value
                    if role not in roles_used:
                        roles_used.append(role)
                    served = True
                    break
                if served:
                    break
            # A unit no role could serve stays pending: the next run resumes it.
        _save_state(self.state_file, state)
        return ScheduleResult(data, roles_used, skipped_roles, failures)
