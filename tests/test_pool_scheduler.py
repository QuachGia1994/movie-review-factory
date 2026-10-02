"""Tests for the shared AGY pool scheduler (roadmap #13)."""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path

import pytest

from movie_review_factory import pool_scheduler
from movie_review_factory.pool_scheduler import (
    BusySignal,
    PoolConfigError,
    PoolScheduler,
    QuotaSignal,
    RotateSignal,
)

ROLES = ("advisor", "executor", "experiment", "reviewer")
WORKERS = [
    (
        role,
        f"http://127.0.0.1:{7411 + number}/run",
        Path("C:/pool/agy-workspaces"),
        f"token-{role}",
    )
    for number, role in enumerate(ROLES)
]


@pytest.fixture(autouse=True)
def _isolated_pool_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep scheduler tests off the shared %LOCALAPPDATA% pool health file."""
    monkeypatch.setenv("MRF_POOL_STATE_FILE", str(tmp_path / "pool_health.json"))
    for name in ("MRF_AGY_RETRIES", "MRF_AGY_COOLDOWN_SECONDS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def state_file(tmp_path: Path) -> Path:
    return tmp_path / "pool_health.json"


def _read_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_quota_role_cools_down_and_is_skipped_on_the_next_run(state_file: Path) -> None:
    calls: list[str] = []

    def attempt(worker: tuple, unit: object) -> dict:
        role = worker[0]
        calls.append(role)
        if role == "advisor":
            raise QuotaSignal("quota exhausted")
        return {"served_by": role}

    scheduler = PoolScheduler(WORKERS, state_file=state_file)
    first = scheduler.run("batch-1", attempt)

    assert first.data == {"job": {"served_by": "executor"}}
    assert first.roles_used == ["executor"]
    assert first.failures == ["advisor quota exhausted"]
    assert first.skipped_roles == []

    calls.clear()
    second = scheduler.run("batch-2", attempt)

    # The quota role never answers with 429 again while it cools down, and every other role still runs.
    assert calls == ["executor"]
    assert second.data == {"job": {"served_by": "executor"}}
    assert second.roles_used == ["executor"]
    assert second.skipped_roles == ["advisor"]
    assert second.failures == ["advisor in cooldown"]

    roles = _read_state(state_file)["roles"]
    assert roles["advisor"]["status"] == "quota"
    assert roles["advisor"]["cooldown_until"] > time.time() + 200


def test_health_telemetry_records_status_latency_and_timestamp(state_file: Path) -> None:
    def attempt(worker: tuple, unit: object) -> str:
        role = worker[0]
        if role == "advisor":
            raise QuotaSignal("quota exhausted")
        if role == "executor":
            raise BusySignal("busy")
        if role == "experiment":
            raise RotateSignal("boom")
        return "fine"

    PoolScheduler(WORKERS, state_file=state_file).run("batch", attempt)

    state = _read_state(state_file)
    assert state["version"] == 1
    roles = state["roles"]
    assert {role: info["status"] for role, info in roles.items()} == {
        "advisor": "quota",
        "executor": "busy",
        "experiment": "error",
        "reviewer": "ok",
    }
    for info in roles.values():
        assert isinstance(info["latency_ms"], int) and info["latency_ms"] >= 0
        datetime.datetime.strptime(info["checked_at"], "%Y-%m-%dT%H:%M:%SZ")
    assert roles["advisor"]["cooldown_until"] > time.time() + 200
    assert roles["executor"]["cooldown_until"] == 0.0
    assert roles["reviewer"]["cooldown_until"] == 0.0


def test_bounded_retries_stop_after_configured_attempts(
    state_file: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MRF_AGY_RETRIES", "2")
    slept: list[float] = []
    monkeypatch.setattr(pool_scheduler, "_sleep", lambda seconds: slept.append(seconds))
    calls: list[str] = []

    def attempt(worker: tuple, unit: object) -> str:
        calls.append(worker[0])
        raise QuotaSignal("quota exhausted")

    outcome = PoolScheduler(WORKERS, state_file=state_file).run("batch", attempt)

    assert calls == [
        "advisor", "advisor",
        "executor", "executor",
        "experiment", "experiment",
        "reviewer", "reviewer",
    ]
    assert slept == [0.5, 0.5, 0.5, 0.5]
    assert outcome.data == {}
    assert outcome.roles_used == []
    assert outcome.failures == [
        "advisor quota exhausted",
        "executor quota exhausted",
        "experiment quota exhausted",
        "reviewer quota exhausted",
    ]


def test_default_retry_count_makes_one_attempt_per_role(state_file: Path) -> None:
    calls: list[str] = []

    def attempt(worker: tuple, unit: object) -> str:
        calls.append(worker[0])
        raise QuotaSignal("quota exhausted")

    PoolScheduler(WORKERS, state_file=state_file).run("batch", attempt)

    assert calls == list(ROLES)


def test_partial_resume_retries_only_unfinished_units(state_file: Path) -> None:
    phase = {"value": 1}
    calls: list[tuple[str, str]] = []

    def attempt(worker: tuple, unit: object) -> str:
        role, unit_name = worker[0], str(unit)
        calls.append((unit_name, role))
        if unit_name == "first":
            return "first-done"  # advisor serves the first unit in run one
        if phase["value"] == 1:
            raise RotateSignal("transient")
        return "second-done"

    scheduler = PoolScheduler(WORKERS, state_file=state_file)
    first = scheduler.run("batch", attempt, units=("first", "second"))

    assert first.data == {"first": "first-done"}
    assert first.roles_used == ["advisor"]
    assert first.failures == [
        "advisor transient",
        "executor transient",
        "experiment transient",
        "reviewer transient",
    ]
    assert [role for _, role in calls] == ["advisor", "advisor", "executor", "experiment", "reviewer"]

    calls.clear()
    phase["value"] = 2
    second = scheduler.run("batch", attempt, units=("first", "second"))

    # Only the unfinished unit is retried, still walking the pool in order.
    assert calls == [("second", "advisor")]
    assert second.data == {"first": "first-done", "second": "second-done"}
    assert second.roles_used == ["advisor"]
    assert second.failures == []


def test_busy_role_rotates_now_and_is_called_again_on_the_next_run(
    state_file: Path,
) -> None:
    calls: list[str] = []

    def attempt(worker: tuple, unit: object) -> str:
        calls.append(worker[0])
        if worker[0] == "advisor":
            raise BusySignal("busy")
        return "ok"

    scheduler = PoolScheduler(WORKERS, state_file=state_file)
    first = scheduler.run("first", attempt)

    assert calls == ["advisor", "executor"]
    assert first.roles_used == ["executor"]
    assert first.failures == ["advisor busy"]

    calls.clear()
    second = scheduler.run("second", attempt)

    assert calls == ["advisor", "executor"]
    assert second.skipped_roles == []
    assert _read_state(state_file)["roles"]["advisor"]["status"] == "busy"


def test_non_rotating_failure_reraises_the_original_error(state_file: Path) -> None:
    def attempt(worker: tuple, unit: object) -> str:
        raise RuntimeError("worker exploded")

    scheduler = PoolScheduler(WORKERS, state_file=state_file)
    with pytest.raises(RuntimeError, match="worker exploded"):
        scheduler.run("batch", attempt, rotate_on_error=False)

    roles = _read_state(state_file)["roles"]
    assert roles["advisor"]["status"] == "error"
    assert roles["advisor"]["cooldown_until"] == 0.0


def test_cooldown_env_override_expires_and_lets_the_role_back_in(
    state_file: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MRF_AGY_COOLDOWN_SECONDS", "30")
    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(pool_scheduler, "_now", lambda: clock["now"])
    calls: list[str] = []

    def attempt(worker: tuple, unit: object) -> str:
        calls.append(worker[0])
        if worker[0] == "advisor":
            raise QuotaSignal("quota exhausted")
        return "ok"

    scheduler = PoolScheduler(WORKERS, state_file=state_file)
    scheduler.run("first", attempt)
    assert calls == ["advisor", "executor"]
    assert _read_state(state_file)["roles"]["advisor"]["cooldown_until"] == 1_000_030.0

    calls.clear()
    clock["now"] = 1_000_010.0
    scheduler.run("second", attempt)
    assert calls == ["executor"]

    calls.clear()
    clock["now"] = 1_000_031.0
    scheduler.run("third", attempt)
    assert calls == ["advisor", "executor"]


def test_resume_disabled_reaches_a_worker_on_every_run(state_file: Path) -> None:
    calls: list[str] = []

    def attempt(worker: tuple, unit: object) -> str:
        calls.append(worker[0])
        return "fresh"

    scheduler = PoolScheduler(WORKERS, state_file=state_file)
    first = scheduler.run("stage", attempt, resume=False)
    second = scheduler.run("stage", attempt, resume=False)

    assert first.data == {"job": "fresh"}
    assert second.data == {"job": "fresh"}
    assert calls == ["advisor", "advisor"]
    assert _read_state(state_file)["progress"] == {}


def test_progress_state_stays_bounded(state_file: Path) -> None:
    scheduler = PoolScheduler(WORKERS, state_file=state_file)
    for index in range(pool_scheduler.MAX_PROGRESS_KEYS + 5):
        scheduler.run(f"key-{index}", lambda worker, unit: "ok")

    progress = _read_state(state_file)["progress"]
    assert len(progress) == pool_scheduler.MAX_PROGRESS_KEYS


def test_corrupt_state_file_is_recovered(state_file: Path) -> None:
    state_file.write_text("{not json", encoding="utf-8")

    outcome = PoolScheduler(WORKERS, state_file=state_file).run(
        "batch", lambda worker, unit: "ok"
    )

    assert outcome.data == {"job": "ok"}
    assert _read_state(state_file)["roles"]["advisor"]["status"] == "ok"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MRF_AGY_RETRIES", "4", "MRF_AGY_RETRIES must be between 0 and 3"),
        ("MRF_AGY_RETRIES", "many", "MRF_AGY_RETRIES must be an integer"),
        ("MRF_AGY_COOLDOWN_SECONDS", "-1", "MRF_AGY_COOLDOWN_SECONDS must be greater than zero"),
        ("MRF_AGY_COOLDOWN_SECONDS", "soon", "MRF_AGY_COOLDOWN_SECONDS must be numeric"),
    ],
)
def test_invalid_scheduler_env_is_rejected(
    state_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv(name, value)
    scheduler = PoolScheduler(WORKERS, state_file=state_file)

    with pytest.raises(PoolConfigError, match=message):
        scheduler.run("batch", lambda worker, unit: "ok")


def test_state_file_env_override_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "runtime" / "health.json"
    monkeypatch.setenv("MRF_POOL_STATE_FILE", str(target))
    assert pool_scheduler.default_state_file() == target

    PoolScheduler(WORKERS).run("batch", lambda worker, unit: "ok")

    assert target.is_file()


def test_default_state_file_follows_localappdata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MRF_POOL_STATE_FILE", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    assert pool_scheduler.default_state_file() == (
        tmp_path / "AppData" / "Local" / "MovieReviewFactory" / "runtime" / "pool_health.json"
    )

    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert pool_scheduler.default_state_file() == (
        Path.home() / ".movie-review-factory" / "runtime" / "pool_health.json"
    )


def test_parallel_run_keeps_every_role_busy_at_once(state_file: Path) -> None:
    import threading

    lock = threading.Lock()
    active = {"now": 0, "peak": 0}
    served: list[str] = []

    def attempt(worker: tuple, unit: object) -> int:
        with lock:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
        time.sleep(0.05)
        with lock:
            active["now"] -= 1
            served.append(worker[0])
        return int(unit) * 10

    result = PoolScheduler(WORKERS, state_file=state_file).run("par", attempt, units=range(8), parallel=True)

    assert result.data == {unit: unit * 10 for unit in range(8)}
    assert active["peak"] == 4
    assert set(served) == set(ROLES)
    assert result.roles_used == list(ROLES)
    assert set(_read_state(state_file)["progress"]["par"]["units"]) == {str(unit) for unit in range(8)}


def test_parallel_run_retires_quota_role_and_reroutes_its_unit(state_file: Path) -> None:
    def attempt(worker: tuple, unit: object) -> str:
        if worker[0] == "advisor":
            raise QuotaSignal("quota exhausted")
        time.sleep(0.01)
        return worker[0]

    result = PoolScheduler(WORKERS, state_file=state_file).run("quota", attempt, units=range(6), parallel=True)

    assert set(result.data) == set(range(6))
    assert "advisor" not in result.data.values()
    assert result.failures.count("advisor quota exhausted") == 1
    assert _read_state(state_file)["roles"]["advisor"]["status"] == "quota"


def test_parallel_run_reraises_non_rotating_failure(state_file: Path) -> None:
    def attempt(worker: tuple, unit: object) -> int:
        if unit == 2:
            raise ValueError("bad batch")
        return int(unit)

    with pytest.raises(ValueError, match="bad batch"):
        PoolScheduler(WORKERS, state_file=state_file).run(
            "fatal", attempt, units=range(4), rotate_on_error=False, parallel=True,
        )


def test_parallel_run_resumes_only_missing_units(state_file: Path) -> None:
    calls: list[object] = []
    scheduler = PoolScheduler(WORKERS, state_file=state_file)
    scheduler.run("resume", lambda worker, unit: unit, units=[0, 1], parallel=True)

    def attempt(worker: tuple, unit: object) -> object:
        calls.append(unit)
        return unit

    result = scheduler.run("resume", attempt, units=[0, 1, 2, 3], parallel=True)

    assert sorted(calls) == [2, 3]
    assert result.data == {0: 0, 1: 1, 2: 2, 3: 3}


def test_concurrent_single_calls_start_on_different_idle_roles(state_file: Path) -> None:
    import threading

    gate = threading.Barrier(2)
    first_roles: list[str] = []

    def attempt(worker: tuple, unit: object) -> str:
        first_roles.append(worker[0])
        gate.wait(timeout=2)
        return worker[0]

    threads = [
        threading.Thread(
            target=PoolScheduler(WORKERS, state_file=state_file).run,
            args=(f"single-{n}", attempt), kwargs={"resume": False},
        )
        for n in range(2)
    ]
    threads[0].start()
    time.sleep(0.05)
    threads[1].start()
    for thread in threads:
        thread.join(timeout=5)

    assert first_roles == ["advisor", "executor"]
