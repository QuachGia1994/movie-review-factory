"""Per-run cooperative cancellation primitives shared by pipeline stages."""
from __future__ import annotations

import contextvars
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol


class ProcessLike(Protocol):
    pid: int


class RunCancelled(RuntimeError):
    """Raised when the operator stops only the active job run."""


@dataclass(frozen=True)
class CancellationContext:
    event: threading.Event
    register_process: Callable[[ProcessLike], None] | None = None
    unregister_process: Callable[[ProcessLike], None] | None = None


_CURRENT: contextvars.ContextVar[CancellationContext | None] = contextvars.ContextVar(
    "mrf_cancellation_context", default=None
)


@contextmanager
def cancellation_scope(context: CancellationContext | None) -> Iterator[None]:
    token = _CURRENT.set(context)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current_context() -> CancellationContext | None:
    return _CURRENT.get()


def checkpoint(message: str = "Project cancelled by user") -> None:
    context = current_context()
    if context is not None and context.event.is_set():
        raise RunCancelled(message)


def cancellable_sleep(seconds: float, message: str = "Project cancelled by user") -> None:
    context = current_context()
    if context is None:
        time.sleep(seconds)
        return
    if context.event.wait(seconds):
        raise RunCancelled(message)
