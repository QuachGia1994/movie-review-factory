"""Overnight batch queue: paste many links, run each through the pipeline in
sequence, fault-tolerant.

The engine here is pure and dependency-injected (``process`` does the real
create -> download -> run in the web app; tests pass a fake), so the queue logic
- ordering, per-item fault isolation, cancellation - is fully unit-tested offline.
It never auto-approves: the web-app worker stops each job at the script-review
gate, so a failed link is logged and skipped without derailing the run.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Callable

MAX_BATCH_ITEMS = 25
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def parse_batch_input(text: str, *, max_items: int = MAX_BATCH_ITEMS) -> list[str]:
    """Pure: split pasted text into a de-duplicated list of http(s) links.

    One URL per line; blank lines and ``#`` comments are ignored. Raises when no
    valid link is found or the count exceeds ``max_items`` so an overnight run
    cannot be pointed at hundreds of jobs by accident.
    """
    urls: list[str] = []
    seen: set[str] = set()
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not _URL_RE.match(line):
            raise ValueError(f"liên kết không hợp lệ (cần http/https): {line[:80]}")
        if line not in seen:
            seen.add(line)
            urls.append(line)
    if not urls:
        raise ValueError("chưa có liên kết hợp lệ nào để chạy hàng đợi")
    if len(urls) > max_items:
        raise ValueError(f"tối đa {max_items} liên kết mỗi hàng đợi (đang có {len(urls)})")
    return urls


def slug_for(url: str, index: int) -> str:
    """Pure: deterministic, OS-safe job id for a queued link."""
    tail = re.split(r"[?#]", url, maxsplit=1)[0].rstrip("/").rsplit("/", 1)[-1]
    token = re.sub(r"[^A-Za-z0-9]+", "-", tail).strip("-").lower()[:24]
    return f"batch-{index + 1:02d}-{token}" if token else f"batch-{index + 1:02d}"


def initial_items(urls: list[str]) -> list[dict]:
    """Pure: the queue's starting rows (all pending)."""
    return [
        {"index": i, "url": u, "status": "pending", "job_id": None, "error": None}
        for i, u in enumerate(urls)
    ]


def _emit(on_event: Callable[[dict], None] | None, item: dict) -> None:
    if on_event is not None:
        on_event(dict(item))


def run_batch(
    urls: list[str],
    *,
    process: Callable[[str, int], str],
    on_event: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[dict]:
    """Run ``process(url, index) -> job_id`` for each url in order, fault-tolerant.

    A failure in one item is captured (``status='failed'`` + ``error``) and the
    queue moves on - it never aborts the whole run. ``should_stop`` (checked before
    each item) lets the caller cancel; the remaining items are marked ``cancelled``.
    Returns the final per-item result rows.
    """
    results = initial_items(urls)
    for item in results:
        if should_stop is not None and should_stop():
            item["status"] = "cancelled"
            _emit(on_event, item)
            continue
        item["status"] = "running"
        _emit(on_event, item)
        try:
            item["job_id"] = process(item["url"], item["index"])
            item["status"] = "ready"
        except Exception as exc:  # fault-tolerant: record on the row, keep going
            item["status"] = "failed"
            item["error"] = str(exc)
        _emit(on_event, item)
    return results


# --- completion webhook (Telegram / Discord) --------------------------------
# When the whole queue finishes, optionally ping a chat so the operator does not
# have to babysit an overnight run. Configured by env; a webhook failure never
# affects the queue result.
DISCORD_WEBHOOK_ENV = "MRF_DISCORD_WEBHOOK_URL"
TELEGRAM_TOKEN_ENV = "MRF_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ENV = "MRF_TELEGRAM_CHAT_ID"


def render_batch_summary(results: list[dict]) -> str:
    """Pure: a short human summary of a finished batch (counts + failed links)."""
    counts: dict[str, int] = {}
    for item in results:
        counts[item.get("status", "")] = counts.get(item.get("status", ""), 0) + 1
    ready = counts.get("ready", 0)
    failed = counts.get("failed", 0)
    cancelled = counts.get("cancelled", 0)
    lines = [
        f"Movie Review Factory — hàng đợi xong: {ready} sẵn sàng duyệt, "
        f"{failed} lỗi, {cancelled} hủy / tổng {len(results)}."
    ]
    fails = [item for item in results if item.get("status") == "failed"]
    if fails:
        lines.append("Lỗi:")
        for item in fails[:10]:
            lines.append(f"  • {item.get('url')} — {item.get('error') or 'lỗi'}")
    return "\n".join(lines)


def webhook_targets(env: dict) -> list[dict]:
    """Pure: build the configured notification targets from an env mapping."""
    targets: list[dict] = []
    discord = str(env.get(DISCORD_WEBHOOK_ENV) or "").strip()
    if discord:
        targets.append({"kind": "discord", "url": discord})
    token = str(env.get(TELEGRAM_TOKEN_ENV) or "").strip()
    chat = str(env.get(TELEGRAM_CHAT_ENV) or "").strip()
    if token and chat:
        targets.append({"kind": "telegram",
                        "url": f"https://api.telegram.org/bot{token}/sendMessage",
                        "chat_id": chat})
    return targets


def _payload_for(target: dict, message: str) -> bytes:
    if target["kind"] == "telegram":
        body = {"chat_id": target["chat_id"], "text": message}
    else:  # discord
        body = {"content": message}
    return json.dumps(body).encode("utf-8")


def _post_json(url: str, data: bytes, *, timeout: float = 10.0) -> int:
    request = urllib.request.Request(
        url, data=data, headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (operator-configured URL)
        return int(getattr(response, "status", None) or response.getcode())


def notify_batch_complete(results, *, env=None, http_post=None, timeout: float = 10.0) -> list[dict]:
    """Fire completion webhooks; never raises (a webhook issue must not fail the run).

    Targets come from env (``MRF_DISCORD_WEBHOOK_URL`` and/or
    ``MRF_TELEGRAM_BOT_TOKEN`` + ``MRF_TELEGRAM_CHAT_ID``). Returns a per-target
    outcome list; ``http_post`` is injectable for offline tests.
    """
    environ = os.environ if env is None else env
    targets = webhook_targets(environ)
    if not targets:
        return []
    message = render_batch_summary(results)
    post = http_post or _post_json
    outcomes: list[dict] = []
    for target in targets:
        try:
            status = int(post(target["url"], _payload_for(target, message), timeout=timeout))
            outcomes.append({"kind": target["kind"], "ok": 200 <= status < 300, "status": status})
        except Exception as exc:
            outcomes.append({"kind": target["kind"], "ok": False, "error": str(exc)})
    return outcomes
