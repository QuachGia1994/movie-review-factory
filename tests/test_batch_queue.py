"""Offline tests for the overnight batch-queue engine (pure, no pipeline)."""
from __future__ import annotations

import pytest

from movie_review_factory import batch_queue


def test_parse_dedupes_and_skips_blanks_and_comments():
    text = "https://a/1\n\n  # a comment\nhttps://a/2\nhttps://a/1\n"
    assert batch_queue.parse_batch_input(text) == ["https://a/1", "https://a/2"]


def test_parse_rejects_non_http_and_empty():
    with pytest.raises(ValueError):
        batch_queue.parse_batch_input("ftp://x/1")
    with pytest.raises(ValueError):
        batch_queue.parse_batch_input("   \n# only comments\n")


def test_parse_enforces_max_items():
    many = "\n".join(f"https://a/{i}" for i in range(50))
    with pytest.raises(ValueError):
        batch_queue.parse_batch_input(many, max_items=25)


def test_slug_is_safe_and_deterministic():
    slug = batch_queue.slug_for("https://youtu.be/AbC123?t=5", 0)
    assert slug == batch_queue.slug_for("https://youtu.be/AbC123?t=5", 0)
    assert slug.startswith("batch-01-")
    assert all(c.isalnum() or c == "-" for c in slug)


def test_run_batch_is_fault_tolerant_and_preserves_order():
    urls = ["https://ok/a", "https://boom/b", "https://ok/c"]
    events: list[tuple[int, str]] = []

    def process(url, index):
        if "boom" in url:
            raise RuntimeError("nguồn hỏng")
        return f"job-{index}"

    results = batch_queue.run_batch(
        urls, process=process, on_event=lambda it: events.append((it["index"], it["status"]))
    )
    assert [r["status"] for r in results] == ["ready", "failed", "ready"]
    assert results[0]["job_id"] == "job-0"
    assert results[1]["error"] == "nguồn hỏng" and results[1]["job_id"] is None
    assert results[2]["job_id"] == "job-2"
    # every item emitted at least a running + terminal event, in order
    assert (1, "failed") in events and (0, "ready") in events


def test_run_batch_cancels_remaining_when_asked_to_stop():
    urls = ["https://a/1", "https://a/2", "https://a/3"]
    processed: list[str] = []

    def process(url, index):
        processed.append(url)
        return f"job-{index}"

    # stop before any item runs -> everything is cancelled, nothing processed
    results = batch_queue.run_batch(urls, process=process, should_stop=lambda: True)
    assert [r["status"] for r in results] == ["cancelled", "cancelled", "cancelled"]
    assert processed == []


# --- completion webhook ------------------------------------------------------

_RESULTS = [
    {"index": 0, "url": "https://a", "status": "ready", "job_id": "j0", "error": None},
    {"index": 1, "url": "https://b", "status": "failed", "job_id": None, "error": "boom"},
    {"index": 2, "url": "https://c", "status": "cancelled", "job_id": None, "error": None},
]


def test_render_batch_summary_counts_and_lists_failures():
    msg = batch_queue.render_batch_summary(_RESULTS)
    assert "1 sẵn sàng duyệt" in msg
    assert "https://b" in msg and "boom" in msg


def test_webhook_targets_from_env():
    assert batch_queue.webhook_targets({}) == []
    only_discord = batch_queue.webhook_targets({"MRF_DISCORD_WEBHOOK_URL": "https://d/hook"})
    assert [t["kind"] for t in only_discord] == ["discord"]
    # telegram needs BOTH token and chat id
    assert batch_queue.webhook_targets({"MRF_TELEGRAM_BOT_TOKEN": "T"}) == []
    both = batch_queue.webhook_targets({
        "MRF_DISCORD_WEBHOOK_URL": "https://d/hook",
        "MRF_TELEGRAM_BOT_TOKEN": "T", "MRF_TELEGRAM_CHAT_ID": "C",
    })
    assert {t["kind"] for t in both} == {"discord", "telegram"}


def test_notify_posts_discord_and_telegram_payloads():
    import json

    calls = []

    def fake_post(url, data, *, timeout=10.0):
        calls.append((url, json.loads(data.decode("utf-8"))))
        return 204

    env = {"MRF_DISCORD_WEBHOOK_URL": "https://discord/hook",
           "MRF_TELEGRAM_BOT_TOKEN": "T", "MRF_TELEGRAM_CHAT_ID": "C"}
    outcomes = batch_queue.notify_batch_complete(_RESULTS, env=env, http_post=fake_post)
    assert {o["kind"] for o in outcomes} == {"discord", "telegram"}
    assert all(o["ok"] for o in outcomes)
    discord_call = next(c for c in calls if "discord" in c[0])
    telegram_call = next(c for c in calls if "sendMessage" in c[0])
    assert "content" in discord_call[1] and "boom" in discord_call[1]["content"]
    assert telegram_call[1]["chat_id"] == "C" and "text" in telegram_call[1]


def test_notify_no_targets_is_noop():
    assert batch_queue.notify_batch_complete(_RESULTS, env={}) == []


def test_notify_is_fault_tolerant():
    def boom_post(url, data, *, timeout=10.0):
        raise RuntimeError("net down")

    outcomes = batch_queue.notify_batch_complete(
        _RESULTS, env={"MRF_DISCORD_WEBHOOK_URL": "https://d/hook"}, http_post=boom_post)
    assert outcomes[0]["ok"] is False and outcomes[0]["error"]
