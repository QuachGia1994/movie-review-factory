from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

import pytest

import movie_review_factory.media_intelligence as mi
from movie_review_factory import pipeline, webapp
from movie_review_factory.media_store import MediaStore
from movie_review_factory.models import JobConfig, MediaAsset, Shot, TranscriptSegment
from movie_review_factory.webapp import JobsService, create_server


def _indexed_job(jobs_root: Path, *, job_id: str = "intel", mode: str = "scaffold") -> Path:
    root = jobs_root / job_id
    source = root / "source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"video")
    pipeline.create_job(root, JobConfig(job_id=job_id, source_video=source, content_agent=mode))
    with MediaStore(root / "media_index.sqlite3") as store:
        store.migrate()
        store.replace_index(
            MediaAsset(path=source, duration_seconds=120),
            [
                Shot(media_asset_id=1, start_seconds=40, end_seconds=120, label="Long climax"),
                Shot(media_asset_id=1, start_seconds=0, end_seconds=10, label="Opening"),
            ],
            [
                TranscriptSegment(media_asset_id=1, start_seconds=1, end_seconds=3, text="mysterious lighthouse appears"),
                TranscriptSegment(media_asset_id=1, start_seconds=41, end_seconds=45, text="hero discovers the secret and saves the town"),
            ],
        )
    return root


def test_highlights_are_deterministic_sorted_and_bounded(tmp_path: Path) -> None:
    root = _indexed_job(tmp_path)
    first = mi.detect_highlights(root / "media_index.sqlite3", limit=20)
    second = mi.detect_highlights(root / "media_index.sqlite3", limit=20)
    assert first == second
    assert [item["id"] for item in first] == ["h-1", "h-2"]
    assert all(0.0 <= item["start"] < item["end"] <= item["start"] + mi.MAX_CLIP_SECONDS for item in first)
    assert first[0]["score"] >= first[1]["score"]


def test_highlight_clip_validates_exports_vertical_and_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _indexed_job(tmp_path)
    svc = JobsService(tmp_path)
    with pytest.raises(ValueError):
        svc.highlight_clip_path("intel", "../bad")
    calls = []
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "ffmpeg.exe")
    def fake_run(command, **krwags):
        calls.append((command, krwags))
        Path(command[-1]).write_bytes(b"clip")
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    first = svc.highlight_clip_path("intel", "h-1")
    second = svc.highlight_clip_path("intel", "h-1")
    assert first == second and first.read_bytes() == b"clip"
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert "scale=1080:1920" in command[command.index("-vf") + 1]
    assert kwargs["shell"] is False


def test_retrieval_ranks_matching_dialogue_before_unrelated_segments(tmp_path: Path) -> None:
    root = _indexed_job(tmp_path)
    matches = mi.retrieve_segments(root / "media_index.sqlite3", "Who saves the town?")
    assert [item["text"] for item in matches] == ["hero discovers the secret and saves the town"]
    assert mi.retrieve_segments(root / "media_index.sqlite3", "quantum banana") == []
    unknown = mi.answer_chat(root, "quantum banana")
    assert unknown["citations"] == []
    assert unknown["mode"] == "local"


def test_real_highlight_export_has_vertical_video(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "data" / "raw" / "ultracode-smoke-sample.mp4"
    if not source.is_file() or not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("owned sample and FFmpeg are required")
    root = tmp_path / "real-highlight"
    pipeline.create_job(root, JobConfig(job_id=root.name, source_video=source))
    with MediaStore(root / "media_index.sqlite3") as store:
        store.migrate()
        store.replace_index(
            MediaAsset(path=source, duration_seconds=2),
            [Shot(media_asset_id=1, start_seconds=0, end_seconds=0.5, label="Opening")],
            [],
        )
    output = JobsService(tmp_path).highlight_clip_path(root.name, "h-1")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert output.stat().st_size > 0
    assert json.loads(probe.stdout)["streams"][0] == {"width": 1080, "height": 1920}


def test_chat_local_fallback_and_claude_citation_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local = _indexed_job(tmp_path, job_id="local")
    answer = mi.answer_chat(local, "Where is the lighthouse?")
    assert answer["mode"] == "local"
    assert "lighthouse" in answer["answer"]
    assert answer["citations"]
    with pytest.raises(ValueError):
        mi.retrieve_segments(local / "media_index.sqlite3", "")

    claude = _indexed_job(tmp_path, job_id="claude", mode="claude")
    captured = {}
    def valid_claude(**kw):
        captured.update(kw)
        return {"answer": "The lighthouse appears first.", "citation_ids": [1]}
    monkeypatch.setattr(mi.content_agent, "run_claude_json", valid_claude)
    answer = mi.answer_chat(claude, "lighthouse")
    assert answer["mode"] == "claude"
    assert [c["id"] for c in answer["citations"]] == [1]
    assert captured["allowed_tools"] == []

    monkeypatch.setattr(mi.content_agent, "run_claude_json", lambda **kw: {"answer": "bad", "citation_ids": [999]})
    fallback = mi.answer_chat(claude, "lighthouse")
    assert fallback["mode"] == "local"
    assert all(c["id"] != 999 for c in fallback["citations"])


def test_dashboard_vtt_sync_and_authenticated_clip_download_contract() -> None:
    html = webapp.INDEX_HTML
    assert '<track id="sourceCaptions" kind="subtitles"' in html
    assert "const captions = await authFetch(data.transcript_vtt_href)" in html
    assert "track.track.mode = 'showing'" in html
    assert "async function downloadHighlight" in html
    assert "const resolved = await authFetch(href)" in html
    assert "downloadHighlight(item.export_href, item.id)" in html
    assert "link.href=item.export_href" not in html
    assert "video.src = data.media_href" in html
    assert "v.src = src" in html
    assert "authFetch(data.media_href)" not in html


def test_http_highlight_clip_chat_and_vtt_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _indexed_job(tmp_path)
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "ffmpeg.exe")
    monkeypatch.setattr(pipeline.subprocess, "run", lambda command, **kw: Path(command[-1]).write_bytes(b"clip"))
    server = create_server("127.0.0.1", 0, tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/api/jobs/intel"
    try:
        with urllib.request.urlopen(base + "/highlights", timeout=5) as resp:
            items = json.loads(resp.read())["highlights"]
        assert [item["id"] for item in items] == ["h-1", "h-2"]
        with urllib.request.urlopen(base + "/highlights/h-1/clip.mp4", timeout=5) as resp:
            assert resp.headers.get_content_type() == "video/mp4"
            assert resp.read() == b"clip"
        with urllib.request.urlopen(base + "/transcript.vtt", timeout=5) as resp:
            assert resp.headers.get_content_type() == "text/vtt"
            assert resp.read().startswith(b"WEBVTT")
        body = json.dumps({"question": "lighthouse"}).encode()
        req = urllib.request.Request(base + "/chat", data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        assert data["mode"] == "local" and data["citations"]
    finally:
        server.shutdown()
        server.server_close()


def test_chat_uses_agy_backend_for_agy_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _indexed_job(tmp_path, job_id="agy", mode="agy")
    captured: dict = {}

    def fake_agy(**kw):
        captured.update(kw)
        return {"answer": "The lighthouse appears first.", "citation_ids": [1]}

    monkeypatch.setattr(mi.agy_agent, "run_agy_json", fake_agy)
    answer = mi.answer_chat(root, "lighthouse")
    assert answer["mode"] == "agy"
    assert [c["id"] for c in answer["citations"]] == [1]
    assert captured["stage"] == "chat"
