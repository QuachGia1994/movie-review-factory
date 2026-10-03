import json
import zipfile
from pathlib import Path

import pytest

from movie_review_factory import packaging
from movie_review_factory import pipeline as pipeline_mod
from movie_review_factory.creator_library import CreatorLibrary
from movie_review_factory.handoff import build_handoff
from movie_review_factory.models import ChannelProfile, JobConfig
from movie_review_factory.narration_style import outline_titles
from movie_review_factory.pipeline import create_job, load_manifest, save_manifest


def test_hashtag_strips_vietnamese_marks() -> None:
    assert packaging.hashtag("Phim Kinh Dị Đài Loan") == "#PhimKinhDiDaiLoan"
    assert packaging.hashtag("!!!") == ""


def test_hashtags_put_format_movie_channel_first_and_dedupe() -> None:
    tags = packaging.hashtags("vi", "Marui Video", "Linh Miu", "kinh dị")
    assert tags[:3] == ["#ReviewPhim", "#MaruiVideo", "#LinhMiu"]
    assert "#PhimKinhDi" in tags and len(tags) == len(set(tags)) <= packaging.HASHTAG_MAX


@pytest.mark.parametrize("url,expected", [
    ("https://www.youtube.com/@kenh", "https://www.youtube.com/@kenh?sub_confirmation=1"),
    ("https://youtube.com/channel/UC1?sub_confirmation=0&x=1", "https://youtube.com/channel/UC1?x=1&sub_confirmation=1"),
    ("https://example.com/@kenh", ""),
    ("http://www.youtube.com/@kenh", ""),
    ("", ""),
])
def test_subscribe_link(url: str, expected: str) -> None:
    assert packaging.subscribe_link(url) == expected


def test_structure_guide_carries_greeting_series_and_compilation() -> None:
    channel = {"name": "Kênh A", "greeting": "Xin chào, đây là Kênh A",
               "series": {"title": "Kinh dị Nhật", "part": 2, "total": 3, "previous": "Phần 1", "next": "Phần 3"}}
    rules = " ".join(packaging.structure_guide("vi", "compilation", channel))
    assert "Xin chào, đây là Kênh A" in rules
    assert "Câu chuyện thứ N" in rules
    assert "2/3" in rules and "Phần 1" in rules and "Phần 3" in rules
    assert "lời đồn" in rules
    assert "chào" not in " ".join(packaging.structure_guide("vi", "single", {})).lower()


def test_compilation_outline_scaffold_titles() -> None:
    titles = outline_titles("vi", "compilation")
    assert titles[1:4] == ("Câu chuyện thứ 1", "Câu chuyện thứ 2", "Câu chuyện thứ 3")
    assert outline_titles("vi") == outline_titles("vi", "single")


def test_channel_profile_rejects_non_https_url() -> None:
    with pytest.raises(ValueError):
        ChannelProfile(name="A", channel_url="javascript:alert(1)")
    assert ChannelProfile(name="A", greeting="  Xin   chào ").greeting == "Xin chào"


def _job(tmp_path: Path, **config) -> Path:
    root = tmp_path / "jobs" / "j1"
    create_job(root, JobConfig(job_id="j1", movie_title="Marui Video", **config))
    (root / "script.json").write_text(json.dumps({"sections": [
        {"title": "Mở đầu", "narration": "Một câu."},
        {"title": "Câu chuyện thứ 1", "narration": "Hai câu."},
    ]}, ensure_ascii=False), encoding="utf-8")
    (root / "render.json").write_text(json.dumps({"clips": [
        {"section_index": 1, "duration_seconds": 30},
        {"section_index": 2, "duration_seconds": 90},
    ]}), encoding="utf-8")
    return root


def test_vietnamese_metadata_packages_titles_chapters_hashtags_and_pinned_comment(tmp_path: Path) -> None:
    root = _job(tmp_path, genre="kinh dị")
    library = CreatorLibrary(root.parent)
    library.save_channel("linh", {"name": "Linh Miu", "channel_url": "https://www.youtube.com/@linhmiu"})
    library.save_series("s1", "Kinh dị Nhật", [{"movie_title": "Marui Video", "job_id": "j1"},
                                               {"movie_title": "Phần 2", "job_id": None}])

    pipeline_mod._metadata(root, load_manifest(root))
    meta = json.loads((root / "youtube_metadata.json").read_text(encoding="utf-8"))

    assert meta["title"] == "Review phim kinh dị | Marui Video"
    assert len(meta["title_options"]) >= 2
    assert "00:00:00 Mở đầu\n00:00:30 Câu chuyện thứ 1" in meta["description"]
    assert "https://www.youtube.com/@linhmiu?sub_confirmation=1" in meta["description"]
    assert "fair use" in meta["description"]
    assert "Phần 1/2 của series Kinh dị Nhật" in meta["description"]
    assert meta["hashtags"][:3] == ["#ReviewPhim", "#MaruiVideo", "#LinhMiu"]
    assert "Marui Video" in meta["tags"] and "kinh dị" in meta["tags"]
    assert "Phần tiếp theo: Phần 2." in meta["pinned_comment"]
    assert "Đăng ký Linh Miu" in meta["pinned_comment"]
    assert meta["approved"] is False


def test_metadata_uses_agy_packaging_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _job(tmp_path, content_agent="agy", review_format="compilation")
    monkeypatch.setenv("MRF_AUTO_PACKAGING", "1")
    calls = []

    def fake_agy(**kwargs):
        calls.append(kwargs)
        return {"titles": ["Review phim kinh dị | Cuốn băng bị nguyền?"], "hook": "Cuốn băng này có gì?",
                "comment_question": "Truyện nào ám ảnh bạn nhất?", "tags": ["marui video", ""]}

    monkeypatch.setattr(pipeline_mod, "run_agy_json", fake_agy)
    _, note = pipeline_mod._metadata(root, load_manifest(root))
    meta = json.loads((root / "youtube_metadata.json").read_text(encoding="utf-8"))

    assert note == "metadata packaged by AGY"
    assert calls[0]["stage"] == "metadata_packaging" and "anthology" in calls[0]["prompt"]
    assert meta["title"] == "Review phim kinh dị | Cuốn băng bị nguyền?"
    assert meta["description"].startswith("Cuốn băng này có gì?")
    assert meta["pinned_comment"].startswith("Truyện nào ám ảnh bạn nhất?")
    assert "" not in meta["tags"]


def test_outline_prompt_receives_story_structure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _job(tmp_path, content_agent="claude", review_format="compilation")
    CreatorLibrary(root.parent).save_channel("a", {"name": "Kênh A", "greeting": "Xin chào cả nhà"})
    seen = {}

    def fake_agent(**kwargs):
        seen.update(kwargs)
        return {"sections": [{"title": "Hook", "budget_minutes": 1, "goal": "g"}], "notes": ""}

    monkeypatch.setattr(pipeline_mod, "_run_reasoning_agent", fake_agent)
    pipeline_mod._outline(root, load_manifest(root))

    assert seen["context"]["review_format"] == "compilation"
    assert seen["context"]["channel_name"] == "Kênh A"
    assert any("Xin chào cả nhà" in rule for rule in seen["context"]["story_structure"])
    assert "anthology" in seen["instruction"]


def test_handoff_writes_pinned_comment_and_title_options(tmp_path: Path) -> None:
    root = _job(tmp_path)
    pipeline_mod._metadata(root, load_manifest(root))
    meta = json.loads((root / "youtube_metadata.json").read_text(encoding="utf-8"))
    meta["approved"] = True
    (root / "youtube_metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    script = json.loads((root / "script.json").read_text(encoding="utf-8"))
    script["approved"] = True
    (root / "script.json").write_text(json.dumps(script, ensure_ascii=False), encoding="utf-8")
    for name, data in {"scene_plan.json": {}, "alignment.json": {}, "qa.json": {"passed": True, "output_file": "final.mp4"}}.items():
        (root / name).write_text(json.dumps(data), encoding="utf-8")
    for name in ("final.mp4", "aligned.srt", "thumbnail.jpg"):
        (root / name).write_bytes(b"x")
    manifest = load_manifest(root)
    for stage in ("script", "scene_plan", "alignment", "render", "qa", "metadata", "thumbnail"):
        manifest.stage(stage).mark("ready")
    save_manifest(root, manifest)

    with zipfile.ZipFile(build_handoff(root)) as archive:
        pinned = archive.read("pinned-comment.txt").decode("utf-8")
        notes = archive.read("upload-notes.txt").decode("utf-8")
    assert pinned.strip() == meta["pinned_comment"]
    assert "Pinned comment:" in notes and "Title options:" in notes
