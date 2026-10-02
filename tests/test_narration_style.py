import json
from pathlib import Path

import pytest

import movie_review_factory.pipeline as pipeline_mod
from movie_review_factory import narration_style
from movie_review_factory.models import JobConfig
from movie_review_factory.pipeline import create_job, load_manifest, run_job
from movie_review_factory.webapp import JobsService


def test_language_name_and_detection() -> None:
    assert narration_style.language_name("en") == "English"
    assert narration_style.language_name("en-US") == "English"
    assert narration_style.language_name("vi") == "Vietnamese"
    assert narration_style.language_name("fr") == "fr"
    assert narration_style.is_vietnamese("vi")
    assert narration_style.is_vietnamese("")
    assert not narration_style.is_vietnamese("en")


def test_prompt_style_is_english_only_and_targets_words() -> None:
    assert narration_style.prompt_style("vi") is None
    style = narration_style.prompt_style("en", [
        {"title": "Hook", "budget_minutes": 0.5},
        {"title": "Story", "budget_minutes": 2},
        "legacy string section",
    ])
    assert style["language"] == "English"
    assert style["words_per_minute"] == 155
    assert any("welcome back" in rule for rule in style["guide"])
    assert style["section_word_targets"] == [
        {"title": "Hook", "target_words": 78},
        {"title": "Story", "target_words": 310},
    ]


def test_english_scaffold_has_no_vietnamese_text(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="en-job", language="en", target_minutes=5, content_agent="scaffold"))
    run_job(tmp_path, until="script")

    outline = json.loads((tmp_path / "outline.json").read_text(encoding="utf-8"))
    script = json.loads((tmp_path / "script.json").read_text(encoding="utf-8"))
    titles = [section["title"] for section in outline["sections"]]
    assert titles[0] == "Hook" and titles[-1] == "Call to action"
    assert sum(section["budget_minutes"] for section in outline["sections"]) == 5.0
    text = json.dumps([outline, script], ensure_ascii=False)
    assert all(ord(ch) < 128 for ch in text)
    assert script["sections"][1]["narration"].startswith("Draft narration for the")


def test_vietnamese_scaffold_is_unchanged(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="vi-job", language="vi", target_minutes=5, content_agent="scaffold"))
    run_job(tmp_path, until="script")

    outline = json.loads((tmp_path / "outline.json").read_text(encoding="utf-8"))
    script = json.loads((tmp_path / "script.json").read_text(encoding="utf-8"))
    assert outline["sections"][0]["title"] == "Mở đầu / hook"
    assert outline["sections"][1]["title"] == "Bối cảnh & tiền đề"
    assert script["sections"][0]["narration"].startswith("Bản thảo lời dẫn cho phần")


def _capture_agent(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake_agent(**kwargs: object) -> dict:
        calls.append(kwargs)
        stage = str(kwargs["stage"])
        if stage == "research":
            return {"brief": "brief", "facts": [], "sources": [], "uncertainties": []}
        if stage == "outline":
            return {
                "sections": [
                    {"title": "Hook", "budget_minutes": 1, "purpose": "a"},
                    {"title": "Story", "budget_minutes": 2, "purpose": "b"},
                ],
                "notes": "",
            }
        return {
            "sections": [
                {"title": "Hook", "narration": "A cop walks into a trap."},
                {"title": "Story", "narration": "Then it gets worse."},
            ],
            "notes": "",
        }

    monkeypatch.setattr(pipeline_mod, "run_claude_json", fake_agent)
    return calls


def test_english_prompts_carry_language_name_and_style(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_agent(monkeypatch)
    create_job(tmp_path, JobConfig(job_id="en-agent", language="en", target_minutes=3, content_agent="claude"))
    run_job(tmp_path, until="script")

    by_stage = {call["stage"]: str(call["prompt"]) for call in calls}
    assert "original English review/recap" in by_stage["research"]
    assert "English review/recap outline" in by_stage["outline"]
    assert "Write natural English narration" in by_stage["script"]
    assert '"narration_style"' in by_stage["outline"]
    assert '"section_word_targets"' in by_stage["script"]
    assert '"target_words": 310' in by_stage["script"]


def test_vietnamese_prompts_have_no_style_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_agent(monkeypatch)
    create_job(tmp_path, JobConfig(job_id="vi-agent", language="vi", target_minutes=3, content_agent="claude"))
    run_job(tmp_path, until="script")

    assert all('"narration_style"' not in str(call["prompt"]) for call in calls)
    assert "Vietnamese review/recap" in str(calls[0]["prompt"])


def test_vieneu_is_not_vietnamese_only() -> None:
    assert "vieneu" not in narration_style.VIETNAMESE_ONLY_TTS


def test_tts_rejects_vietnamese_only_provider_for_english(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="en-tts", language="en", tts_provider="fptai"))
    (tmp_path / "script.json").write_text(json.dumps({
        "approved": True,
        "sections": [{"title": "Hook", "narration": "A short line."}],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="only speaks Vietnamese"):
        pipeline_mod._tts(tmp_path, load_manifest(tmp_path))


def test_english_metadata_uses_movie_title_and_english_copy(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="en-meta", language="en", movie_title="Heat"))
    (tmp_path / "script.json").write_text(json.dumps({
        "sections": [{"title": "Hook", "narration": "x"}],
    }), encoding="utf-8")
    pipeline_mod._metadata(tmp_path, load_manifest(tmp_path))

    meta = json.loads((tmp_path / "youtube_metadata.json").read_text(encoding="utf-8"))
    assert meta["title"] == "Heat – Movie Recap"
    assert "In this video:" in meta["description"]
    assert "phim" not in meta["tags"]
    assert meta["approved"] is False


def test_fallback_thumbnail_headlines_follow_language() -> None:
    english = JobsService._fallback_thumbnail_headlines("Heat", "en")
    assert english[0] == "HEAT: THE DARK TRUTH"
    assert all(ord(ch) < 128 for line in english for ch in line)
    assert JobsService._fallback_thumbnail_headlines("Heat")[0] == "HEAT: SỰ THẬT KINH HOÀNG"
