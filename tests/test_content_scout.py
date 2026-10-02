"""Tests for Content Scout Engine (content_scout.py) and web integration."""
from __future__ import annotations

import pytest

from movie_review_factory import content_scout
from movie_review_factory.webapp import JobsService


def test_scored_seed_includes_vietnamese_summary():
    seed = {
        "id": "localized-seed",
        "title": "Original title",
        "source": "tmdb_douban",
        "summary": "Raw source summary.",
        "vietnamese_summary": "Tóm tắt bằng tiếng Việt.",
    }

    candidate = content_scout.score_movie_candidate(seed)

    assert candidate.vietnamese_summary == "Tóm tắt bằng tiếng Việt."
    assert candidate.to_dict()["vietnamese_summary"] == "Tóm tắt bằng tiếng Việt."


def test_normalization_keeps_vietnamese_and_han_but_removes_markup():
    normalized = content_scout.normalize_scout_text("<b>ＴＲẢ&nbsp; THÙ！</b>  重生")
    assert normalized == "trả thù 重生"


def test_multilingual_concepts_are_equivalent_and_deduplicated():
    vi = content_scout.calculate_story_twist_index("Cô trở về để báo thù")
    en = content_scout.calculate_story_twist_index("She returns for vengeance")
    zh = content_scout.calculate_story_twist_index("她重生后复仇")
    assert vi == en == 2.4
    assert zh > 1.0
    assert content_scout.calculate_story_twist_index("revenge vengeance 复仇 báo thù") == 2.4
    assert content_scout.calculate_story_twist_index("reborn after betrayal") > 3.0


def test_vietnamese_copy_is_meaningful_and_does_not_leak_source_prose():
    title, summary = content_scout.build_vietnamese_copy(
        "CEO's secret revenge", "The heiress was betrayed", ["Drama"],
        "ceo_romance", "youtube_obscure", 70,
    )
    assert "Báo Thù" in title
    assert "tổng tài" in summary and "thiên kim" in summary
    assert "CEO's secret revenge" not in title + summary

    fallback_title, fallback_summary = content_scout.build_vietnamese_copy(
        "Unknown foreign sentence", "Nothing recognizable", [],
        "horror", "douyin_bilibili", 45,
    )
    assert "Kinh" in fallback_title
    assert "45 phút" in fallback_summary and "Bilibili" in fallback_summary


def test_localize_live_candidates_gives_curated_style_copy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MRF_SCOUT_LOCALIZE", "1")
    prompts: list[str] = []

    def runner(prompt: str) -> dict:
        prompts.append(prompt)
        return {"items": [{"id": "yt-live-a", "vietnamese_title": "Lễ Hội Của Những Linh Hồn (1962)",
                           "vietnamese_summary": "Một cô gái sống sót sau tai nạn bị ám bởi bóng ma."}]}

    candidates = [
        {"id": "yt-live-a", "title": "Carnival of Souls (1962) | Full Movie", "summary": "Mary Henry...",
         "vietnamese_title": "Phim xưa độc lạ: Câu Chuyện Bí Ẩn", "vietnamese_summary": "generic"},
        {"id": "bili-b", "title": "短剧", "summary": "短剧", "vietnamese_title": "x", "vietnamese_summary": "y"},
        {"id": "tmdb-1", "title": "The Vanishing", "vietnamese_title": "Biến Mất Không Dấu Vết (1988)"},
    ]
    content_scout.localize_live_candidates(candidates, runner=runner)

    assert len(prompts) == 1 and "Carnival of Souls" in prompts[0] and "The Vanishing" not in prompts[0]
    assert candidates[0]["vietnamese_title"] == "Lễ Hội Của Những Linh Hồn (1962)"
    assert candidates[0]["localized"] == "agy"
    assert candidates[1]["localized"] == "failed" and candidates[1]["vietnamese_title"] == "x"
    assert "localized" not in candidates[2]

    content_scout.localize_live_candidates(candidates, runner=runner)
    assert len(prompts) == 1


def test_localize_live_candidates_survives_agy_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MRF_SCOUT_LOCALIZE", "1")

    def runner(prompt: str) -> dict:
        raise RuntimeError("pool offline")

    candidates = [{"id": "yt-live-a", "title": "T", "vietnamese_title": "generic", "vietnamese_summary": "s"}]
    content_scout.localize_live_candidates(candidates, runner=runner)
    assert candidates[0]["vietnamese_title"] == "generic" and candidates[0]["localized"] == "failed"


def test_story_twist_index_baseline():
    """Text without any trigger keywords returns baseline 1.0."""
    score = content_scout.calculate_story_twist_index("Một bộ phim tài liệu về thế giới động vật.")
    assert score == 1.0


def test_story_twist_index_with_triggers():
    """Text with revenge and transmigration triggers increases score."""
    summary = "Sau khi bị phản bội và đâm sau lưng, cô gái trùng sinh về quá khứ để trả thù."
    score = content_scout.calculate_story_twist_index(summary)
    assert score > 3.0
    assert score <= 10.0


def test_story_twist_index_capped_at_ten():
    """Twist index cannot exceed 10.0 even with many trigger keywords."""
    summary = (
        "trả thù báo thù phản bội đâm sau lưng oan khuất hãm hại xuyên không trùng sinh "
        "tái sinh đại gia ngầm ẩn giấu thân phận giả nghèo nghịch tập tổng tài thiên kim "
        "quái vật biến dị ký sinh ma quái nguyền rủa tà thuật sát nhân"
    )
    score = content_scout.calculate_story_twist_index(summary)
    assert score == 10.0


def test_copyright_risk_pre_2008_indie():
    """Pre-2008 indie or Asian cinema has minimal risk = 0.1."""
    risk = content_scout.calculate_copyright_risk(1994, studio="Full Moon Entertainment", country="US")
    assert risk == 0.1

    risk_hk = content_scout.calculate_copyright_risk(1985, studio="Golden Harvest", country="HK")
    assert risk_hk == 0.1


def test_copyright_risk_major_studios():
    """Major modern studios carry high copyright risk."""
    risk_modern = content_scout.calculate_copyright_risk(2022, studio="Walt Disney Pictures", country="US")
    assert risk_modern >= 0.9

    risk_pre_2008_major = content_scout.calculate_copyright_risk(2000, studio="Warner Bros", country="US")
    assert risk_pre_2008_major == 0.5


def test_popularity_index_vietnam():
    """Zero reviews in VN yields popularity = 0.1 for maximum viral score spike."""
    pop_zero = content_scout.calculate_popularity_index("Unknown Gem", review_count_vn=0)
    assert pop_zero == 0.1

    pop_one = content_scout.calculate_popularity_index("Some Gem", review_count_vn=1)
    assert pop_one == 0.5

    pop_saturated = content_scout.calculate_popularity_index("Famous Movie", review_count_vn=10, is_major_reviewed=True)
    assert pop_saturated >= 2.0


def test_viral_scoring_formula():
    """Formula Score = (Story_Twist_Index * Rating) / (Popularity_Index * Copyright_Risk)."""
    # (5.0 * 8.0) / (0.1 * 0.1) = 4000.0
    score = content_scout.calculate_viral_score(5.0, 8.0, 0.1, 0.1)
    assert score == 4000.0


def test_query_builders():
    """Query builders produce well-formed queries with exclusion tokens."""
    yt_query = content_scout.build_youtube_obscure_query("horror", 1990, 2005)
    assert '"Full Movie"' in yt_query
    assert "-marvel" in yt_query
    assert "-netflix" in yt_query
    assert "1990..2005" in yt_query

    douyin_query = content_scout.build_douyin_short_drama_query("ceo_romance")
    assert "#短剧" in douyin_query
    assert "#逆袭" in douyin_query

    tmdb_params = content_scout.build_tmdb_discover_params("horror", 1985, 2008, 6.8, 3000)
    assert tmdb_params["with_genres"] == "27"
    assert tmdb_params["vote_average.gte"] == 6.8
    assert tmdb_params["vote_count.lte"] == 3000


def _live_candidates(source, topic, count):
    host = "youtube.com" if source == "youtube_obscure" else "bilibili.com"
    return [
        {
            "id": f"live-{source}-{index}",
            "title": f"Live {source} {index}",
            "source": source,
            "topic": topic,
            "source_url": f"https://{host}/video/{source}-{index}",
            "viral_score": 1000.0 - index,
        }
        for index in range(count)
    ]


def test_discover_hidden_gems_all_uses_exact_source_quotas(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(content_scout, "_CACHE", content_scout.ScoutCache(tmp_path / "scout.json"))

    def fake_youtube(topic, limit):
        calls.append(("youtube", topic, limit))
        return _live_candidates("youtube_obscure", topic, limit + 2)

    def fake_bilibili(topic, limit):
        calls.append(("bilibili", topic, limit))
        return _live_candidates("douyin_bilibili", topic, limit + 2)

    monkeypatch.setattr(content_scout, "search_youtube_live", fake_youtube)
    monkeypatch.setattr(content_scout, "search_bilibili_short_dramas", fake_bilibili)

    gems = content_scout.discover_hidden_gems(limit=10, refresh=True)

    assert calls == [("youtube", "all", 4), ("bilibili", "all", 3)]
    assert [gem["source"] for gem in gems] == ["youtube_obscure"] * 4 + ["douyin_bilibili"] * 3 + ["tmdb_douban"] * 3
    assert len({gem["source_url"] for gem in gems}) == 10


def test_discover_hidden_gems_strict_source_dispatch_and_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(content_scout, "_CACHE", content_scout.ScoutCache(tmp_path / "scout.json"))
    youtube_calls = []
    bilibili_calls = []
    monkeypatch.setattr(content_scout, "search_youtube_live", lambda topic, limit: youtube_calls.append((topic, limit)) or [])
    monkeypatch.setattr(content_scout, "search_bilibili_short_dramas", lambda topic, limit: bilibili_calls.append((topic, limit)) or [])

    tmdb = content_scout.discover_hidden_gems(source="tmdb_douban", limit=3, refresh=True)
    assert youtube_calls == []
    assert bilibili_calls == []
    assert len(tmdb) == 3
    assert {gem["source"] for gem in tmdb} == {"tmdb_douban"}

    bilibili = content_scout.discover_hidden_gems(source="bilibili", limit=3, refresh=True)
    assert youtube_calls == []
    assert bilibili_calls == [("all", 3)]
    assert len(bilibili) == 3
    assert {gem["source"] for gem in bilibili} == {"douyin_bilibili"}

    youtube = content_scout.discover_hidden_gems(source="youtube", limit=3, refresh=True)
    assert youtube_calls == [("all", 3)]
    assert bilibili_calls == [("all", 3)]
    assert len(youtube) == 3
    assert {gem["source"] for gem in youtube} == {"youtube_obscure"}


def test_discover_hidden_gems_live_results_fill_with_matching_seeds(monkeypatch, tmp_path):
    monkeypatch.setattr(content_scout, "_CACHE", content_scout.ScoutCache(tmp_path / "scout.json"))
    monkeypatch.setattr(content_scout, "search_youtube_live", lambda topic, limit: _live_candidates("youtube_obscure", topic, 1))
    monkeypatch.setattr(content_scout, "search_bilibili_short_dramas", lambda topic, limit: [])

    youtube = content_scout.discover_hidden_gems(source="youtube_obscure", limit=3, refresh=True)
    bilibili = content_scout.discover_hidden_gems(source="douyin_bilibili", limit=3, refresh=True)

    assert youtube[0]["id"].startswith("live-youtube_obscure")
    assert len(youtube) == 3
    assert all(gem["source"] == "youtube_obscure" for gem in youtube)
    assert len(bilibili) == 3
    assert all(gem["id"].startswith("gem-drama-") for gem in bilibili)


def test_discover_hidden_gems_cache_hit_reapplies_source_filter(monkeypatch, tmp_path):
    cache = content_scout.ScoutCache(tmp_path / "scout.json")
    cache.set("all", "youtube", _live_candidates("youtube_obscure", "all", 2) + _live_candidates("douyin_bilibili", "all", 2))
    monkeypatch.setattr(content_scout, "_CACHE", cache)
    monkeypatch.setattr(content_scout, "search_youtube_live", lambda topic, limit: pytest.fail("cache hit should not search YouTube"))
    monkeypatch.setattr(content_scout, "search_bilibili_short_dramas", lambda topic, limit: pytest.fail("cache hit should not search Bilibili"))

    gems = content_scout.discover_hidden_gems(source="youtube", limit=10)

    assert len(gems) == 2
    assert {gem["source"] for gem in gems} == {"youtube_obscure"}


@pytest.mark.parametrize(
    ("limit", "expected_sources"),
    [
        (1, ["youtube_obscure"]),
        (2, ["youtube_obscure", "douyin_bilibili"]),
        (3, ["youtube_obscure", "douyin_bilibili", "tmdb_douban"]),
        (4, ["youtube_obscure", "youtube_obscure", "douyin_bilibili", "tmdb_douban"]),
        (5, ["youtube_obscure", "youtube_obscure", "douyin_bilibili", "douyin_bilibili", "tmdb_douban"]),
    ],
)
def test_discover_hidden_gems_small_limits_are_deterministic(monkeypatch, tmp_path, limit, expected_sources):
    monkeypatch.setattr(content_scout, "_CACHE", content_scout.ScoutCache(tmp_path / f"scout-{limit}.json"))
    monkeypatch.setattr(content_scout, "search_youtube_live", lambda topic, limit: _live_candidates("youtube_obscure", topic, limit))
    monkeypatch.setattr(content_scout, "search_bilibili_short_dramas", lambda topic, limit: _live_candidates("douyin_bilibili", topic, limit))

    gems = content_scout.discover_hidden_gems(limit=limit, refresh=True)

    assert [gem["source"] for gem in gems] == expected_sources


def test_discover_hidden_gems_drama_topic_all_keeps_live_source_diversity(monkeypatch, tmp_path):
    monkeypatch.setattr(content_scout, "_CACHE", content_scout.ScoutCache(tmp_path / "scout.json"))
    monkeypatch.setattr(content_scout, "search_youtube_live", lambda topic, limit: _live_candidates("youtube_obscure", topic, limit))
    monkeypatch.setattr(content_scout, "search_bilibili_short_dramas", lambda topic, limit: _live_candidates("douyin_bilibili", topic, limit))

    gems = content_scout.discover_hidden_gems(topic="ceo_romance", source="all", limit=4, refresh=True)

    assert [gem["source"] for gem in gems] == ["youtube_obscure", "youtube_obscure", "douyin_bilibili"]
    assert all(gem["topic"] == "ceo_romance" for gem in gems)


def test_enqueue_gem_for_review():
    """Enqueueing valid candidate returns ready metadata; invalid ID raises error."""
    entry = content_scout.enqueue_gem_for_review("gem-tmdb-01")
    assert entry["status"] == "ready"
    assert "batch_entry" in entry
    assert "candidate" in entry

    with pytest.raises(ValueError, match="Không tìm thấy"):
        content_scout.enqueue_gem_for_review("non-existent-id")


def test_webapp_service_scout_integration(tmp_path):
    """Test JobsService scout_discover and scout_enqueue methods."""
    service = JobsService(jobs_root=tmp_path)
    res = service.scout_discover(topic="all", source="all")
    assert "candidates" in res
    assert res["total"] > 0

    first_id = res["candidates"][0]["id"]
    enqueue_res = service.scout_enqueue({"candidate_id": first_id, "auto_create": True})
    assert enqueue_res["status"] == "ready"
    assert "created_job" in enqueue_res
    assert (tmp_path / enqueue_res["created_job"]["job_id"]).exists()
