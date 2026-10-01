"""Content Scout Engine: Automated discovery and viral scoring for 'Hidden Gem' movies.

Replaces manual search/link-pasting with algorithmic discovery across 3 sources:
1. TMDb / Douban Hidden Gem Discovery (Horror/Fantasy/Mystery 1985-2008, rating >= 6.8, votes < 3000)
2. Douyin / Bilibili Mini Short Drama (#短剧, #逆袭, #穿越, #重生)
3. YouTube Obscure Search ("Full Movie" + topic + 1990..2005, 45-90m, 20k-200k views)

Formula:
    Score = (Story_Twist_Index * Rating) / (Popularity_Index * Copyright_Risk)
"""
from __future__ import annotations

import copy
import html
import json
import logging
import os
import re
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote, urlparse

import httpx

logger = logging.getLogger(__name__)

# Standard desktop browser User-Agent & headers to avoid Bilibili HTTP 412 WAF blocking
BILIBILI_PROBE_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8,zh-CN;q=0.7",
}

# Markers indicating soft-deleted or unavailable videos returned with HTTP 200 on Bilibili
BILIBILI_UNAVAILABLE_MARKERS: tuple[str, ...] = (
    "视频去nah了呢",
    "视频去哪了呢",
    "稿件不可见",
    "视频不存在",
    "已失效",
)

# Viral trigger keywords indicating dramatic twists and retention hooks
TWIST_TRIGGER_KEYWORDS: dict[str, float] = {
    # Revenge / betrayal
    "trả thù": 1.4,
    "báo thù": 1.4,
    "phản bội": 1.3,
    "đâm sau lưng": 1.2,
    "oan khuất": 1.1,
    "hãm hại": 1.2,
    "revenge": 1.4,
    "betrayal": 1.3,
    "vengeance": 1.4,
    # Transmigration / rebirth / identity
    "xuyên không": 1.5,
    "trùng sinh": 1.5,
    "tái sinh": 1.3,
    "đại gia ngầm": 1.6,
    "ẩn giấu thân phận": 1.5,
    "giả nghèo": 1.4,
    "nghịch tập": 1.5,
    "tổng tài": 1.3,
    "thiên kim": 1.2,
    "chiến thần": 1.3,
    "rebirth": 1.5,
    "transmigration": 1.5,
    "billionaire": 1.4,
    # Monster / horror / occult
    "quái vật": 1.3,
    "biến dị": 1.4,
    "ký sinh": 1.4,
    "ma quái": 1.2,
    "nguyền rủa": 1.2,
    "tà thuật": 1.3,
    "thí nghiệm": 1.3,
    "cương thi": 1.3,
    "monster": 1.3,
    "mutation": 1.4,
    "parasite": 1.4,
    "curse": 1.2,
    "experiment": 1.3,
    # Suspense / mystery
    "sát nhân": 1.2,
    "mất tích": 1.1,
    "phòng kín": 1.2,
    "mê cung": 1.1,
    "âm mưu": 1.2,
    "serial killer": 1.2,
    "mystery": 1.1,
    "conspiracy": 1.2,
}

MAJOR_STUDIOS: set[str] = {
    "disney", "marvel", "warner", "netflix", "universal", "sony pictures",
    "paramount", "columbia pictures", "20th century", "mgm",
}

# Offline Vietnamese genre labels (lowercase key) for every seed and live-discovery genre.
_GENRE_VI: dict[str, str] = {
    "horror": "Kinh Dị",
    "mystery": "Bí Ẩn",
    "thriller": "Ly Kỳ",
    "fantasy": "Huyền Ảo",
    "comedy": "Hài Hước",
    "sci-fi": "Khoa Học Viễn Tưởng",
    "action": "Hành Động",
    "romance": "Lãng Mạn",
    "drama": "Chính Kịch",
    "short drama": "Đoản Kịch",
    "mini drama": "Đoản Kịch",
    "revenge": "Báo Thù",
    "retro cinema": "Điện Ảnh Cổ Điển",
    "cult classic": "Phim B-Movie Độc Lạ",
    "fantasy mystery": "Huyền Ảo Bí Ẩn",
    "ceo romance": "Tổng Tài Lãng Mạn",
    "isekai rebirth": "Xuyên Không Trùng Sinh",
    "action thriller": "Hành Động Ly Kỳ",
    "all": "Tổng Hợp",
}


def _translate_genres(genres: list[str]) -> list[str]:
    """Map English genre labels to Vietnamese using the static lookup table."""
    return [_GENRE_VI.get(g.lower(), g) for g in genres]


# Score per concept, not per alias, so repeated words or translations cannot inflate the result.
CONCEPT_GLOSSARY: dict[str, dict[str, Any]] = {
    "revenge": {"vi": "báo thù", "weight": 1.4, "aliases": ("báo thù", "trả thù", "revenge", "vengeance", "复仇", "報仇")},
    "betrayal": {"vi": "phản bội", "weight": 1.3, "aliases": ("phản bội", "đâm sau lưng", "betrayal", "betrayed", "背叛")},
    "rebirth": {"vi": "trùng sinh", "weight": 1.5, "aliases": ("trùng sinh", "tái sinh", "rebirth", "reborn", "重生")},
    "transmigration": {"vi": "xuyên không", "weight": 1.5, "aliases": ("xuyên không", "transmigration", "isekai", "穿越", "穿书")},
    "counterattack": {"vi": "nghịch tập", "weight": 1.5, "aliases": ("nghịch tập", "lật ngược thế cờ", "counterattack", "comeback", "逆袭", "打脸")},
    "hidden_identity": {"vi": "ẩn giấu thân phận", "weight": 1.5, "aliases": ("ẩn giấu thân phận", "thân phận bí mật", "hidden identity", "secret identity", "隐藏身份", "隐瞒身份")},
    "fake_poor": {"vi": "giả nghèo", "weight": 1.4, "aliases": ("giả nghèo", "pretends to be poor", "fake poor", "装穷")},
    "ceo": {"vi": "tổng tài", "weight": 1.3, "aliases": ("tổng tài", "ceo", "chief executive", "总裁", "霸总")},
    "heiress": {"vi": "thiên kim", "weight": 1.2, "aliases": ("thiên kim", "heiress", "rich daughter", "千金")},
    "war_god": {"vi": "chiến thần", "weight": 1.3, "aliases": ("chiến thần", "war god", "god of war", "战神")},
    "monster": {"vi": "quái vật", "weight": 1.3, "aliases": ("quái vật", "monster", "creature", "怪物")},
    "mutation": {"vi": "biến dị", "weight": 1.4, "aliases": ("biến dị", "đột biến", "mutation", "mutant", "变异")},
    "parasite": {"vi": "ký sinh", "weight": 1.4, "aliases": ("ký sinh", "parasite", "parasitic", "寄生")},
    "curse": {"vi": "lời nguyền", "weight": 1.2, "aliases": ("lời nguyền", "nguyền rủa", "curse", "cursed", "诅咒")},
    "occult": {"vi": "tà thuật", "weight": 1.3, "aliases": ("tà thuật", "ma quái", "occult", "black magic", "邪术", "灵异")},
    "experiment": {"vi": "thí nghiệm", "weight": 1.3, "aliases": ("thí nghiệm", "experiment", "experimental", "实验")},
    "serial_killer": {"vi": "sát nhân hàng loạt", "weight": 1.2, "aliases": ("sát nhân hàng loạt", "sát nhân", "serial killer", "连环杀手")},
    "missing": {"vi": "mất tích", "weight": 1.1, "aliases": ("mất tích", "missing person", "disappearance", "失踪")},
    "locked_room": {"vi": "phòng kín", "weight": 1.2, "aliases": ("phòng kín", "căn phòng khóa kín", "locked room", "密室")},
    "maze": {"vi": "mê cung", "weight": 1.1, "aliases": ("mê cung", "maze", "labyrinth", "迷宫")},
    "conspiracy": {"vi": "âm mưu", "weight": 1.2, "aliases": ("âm mưu", "conspiracy", "plot against", "阴谋")},
}


def normalize_scout_text(value: Any) -> str:
    """Normalize scout metadata without stripping Vietnamese or Han characters."""
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]*>", " ", text)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(ch)[0] in {"P", "S", "C"} else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def _matched_story_concepts(*values: Any) -> list[str]:
    text = normalize_scout_text(" ".join(str(value or "") for value in values))
    padded = f" {text} "
    matched: list[str] = []
    for concept, details in CONCEPT_GLOSSARY.items():
        for raw_alias in details["aliases"]:
            alias = normalize_scout_text(raw_alias)
            has_han = any("CJK" in unicodedata.name(ch, "") for ch in alias)
            if alias and ((alias in text) if has_han else (f" {alias} " in padded)):
                matched.append(concept)
                break
    return matched


def build_vietnamese_copy(
    title: Any = "",
    summary: Any = "",
    genres: Any = None,
    topic: str = "all",
    source: str = "",
    duration_minutes: int = 0,
) -> tuple[str, str]:
    """Build deterministic Vietnamese card copy from concepts, never raw prose."""
    genre_values = genres if isinstance(genres, (list, tuple)) else [genres or ""]
    concepts = _matched_story_concepts(title, summary, *genre_values)
    labels = [str(CONCEPT_GLOSSARY[key]["vi"]) for key in concepts]
    topic_label = _TOPIC_VI_LABELS.get(topic, _GENRE_VI.get(topic.replace("_", " ").lower(), "Phim độc lạ"))
    source_label = "Bilibili" if "bilibili" in source else ("YouTube" if "youtube" in source else "nguồn tuyển chọn")
    duration_label = f"{int(duration_minutes)} phút" if duration_minutes else "thời lượng vừa phải"

    if labels:
        hook = " và ".join(labels[:2]).title()
        vi_title = f"{hook}: {topic_label}"
        vi_summary = f"Câu chuyện khai thác mô-típ {', '.join(labels[:4])}, theo nhịp {topic_label.lower()} trong {duration_label}; tuyển chọn từ {source_label}."
    else:
        vi_title = f"{topic_label}: Câu Chuyện Bí Ẩn"
        vi_summary = f"Một câu chuyện {topic_label.lower()} có cao trào và nút thắt, thời lượng {duration_label}; tuyển chọn từ {source_label}."
    return vi_title, vi_summary


@dataclass
class CandidateMovie:
    """Represents a discovered hidden-gem movie candidate."""
    id: str
    title: str
    vietnamese_title: str
    source: str  # "tmdb_douban" | "douyin_bilibili" | "youtube_obscure"
    topic: str   # "horror" | "fantasy_mystery" | "ceo_romance" | "isekai_rebirth" | "cult_classic"
    release_year: int
    genres: list[str]
    rating: float
    vote_count: int
    summary: str
    vietnamese_summary: str
    source_url: str
    duration_minutes: int
    views: int
    country: str
    studio: str
    story_twist_index: float = 1.0
    popularity_index: float = 0.1
    copyright_risk: float = 0.1
    viral_score: float = 0.0
    reasoning: str = ""
    is_live: bool = True
    fallback_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_story_twist_index(summary: str, keywords: Optional[list[str]] = None) -> float:
    """Evaluate story motifs once per multilingual narrative concept."""
    score = 1.0 + sum(float(CONCEPT_GLOSSARY[key]["weight"]) for key in _matched_story_concepts(summary))

    # Keep the public extension point, while deduplicating normalized custom terms.
    text = f" {normalize_scout_text(summary)} "
    seen_extra: set[str] = set()
    for keyword in keywords or []:
        normalized = normalize_scout_text(keyword)
        if normalized and normalized not in seen_extra and f" {normalized} " in text:
            seen_extra.add(normalized)
            score += 1.0

    return round(min(10.0, max(1.0, score)), 2)


def calculate_copyright_risk(
    release_year: int,
    studio: str = "",
    country: str = "",
) -> float:
    """Calculate copyright strike risk.

    Pre-2008 indie, Hong Kong, Taiwan, or cult B-movies are extremely safe (Risk = 0.1).
    Modern major-studio releases carry high risk (0.8 - 1.0).
    """
    clean_studio = (studio or "").lower()
    is_major = any(major in clean_studio for major in MAJOR_STUDIOS)

    if release_year <= 2008:
        if is_major:
            return 0.5
        # Independent, HK, Taiwan, European, or cult cinema
        return 0.1

    # Post-2008
    if is_major:
        return 0.95
    if country.upper() in ("CN", "HK", "TW", "VN"):
        # Mini short dramas or Asian web series
        return 0.25
    return 0.4


def calculate_popularity_index(
    title: str,
    review_count_vn: int = 0,
    is_major_reviewed: bool = False,
) -> float:
    """Calculate popularity index in Vietnam market.

    If unreviewed by major Vietnamese channels => Popularity = 0.1 (Score spikes!).
    If already saturated with reviews => Popularity = 1.0 - 3.0.
    """
    if is_major_reviewed or review_count_vn >= 5:
        return 2.5
    if review_count_vn >= 2:
        return 1.2
    if review_count_vn == 1:
        return 0.5
    # Unreviewed hidden gem in Vietnam
    return 0.1


def calculate_viral_score(
    story_twist_index: float,
    rating: float,
    popularity_index: float,
    copyright_risk: float,
) -> float:
    """Calculate potential viral score using the formula:

        Score = (Story_Twist_Index * Rating) / (Popularity_Index * Copyright_Risk)
    """
    twist = min(10.0, max(1.0, float(story_twist_index)))
    rate = min(10.0, max(1.0, float(rating)))
    pop = max(0.05, float(popularity_index))
    risk = max(0.05, float(copyright_risk))

    raw_score = (twist * rate) / (pop * risk)
    return round(raw_score, 2)


def build_youtube_obscure_query(
    topic: str,
    year_start: int = 1990,
    year_end: int = 2005,
) -> str:
    """Construct search query for obscure full movies on YouTube."""
    topic_map = {
        "horror": "Kinh dị quái vật",
        "fantasy_mystery": "Huyền ảo bí ẩn",
        "cult_classic": "Phim xưa độc lạ",
        "action_thriller": "Hành động ly kỳ",
    }
    term = topic_map.get(topic, topic)
    return f'"Full Movie" "{term}" {year_start}..{year_end} -marvel -netflix'


def build_douyin_short_drama_query(subtopic: str) -> str:
    """Construct Douyin/Bilibili short drama search query."""
    subtopic_map = {
        "ceo_romance": "#短剧 #逆袭 #总监 #总裁",
        "isekai_rebirth": "#短剧 #穿越 #重生 #女帝",
        "revenge": "#短剧 #复仇 #打脸 #战神",
    }
    return subtopic_map.get(subtopic, f"#短剧 #{subtopic}")


YOUTUBE_TOPIC_QUERIES: dict[str, str] = {
    "horror": "retro horror full movie",
    "cult_classic": "cult classic full movie",
    "fantasy_mystery": "fantasy mystery full movie 1980s 1990s",
    "action_thriller": "retro action thriller full movie",
    "ceo_romance": "short drama full movie",
    "isekai_rebirth": "rebirth drama full movie",
    "revenge": "revenge drama full movie",
    "all": "cult classic retro full movie",
}

# Vietnamese topic labels, prefixed to live results that cannot be translated offline.
_TOPIC_VI_LABELS: dict[str, str] = {
    "horror": "Kinh dị",
    "fantasy_mystery": "Huyền ảo / Bí ẩn",
    "cult_classic": "Phim xưa độc lạ",
    "action_thriller": "Hành động ly kỳ",
    "ceo_romance": "Tổng tài / Nghịch tập",
    "isekai_rebirth": "Xuyên không / Trùng sinh",
    "revenge": "Báo thù",
    "all": "Phim độc lạ",
}


def build_youtube_live_query(topic: str = "all") -> str:
    """Build optimal high-yield keyword query for YouTube full-length movie discovery."""
    return YOUTUBE_TOPIC_QUERIES.get(topic, f"{topic} full movie")


def build_tmdb_discover_params(
    genre: str = "horror",
    min_year: int = 1985,
    max_year: int = 2008,
    min_rating: float = 6.8,
    max_votes: int = 3000,
) -> dict[str, Any]:
    """Build TMDb discover API filter parameters for hidden gems."""
    genre_ids = {
        "horror": "27",
        "fantasy": "14",
        "mystery": "9648",
        "thriller": "53",
    }
    return {
        "with_genres": genre_ids.get(genre, "27"),
        "primary_release_date.gte": f"{min_year}-01-01",
        "primary_release_date.lte": f"{max_year}-12-31",
        "vote_average.gte": min_rating,
        "vote_count.lte": max_votes,
        "vote_count.gte": 50,  # Ensure at least minimal verified rating
        "sort_by": "vote_average.desc",
    }


# Curated high-potential seed catalog (100% offline-ready & verified playable)
_SEEDS: list[dict[str, Any]] = [
    # Nguồn 1: TMDb / Douban Hidden Gem Discovery
    {
        "id": "gem-tmdb-01",
        "title": "In the Mouth of Madness",
        "vietnamese_title": "Cơn Điên Loạn Tột Cùng (1994)",
        "source": "tmdb_douban",
        "topic": "horror",
        "release_year": 1994,
        "genres": ["Horror", "Mystery"],
        "rating": 7.1,
        "vote_count": 1420,
        "summary": "Một điều tra viên bảo hiểm tìm kiếm tiểu thuyết gia kinh dị mất tích, phát hiện cuốn sách có tà thuật nguyền rủa khiến độc giả biến dị thành quái vật và thế giới bị nuốt chửng bởi ảo giác phản bội và ma quái.",
        "source_url": "https://www.youtube.com/watch?v=LKVe16ppDhs",
        "duration_minutes": 95,
        "views": 400156,
        "country": "US",
        "studio": "New Line Cinema (Indie Era)",
        "popularity_vn_reviews": 0,
    },
    {
        "id": "gem-tmdb-02",
        "title": "The Vanishing (Spoorloos)",
        "vietnamese_title": "Biến Mất Không Dấu Vết (1988)",
        "source": "tmdb_douban",
        "topic": "cult_classic",
        "release_year": 1988,
        "genres": ["Mystery", "Thriller"],
        "rating": 7.7,
        "vote_count": 1850,
        "summary": "Người bạn gái đột ngột mất tích tại trạm dừng chân. Ba năm sau kẻ sát nhân tâm thần liên lạc, đề nghị anh trải nghiệm chính xác số phận oan khuất và sự thật kinh hoàng trong căn phòng kín chôn sống.",
        "source_url": "https://www.youtube.com/watch?v=yRIjcSDh0Tc",
        "duration_minutes": 106,
        "views": 39727,
        "country": "NL",
        "studio": "Golden Harvest Europe",
        "popularity_vn_reviews": 0,
    },
    {
        "id": "gem-tmdb-03",
        "title": "Mr. Vampire",
        "vietnamese_title": "Cương Thi Tiên Sinh (1985)",
        "source": "tmdb_douban",
        "topic": "fantasy_mystery",
        "release_year": 1985,
        "genres": ["Fantasy", "Comedy", "Horror"],
        "rating": 7.4,
        "vote_count": 2100,
        "summary": "Đạo sĩ Mao Sơn làm lễ cải táng cho một gia tộc đại gia ngầm, phát hiện thi thể không phân hủy biến dị thành cương thi khát máu, bắt đầu cuộc chiến bùa chú tà thuật trừ ma quái đầy kịch tính.",
        "source_url": "https://www.youtube.com/watch?v=6injkGWQy-g",
        "duration_minutes": 90,
        "views": 1287846,
        "country": "HK",
        "studio": "Bo Ho Films / Golden Harvest",
        "popularity_vn_reviews": 1,
    },
    {
        "id": "gem-tmdb-04",
        "title": "Demons (Demoni)",
        "vietnamese_title": "Ác Quỷ Rạp Chiếu Phim (1985)",
        "source": "tmdb_douban",
        "topic": "horror",
        "release_year": 1985,
        "genres": ["Horror", "Mystery", "Thriller"],
        "rating": 7.0,
        "vote_count": 1800,
        "summary": "Một nhóm khán giả tại rạp chiếu phim bị phong tỏa khi chiếc mặt nạ ma quái biến người đeo thành quái vật nhiễm trùng khát máu, bắt đầu cuộc chiến tàn sát sinh tồn đầy kinh hoàng.",
        "source_url": "https://www.youtube.com/watch?v=9CaCk2OizKQ",
        "duration_minutes": 84,
        "views": 37308,
        "country": "IT",
        "studio": "DACfilm / Dario Argento (Indie Classic)",
        "popularity_vn_reviews": 0,
    },
    # Nguồn 2: Douyin / Bilibili Mini Short Drama
    {
        "id": "gem-drama-01",
        "title": "Undercover CEO: The Disguise Unveiled",
        "vietnamese_title": "Tổng Tài Ẩn Danh: 3 Năm Ẩn Mình Đại Nghịch Tập (2024)",
        "source": "douyin_bilibili",
        "topic": "ceo_romance",
        "release_year": 2024,
        "genres": ["Short Drama", "Romance", "Action"],
        "rating": 7.6,
        "vote_count": 12500,
        "summary": "Chủ tịch tập đoàn nghìn tỷ ẩn giấu thân phận làm nhân viên bình thường suốt 3 năm, tại tiệc thường niên bị sếp nhỏ và vợ phản bội sa thải. Đúng lúc đó, thân phận thật sự bại lộ tạo cú lật kèo nghịch tập chấn động.",
        "source_url": "https://www.bilibili.com/video/BV1vtaV6VEgo",
        "duration_minutes": 51,
        "views": 850000,
        "country": "CN",
        "studio": "Bilibili Short Theater",
        "popularity_vn_reviews": 0,
    },
    {
        "id": "gem-drama-02",
        "title": "Rebirth of the Real Heiress",
        "vietnamese_title": "Trùng Sinh Báo Thù: Thiên Kim Thật Trở Về (2024)",
        "source": "douyin_bilibili",
        "topic": "isekai_rebirth",
        "release_year": 2024,
        "genres": ["Short Drama", "Revenge", "Drama"],
        "rating": 7.8,
        "vote_count": 18200,
        "summary": "Thiên kim thật bị gia đình thiên vị hãm hại oan khuất, cô trùng sinh trở về vạch trần tâm kế của giả thiên kim, nghịch tập xuất sắc bắt kẻ phản bội đền tội từng người một.",
        "source_url": "https://www.bilibili.com/video/BV1rUtE68EPY",
        "duration_minutes": 59,
        "views": 1200000,
        "country": "CN",
        "studio": "Bilibili Short Theater",
        "popularity_vn_reviews": 0,
    },
    {
        "id": "gem-drama-03",
        "title": "Empress Transmigration: The Bag of Miracles",
        "vietnamese_title": "Nữ Đế Xuyên Không: Cùng Ta Đánh Thiên Hạ (2024)",
        "source": "douyin_bilibili",
        "topic": "isekai_rebirth",
        "release_year": 2024,
        "genres": ["Short Drama", "Fantasy", "Romance"],
        "rating": 7.3,
        "vote_count": 9400,
        "summary": "Chàng trai thời hiện đại vô tình kết nối xuyên không với Nữ đế cổ đại, dùng bách bảo túi và tri thức hiện đại hỗ trợ Nữ đế vượt qua phản bội sát phạt, tạo nên kỳ tích nghịch tập thâu tóm thiên hạ.",
        "source_url": "https://www.bilibili.com/video/BV1oTtc6eEJA",
        "duration_minutes": 53,
        "views": 396608,
        "country": "CN",
        "studio": "Bilibili Short Theater",
        "popularity_vn_reviews": 0,
    },
    # Nguồn 3: YouTube Obscure Search
    {
        "id": "gem-yt-01",
        "title": "Subspecies",
        "vietnamese_title": "Huyết Tộc Ma Cà Rồng (1991)",
        "source": "youtube_obscure",
        "topic": "horror",
        "release_year": 1991,
        "genres": ["Horror", "Fantasy"],
        "rating": 6.8,
        "vote_count": 3100,
        "summary": "Ba nữ sinh viên nghiên cứu văn hóa dân gian tại Romania bị cuốn vào cuộc chiến đẫm máu giữa hai anh em ma cà rồng cổ xưa trong lâu đài Transylvania với bùa chú tà thuật và sự phản bội đẫm máu.",
        "source_url": "https://www.youtube.com/watch?v=XboFipFkiDw",
        "duration_minutes": 83,
        "views": 68000,
        "country": "US",
        "studio": "Full Moon Entertainment (Indie Classic)",
        "popularity_vn_reviews": 0,
    },
    {
        "id": "gem-yt-02",
        "title": "Bad Taste",
        "vietnamese_title": "Thực Khách Ngoài Hành Tinh (1987)",
        "source": "youtube_obscure",
        "topic": "cult_classic",
        "release_year": 1987,
        "genres": ["Comedy", "Horror", "Sci-Fi"],
        "rating": 7.1,
        "vote_count": 3200,
        "summary": "Một tiểu đội đặc nhiệm nghiệp dư được phái đi điều tra một thị trấn ven biển bị người ngoài hành tinh quái vật biến dị đổ bộ thảm sát nhằm biến nhân loại thành nguyên liệu thực phẩm, bắt đầu cuộc chiến phản bội và sinh tồn rùng rợn.",
        "source_url": "https://www.youtube.com/watch?v=GrmC-1tGgw4",
        "duration_minutes": 91,
        "views": 94000,
        "country": "NZ",
        "studio": "WingNut Films / Peter Jackson (Indie Cult Classic)",
        "popularity_vn_reviews": 0,
    },
    {
        "id": "gem-yt-03",
        "title": "Castle Freak",
        "vietnamese_title": "Quái Vật Lâu Đài Cổ (1995)",
        "source": "youtube_obscure",
        "topic": "horror",
        "release_year": 1995,
        "genres": ["Horror", "Thriller"],
        "rating": 6.8,
        "vote_count": 2950,
        "summary": "Một gia đình Mỹ thừa kế tòa lâu đài cổ ở Ý mà không biết dưới tầng hầm giam giữ một quái vật biến dị hung bạo là kết quả của sự phản bội và thí nghiệm tàn nhẫn.",
        "source_url": "https://www.youtube.com/watch?v=zkqXzvysUs0",
        "duration_minutes": 93,
        "views": 22223,
        "country": "US",
        "studio": "Full Moon Entertainment (B-Movie Indie)",
        "popularity_vn_reviews": 0,
    },
]


def build_fallback_search_url(title: str) -> str:
    """Generate YouTube search fallback query URL for a candidate movie."""
    clean_title = (title or "").strip().replace(" ", "+")
    return f"https://www.youtube.com/results?search_query={clean_title}+full+movie"


DISALLOWED_URL_CHARS_REGEX = re.compile(r"[\s;|`$<>'\"\\{}()\[\]^~]")
YOUTUBE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{11,32}$")
BILIBILI_BV_REGEX = re.compile(r"^BV[a-zA-Z0-9]{10}$", re.IGNORECASE)
QUERY_KEY_REGEX = re.compile(r"^[a-zA-Z0-9_]+$")
QUERY_VAL_REGEX = re.compile(r"^[a-zA-Z0-9_.-]*$")


def _parse_video_target(raw_url: str) -> tuple[str, str] | None:
    """Pre-validate and normalize candidate video URL.

    Returns (platform, normalized_url) or None if invalid or unsupported.
    Supported platforms: 'youtube', 'bilibili'.
    """
    if not raw_url or not isinstance(raw_url, str):
        return None
    url = raw_url.strip()
    if not url:
        return None

    # Disallow unencoded whitespace and shell/injection metacharacters
    if DISALLOWED_URL_CHARS_REGEX.search(url):
        return None

    try:
        parsed = urlparse(url)
    except Exception:
        return None

    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        return None

    netloc = (parsed.netloc or "").lower()
    if ":" in netloc:
        netloc = netloc.split(":")[0]

    # Validate query string parameters against injection
    if parsed.query:
        qs = parse_qs(parsed.query, keep_blank_values=True)
        for qk, qvs in qs.items():
            if not QUERY_KEY_REGEX.match(qk):
                return None
            for qv in qvs:
                if not QUERY_VAL_REGEX.match(qv):
                    return None
    else:
        qs = {}

    # YouTube: youtube.com, www.youtube.com, m.youtube.com, youtu.be, www.youtu.be
    if netloc in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            v_list = qs.get("v")
            if not v_list or not v_list[0] or not YOUTUBE_ID_REGEX.match(v_list[0]):
                return None
            return ("youtube", f"https://www.youtube.com/watch?v={v_list[0]}")
        elif parsed.path.startswith("/shorts/"):
            parts = [p for p in parsed.path.split("/shorts/")[1].split("/") if p]
            if not parts or not YOUTUBE_ID_REGEX.match(parts[0]):
                return None
            return ("youtube", f"https://www.youtube.com/watch?v={parts[0]}")
        elif parsed.path.startswith("/embed/"):
            parts = [p for p in parsed.path.split("/embed/")[1].split("/") if p]
            if not parts or not YOUTUBE_ID_REGEX.match(parts[0]):
                return None
            return ("youtube", f"https://www.youtube.com/watch?v={parts[0]}")
        return None

    elif netloc in ("youtu.be", "www.youtu.be"):
        parts = [p for p in parsed.path.split("/") if p]
        if not parts or not YOUTUBE_ID_REGEX.match(parts[0]):
            return None
        return ("youtube", f"https://www.youtube.com/watch?v={parts[0]}")

    # Bilibili: bilibili.com, www.bilibili.com, m.bilibili.com
    elif netloc in ("www.bilibili.com", "bilibili.com", "m.bilibili.com"):
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) == 2 and parts[0].lower() == "video" and BILIBILI_BV_REGEX.match(parts[1]):
            return ("bilibili", f"https://www.bilibili.com/video/{parts[1]}")
        elif len(parts) == 1 and BILIBILI_BV_REGEX.match(parts[0]):
            return ("bilibili", f"https://www.bilibili.com/video/{parts[0]}")
        return None

    return None


def _check_tier2_ytdlp(url: str, timeout: float) -> bool:
    """Fallback probe using yt-dlp extract_flat metadata simulation."""
    try:
        import yt_dlp
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
            "socket_timeout": max(1, int(timeout)),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return bool(info and info.get("title"))
    except Exception:
        return False


def check_link_liveness(url: str, timeout: float = 3.0) -> bool:
    """Verifies if YouTube or Bilibili URL is active and playable.

    Employs a two-tier verification strategy:
    - Tier 1: Fast HTTP probe (YouTube oEmbed endpoint, Bilibili web page with browser User-Agent).
    - Tier 2: Fallback probe (yt-dlp extract_flat) for ambiguous rate-limit / transient server errors.

    Returns:
        True if video is confirmed active and playable.
        False if deleted, private, broken, malformed, or timed out.
    """
    if timeout is None:
        return False
    try:
        budget = float(timeout)
        if budget <= 0.0:
            return False
    except (TypeError, ValueError):
        return False

    target = _parse_video_target(url)
    if target is None:
        return False

    platform, normalized_url = target
    start_time = time.time()
    budget = max(0.1, budget)

    # Tier 1: Fast HTTP Probe
    try:
        with httpx.Client(timeout=budget, follow_redirects=True) as client:
            if platform == "youtube":
                oembed_url = f"https://www.youtube.com/oembed?url={quote(normalized_url, safe='')}&format=json"
                resp = client.get(oembed_url)
                if resp.status_code == 200:
                    return True
                elif resp.status_code in (400, 401, 403, 404):
                    return False
            elif platform == "bilibili":
                resp = client.get(normalized_url, headers=BILIBILI_PROBE_HEADERS)
                if resp.status_code == 200:
                    resp_text = getattr(resp, "text", "") or ""
                    if any(marker in resp_text for marker in BILIBILI_UNAVAILABLE_MARKERS):
                        return False
                    return True
                elif resp.status_code in (400, 401, 403, 404):
                    return False
    except (TimeoutError, httpx.TimeoutException):
        return False
    except Exception:
        pass

    # Tier 2: Opportunistic yt-dlp Fallback Probe
    elapsed = time.time() - start_time
    remaining = budget - elapsed
    if remaining > 0.5:
        return _check_tier2_ytdlp(normalized_url, timeout=remaining)

    return False


DEFAULT_CACHE_PATH = Path(".cache/mrf/scout_cache.json")
DEFAULT_CACHE_TTL = 21600  # 6 hours in seconds
SCOUT_CACHE_VERSION = 2


class ScoutCache:
    """Thread-safe, disk-backed cache for movie scout discovery results with atomic writes and corrupt recovery."""

    def __init__(
        self,
        cache_path: Path | str | None = None,
        ttl_seconds: int | float = DEFAULT_CACHE_TTL,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path is not None else DEFAULT_CACHE_PATH
        try:
            self.ttl_seconds = float(ttl_seconds)
        except (TypeError, ValueError):
            self.ttl_seconds = float(DEFAULT_CACHE_TTL)
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._last_mtime: float = -1.0
        self._load()

    def _normalize_key(self, topic: str | None, source: str | None) -> str:
        t = str(topic if topic is not None else "all").strip().lower()
        s = str(source if source is not None else "all").strip().lower()
        return f"{t}:{s}"

    def _load(self) -> None:
        """Load cache entries from disk file, recovering gracefully if file is corrupt or empty."""
        if not self.cache_path.exists():
            self._data = {}
            self._last_mtime = -1.0
            return

        try:
            st = self.cache_path.stat()
            if st.st_mtime == self._last_mtime:
                return

            if st.st_size == 0:
                self._data = {}
                self._last_mtime = st.st_mtime
                return

            with open(self.cache_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    self._data = {}
                    self._last_mtime = st.st_mtime
                    return
                loaded = json.loads(content)

            clean_data: dict[str, dict[str, Any]] = {}
            if isinstance(loaded, dict):
                for k, v in loaded.items():
                    if isinstance(v, dict):
                        raw_ts = v.get("timestamp")
                        raw_cands = v.get("candidates")
                        if (
                            isinstance(raw_ts, (int, float))
                            and not isinstance(raw_ts, bool)
                            and isinstance(raw_cands, list)
                        ):
                            if v.get("version") == SCOUT_CACHE_VERSION:
                                clean_data[str(k)] = {
                                    "version": SCOUT_CACHE_VERSION,
                                    "timestamp": float(raw_ts),
                                    "candidates": raw_cands,
                                }
            self._data = clean_data
            self._last_mtime = st.st_mtime
        except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError):
            self._data = {}
            try:
                self._last_mtime = self.cache_path.stat().st_mtime
            except OSError:
                self._last_mtime = -1.0

    def _save(self) -> None:
        """Persist cache entries to disk using an atomic write pattern (.tmp file + os.replace)."""
        tmp_path = None
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            unique_suffix = f"{os.getpid()}_{threading.get_ident()}_{time.time_ns()}"
            tmp_path = self.cache_path.parent / f"{self.cache_path.name}.{unique_suffix}.tmp"

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Retry loop for Windows transient file locks / concurrent replacement
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    os.replace(tmp_path, self.cache_path)
                    tmp_path = None
                    break
                except (PermissionError, OSError):
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.02 * (attempt + 1))

            try:
                self._last_mtime = self.cache_path.stat().st_mtime
            except OSError:
                self._last_mtime = -1.0
        except Exception as err:
            logger.error("Failed to write ScoutCache to %s: %s", self.cache_path, err)
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def get(self, topic: str | None, source: str | None) -> list[dict[str, Any]] | None:
        """Retrieve cached candidates for given topic and source if not expired."""
        key = self._normalize_key(topic, source)
        with self._lock:
            self._load()
            entry = self._data.get(key)
            if not isinstance(entry, dict):
                return None

            timestamp = entry.get("timestamp")
            candidates = entry.get("candidates")

            if (
                not isinstance(timestamp, (int, float))
                or isinstance(timestamp, bool)
                or not isinstance(candidates, list)
            ):
                return None

            if time.time() - float(timestamp) > self.ttl_seconds:
                self._data.pop(key, None)
                return None

            return copy.deepcopy(candidates)

    def set(self, topic: str | None, source: str | None, candidates: list[dict[str, Any]]) -> None:
        """Store candidates for given topic and source with current timestamp and persist to disk."""
        key = self._normalize_key(topic, source)
        with self._lock:
            self._load()

            # Clean expired or corrupted entries while setting
            now = time.time()
            expired_keys: list[str] = []
            for k, v in list(self._data.items()):
                if not isinstance(v, dict):
                    expired_keys.append(k)
                    continue
                raw_ts = v.get("timestamp")
                if raw_ts is None:
                    expired_keys.append(k)
                    continue
                try:
                    ts = float(raw_ts)
                    if (now - ts) > self.ttl_seconds:
                        expired_keys.append(k)
                except (TypeError, ValueError):
                    expired_keys.append(k)

            for k in expired_keys:
                self._data.pop(k, None)

            self._data[key] = {
                "version": SCOUT_CACHE_VERSION,
                "timestamp": now,
                "candidates": copy.deepcopy(candidates),
            }
            self._save()

    def clear(self) -> None:
        """Clear all cache entries in memory and remove disk artifacts."""
        with self._lock:
            self._data.clear()
            self._last_mtime = -1.0
            if self.cache_path.exists():
                try:
                    self.cache_path.unlink()
                except OSError:
                    pass

            try:
                parent = self.cache_path.parent
                if parent.exists():
                    for f in parent.glob(f"{self.cache_path.name}.*tmp"):
                        try:
                            f.unlink()
                        except OSError:
                            pass
            except OSError:
                pass


# Module-level default singleton cache instance
_CACHE: ScoutCache = ScoutCache()


def score_movie_candidate(seed: dict[str, Any]) -> CandidateMovie:
    """Evaluate and enrich candidate movie with viral metrics."""
    scoring_text = " ".join(
        str(value or "")
        for value in (
            seed.get("title"),
            seed.get("summary"),
            " ".join(str(g) for g in (seed.get("genres") or [])),
            seed.get("vietnamese_title"),
            seed.get("vietnamese_summary"),
        )
    )
    twist_idx = calculate_story_twist_index(scoring_text)
    pop_idx = calculate_popularity_index(
        title=seed.get("title", ""),
        review_count_vn=seed.get("popularity_vn_reviews", 0),
    )
    copy_risk = calculate_copyright_risk(
        release_year=seed.get("release_year", 2000),
        studio=seed.get("studio", ""),
        country=seed.get("country", ""),
    )
    rating = float(seed.get("rating", 6.8))
    viral_score = calculate_viral_score(twist_idx, rating, pop_idx, copy_risk)

    reasoning_parts = []
    if twist_idx >= 4.0:
        reasoning_parts.append(f"Cốt truyện cuốn (Điểm Twist: {twist_idx})")
    if pop_idx <= 0.2:
        reasoning_parts.append("Chưa có kênh lớn VN review (Thị trường trắng)")
    if copy_risk <= 0.15:
        reasoning_parts.append("Bản quyền an toàn (Phim xưa/Indie)")

    reasoning = "; ".join(reasoning_parts) or "Phim độc lạ tiềm năng cao"
    fallback_url = str(seed.get("fallback_url") or build_fallback_search_url(str(seed.get("title", ""))))
    is_live = bool(seed.get("is_live", True))

    return CandidateMovie(
        id=str(seed["id"]),
        title=str(seed["title"]),
        vietnamese_title=str(seed.get("vietnamese_title") or seed["title"]),
        source=str(seed["source"]),
        topic=str(seed.get("topic") or "horror"),
        release_year=int(seed.get("release_year", 2000)),
        genres=list(seed.get("genres") or []),
        rating=rating,
        vote_count=int(seed.get("vote_count", 0)),
        summary=str(seed.get("summary", "")),
        vietnamese_summary=str(seed.get("vietnamese_summary") or seed.get("summary", "")),
        source_url=str(seed.get("source_url", "")),
        duration_minutes=int(seed.get("duration_minutes", 60)),
        views=int(seed.get("views", 0)),
        country=str(seed.get("country", "")),
        studio=str(seed.get("studio", "")),
        story_twist_index=twist_idx,
        popularity_index=pop_idx,
        copyright_risk=copy_risk,
        viral_score=viral_score,
        reasoning=reasoning,
        is_live=is_live,
        fallback_url=fallback_url,
    )


class _SilentYtDlpLogger:
    """Logger to suppress yt-dlp stderr output during discovery scans."""

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass


def _get_ytdlp_search_opts(timeout: float = 8.0) -> dict[str, Any]:
    return {
        "logger": _SilentYtDlpLogger(),
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "socket_timeout": max(1, int(timeout)),
        "ignoreerrors": True,
        "no_color": True,
    }


def search_youtube_live(
    topic: str = "all",
    limit: int = 10,
    timeout: float = 8.0,
) -> list[dict[str, Any]]:
    """Query YouTube for live full movies via yt-dlp flat extraction with duration filtering.

    Filters candidates by 45-120 minutes (2700s - 7200s), extracts live metadata,
    calculates viral scores dynamically, and returns formatted candidate dictionaries.
    """
    try:
        import yt_dlp
    except ImportError:
        logger.warning("yt-dlp is not installed; skipping YouTube live discovery")
        return []

    query = build_youtube_live_query(topic)
    fetch_count = max(limit * 2, 20)
    ydl_opts = _get_ytdlp_search_opts(timeout=timeout)

    candidates: list[dict[str, Any]] = []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_target = f"ytsearch{fetch_count}:{query}"
            res = ydl.extract_info(search_target, download=False)
            entries = res.get("entries", []) if isinstance(res, dict) else []

            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                vid = str(entry.get("id") or "").strip()
                if not vid:
                    continue

                raw_dur = entry.get("duration")
                if raw_dur is None:
                    continue
                try:
                    dur_sec = int(raw_dur)
                except (ValueError, TypeError):
                    continue

                # Strict duration filter: 45 - 120 minutes for feature films
                if not (2700 <= dur_sec <= 7200):
                    continue

                title = str(entry.get("title") or "Untitled Movie").strip()
                dur_min = dur_sec // 60
                views = int(entry.get("view_count") or 0)
                uploader = str(entry.get("channel") or entry.get("uploader") or "YouTube").strip()
                summary = str(entry.get("description") or title).strip()[:500]

                # High-res thumbnail extraction with fallback
                thumbs = entry.get("thumbnails")
                if isinstance(thumbs, list) and thumbs and isinstance(thumbs[-1], dict) and thumbs[-1].get("url"):
                    thumbnail_url = str(thumbs[-1]["url"])
                else:
                    thumbnail_url = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"

                # Year detection from title (fallback to 2000)
                year_match = re.search(r"\b(19\d{2}|20[0-2]\d)\b", title)
                release_year = int(year_match.group(1)) if year_match else 2000

                # Dynamic Viral Scoring
                twist_idx = calculate_story_twist_index(f"{title} {summary}")
                pop_idx = 0.1 if views < 50000 else (0.2 if views < 200000 else (0.5 if views < 1000000 else 1.0))
                copy_risk = calculate_copyright_risk(release_year, studio=uploader, country="US")
                rating = 7.0
                viral_score = calculate_viral_score(twist_idx, rating, pop_idx, copy_risk)

                reasoning_parts = []
                if twist_idx >= 4.0:
                    reasoning_parts.append(f"Cốt truyện cuốn hút (điểm kịch tính: {twist_idx})")
                if pop_idx <= 0.2:
                    reasoning_parts.append("Chưa có kênh lớn tại Việt Nam đánh giá (thị trường còn trống)")
                if copy_risk <= 0.15:
                    reasoning_parts.append("Rủi ro bản quyền thấp (phim xưa hoặc phim độc lập)")
                reasoning = "; ".join(reasoning_parts) or "Phim độc lạ có tiềm năng cao từ YouTube"

                source_url = f"https://www.youtube.com/watch?v={vid}"
                effective_topic = topic if topic != "all" else "cult_classic"
                localized_title, localized_summary = build_vietnamese_copy(
                    title, summary, [topic.replace("_", " "), "Retro Cinema"],
                    effective_topic, "youtube_obscure", dur_min,
                )
                candidate_dict = {
                    "id": f"yt-live-{vid}",
                    "title": title,
                    "vietnamese_title": localized_title,
                    "source": "youtube_obscure",
                    "topic": topic if topic != "all" else "cult_classic",
                    "release_year": release_year,
                    "genres": _translate_genres([topic.replace("_", " ").title(), "Retro Cinema"]),
                    "rating": rating,
                    "vote_count": max(100, views // 50),
                    "summary": summary,
                    "vietnamese_summary": localized_summary,
                    "source_url": source_url,
                    "duration_minutes": dur_min,
                    "views": views,
                    "country": "US",
                    "studio": uploader,
                    "story_twist_index": twist_idx,
                    "popularity_index": pop_idx,
                    "copyright_risk": copy_risk,
                    "viral_score": viral_score,
                    "reasoning": reasoning,
                    "is_live": True,
                    "fallback_url": build_fallback_search_url(title),
                    "thumbnail": thumbnail_url,
                }
                candidates.append(candidate_dict)

                if len(candidates) >= limit:
                    break

        candidates.sort(key=lambda x: x["viral_score"], reverse=True)
    except Exception as exc:
        logger.warning("YouTube live discovery failed (%s): falling back to static seeds", exc)
        return []

    return candidates


BILIBILI_SEARCH_ENDPOINT = "https://api.bilibili.com/x/web-interface/search/type"
BILIBILI_SPI_ENDPOINT = "https://api.bilibili.com/x/frontend/finger/spi"

_bilibili_buvid_lock = threading.Lock()
_bilibili_buvid_cookie: str | None = None

BILIBILI_TOPIC_QUERIES: dict[str, str] = {
    "ceo_romance": "短剧 总裁 逆袭",
    "isekai_rebirth": "短剧 穿越 重生",
    "revenge": "短剧 复仇 打脸",
    "all": "短剧 逆袭 全集",
}


def get_bilibili_cookie(timeout: float = 3.0) -> str | None:
    """Fetch or reuse buvid3/buvid4 cookie required by Bilibili WAF."""
    global _bilibili_buvid_cookie
    with _bilibili_buvid_lock:
        if _bilibili_buvid_cookie:
            return _bilibili_buvid_cookie
        try:
            with httpx.Client(timeout=timeout, headers=BILIBILI_PROBE_HEADERS) as client:
                res = client.get(BILIBILI_SPI_ENDPOINT)
                if res.status_code == 200:
                    data = res.json().get("data", {})
                    b_3 = data.get("b_3")
                    b_4 = data.get("b_4")
                    if b_3:
                        _bilibili_buvid_cookie = f"buvid3={b_3}; buvid4={b_4}" if b_4 else f"buvid3={b_3}"
                        return _bilibili_buvid_cookie
        except Exception as err:
            logger.warning("Failed to obtain Bilibili buvid cookie: %s", err)
        return None


def parse_bilibili_duration(dur_str: str | None) -> int:
    """Parse Bilibili duration string ('MM:SS' or 'HH:MM:SS') into total seconds.

    Supports minutes > 59 (e.g. '73:37' -> 4417 seconds).
    Returns 0 if dur_str is missing or malformed.
    """
    if not dur_str or not isinstance(dur_str, str):
        return 0
    clean = dur_str.strip()
    if not clean:
        return 0
    parts = clean.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, TypeError):
        return 0
    return 0


def parse_bilibili_play_count(play: Any) -> int:
    """Safely convert play count (integer or string with '万' or '亿') to integer."""
    if play is None:
        return 0
    if isinstance(play, (int, float)):
        return int(play)
    if isinstance(play, str):
        clean = play.strip()
        if not clean or clean == "--":
            return 0
        if clean.endswith("万"):
            try:
                return int(float(clean[:-1].strip()) * 10000)
            except ValueError:
                return 0
        if clean.endswith("亿"):
            try:
                return int(float(clean[:-1].strip()) * 100000000)
            except ValueError:
                return 0
        try:
            return int(float(clean))
        except ValueError:
            return 0
    return 0


def search_bilibili_short_dramas(
    topic: str = "all",
    limit: int = 10,
    timeout: float = 6.0,
) -> list[dict[str, Any]]:
    """Query Bilibili public search API for live mini short dramas (30-60m).

    Returns candidate dictionaries conforming to CandidateMovie schema.
    Falls back gracefully to verified _SEEDS on any error.
    """
    keyword = BILIBILI_TOPIC_QUERIES.get(
        topic,
        f"短剧 {topic} 全集" if topic != "all" else "短剧 逆袭 全集",
    )

    cookie = get_bilibili_cookie(timeout=min(3.0, timeout))
    headers = dict(BILIBILI_PROBE_HEADERS)
    if cookie:
        headers["Cookie"] = cookie

    params = {
        "search_type": "video",
        "keyword": keyword,
        "duration": "3",  # Bilibili native 30-60 min filter
        "page": 1,
        "pagesize": max(20, limit),
        "order": "totalrank",
    }

    candidates: list[dict[str, Any]] = []

    try:
        with httpx.Client(timeout=timeout, headers=headers) as client:
            resp = client.get(BILIBILI_SEARCH_ENDPOINT, params=params)

            # If 412, refresh buvid and retry once
            if resp.status_code == 412:
                global _bilibili_buvid_cookie
                with _bilibili_buvid_lock:
                    _bilibili_buvid_cookie = None
                fresh_cookie = get_bilibili_cookie(timeout=2.0)
                if fresh_cookie:
                    headers["Cookie"] = fresh_cookie
                    resp = client.get(BILIBILI_SEARCH_ENDPOINT, params=params, headers=headers)

            if resp.status_code != 200:
                logger.warning("Bilibili search returned HTTP %d", resp.status_code)
                return []

            data = resp.json().get("data", {})
            results = data.get("result", [])
            if not isinstance(results, list):
                return []

            for item in results:
                if not isinstance(item, dict):
                    continue

                bvid = item.get("bvid")
                if not bvid:
                    continue

                duration_sec = parse_bilibili_duration(item.get("duration"))
                # Strict boundary filter: 30m (1800s) to 60m (3600s)
                if not (1800 <= duration_sec <= 3600):
                    continue

                raw_title = item.get("title", "")
                clean_title = re.sub(r"<[^>]+>", "", raw_title).strip()
                if not clean_title:
                    continue

                play_count = parse_bilibili_play_count(item.get("play"))
                pic = str(item.get("pic") or "")
                if pic.startswith("//"):
                    pic = f"https:{pic}"

                author = str(item.get("author") or "Bilibili Short Drama")
                desc = str(item.get("description") or "").strip() or clean_title
                tag_str = str(item.get("tag") or "")
                genres = _translate_genres([t.strip() for t in tag_str.split(",") if t.strip()] or ["Short Drama", "Mini Drama"])

                # Extract year
                pubdate = item.get("pubdate")
                release_year = 2024
                if isinstance(pubdate, (int, float)) and pubdate > 0:
                    try:
                        release_year = datetime.fromtimestamp(pubdate, tz=timezone.utc).year
                    except Exception:
                        release_year = 2024

                # Dynamic Viral Scoring
                summary_for_score = f"{desc} {clean_title} {tag_str}"
                twist_idx = calculate_story_twist_index(summary_for_score)
                pop_idx = 0.1 if play_count < 50000 else (0.2 if play_count < 200000 else 0.5)
                copy_risk = calculate_copyright_risk(release_year, studio=author, country="CN")
                rating = 7.5
                viral_score = calculate_viral_score(twist_idx, rating, pop_idx, copy_risk)

                bili_topic = topic if topic != "all" else "ceo_romance"
                localized_title, localized_summary = build_vietnamese_copy(
                    clean_title, desc, genres, bili_topic,
                    "douyin_bilibili", duration_sec // 60,
                )
                cand = {
                    "id": f"bili-{bvid}",
                    "title": clean_title,
                    "vietnamese_title": localized_title,
                    "source": "douyin_bilibili",
                    "topic": topic if topic != "all" else "ceo_romance",
                    "release_year": release_year,
                    "genres": genres,
                    "rating": rating,
                    "vote_count": max(100, int(item.get("favorites") or 1000)),
                    "summary": desc[:500],
                    "vietnamese_summary": localized_summary,
                    "source_url": f"https://www.bilibili.com/video/{bvid}",
                    "duration_minutes": duration_sec // 60,
                    "views": play_count,
                    "country": "CN",
                    "studio": author,
                    "story_twist_index": twist_idx,
                    "popularity_index": pop_idx,
                    "copyright_risk": copy_risk,
                    "viral_score": viral_score,
                    "reasoning": f"Đoản kịch từ Bilibili · Điểm kịch tính: {twist_idx} · Lượt xem: {play_count:,}",
                    "is_live": True,
                    "fallback_url": build_fallback_search_url(clean_title),
                    "thumbnail": pic,
                }
                candidates.append(cand)
                if len(candidates) >= limit:
                    break

        candidates.sort(key=lambda x: x["viral_score"], reverse=True)
    except Exception as err:
        logger.warning("Bilibili live search error: %s", err)
        return []

    return candidates


_SCOUT_SOURCE_ALIASES: dict[str, str] = {
    "tmdb_douban": "tmdb_douban",
    "douyin_bilibili": "douyin_bilibili",
    "bilibili": "douyin_bilibili",
    "youtube_obscure": "youtube_obscure",
    "youtube": "youtube_obscure",
}


def _candidate_rank_key(candidate: dict[str, Any]) -> tuple[float, str, str]:
    return (-float(candidate.get("viral_score", 0.0)), str(candidate.get("source_url", "")), str(candidate.get("id", "")))


def _matches_scout_filters(candidate: dict[str, Any], topic: str, source: str, min_score: float) -> bool:
    expected_source = _SCOUT_SOURCE_ALIASES.get(source, source)
    return (
        (topic == "all" or candidate.get("topic") == topic)
        and (source == "all" or candidate.get("source") == expected_source)
        and float(candidate.get("viral_score", 0.0)) >= min_score
    )


def _rank_and_deduplicate(candidates: list[dict[str, Any]], topic: str, source: str, min_score: float, seen_urls: set[str] | None = None) -> list[dict[str, Any]]:
    seen = seen_urls if seen_urls is not None else set()
    ranked: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=_candidate_rank_key):
        if not _matches_scout_filters(candidate, topic, source, min_score):
            continue
        source_url = str(candidate.get("source_url", ""))
        if source_url and source_url in seen:
            continue
        if source_url:
            seen.add(source_url)
        ranked.append(candidate)
    return ranked


def _source_pool(topic: str, source: str, limit: int, min_score: float, live_candidates: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    seen_urls: set[str] = set()
    live_ranked = _rank_and_deduplicate(live_candidates or [], topic, source, min_score, seen_urls)
    selected = live_ranked[:limit]
    if len(selected) >= limit:
        return selected

    seed_candidates = [score_movie_candidate(seed).to_dict() for seed in _SEEDS]
    seed_ranked = _rank_and_deduplicate(seed_candidates, topic, source, min_score, seen_urls)
    return selected + seed_ranked[:limit - len(selected)]


def _all_source_quotas(limit: int) -> tuple[int, int, int]:
    base, remainder = divmod(limit, 3)
    return base + (1 if remainder > 0 else 0), base + (1 if remainder > 1 else 0), base


def discover_hidden_gems(
    topic: str = "all",
    source: str = "all",
    min_score: float = 0.0,
    limit: int = 10,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Discover and rank candidate hidden-gem movies across cache, live search, and static seeds."""
    global _CACHE

    if limit <= 0:
        return []

    normalized_source = str(source or "all").strip().lower()
    if normalized_source not in {"all", *_SCOUT_SOURCE_ALIASES}:
        return []

    if not refresh and _CACHE is not None:
        cached = _CACHE.get(topic, source)
        if cached is not None:
            seen_urls: set[str] = set()
            filtered: list[dict[str, Any]] = []
            for candidate in cached:
                if not isinstance(candidate, dict) or not _matches_scout_filters(candidate, topic, normalized_source, min_score):
                    continue
                source_url = str(candidate.get("source_url", ""))
                if source_url and source_url in seen_urls:
                    continue
                if source_url:
                    seen_urls.add(source_url)
                filtered.append(candidate)
            return filtered[:limit]

    if normalized_source == "all":
        youtube_quota, bilibili_quota, tmdb_quota = _all_source_quotas(limit)
        youtube_live = search_youtube_live(topic=topic, limit=youtube_quota) if youtube_quota else []
        bilibili_live = search_bilibili_short_dramas(topic=topic, limit=bilibili_quota) if bilibili_quota else []
        candidates = (
            _source_pool(topic, "youtube_obscure", youtube_quota, min_score, youtube_live)
            + _source_pool(topic, "douyin_bilibili", bilibili_quota, min_score, bilibili_live)
            + _source_pool(topic, "tmdb_douban", tmdb_quota, min_score)
        )
    elif normalized_source == "tmdb_douban":
        candidates = _source_pool(topic, normalized_source, limit, min_score)
    elif normalized_source in ("douyin_bilibili", "bilibili"):
        bilibili_live = search_bilibili_short_dramas(topic=topic, limit=limit)
        candidates = _source_pool(topic, normalized_source, limit, min_score, bilibili_live)
    else:
        youtube_live = search_youtube_live(topic=topic, limit=limit)
        candidates = _source_pool(topic, normalized_source, limit, min_score, youtube_live)

    if _CACHE is not None:
        _CACHE.set(topic, source, candidates)

    return candidates[:limit]




def enqueue_gem_for_review(candidate_id: str) -> dict[str, Any]:
    """Prepare a scouted candidate for batch queue processing or project creation."""
    global _CACHE
    match: dict[str, Any] | None = next((s for s in _SEEDS if s.get("id") == candidate_id), None)

    if match:
        candidate_dict = score_movie_candidate(match).to_dict()
    else:
        # Search active cache entries for live-discovered candidate
        candidate_dict = None
        if _CACHE is not None:
            with _CACHE._lock:
                for entry in _CACHE._data.values():
                    if isinstance(entry, dict) and isinstance(entry.get("candidates"), list):
                        for c in entry["candidates"]:
                            if isinstance(c, dict) and c.get("id") == candidate_id:
                                candidate_dict = copy.deepcopy(c)
                                break
                    if candidate_dict is not None:
                        break

    if not candidate_dict:
        raise ValueError(f"Không tìm thấy phim candidate id: {candidate_id}")

    return {
        "status": "ready",
        "candidate": candidate_dict,
        "batch_entry": {
            "url": candidate_dict.get("source_url", ""),
            "title": candidate_dict.get("vietnamese_title") or candidate_dict.get("title", ""),
            "topic": candidate_dict.get("topic", "horror"),
            "viral_score": candidate_dict.get("viral_score", 0.0),
        },
    }
