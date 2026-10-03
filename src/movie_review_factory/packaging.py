"""YouTube packaging: retention structure rules, titles, description, hashtags, chapters, pinned comment."""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit

from . import genre_tone, narration_style

REVIEW_FORMATS = ("single", "compilation")
TITLE_MAX = 100
HASHTAG_MAX = 5
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com"}

PACKAGING_SCHEMA = {
    "type": "object",
    "properties": {
        "titles": {"type": "array", "items": {"type": "string"}},
        "hook": {"type": "string"},
        "comment_question": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["titles", "hook", "comment_question", "tags"],
    "additionalProperties": False,
}


def _clean(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    text = "".join(ch for ch in text if ord(ch) >= 32 and ord(ch) != 127)
    return text[:limit].rstrip()


def timecode(seconds: float) -> str:
    rounded = max(0, int(seconds))
    minutes, seconds = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def chapters(plan: dict, script: dict) -> str:
    """YouTube chapter lines from the render plan; midroll CTA and continued sections are skipped."""
    sections = script.get("sections") or []
    lines: list[str] = []
    seen: set[int] = set()
    cursor = 0.0
    for clip in plan.get("clips") or []:
        if not isinstance(clip, dict):
            continue
        try:
            index = int(clip.get("section_index"))
            duration = float(clip.get("duration_seconds"))
        except (TypeError, ValueError):
            continue
        if index < 1 or index > len(sections) or duration <= 0:
            continue
        start = cursor
        cursor += duration
        if index in seen:
            continue
        seen.add(index)
        section = sections[index - 1]
        if isinstance(section, dict) and (section.get("midroll") or section.get("continued")):
            continue
        title = section.get("title", "") if isinstance(section, dict) else str(section)
        title = " ".join(str(title).split()) or f"Phần {index}"
        lines.append(f"{timecode(start)} {title}")
    if lines and not lines[0].startswith("00:00:00 "):
        lines.insert(0, "00:00:00 Mở đầu")
    return "\n".join(lines) + ("\n" if lines else "")


def hashtag(text: str) -> str:
    """'Phim Kinh Dị' -> '#PhimKinhDi'; empty when nothing usable remains."""
    ascii_text = unicodedata.normalize("NFKD", str(text or "").replace("đ", "d").replace("Đ", "D"))
    ascii_text = "".join(ch for ch in ascii_text if not unicodedata.combining(ch))
    words = re.findall(r"[A-Za-z0-9]+", ascii_text)
    body = "".join(word[:1].upper() + word[1:] for word in words)[:40]
    return f"#{body}" if body else ""


def hashtags(language: str, movie: str, channel_name: str = "", genre: str = "") -> list[str]:
    """YouTube shows the first three above the title: format tag, movie, channel."""
    base = ("Review Phim", "Tóm Tắt Phim") if narration_style.is_vietnamese(language) else ("Movie Recap", "Movie Review")
    candidates = [base[0], movie, channel_name, f"Phim {genre}" if genre and narration_style.is_vietnamese(language) else genre, base[1]]
    tags: list[str] = []
    for item in candidates:
        tag = hashtag(item)
        if tag and tag.casefold() not in {t.casefold() for t in tags}:
            tags.append(tag)
    return tags[:HASHTAG_MAX]


def subscribe_link(channel_url: str) -> str:
    """YouTube channel URL with the subscribe-confirmation prompt; empty for non-YouTube links."""
    parts = urlsplit(str(channel_url or "").strip())
    if parts.scheme != "https" or parts.netloc.lower() not in _YOUTUBE_HOSTS or not parts.path.strip("/"):
        return ""
    kept = [item for item in parts.query.split("&") if item and not item.startswith("sub_confirmation=")]
    query = "&".join([*kept, "sub_confirmation=1"])
    return urlunsplit(parts._replace(query=query, fragment=""))


def fair_use_notice(language: str) -> str:
    if narration_style.is_vietnamese(language):
        return (
            "Video chỉ nhằm mục đích bình luận, đánh giá và phân tích phim theo nguyên tắc sử dụng hợp lý (fair use). "
            "Bản quyền hình ảnh và âm thanh thuộc về nhà sản xuất. Nếu có vấn đề bản quyền, vui lòng liên hệ kênh để được xử lý."
        )
    return (
        "This video is commentary, criticism and analysis under fair use. All footage and audio belong to their "
        "rights holders. For any copyright concern, please contact the channel."
    )


def fallback_titles(language: str, movie: str, genre: str = "", review_format: str = "single") -> list[str]:
    movie = _clean(movie, 70)
    genre = _clean(genre, 30)
    if narration_style.is_vietnamese(language):
        kind = f"Review phim {genre.lower()}" if genre else "Review phim"
        titles = [f"{kind} | {movie}", f"Tóm tắt phim {movie} | Giải thích cái kết"]
        if review_format == "compilation":
            titles.insert(0, f"{kind} | {movie} | Tuyển tập truyện ngắn")
    else:
        kind = f"{genre} movie recap".strip().capitalize() if genre else "Movie recap"
        titles = [f"{movie} | {kind}", f"{movie} explained | Story and ending"]
        if review_format == "compilation":
            titles.insert(0, f"{movie} | Every story explained")
    return [_clean(title, TITLE_MAX) for title in titles]


def fallback_comment_question(language: str, movie: str, review_format: str = "single", genre: str = "") -> str:
    tuned = genre_tone.comment_question(language, genre, movie, review_format)
    if tuned:
        return tuned
    if narration_style.is_vietnamese(language):
        if review_format == "compilation":
            return "Câu chuyện nào ám ảnh bạn nhất? Để lại số thứ tự câu chuyện dưới bình luận nhé."
        return f"Chi tiết nào trong {movie} khiến bạn ấn tượng nhất? Chỗ nào chưa hiểu thì để lại bình luận, mình giải đáp."
    if review_format == "compilation":
        return "Which story stayed with you the most? Drop its number in the comments."
    return f"Which moment in {movie} hit you the hardest? Ask anything you missed in the comments."


def pinned_comment(language: str, question: str, channel_name: str = "", subscribe_url: str = "", series_next: str = "") -> str:
    vi = narration_style.is_vietnamese(language)
    lines = [_clean(question, 300)]
    if series_next:
        lines.append(f"Phần tiếp theo: {series_next}." if vi else f"Next up: {series_next}.")
    if subscribe_url:
        name = _clean(channel_name, 40) or ("kênh" if vi else "the channel")
        label = f"Đăng ký {name} để không bỏ lỡ video mới" if vi else f"Subscribe to {name} for the next one"
        lines.append(f"{label}: {subscribe_url}")
    return "\n".join(line for line in lines if line)


def description(
    *,
    language: str,
    hook: str,
    movie: str,
    chapter_text: str,
    tags: list[str],
    subscribe_url: str = "",
    series: dict | None = None,
) -> str:
    vi = narration_style.is_vietnamese(language)
    blocks = [_clean(hook, 900) or (f"Review và tóm tắt phim {movie}." if vi else f"Recap and review of {movie}.")]
    if series and series.get("title"):
        part = f"Phần {series['part']}/{series['total']} của series {series['title']}." if vi else f"Part {series['part']}/{series['total']} of the {series['title']} series."
        if series.get("previous"):
            part += (f" Phần trước: {series['previous']}." if vi else f" Previous: {series['previous']}.")
        blocks.append(part)
    if chapter_text.strip():
        blocks.append(("Mốc thời gian:\n" if vi else "Chapters:\n") + chapter_text.rstrip())
    if subscribe_url:
        blocks.append(("Đăng ký kênh: " if vi else "Subscribe: ") + subscribe_url)
    blocks.append(fair_use_notice(language))
    if tags:
        blocks.append(" ".join(tags))
    return "\n\n".join(blocks)


def structure_guide(language: str, review_format: str = "single", channel: dict | None = None, genre: str = "") -> list[str]:
    """Retention rules for outline/script prompts, distilled from top Vietnamese review channels."""
    return _structure_rules(language, review_format, channel) + genre_tone.tone_rules(language, genre)


def _structure_rules(language: str, review_format: str, channel: dict | None) -> list[str]:
    channel = channel or {}
    name = _clean(channel.get("name"), 40)
    greeting = _clean(channel.get("greeting"), 200)
    series = channel.get("series") or {}
    vi = narration_style.is_vietnamese(language)
    rules: list[str] = []
    if vi:
        if greeting:
            rules.append(f"Câu đầu tiên là lời chào riêng của kênh, giữ đúng ý: '{greeting}'. Chỉ một câu.")
        elif name:
            rules.append(f"Mở bằng đúng một câu chào ngắn có tên kênh {name}, rồi vào hook ngay.")
        rules += [
            "Hook: 2 đến 3 câu hỏi gây tò mò lấy từ chính tình tiết phim, rồi một câu mời người xem ở lại đến cuối để có câu trả lời. Vào truyện trước giây thứ 40.",
            "Cuối mỗi phần để lại một câu treo (ai đứng sau, chuyện gì sắp xảy ra) kéo sang phần sau.",
            "Giữ cú twist và lời giải cho phần cuối; hook chỉ gợi, không tiết lộ.",
            "Chỉ hứa điều video thật sự trả lời; tin đồn hay truyền thuyết phải nói rõ là lời đồn.",
            "Phần kết: trả lời các câu hỏi đã đặt ở hook, nêu một nhận định riêng, đặt một câu hỏi bình luận cụ thể về phim, mời like và đăng ký trong tối đa hai câu.",
        ]
        if review_format == "compilation":
            rules.append("Đây là tuyển tập nhiều truyện: mỗi truyện là một phần riêng, tiêu đề dạng 'Câu chuyện thứ N: tên truyện', mỗi truyện đủ mở đầu, cao trào, kết và một câu chuyển sang truyện sau.")
        if series.get("title"):
            rules.append(f"Video là phần {series.get('part')}/{series.get('total')} của series '{series['title']}'."
                         + (f" Nhắc một câu về phần trước '{series['previous']}' sau hook." if series.get("previous") else "")
                         + (f" Câu cuối hé lộ phần sau '{series['next']}'." if series.get("next") else ""))
        return rules
    if greeting:
        rules.append(f"After the hook, one channel greeting that keeps this meaning: '{greeting}'.")
    rules += [
        "Hook: two or three curiosity questions taken from the film itself, then invite viewers to stay to the end for the answer. Enter the story within 40 seconds.",
        "Keep the twist and the explanation for the final section; the hook teases, never reveals.",
        "Promise only what the video answers; label rumours and legends as rumours.",
        "Ending: answer the hook questions, give one personal verdict, ask one specific comment question about the film, then a like/subscribe ask of at most two sentences.",
    ]
    if review_format == "compilation":
        rules.append("This is an anthology: one section per story titled 'Story N: name', each with setup, climax, ending and a bridge to the next story.")
    if series.get("title"):
        rules.append(f"This is part {series.get('part')}/{series.get('total')} of the '{series['title']}' series."
                     + (f" Mention the previous part '{series['previous']}' once after the hook." if series.get("previous") else "")
                     + (f" Tease the next part '{series['next']}' in the last line." if series.get("next") else ""))
    return rules


def packaging_prompt(language: str, movie: str, genre: str, channel_name: str, review_format: str) -> str:
    lang = narration_style.language_name(language)
    shape = "Review phim {thể loại} | {câu gây tò mò}" if narration_style.is_vietnamese(language) else "{Movie} | {curiosity line}"
    return (
        f"You package a {lang} YouTube movie review of '{movie}'"
        + (f" (genre: {genre})" if genre else "")
        + (" that is an anthology of short stories" if review_format == "compilation" else "")
        + (f", narrated in a {genre_tone.voice_label('en', genre)} voice whose mood the titles and hook must match" if genre_tone.detect(genre) else "")
        + f" for the channel '{channel_name or 'this channel'}'. Using only the script context, return JSON with: "
        f"titles: 3 {lang} titles under 70 characters following the shape '{shape}', each naming the film, "
        "promising only what the video really answers, no all-caps titles, no fake claims; "
        f"hook: a 2-3 sentence {lang} description opening that names the film and raises the core question without spoiling the twist; "
        f"comment_question: one specific {lang} question about the film that invites comments; "
        "tags: 8-12 search tags (film title, genre, review/recap keywords in the video language)."
    )


def clean_packaging(agent: dict | None) -> dict:
    agent = agent if isinstance(agent, dict) else {}
    titles = [_clean(title, TITLE_MAX) for title in agent.get("titles") or [] if _clean(title, TITLE_MAX)]
    tags = [_clean(tag, 60) for tag in agent.get("tags") or [] if _clean(tag, 60)]
    return {
        "titles": titles[:3],
        "hook": _clean(agent.get("hook"), 900),
        "comment_question": _clean(agent.get("comment_question"), 300),
        "tags": list(dict.fromkeys(tags))[:15],
    }
