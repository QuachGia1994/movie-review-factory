"""Language-specific narration style for content-agent prompts and offline scaffolds."""
from __future__ import annotations

from dataclasses import dataclass

LANGUAGE_NAMES = {"vi": "Vietnamese", "en": "English"}

# VieNeu v3 Turbo is trained on English-Vietnamese speech, so only FPT.AI is Vietnamese-only.
VIETNAMESE_ONLY_TTS = frozenset({"fptai"})


def language_name(code: str | None) -> str:
    key = (code or "").strip().lower()
    return LANGUAGE_NAMES.get(key.split("-")[0], code or "Vietnamese")


def is_vietnamese(code: str | None) -> bool:
    return (code or "vi").strip().lower().split("-")[0] == "vi"


@dataclass(frozen=True)
class NarrationStyle:
    words_per_minute: int
    guide: tuple[str, ...]


STYLES = {
    "en": NarrationStyle(
        words_per_minute=155,
        guide=(
            "Open with the hook in the first two sentences: a concrete question, stake, or twist. "
            "Never open with 'welcome back', 'hey guys', or 'let's dive in'.",
            "Narrate in present tense, like telling a friend what happens on screen.",
            "Write for the ear: short sentences, one idea each, no parentheses, no em-dash asides, "
            "no abbreviations a TTS voice would misread.",
            "Write numbers and years the way they are spoken.",
            "Tag each character on first mention with a short descriptor (the rookie cop, the sister).",
            "Use light wordplay and callbacks sparingly; never mock the audience.",
            "Every section adds original analysis: why a choice works, what it sets up, or what it says.",
            "End each section with an open loop that pulls into the next one.",
            "Keep the call to action brief and natural, at most two sentences.",
        ),
    ),
}

_EN_OUTLINE_TITLES = (
    "Hook",
    "Setup & premise",
    "Main story (light spoilers)",
    "Key analysis",
    "Verdict",
    "Call to action",
)
_VI_OUTLINE_TITLES = (
    "Mở đầu / hook",
    "Bối cảnh & tiền đề",
    "Diễn biến chính (hạn chế spoiler)",
    "Điểm nhấn phân tích",
    "Đánh giá & kết luận",
    "Call to action",
)


def outline_titles(code: str | None) -> tuple[str, ...]:
    """Hook, body titles..., CTA for the offline outline scaffold."""
    return _VI_OUTLINE_TITLES if is_vietnamese(code) else _EN_OUTLINE_TITLES


def fallback_script_titles(code: str | None) -> tuple[str, str, str]:
    if is_vietnamese(code):
        return ("Mở đầu / hook", "Nội dung chính", "Kết luận & CTA")
    return ("Hook", "Main story", "Verdict & CTA")


def placeholder_narration(code: str | None, title: str) -> str:
    if is_vietnamese(code):
        return (
            f"Bản thảo lời dẫn cho phần {title}. "
            "Hãy rà soát và chỉnh sửa nội dung này trước khi duyệt."
        )
    return f"Draft narration for the {title} section. Review and edit it before approval."


def prompt_style(code: str | None, sections: list | None = None) -> dict | None:
    """Style guide plus per-section word targets; None keeps the legacy prompt shape."""
    style = STYLES.get((code or "").strip().lower().split("-")[0])
    if style is None:
        return None
    data: dict = {
        "language": language_name(code),
        "words_per_minute": style.words_per_minute,
        "guide": list(style.guide),
    }
    if sections:
        targets = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            budget = float(section.get("budget_minutes") or 0)
            targets.append({
                "title": str(section.get("title") or ""),
                "target_words": round(budget * style.words_per_minute),
            })
        data["section_word_targets"] = targets
    return data
