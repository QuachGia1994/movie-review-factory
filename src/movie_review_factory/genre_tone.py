"""Genre-specific narration voice: tone, rhythm, vocabulary and comment hooks per film genre."""
from __future__ import annotations

import re
import unicodedata

from . import narration_style

MAX_BLEND = 2

_ALIASES: dict[str, tuple[str, ...]] = {
    "scifi": ("khoa hoc vien tuong", "vien tuong", "sci fi", "scifi", "science fiction", "cyberpunk", "khong gian"),
    "horror": ("kinh di", "phim ma", "ma quai", "tam linh", "zombie", "xac song", "horror", "ghost", "haunted", "slasher"),
    "thriller": ("giat gan", "ly ky", "hoi hop", "cang thang", "thriller", "suspense"),
    "crime": ("trinh tham", "toi pham", "hinh su", "pha an", "bi an", "crime", "mystery", "detective", "heist"),
    "action": ("hanh dong", "vo thuat", "kiem hiep", "sieu anh hung", "action", "martial arts", "superhero"),
    "comedy": ("hai huoc", "phim hai", "hai", "comedy", "sitcom"),
    "romance": ("tinh cam", "lang man", "ngon tinh", "tinh yeu", "romance", "romantic", "love"),
    "fantasy": ("gia tuong", "ky ao", "than thoai", "tien hiep", "huyen huyen", "phep thuat", "fantasy", "magic"),
    "animation": ("hoat hinh", "anime", "animation", "cartoon"),
    "war": ("chien tranh", "lich su", "co trang", "war", "historical", "history"),
    "disaster": ("tham hoa", "sinh ton", "disaster", "survival"),
    "drama": ("tam ly", "chinh kich", "gia dinh", "xa hoi", "drama", "family"),
}

GENRE_LABELS_VI = (
    "Kinh dị", "Giật gân", "Trinh thám", "Hành động", "Hài", "Tình cảm", "Tâm lý",
    "Khoa học viễn tưởng", "Giả tưởng", "Hoạt hình", "Chiến tranh", "Thảm họa",
)

_VI: dict[str, dict] = {
    "horror": {
        "label": "kinh dị",
        "rules": [
            "Giọng kể thấp, chậm, như kể chuyện ma lúc nửa đêm; gợi nhiều hơn tả, để khán giả tự tưởng tượng.",
            "Câu ngắn ở đoạn căng, có khoảng lặng trước cú hù; mô tả âm thanh, bóng tối, chi tiết lạ trong khung hình.",
            "Từ vựng: rợn người, lạnh sống lưng, ám ảnh, thứ gì đó, không ai biết. Không đùa cợt làm mất không khí, không mô tả máu me quá đà.",
            "Nếu có truyền thuyết hay tín ngưỡng dân gian, kể như lời đồn và nói rõ nguồn gốc.",
        ],
        "question": "Đoạn nào trong {movie} làm bạn rợn người nhất? Bạn có dám xem một mình lúc nửa đêm không?",
        "compilation_question": "Câu chuyện nào ám ảnh bạn nhất? Để lại số thứ tự câu chuyện dưới bình luận nhé.",
    },
    "thriller": {
        "label": "giật gân",
        "rules": [
            "Nhịp nhanh, dồn dập; mỗi đoạn kết bằng một nguy cơ mới hoặc một câu hỏi chưa có lời đáp.",
            "Ưu tiên câu ngắn, động từ mạnh, đếm ngược thời gian khi phim có áp lực thời gian.",
            "Gieo nghi ngờ cho nhiều nhân vật, chỉ lật bài ở phần cuối.",
        ],
        "question": "Bạn đoán ra sự thật của {movie} từ lúc nào? Thú thật dưới bình luận nhé.",
    },
    "crime": {
        "label": "trinh thám",
        "rules": [
            "Giọng điềm tĩnh, logic như người phá án; trình bày manh mối theo trình tự và đánh dấu chi tiết quan trọng.",
            "Mời khán giả cùng suy luận: nêu nghi phạm, động cơ, chứng cứ trước khi giải.",
            "Phần giải thích cuối nối lại từng manh mối đã gieo; không đưa ra chứng cứ mà phim không có.",
        ],
        "question": "Bạn nghi ai trong {movie} ngay từ đầu? Manh mối nào khiến bạn đoán trúng hoặc đoán trật?",
    },
    "action": {
        "label": "hành động",
        "rules": [
            "Năng lượng cao, nhịp nhanh; câu ngắn, động từ mạnh, tả pha hành động theo đúng hình đang chiếu.",
            "Nhấn vào độ khó, cái giá phải trả và mục tiêu của nhân vật, không chỉ liệt kê cảnh đánh.",
            "Dùng từ gợi tốc độ và sức mạnh nhưng tránh lặp từ cảm thán.",
        ],
        "question": "Pha hành động nào trong {movie} đã nhất với bạn? Bình luận mốc thời gian cho mọi người cùng xem lại.",
    },
    "comedy": {
        "label": "hài",
        "rules": [
            "Giọng vui, dí dỏm, gần gũi như kể với bạn bè; được phép bình luận hài hước, so sánh đời thường.",
            "Giữ nhịp cho câu chốt: dẫn ngắn rồi tung câu chốt ở cuối câu, không giải thích lại miếng hài.",
            "Không châm biếm ngoại hình, vùng miền hay nhóm người thật.",
        ],
        "question": "Cảnh nào trong {movie} làm bạn cười nhiều nhất? Kể mình nghe dưới bình luận nhé.",
    },
    "romance": {
        "label": "tình cảm",
        "rules": [
            "Giọng ấm, nhẹ, giàu cảm xúc; tập trung vào ánh mắt, khoảng cách và những điều nhân vật không nói ra.",
            "Nhịp chậm vừa phải, câu dài hơn ở đoạn lắng; để lại dư âm thay vì chốt ý quá nhanh.",
            "Tránh sến và tránh phán xét nhân vật; nêu cả hai phía của mâu thuẫn.",
        ],
        "question": "Nếu là nhân vật chính trong {movie}, bạn sẽ chọn như thế nào? Chia sẻ dưới bình luận nhé.",
    },
    "drama": {
        "label": "tâm lý",
        "rules": [
            "Giọng trầm, suy ngẫm; đi sâu vào động cơ, tổn thương và lựa chọn của nhân vật.",
            "Kết mỗi phần bằng một câu hỏi đạo đức hoặc một nhận định khiến người xem phải nghĩ.",
            "Liên hệ đời thật vừa phải, không lên lớp.",
        ],
        "question": "Theo bạn, nhân vật trong {movie} đúng hay sai ở lựa chọn cuối cùng? Vì sao?",
    },
    "scifi": {
        "label": "khoa học viễn tưởng",
        "rules": [
            "Giọng tò mò, khám phá; giải thích luật chơi của thế giới phim thật dễ hiểu trước khi vào cao trào.",
            "Dùng ví dụ đời thường để giải thích khái niệm khó; nói rõ đâu là khoa học thật, đâu là giả tưởng.",
            "Phần cuối đặt câu hỏi về tương lai hoặc công nghệ mà phim gợi ra.",
        ],
        "question": "Nếu công nghệ trong {movie} có thật, bạn sẽ dùng nó hay tránh xa? Bình luận cho mình biết nhé.",
    },
    "fantasy": {
        "label": "giả tưởng",
        "rules": [
            "Giọng kể sử thi, giàu hình ảnh như kể truyền thuyết; giới thiệu thế giới, phe phái và luật phép thuật ngắn gọn.",
            "Nhấn vào hành trình và cái giá của sức mạnh; giữ tên riêng nhất quán.",
            "Phân biệt rõ nội dung phim với nguyên tác hoặc thần thoại gốc.",
        ],
        "question": "Nếu được sống trong thế giới của {movie}, bạn muốn đứng về phe nào?",
    },
    "animation": {
        "label": "hoạt hình",
        "rules": [
            "Giọng tươi sáng, trong trẻo, hợp cả gia đình; nhấn vào thông điệp và cảm xúc.",
            "Khen hoặc chê phần hình ảnh, màu sắc và âm nhạc bằng chi tiết cụ thể.",
            "Tránh từ ngữ bạo lực hoặc khó nghe.",
        ],
        "question": "Nhân vật nào trong {movie} bạn thích nhất? Bạn sẽ xem lại cùng gia đình chứ?",
    },
    "war": {
        "label": "chiến tranh",
        "rules": [
            "Giọng trang nghiêm, tôn trọng; cung cấp bối cảnh lịch sử ngắn và tách rõ sự kiện thật với chi tiết hư cấu.",
            "Tập trung vào con người và cái giá của chiến tranh, không tô vẽ bạo lực.",
            "Cẩn trọng với số liệu, địa danh và tên thật; chỉ nêu khi có trong research.",
        ],
        "question": "Khoảnh khắc nào trong {movie} khiến bạn lặng người nhất?",
    },
    "disaster": {
        "label": "thảm họa",
        "rules": [
            "Nhịp tăng dần như đồng hồ đếm ngược; nhấn vào quy mô thảm họa và lựa chọn sinh tồn.",
            "Câu ngắn ở đoạn khẩn cấp, dừng lại ở khoảnh khắc nhân văn.",
            "Nêu rõ chi tiết nào phi thực tế nếu phim cường điệu khoa học.",
        ],
        "question": "Nếu rơi vào tình huống trong {movie}, bạn nghĩ mình sống sót được bao lâu?",
    },
}

_EN: dict[str, dict] = {
    "horror": {"label": "horror", "rules": [
        "Low, slow, campfire storytelling voice; suggest more than you show.",
        "Short sentences in tense moments with a beat of silence before each scare; describe sounds, darkness and odd details on screen.",
        "No jokes that break the dread and no excessive gore; tell legends as legends.",
    ], "question": "Which moment in {movie} gave you chills? Would you watch it alone at midnight?"},
    "thriller": {"label": "thriller", "rules": [
        "Fast, relentless pacing; end every section on a new threat or an open question.",
        "Short sentences and strong verbs; keep several suspects in doubt until the end.",
    ], "question": "When did you figure out the truth in {movie}? Be honest in the comments."},
    "crime": {"label": "crime", "rules": [
        "Calm, logical detective voice; present clues in order and flag the important ones.",
        "Invite viewers to deduce: suspects, motives and evidence before the reveal; never invent evidence.",
    ], "question": "Who did you suspect first in {movie}, and which clue gave it away?"},
    "action": {"label": "action", "rules": [
        "High energy, fast cuts in the prose; short sentences and strong verbs that match the picture.",
        "Stress stakes and cost, not just a list of fights.",
    ], "question": "Which action scene in {movie} was the best? Drop a timestamp."},
    "comedy": {"label": "comedy", "rules": [
        "Playful, friendly voice; witty asides are welcome.",
        "Land punchlines at the end of sentences and never explain the joke; no jokes about real groups of people.",
    ], "question": "Which scene in {movie} made you laugh the most?"},
    "romance": {"label": "romance", "rules": [
        "Warm, gentle, emotional voice; focus on glances, distance and what stays unsaid.",
        "Slower rhythm in quiet beats; avoid melodrama and judging characters.",
    ], "question": "In the lead's shoes in {movie}, what would you have chosen?"},
    "drama": {"label": "drama", "rules": [
        "Reflective voice; dig into motives, wounds and choices.",
        "Close sections on a moral question; relate to real life without preaching.",
    ], "question": "Was the final choice in {movie} right or wrong? Why?"},
    "scifi": {"label": "sci-fi", "rules": [
        "Curious, exploratory voice; explain the world's rules simply before the climax.",
        "Use everyday analogies and separate real science from fiction.",
    ], "question": "If the technology in {movie} were real, would you use it?"},
    "fantasy": {"label": "fantasy", "rules": [
        "Epic, vivid storyteller voice; introduce the world, factions and magic rules briefly.",
        "Keep names consistent and separate the film from its source myth or book.",
    ], "question": "Which side would you join in the world of {movie}?"},
    "animation": {"label": "animation", "rules": [
        "Bright, family-friendly voice; highlight message, emotion, visuals and music with specifics.",
    ], "question": "Who is your favourite character in {movie}?"},
    "war": {"label": "war", "rules": [
        "Solemn, respectful voice; brief historical context, clearly separating fact from fiction.",
        "Focus on people and cost; only cite figures and real names found in research.",
    ], "question": "Which moment in {movie} left you speechless?"},
    "disaster": {"label": "disaster", "rules": [
        "Countdown pacing that builds; stress scale and survival choices, pause on human moments.",
    ], "question": "How long do you think you would survive in {movie}?"},
}


# Edge-TTS prosody per genre; rate also scales the script word budget so the video length holds.
_PROSODY: dict[str, dict[str, str]] = {
    "horror": {"rate": "-10%", "pitch": "-6Hz"},
    "thriller": {"rate": "+4%", "pitch": "-2Hz"},
    "crime": {"rate": "-4%", "pitch": "-3Hz"},
    "action": {"rate": "+10%", "pitch": "+4Hz"},
    "comedy": {"rate": "+8%", "pitch": "+6Hz"},
    "romance": {"rate": "-6%", "pitch": "+2Hz"},
    "drama": {"rate": "-6%", "pitch": "-3Hz"},
    "scifi": {"rate": "-2%"},
    "fantasy": {"rate": "-4%", "pitch": "-2Hz"},
    "animation": {"rate": "+6%", "pitch": "+8Hz"},
    "war": {"rate": "-8%", "pitch": "-5Hz"},
    "disaster": {"rate": "+6%", "pitch": "-2Hz"},
}
_DARK = {"horror", "thriller", "crime", "action", "war", "disaster", "scifi", "fantasy"}
_VOICES = {
    "edge": {"vi": ("vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"), "en": ("en-US-GuyNeural", "en-US-AriaNeural")},
    "fptai": {"vi": ("leminh", "banmai")},
}
RATE_PROVIDERS = ("edge", "fptai", "elevenlabs")
_RATE_RE = re.compile(r"^([+-]\d{1,3})%$")


def prosody(genre: str) -> dict[str, str]:
    keys = detect(genre)
    return dict(_PROSODY.get(keys[0], {})) if keys else {}


def voice(provider: str, language: str, genre: str) -> str:
    """Default voice for the genre mood (deep male for dark genres), or '' to keep the provider default."""
    keys = detect(genre)
    base = str(language or "").strip().lower().split("-")[0]
    pair = _VOICES.get((provider or "").lower(), {}).get(base)
    if not keys or not pair:
        return ""
    return pair[0] if keys[0] in _DARK else pair[1]


def rate_factor(rate: str) -> float:
    match = _RATE_RE.match(str(rate or "").strip())
    return max(0.5, min(1.5, 1 + int(match.group(1)) / 100)) if match else 1.0


def _fold(text: str) -> str:
    text = str(text or "").lower().replace("đ", "d")
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    return " " + " ".join(re.sub(r"[^a-z0-9]+", " ", text).split()) + " "


def detect(genre: str) -> list[str]:
    """Genre keys found in free text, ordered by where they appear, at most MAX_BLEND."""
    folded = _fold(genre)
    hits: dict[str, int] = {}
    for key, aliases in _ALIASES.items():
        for alias in aliases:
            pos = folded.find(f" {alias} ")
            if pos >= 0 and (key not in hits or pos < hits[key]):
                hits[key] = pos
                masked = folded[:pos] + " " * (len(alias) + 2) + folded[pos + len(alias) + 2:]
                folded = masked
    return sorted(hits, key=hits.get)[:MAX_BLEND]


def _table(language: str) -> dict[str, dict]:
    return _VI if narration_style.is_vietnamese(language) else _EN


def tone_rules(language: str, genre: str) -> list[str]:
    keys = detect(genre)
    if not keys:
        return []
    table = _table(language)
    vi = narration_style.is_vietnamese(language)
    rules: list[str] = []
    if len(keys) > 1:
        main, extra = table[keys[0]]["label"], table[keys[1]]["label"]
        rules.append(
            f"Phim pha trộn {main} và {extra}: giọng chính là {main}, chỉ mượn chất {extra} ở những cảnh phù hợp."
            if vi else f"Blend of {main} and {extra}: lead with the {main} voice, borrow {extra} only where scenes call for it."
        )
    rules += [f"[{table[keys[0]]['label']}] {rule}" for rule in table[keys[0]]["rules"]]
    if len(keys) > 1:
        rules += [f"[{table[keys[1]]['label']}] {rule}" for rule in table[keys[1]]["rules"][:1]]
    return rules


def comment_question(language: str, genre: str, movie: str, review_format: str = "single") -> str:
    keys = detect(genre)
    if not keys:
        return ""
    entry = _table(language)[keys[0]]
    template = entry.get("compilation_question") if review_format == "compilation" else None
    return (template or entry["question"]).format(movie=movie)


def voice_label(language: str, genre: str) -> str:
    keys = detect(genre)
    return _table(language)[keys[0]]["label"] if keys else ""
