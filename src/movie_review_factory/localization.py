"""Vietnamese localization for pipeline stages, statuses, and messages.

The pipeline (``pipeline.py``) emits deterministic English stage messages and
skip/failure reasons. The local web UI is Vietnamese, so this module maps stage
names and statuses to Vietnamese labels and best-effort translates the known
English messages.

Translation is pattern based and additive: an unrecognised message is returned
unchanged rather than mangled, so a new pipeline message can never break the UI
- it just shows in English until a rule is added here. This keeps the mapping
honest (we never invent a Vietnamese sentence for a message we do not know).
"""

from __future__ import annotations

import re

# Canonical stage order lives in pipeline.STAGES; this dict must cover all of
# them. tests/test_localization.py asserts the two stay in sync.
STAGE_LABELS: dict[str, str] = {
    "ingest": "Nạp nguồn",
    "research": "Nghiên cứu",
    "transcript": "Bóc lời thoại",
    "scenes": "Chỉ mục cảnh",
    "outline": "Dàn ý",
    "script": "Kịch bản",
    "scene_plan": "Kế hoạch cảnh",
    "tts": "Lồng tiếng",
    "alignment": "Căn chỉnh phụ đề",
    "render": "Kết xuất video",
    "qa": "Kiểm tra chất lượng",
    "metadata": "Siêu dữ liệu",
    "publish": "Xuất bản",
}

STATUS_LABELS: dict[str, str] = {
    "pending": "Chờ xử lý",
    "running": "Đang chạy",
    "ready": "Hoàn tất",
    "failed": "Lỗi",
    "skipped": "Bỏ qua",
}

# Short Vietnamese hint per status, used for badge tooltips in the UI.
STATUS_HINTS: dict[str, str] = {
    "pending": "Chưa chạy",
    "running": "Đang xử lý…",
    "ready": "Đã hoàn tất",
    "failed": "Đã xảy ra lỗi",
    "skipped": "Bỏ qua có chủ đích",
}


def stage_label(stage: str) -> str:
    """Vietnamese label for a stage name; falls back to the raw name."""
    return STAGE_LABELS.get(stage, stage)


def status_label(status: str) -> str:
    """Vietnamese label for a status; falls back to the raw status."""
    return STATUS_LABELS.get(status, status)


def status_hint(status: str) -> str:
    return STATUS_HINTS.get(status, "")


# --- message translation -----------------------------------------------------

# (pattern, Vietnamese template). The first match wins, so specific patterns
# come before generic ones. Templates use \1, \2 ... for the captured dynamic
# parts (counts, paths) via ``re.Match.expand``.
_MESSAGE_RULES: list[tuple[re.Pattern[str], str]] = [
    # approval / gate fragments (also emitted inside "publish blocked: ...")
    (re.compile(r"^youtube_metadata\.json not approved.*$"),
     "siêu dữ liệu chưa được duyệt (đặt approved=true)"),
    (re.compile(r"^script\.json not approved.*$"),
     "kịch bản chưa được duyệt (đặt approved=true)"),
    (re.compile(r"^youtube_metadata\.json missing.*$"), "thiếu tệp siêu dữ liệu"),
    (re.compile(r"^script\.json missing.*$"), "thiếu tệp kịch bản"),
    # ready messages
    (re.compile(r"^probed source video with ffprobe$"),
     "đã đọc thông số video nguồn bằng ffprobe"),
    (re.compile(r"^research brief scaffold written$"), "đã tạo khung nghiên cứu"),
    (re.compile(r"^transcribed (\d+) segments$"), r"đã bóc \1 đoạn lời thoại"),
    (re.compile(r"^indexed (\d+) scenes from transcript$"),
     r"đã lập chỉ mục \1 cảnh từ lời thoại"),
    (re.compile(r"^outline scaffold written$"), "đã tạo khung dàn ý"),
    (re.compile(r"^script scaffold written.*$"),
     "đã tạo khung kịch bản (đặt approved=true trước khi lồng tiếng)"),
    (re.compile(r"^scene_plan written \((\d+) clips, ([\d.]+) min total\)$"),
     r"đã tạo kế hoạch cảnh (\1 phân đoạn, tổng \2 phút)"),
    (re.compile(r"^synthesized narration\.mp3 from (\d+) sections$"),
     r"đã tổng hợp narration.mp3 từ \1 phần lời dẫn"),
    (re.compile(r"^aligned (\d+) cues within narration bounds \(dropped (\d+)\) from script$"),
     r"đã căn \1 phụ đề trong giới hạn narration (bỏ \2) từ kịch bản"),
    (re.compile(r"^aligned (\d+) cues within narration bounds \(dropped (\d+)\)$"),
     r"đã căn \1 phụ đề trong giới hạn narration (bỏ \2)"),
    (re.compile(r"^rendered (\d+) clips to final\.mp4$"),
     r"đã kết xuất \1 phân đoạn vào final.mp4"),
    (re.compile(r"^qa passed .+? (\d+) checks OK$"), r"kiểm tra đạt — \1 mục OK"),
    (re.compile(r"^draft metadata written$"), "đã tạo bản nháp siêu dữ liệu"),
    (re.compile(r"^publish record written.*$"),
     "đã ghi bản ghi xuất bản — sẵn sàng bàn giao để tải lên"),
    # skip / failure reasons
    (re.compile(r"^no source_video set.*$"), "chưa đặt video nguồn"),
    (re.compile(r"^source_video not found: (.+)$"), r"không tìm thấy video nguồn: \1"),
    (re.compile(r"^ffprobe not on PATH.*$"), "ffprobe không có trong PATH — cài FFmpeg"),
    (re.compile(r"^ffmpeg not on PATH.*$"), "ffmpeg không có trong PATH — cài FFmpeg"),
    (re.compile(r"^faster-whisper not installed.*$"),
     "chưa cài faster-whisper (cài gói media)"),
    (re.compile(r"^edge-tts not installed.*$"), "chưa cài edge-tts (cài gói tts)"),
    (re.compile(r"^script has no non-empty narration.*$"),
     "kịch bản chưa có lời dẫn để tổng hợp"),
    (re.compile(r"^narration has no positive duration.*$"),
     "narration không có thời lượng hợp lệ"),
    (re.compile(r"^narration\.mp3 has no positive duration.*$"),
     "narration.mp3 không có thời lượng hợp lệ"),
    (re.compile(r"^(\S+) missing - run (\S+) before (\S+)$"),
     r"thiếu \1 — chạy bước \2 trước bước \3"),
    (re.compile(r"^needs source video/media - skipped \(not implemented\)$"),
     "cần video/media nguồn — bỏ qua (chưa triển khai)"),
    (re.compile(r"^no handler - skipped$"), "không có bộ xử lý — bỏ qua"),
]

_PUBLISH_BLOCKED = re.compile(r"^publish blocked: (.+)$")


def _localize_fragment(fragment: str) -> str:
    for pattern, template in _MESSAGE_RULES:
        match = pattern.match(fragment)
        if match:
            return match.expand(template)
    return fragment


def localize_message(message: str) -> str:
    """Translate a known pipeline message into Vietnamese.

    "publish blocked: A; B" carries a ``; ``-joined list of sub-reasons; each is
    translated on its own so the compound gate message reads naturally. Anything
    unrecognised is returned unchanged.
    """
    if not message:
        return message
    blocked = _PUBLISH_BLOCKED.match(message)
    if blocked:
        reasons = [_localize_fragment(part.strip()) for part in blocked.group(1).split(";")]
        return "Chưa thể xuất bản: " + "; ".join(reasons)
    return _localize_fragment(message)
