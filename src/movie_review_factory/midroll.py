"""Insert a short approved-by-editor CTA at a visual section boundary near 50%."""
from __future__ import annotations

import copy
import math


def prepare(script: dict, plan: dict, line: str, seconds: float = 10,
            narration_seconds: float | None = None) -> tuple[dict, dict, float]:
    """Return draft script and visual plan; never mutate the approved originals."""
    text = " ".join(str(line).split())
    if not text or len(text) > 250 or len(text.split()) > 50 or not (6 <= seconds <= 25):
        raise ValueError("CTA cần 1–50 từ và thời lượng 6–25 giây.")
    sections = list(script.get("sections") or [])
    if any(section.get("midroll") for section in sections):
        raise ValueError("Video đã có CTA giữa phim.")
    if not script.get("approved"):
        raise ValueError("Kịch bản hiện tại cần được duyệt trước khi thêm CTA.")
    clips = list(plan.get("clips") or [])
    if len(sections) < 2 or not clips:
        raise ValueError("Cần kịch bản và scene plan hợp lệ.")
    total = sum(float(clip.get("duration_seconds") or 0) for clip in clips)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Scene plan không có thời lượng hợp lệ.")
    # Section boundary nearest the actual midpoint; avoids cutting a spoken sentence.
    boundaries: list[tuple[float, int, int]] = []
    for position in range(1, len(clips)):
        left = clips[position - 1]
        right = clips[position]
        left_section = left.get("section_index")
        right_section = right.get("section_index")
        if left_section != right_section and isinstance(left_section, int):
            point = sum(float(clip["duration_seconds"]) for clip in clips[:position])
            boundaries.append((point, left_section, position))
    if not boundaries:
        raise ValueError("Không tìm thấy ranh giới phần ở 50% timeline.")

    # The renderer re-times every section to its spoken narration, and QA measures the
    # CTA against that narration timeline. So when the measured narration length is known
    # we must pick and validate the boundary on the narration timeline -- not on the
    # nominal visual budget, which can be on a completely different scale (e.g. a 10-minute
    # target that only speaks ~3.6 minutes). Choosing on the visual budget can drop the CTA
    # outside the central 40-60% of actual narration and fail editorial QA.
    words_total = sum(len(str(s.get("narration") or "").split()) for s in sections)
    use_voice = bool(
        narration_seconds and math.isfinite(narration_seconds)
        and narration_seconds > 0 and words_total
    )

    def voice_fraction(after: int) -> float:
        """Estimate the rendered CTA-midpoint fraction of narration for a boundary."""
        words_before = sum(len(str(s.get("narration") or "").split()) for s in sections[:after])
        voice_before = narration_seconds * words_before / words_total
        return (voice_before + seconds / 2) / (narration_seconds + seconds)

    if use_voice:
        midpoint, after_section, clip_position = min(
            boundaries, key=lambda item: abs(voice_fraction(item[1]) - 0.5)
        )
        if abs(voice_fraction(after_section) - 0.5) > 0.1:
            raise ValueError("Không có ranh giới phần nào nằm trong khoảng giữa 40–60% lời đọc.")
    else:
        midpoint, after_section, clip_position = min(
            boundaries, key=lambda item: abs(item[0] - total / 2)
        )
        if abs(midpoint - total / 2) > total * .1:
            raise ValueError("Không có ranh giới phần đủ gần 50% timeline.")
    if not 1 <= after_section < len(sections):
        raise ValueError("Chỉ số phần phim trong scene plan không khớp kịch bản.")

    extra_before = 0.0
    if use_voice:
        words_before = sum(len(str(s.get("narration") or "").split()) for s in sections[:after_section])
        expected_voice_start = narration_seconds * words_before / words_total
        extra_before = round(max(0.0, expected_voice_start - midpoint), 3)
        if extra_before > total * .05:
            raise ValueError("Lời đọc và hình lệch quá xa ở mốc 50% timeline.")
    amended_script = copy.deepcopy(script)
    amended_plan = copy.deepcopy(plan)
    amended_script["approved"] = False
    amended_script["target_minutes"] = round(float(script.get("target_minutes") or total / 60) + seconds / 60, 2)
    amended_script["sections"].insert(after_section, {
        "title": "Giữa phim: Màn Kể",
        "budget_minutes": round(seconds / 60, 2),
        "duration_seconds": seconds,
        "narration": text,
        "midroll": True,
        "generator": "agy",
    })
    source = amended_plan["clips"][clip_position]["source_clip"]
    if not isinstance(source, dict) or not isinstance(source.get("start_seconds"), (int, float)):
        raise ValueError("Clip tiếp theo chưa có source_clip cho CTA.")
    source_start = float(source["start_seconds"])
    source_end = min(float(source["end_seconds"]), source_start + seconds)
    if source_end <= source_start:
        raise ValueError("source_clip cho CTA không hợp lệ.")
    insert = {
        "section": "Giữa phim: Màn Kể",
        "section_index": after_section + 1,
        "shot_index": 1,
        "shot_count": 1,
        "type": "midroll",
        "start_seconds": midpoint,
        "duration_seconds": seconds,
        "source_clip": {"start_seconds": source_start, "end_seconds": source_end},
        "notes": "Lời kêu gọi bấm thích và đăng ký do AGY soạn, chờ duyệt.",
    }
    if extra_before:
        amended_plan["clips"][clip_position - 1]["duration_seconds"] = round(
            float(amended_plan["clips"][clip_position - 1]["duration_seconds"]) + extra_before, 3
        )
    insert["start_seconds"] = round(midpoint + extra_before, 3)
    for clip in amended_plan["clips"][clip_position:]:
        clip["start_seconds"] = round(float(clip["start_seconds"]) + extra_before + seconds, 3)
        if isinstance(clip.get("section_index"), int):
            clip["section_index"] += 1
    amended_plan["clips"].insert(clip_position, insert)
    amended_plan["total_seconds"] = round(total + extra_before + seconds, 3)
    amended_plan["midroll"] = {"start_seconds": insert["start_seconds"], "duration_seconds": seconds, "generator": "agy"}
    return amended_script, amended_plan, insert["start_seconds"]


def stage(root, line: str, seconds: float = 10) -> dict:
    """Stage CTA behind script approval, preserving the last successful video."""
    import json
    import shutil
    from pathlib import Path
    from . import pipeline

    root = Path(root)
    script_path, plan_path = root / "script.json", root / "scene_plan.json"
    script = json.loads(script_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    render = json.loads((root / "render.json").read_text(encoding="utf-8")) if (root / "render.json").exists() else {}
    new_script, new_plan, midpoint = prepare(
        script, plan, line, seconds, render.get("narration_duration_seconds")
    )
    for name in ("script.json", "scene_plan.json", "final.mp4"):
        source = root / name
        if source.exists():
            backup = root / (name.rsplit(".", 1)[0] + ".pre-midroll." + name.rsplit(".", 1)[-1])
            if backup.exists():
                raise ValueError("Đã có bản sao trước CTA; hãy kiểm tra trước khi chèn tiếp.")
            shutil.copy2(source, backup)
    pipeline._write_json(root, "script.json", new_script)
    pipeline._write_text(root, "script.md", pipeline._script_markdown(new_script))
    pipeline._write_json(root, "scene_plan.json", new_plan)
    manifest = pipeline.load_manifest(root)
    for stage_item in manifest.stages:
        if stage_item.stage in ("tts", "alignment", "render", "qa", "metadata", "thumbnail", "publish"):
            stage_item.status = "pending"
            stage_item.message = ""
            stage_item.artifacts = []
            stage_item.updated_at = None
    pipeline.save_manifest(root, manifest)
    return {"start_seconds": midpoint, "duration_seconds": seconds, "line": line, "approved": False}
