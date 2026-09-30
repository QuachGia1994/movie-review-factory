from __future__ import annotations

from copy import deepcopy
import re
from typing import Literal

from .models import JobConfig


def prompt_creative_brief(config: JobConfig, *, retention_advice: dict | None = None) -> dict:
    """Expose the saved editorial choices to outline and script generation.

    ``retention_advice`` (optional, from :func:`analytics.retention_advice`) is
    surfaced as read-only ADVISORY guidance under ``retention_advisory`` — it never
    mutates the brief or the generated script; the creator / content agent decides
    whether to act on it. Omit it (the default) to keep the exact legacy shape.
    """
    brief = config.creative_brief
    result = {
        "desired_length_minutes": config.target_minutes,
        "review_thesis": brief.review_thesis.strip(),
        "tone": brief.tone.strip(),
        "target_audience": brief.target_audience.strip(),
        "spoiler_policy": brief.spoiler_policy,
        "forbidden_claims": [
            item.strip() for item in brief.forbidden_claims if item.strip()
        ],
    }
    prompt = {key: value for key, value in result.items() if value and value != "unspecified"}
    advisory = _retention_advisory(retention_advice)
    if advisory:
        prompt["retention_advisory"] = advisory
    return prompt


def _retention_advisory(advice: dict | None) -> list[dict]:
    """Flatten :func:`analytics.retention_advice` output into read-only advisory rows."""
    if not isinstance(advice, dict) or not advice.get("enabled"):
        return []
    rows = [
        {"code": item.get("code"), "severity": item.get("severity"),
         "target": item.get("target"), "message": item["message_vi"]}
        for item in advice.get("suggestions") or []
        if isinstance(item, dict) and item.get("message_vi")
    ]
    if rows and advice.get("low_confidence"):
        rows.append({"code": "low_confidence", "severity": "info", "target": "all",
                     "message": "Mẫu số liệu còn ít — chỉ nên tham khảo, không đổi lớn."})
    return rows


def tag_script_span(
    script: dict,
    section_index: int,
    start: int,
    end: int,
    kind: Literal["plot_recap", "opinion"],
    evidence_refs: list[str] | None = None,
) -> dict:
    """Return a script copy with an editorial tag anchored to exact narration text."""
    if kind not in ("plot_recap", "opinion"):
        raise ValueError("kind must be plot_recap or opinion")
    sections = script.get("sections") or []
    if not 0 <= section_index < len(sections):
        raise ValueError("unknown script section")
    narration = sections[section_index].get("narration")
    if not isinstance(narration, str) or not 0 <= start < end <= len(narration):
        raise ValueError("invalid narration span")
    refs = [ref.strip() for ref in (evidence_refs or []) if isinstance(ref, str) and ref.strip()]
    tag = {
        "start": start,
        "end": end,
        "quote": narration[start:end],
        "kind": kind,
        "evidence_refs": refs,
    }
    updated = deepcopy(script)
    annotations = updated["sections"][section_index].setdefault("annotations", [])
    if tag in annotations:
        return updated
    annotations[:] = [
        item for item in annotations
        if item.get("start") != start or item.get("end") != end
    ]
    if any(start < item["end"] and item["start"] < end for item in annotations):
        raise ValueError("narration tags cannot overlap")
    annotations.append(tag)
    annotations.sort(key=lambda item: item["start"])
    updated["approved"] = False
    return updated


def stale_script_tags(script: dict) -> list[tuple[int, int]]:
    """Locate tags whose quoted text no longer matches the current narration."""
    stale = []
    for section_index, section in enumerate(script.get("sections") or []):
        narration = section.get("narration") or ""
        for tag_index, tag in enumerate(section.get("annotations") or []):
            if not isinstance(tag, dict):
                stale.append((section_index, tag_index))
                continue
            start, end = tag.get("start"), tag.get("end")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start < end <= len(narration)
                or narration[start:end] != tag.get("quote")
            ):
                stale.append((section_index, tag_index))
    return stale


def script_evidence_issues(script: dict, scenes: dict, transcript: dict) -> list[str]:
    """Check reference existence only; a matching ID does not prove a plot claim."""
    scene_refs = {
        f"scene:{item['index']}"
        for item in scenes.get("scenes", [])
        if isinstance(item, dict) and isinstance(item.get("index"), int)
    }
    transcript_refs = {
        f"transcript:{index}"
        for index, _ in enumerate(transcript.get("segments", []), start=1)
    }
    known = scene_refs | transcript_refs
    issues = []
    for section_index, section in enumerate(script.get("sections") or [], start=1):
        if not isinstance(section, dict):
            continue
        for tag_index, tag in enumerate(section.get("annotations") or [], start=1):
            if not isinstance(tag, dict):
                issues.append(f"section {section_index}, tag {tag_index}: invalid annotation")
                continue
            refs = tag.get("evidence_refs") or []
            if tag.get("kind") == "plot_recap" and not refs:
                issues.append(f"section {section_index}, tag {tag_index}: plot recap needs source evidence")
            for ref in refs:
                if not isinstance(ref, str) or not re.fullmatch(r"(?:scene|transcript):[1-9][0-9]*", ref) or ref not in known:
                    issues.append(f"section {section_index}, tag {tag_index}: missing source evidence {ref!r}")
    return issues
