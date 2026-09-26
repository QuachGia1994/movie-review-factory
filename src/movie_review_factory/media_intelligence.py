from __future__ import annotations

import json
import re
from pathlib import Path

from . import agy_agent, content_agent, pipeline
from .media_store import MediaStore

MAX_CLIP_SECONDS = 60.0
WORD = re.compile(r"[^\W_]+", re.UNICODE)


def tokens(text: str) -> set[str]:
    return {word.casefold() for word in WORD.findall(text) if len(word) > 1}


def detect_highlights(database: Path, limit: int = 8) -> list[dict]:
    limit = max(1, min(int(limit), 20))
    with MediaStore(database) as store:
        shots = store.list_shots(1)
        transcript = store.list_transcript(1)

    highlights = []
    for shot in shots:
        if shot.id is None:
            continue
        start = max(0.0, float(shot.start_seconds))
        end = min(float(shot.end_seconds), start + MAX_CLIP_SECONDS)
        if end <= start:
            continue
        related = [
            segment for segment in transcript
            if segment.end_seconds > start and segment.start_seconds < end
        ]
        word_count = sum(len(tokens(segment.text)) for segment in related)
        score = round(min(1.0, 0.25 + min(word_count, 60) / 100 + min(end - start, 30) / 100), 4)
        title = (shot.label or (related[0].text if related else f"Shot {shot.id}")).strip()[:80]
        highlights.append({
            "id": f"h-{shot.id}",
            "shot_id": shot.id,
            "start": start,
            "end": end,
            "start_seconds": start,
            "end_seconds": end,
            "title": title,
            "reason": f"Shot has {len(related)} related transcript segments",
            "score": score,
        })
    return sorted(highlights, key=lambda item: (-item["score"], item["start"], item["id"]))[:limit]


def retrieve_segments(database: Path, question: str) -> list[dict]:
    question = question.strip()
    if not question or len(question) > 500:
        raise ValueError("question must contain 1 to 500 characters")
    query = tokens(question)
    with MediaStore(database) as store:
        segments = store.list_transcript(1)
    ranked = sorted(
        ((len(query & tokens(segment.text)), segment) for segment in segments),
        key=lambda item: (-item[0], item[1].start_seconds, item[1].id or 0),
    )
    chosen = [segment for score, segment in ranked if score > 0][:8]
    return [{
        "id": int(segment.id or 0),
        "start": float(segment.start_seconds),
        "end": float(segment.end_seconds),
        "start_seconds": float(segment.start_seconds),
        "end_seconds": float(segment.end_seconds),
        "text": segment.text,
    } for segment in chosen]


def answer_chat(root: Path, question: str) -> dict:
    database = root / "media_index.sqlite3"
    if not database.is_file():
        raise FileNotFoundError("media index is not available")
    citations = retrieve_segments(database, question)
    if not citations:
        return {"answer": "No matching indexed transcript is available.", "citations": [], "mode": "local"}

    mode = pipeline.load_manifest(root).config.content_agent
    if mode in ("claude", "agy"):
        ids = {citation["id"] for citation in citations}
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "citation_ids"],
            "properties": {
                "answer": {"type": "string", "minLength": 1, "maxLength": 2000},
                "citation_ids": {
                    "type": "array", "minItems": 1, "maxItems": 8,
                    "items": {"type": "integer"},
                },
            },
        }
        prompt = (
            "Answer using ONLY this transcript JSON and cite supplied IDs. QUESTION: "
            + question + " TRANSCRIPT: " + json.dumps(citations, ensure_ascii=False)
        )
        try:
            if mode == "agy":
                result = agy_agent.run_agy_json(stage="chat", prompt=prompt, schema=schema)
            else:
                result = content_agent.run_claude_json(
                    root=root, stage="chat", prompt=prompt, schema=schema, allowed_tools=[]
                )
            cited = result.get("citation_ids")
            if (
                not isinstance(result.get("answer"), str)
                or not isinstance(cited, list)
                or not cited
                or any(citation_id not in ids for citation_id in cited)
            ):
                raise ValueError("invalid citations")
            return {
                "answer": result["answer"].strip(),
                "citations": [citation for citation in citations if citation["id"] in cited],
                "mode": mode,
            }
        except (content_agent.ContentAgentError, ValueError, TypeError):
            pass

    return {
        "answer": "According to the indexed transcript: "
        + " ".join(citation["text"] for citation in citations[:3]),
        "citations": citations[:3],
        "mode": "local",
    }
