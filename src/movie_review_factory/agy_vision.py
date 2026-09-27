from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from contextlib import ExitStack, contextmanager
from pathlib import Path

from . import pool_scheduler
from .pool_scheduler import QUOTA_ERROR

BATCH_SIZE = 6
MODEL = "gemini-3.7-flash-medium"


class VisionUnavailable(RuntimeError):
    pass


class QuotaExhausted(VisionUnavailable):
    pass


class VisionUnsupported(VisionUnavailable):
    pass


WORKER_ROLES = ("advisor", "executor", "experiment", "reviewer")


def _source_stamp(source: Path) -> str:
    try:
        stat = source.stat()
    except OSError:
        return str(source)
    return f"{source}:{stat.st_mtime_ns}:{stat.st_size}"


def _resume_key(prefix: str, basis: str) -> str:
    """Stable resume key so a re-run only redoes the units that failed."""
    return f"{prefix}:{hashlib.sha256(basis.encode('utf-8')).hexdigest()[:32]}"


def _extract_frames(
    source: Path,
    ffmpeg: str,
    scenes: list[dict],
    offset: int,
    folder: Path,
    *,
    prefix: str,
    scale: str,
    id_field: str,
    verb: str,
) -> list[dict]:
    batch: list[dict] = []
    for scene in scenes[offset:offset + BATCH_SIZE]:
        index = int(scene["index"])
        frame = folder / f"{prefix}-{index}.jpg"
        start, end = float(scene["start_seconds"]), float(scene["end_seconds"])
        midpoint = start + (end - start) / 2
        try:
            subprocess.run([
                ffmpeg, "-y", "-ss", f"{midpoint:.6f}", "-i", str(source),
                "-frames:v", "1", "-vf", f"scale={scale}", str(frame),
            ], capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise VisionUnavailable(f"could not extract {verb} frame for scene {index}") from exc
        if not frame.is_file() or frame.stat().st_size == 0:
            raise VisionUnavailable(f"{verb} frame for scene {index} is empty")
        batch.append({id_field: index, "path": str(frame)})
    return batch


def _copy_frames(
    batch: list[dict],
    folder: Path,
    id_field: str,
    unavailable: str = "AGY frame workspace is unavailable",
) -> list[dict]:
    staged = [{id_field: item[id_field], "path": str(folder / Path(item["path"]).name)} for item in batch]
    try:
        for original, target in zip(batch, staged):
            shutil.copyfile(original["path"], target["path"])
    except OSError as exc:
        raise VisionUnavailable(unavailable) from exc
    return staged


def pool_workers(pool_dir: Path | None) -> list[tuple[str, str, Path, str]]:
    location = pool_dir or Path.home() / ".aki" / "mcpsv"
    try:
        settings = json.loads((location / "setting.json").read_text(encoding="utf-8"))
        configured = settings["agy"]["workers"]
        secrets = json.loads((location / "agy-pool-secrets.json").read_text(encoding="utf-8"))
        if not isinstance(configured, dict) or not isinstance(secrets, dict):
            raise ValueError("invalid AGY pool configuration")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VisionUnavailable("AGY pool is not configured") from exc
    workers = []
    for role in WORKER_ROLES:
        worker = configured.get(role)
        if not isinstance(worker, dict) or not isinstance(worker.get("allowedModes", ["plan"]), list) or "plan" not in worker.get("allowedModes", ["plan"]):
            continue
        try:
            token = secrets[worker["secretRef"]]
            url = urllib.parse.urlsplit(worker["url"])
            root = Path(worker["root"])
            valid = (url.scheme == "http" and url.hostname == "127.0.0.1"
                     and url.port is not None and url.port > 0 and url.path in ("", "/")
                     and not url.query and not url.fragment)
        except (KeyError, TypeError, ValueError):
            continue
        if valid and isinstance(token, str) and token and root.is_dir():
            workers.append((role, f"http://127.0.0.1:{url.port}/run", root, token))
    if not workers:
        raise VisionUnavailable("AGY pool has no configured plan workers")
    return workers


def _clean_terms(value: object, *, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    terms: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            continue
        term = " ".join(raw.split()).strip()
        if not term or len(term) > 60 or term.casefold() in {item.casefold() for item in terms}:
            continue
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def _first_json_object_array(text: str) -> list[dict] | None:
    """Skip AGY rule receipts/fences without combining unrelated brackets."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "[":
            continue
        try:
            data, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
            return data
    return None


def _parse_observations(text: str, expected_ids: set[int]) -> dict[int, dict]:
    data = _first_json_object_array(text)
    if data is None:
        raise VisionUnavailable("AGY returned invalid visual descriptions")
    observations: dict[int, dict] = {}
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int) or isinstance(item["id"], bool):
            raise VisionUnavailable("AGY returned invalid scene IDs")
        index = item["id"]
        if index in observations or index not in expected_ids:
            raise VisionUnavailable("AGY returned unmatched scene IDs")
        description = item.get("description")
        if not isinstance(description, str) or not description.strip() or len(description) > 240:
            raise VisionUnavailable("AGY returned an empty visual description")
        normalized = description.strip()
        if "unavailable" in normalized.casefold() or "unsupported" in normalized.casefold():
            raise VisionUnsupported("AGY could not inspect a source frame")
        observations[index] = {
            "description": normalized,
            "tags": _clean_terms(item.get("tags")),
            "people": _clean_terms(item.get("people")),
            "actions": _clean_terms(item.get("actions")),
        }
    if observations.keys() != expected_ids:
        raise VisionUnavailable("AGY returned unmatched scene IDs")
    return observations


def _parse_descriptions(text: str, expected_ids: set[int]) -> dict[int, str]:
    return {
        index: observation["description"]
        for index, observation in _parse_observations(text, expected_ids).items()
    }


def _describe_batch(url: str, root: Path, token: str, frames: list[dict]) -> dict[int, dict]:
    prompt = (
        "Inspect the actual pixels in each local image using image vision. "
        "Return only a JSON array with exactly these fields per object: "
        "id (integer), description (short visible description), tags (visible objects/settings), "
        "people (visible person labels only when visually supported), actions (visible actions). "
        "Keep exactly the supplied IDs. Do not infer from dialogue, names, filenames, or timestamps. "
        "Use empty arrays when a category is absent. If a frame cannot be viewed, use description unavailable. "
        "FILES: " + json.dumps(frames, ensure_ascii=False)
    )
    request = urllib.request.Request(
        url,
        data=json.dumps({
            "prompt": prompt, "mode": "plan", "model": MODEL, "cwd": str(root),
        }).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=135) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        if exc.code == 429 or QUOTA_ERROR.search(f"{exc.reason} {detail}"):
            raise QuotaExhausted("AGY account quota exhausted") from exc
        if "agy returned no output" in detail.casefold():
            raise VisionUnsupported("AGY worker returned no image output") from exc
        raise VisionUnavailable("AGY worker could not process source frames") from exc
    except (OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise VisionUnavailable("AGY worker could not process source frames") from exc
    if not isinstance(result, dict) or result.get("ok") is not True or not isinstance(result.get("text"), str):
        detail = result.get("error", "") if isinstance(result, dict) else ""
        if isinstance(detail, str) and QUOTA_ERROR.search(detail):
            raise QuotaExhausted("AGY account quota exhausted")
        raise VisionUnavailable("AGY worker did not return visual descriptions")
    return _parse_observations(result["text"], {item["id"] for item in frames})


@contextmanager
def _shared_frames(root: Path):
    folder = root / f"mrf-vision-{secrets.token_hex(16)}"
    try:
        folder.mkdir()
    except OSError as exc:
        raise VisionUnavailable("AGY frame workspace is unavailable") from exc
    try:
        yield folder
    finally:
        shutil.rmtree(folder)


def describe_candidate_observations(
    source: Path,
    candidates: list[list[dict]],
    *,
    pool_dir: Path | None = None,
) -> dict[int, dict]:
    workers = pool_workers(pool_dir)
    ffmpeg = shutil.which("ffmpeg")
    if not source.is_file() or not ffmpeg:
        raise VisionUnavailable("source video or FFmpeg is unavailable for AGY vision")
    unique = {scene["index"]: scene for group in candidates for scene in group}
    ordered = list(unique.values())
    offsets = list(range(0, len(ordered), BATCH_SIZE))
    scheduler = pool_scheduler.PoolScheduler(workers)
    unsupported: set[str] = set()
    key = _resume_key(
        "vision:describe",
        f"{_source_stamp(source)}|"
        + ";".join(f"{scene['index']}:{scene['start_seconds']}-{scene['end_seconds']}" for scene in ordered),
    )
    batches: dict[int, list[dict]] = {}
    with ExitStack() as stack:
        folders = {root: stack.enter_context(_shared_frames(root))
                   for root in {worker[2] for worker in workers}}
        primary = folders[workers[0][2]]
        for offset in offsets:
            if scheduler.completed(key, offset):
                continue  # A resumed run replays this unit from pool health state.
            batches[offset] = _extract_frames(
                source, ffmpeg, ordered, offset, primary,
                prefix="frame", scale="320:-2", id_field="id", verb="source",
            )

        def attempt(worker: tuple, unit: object) -> dict[int, dict]:
            role, url, root, token = worker
            batch = batches[int(unit)]
            staged = batch if folders[root] == primary else _copy_frames(batch, folders[root], "id")
            try:
                return _describe_batch(url, root, token, staged)
            except VisionUnsupported as exc:
                unsupported.add(role)
                raise pool_scheduler.RotateSignal(str(exc)) from exc

        result = scheduler.run(key, attempt, units=offsets, rotate_on_error=False)

    if len(result.data) != len(offsets):
        if unsupported:
            raise VisionUnavailable("AGY vision unavailable across configured accounts")
        raise VisionUnavailable("AGY vision quota exhausted across configured accounts")
    observations: dict[int, dict] = {}
    for offset in offsets:
        # Resumed units come back through JSON, so their scene keys are strings.
        observations.update({int(index): value for index, value in result.data[offset].items()})
    return observations


def _parse_identity_batch(
    text: str,
    expected_scene_ids: set[int],
    allowed_labels: set[str],
) -> list[dict]:
    data = _first_json_object_array(text)
    if data is None:
        raise VisionUnavailable("AGY returned invalid identity tracking result")
    result: list[dict] = []
    seen_scenes: set[int] = set()
    for item in data:
        if not isinstance(item, dict):
            raise VisionUnavailable("AGY returned invalid identity tracking item")
        scene_id = item.get("scene_id")
        if not isinstance(scene_id, int) or isinstance(scene_id, bool) or scene_id not in expected_scene_ids:
            raise VisionUnavailable("AGY returned unmatched identity scene IDs")
        if scene_id in seen_scenes:
            raise VisionUnavailable("AGY returned duplicate identity scene IDs")
        seen_scenes.add(scene_id)
        raw_people = item.get("people")
        if not isinstance(raw_people, list):
            raise VisionUnavailable("AGY returned invalid identity people list")
        people: list[dict] = []
        seen_labels: set[str] = set()
        for person in raw_people:
            if not isinstance(person, dict):
                raise VisionUnavailable("AGY returned invalid identity person")
            label = str(person.get("label") or "").strip()
            if label not in allowed_labels or label in seen_labels:
                raise VisionUnavailable("AGY returned invalid anonymous person label")
            description = " ".join(str(person.get("description") or "").split()).strip()
            clothing = " ".join(str(person.get("clothing") or "").split()).strip()
            evidence = " ".join(str(person.get("evidence") or "").split()).strip()
            ambiguous = bool(person.get("ambiguous", False))
            try:
                confidence = float(person.get("confidence"))
            except (TypeError, ValueError) as exc:
                raise VisionUnavailable("AGY returned invalid identity confidence") from exc
            if not description or not (0.0 <= confidence <= 1.0):
                raise VisionUnavailable("AGY returned invalid identity person metadata")
            seen_labels.add(label)
            people.append({
                "label": label,
                "description": description[:240],
                "clothing": clothing[:180],
                "ambiguous": ambiguous,
                "confidence": confidence,
                "evidence": evidence[:240],
            })
        result.append({"scene_id": scene_id, "people": people})
    if seen_scenes != expected_scene_ids:
        raise VisionUnavailable("AGY returned unmatched identity scene IDs")
    return result


def _identity_batch(
    url: str,
    root: Path,
    token: str,
    frames: list[dict],
    roster: list[dict],
    allowed_labels: list[str],
) -> list[dict]:
    prompt = (
        "Track anonymous on-screen people across these images. Never identify or guess any real-world "
        "name, celebrity, actor, account, or identity. Reuse an existing Person N only when the visible "
        "individual is sufficiently supported by appearance continuity; otherwise assign a fresh allowed "
        "Person N. Return only a JSON array with one object per scene: "
        "{scene_id:int, people:[{label,description,clothing,ambiguous,confidence,evidence}]}. "
        "confidence is 0..1. description is stable visible appearance only; clothing is current visible "
        "clothing/accessory summary. Set ambiguous=true whenever continuity is uncertain or the face/body "
        "is too obscured to confidently reuse a label. evidence briefly states visible continuity; "
        "do not use dialogue or filenames. "
        f"EXISTING_ROSTER: {json.dumps(roster, ensure_ascii=False)} "
        f"ALLOWED_LABELS: {json.dumps(allowed_labels, ensure_ascii=False)} "
        f"FILES: {json.dumps(frames, ensure_ascii=False)}"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps({
            "prompt": prompt, "mode": "plan", "model": MODEL, "cwd": str(root),
        }).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=135) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        if exc.code == 429 or QUOTA_ERROR.search(f"{exc.reason} {detail}"):
            raise QuotaExhausted("AGY account quota exhausted") from exc
        raise VisionUnavailable("AGY worker could not track anonymous people") from exc
    except (OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise VisionUnavailable("AGY worker could not track anonymous people") from exc
    if not isinstance(result, dict) or result.get("ok") is not True or not isinstance(result.get("text"), str):
        raise VisionUnavailable("AGY worker did not return identity tracking output")
    return _parse_identity_batch(
        result["text"],
        {item["scene_id"] for item in frames},
        set(allowed_labels),
    )


def track_anonymous_people(
    source: Path,
    scenes: list[dict],
    *,
    pool_dir: Path | None = None,
) -> dict:
    workers = pool_workers(pool_dir)
    ffmpeg = shutil.which("ffmpeg")
    if not source.is_file() or not ffmpeg:
        raise VisionUnavailable("source video or FFmpeg is unavailable for AGY identity tracking")
    ordered = [scene for scene in scenes if isinstance(scene.get("index"), int)]
    roster: dict[str, str] = {}
    appearances: list[dict] = []
    next_label = 1
    scheduler = pool_scheduler.PoolScheduler(workers)
    key = _resume_key(
        "vision:identity",
        f"{_source_stamp(source)}|"
        + ";".join(f"{scene['index']}:{scene['start_seconds']}-{scene['end_seconds']}" for scene in ordered),
    )
    with ExitStack() as stack:
        folders = {root: stack.enter_context(_shared_frames(root))
                   for root in {worker[2] for worker in workers}}
        primary = folders[workers[0][2]]
        for offset in range(0, len(ordered), BATCH_SIZE):
            if scheduler.completed(key, offset):
                # Resumed unit: replay the stored result so the roster below is rebuilt exactly as the successful run built it.
                batch = []
            else:
                batch = _extract_frames(
                    source, ffmpeg, ordered, offset, primary,
                    prefix="identity", scale="384:-2", id_field="scene_id", verb="identity",
                )

            def attempt(worker: tuple, unit: object) -> list[dict]:
                _, url, root, token = worker
                staged = batch if folders[root] == primary else _copy_frames(
                    batch, folders[root], "scene_id",
                    unavailable="AGY identity frame workspace is unavailable",
                )
                allowed = list(roster)
                allowed.extend(f"Person {number}" for number in range(next_label, next_label + 12))
                roster_payload = [
                    {"label": label, "description": description}
                    for label, description in roster.items()
                ]
                return _identity_batch(url, root, token, staged, roster_payload, allowed)

            result = scheduler.run(key, attempt, units=(offset,), rotate_on_error=False)
            if offset not in result.data:
                raise VisionUnavailable("AGY identity tracking unavailable across configured accounts")
            for scene_result in result.data[offset]:
                for person in scene_result["people"]:
                    label = person["label"]
                    roster.setdefault(label, person["description"])
                    if label.startswith("Person "):
                        try:
                            next_label = max(next_label, int(label.split(" ", 1)[1]) + 1)
                        except ValueError:
                            pass
                    appearances.append({
                        "scene_index": scene_result["scene_id"],
                        **person,
                    })
    tracks = []
    for label, description in sorted(
        roster.items(),
        key=lambda item: int(item[0].split(" ", 1)[1]) if item[0].startswith("Person ") else 999999,
    ):
        items = [item for item in appearances if item["label"] == label]
        tracks.append({
            "label": label,
            "description": description,
            "appearance_summary": description,
            "ambiguous": any(bool(item.get("ambiguous")) for item in items),
            "source": "agy",
        })
    return {"tracks": tracks, "appearances": appearances}


_STORY_TYPES = {"person", "location", "event", "object"}


def _parse_story_graph(
    text: str,
    expected_scene_ids: set[int],
    allowed_people: set[str],
) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise VisionUnavailable("AGY returned no story graph")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise VisionUnavailable("AGY returned invalid story graph JSON") from exc
    if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
        raise VisionUnavailable("AGY returned invalid story graph")

    entities: dict[tuple[str, str], dict] = {}
    scene_entities: list[dict] = []
    relations: list[dict] = []
    seen_scenes: set[int] = set()
    for item in data["scenes"]:
        if not isinstance(item, dict):
            raise VisionUnavailable("AGY returned invalid story scene")
        scene_id = item.get("scene_id")
        if not isinstance(scene_id, int) or isinstance(scene_id, bool) or scene_id not in expected_scene_ids:
            raise VisionUnavailable("AGY returned unmatched story scene IDs")
        if scene_id in seen_scenes:
            raise VisionUnavailable("AGY returned duplicate story scene IDs")
        seen_scenes.add(scene_id)

        local_keys: set[tuple[str, str]] = set()
        raw_entities = item.get("entities") or []
        if not isinstance(raw_entities, list):
            raise VisionUnavailable("AGY returned invalid story entities")
        for raw in raw_entities:
            if not isinstance(raw, dict):
                raise VisionUnavailable("AGY returned invalid story entity")
            entity_type = str(raw.get("type") or "").strip().casefold()
            label = " ".join(str(raw.get("label") or "").split()).strip()
            description = " ".join(str(raw.get("description") or "").split()).strip()
            try:
                confidence = float(raw.get("confidence"))
            except (TypeError, ValueError) as exc:
                raise VisionUnavailable("AGY returned invalid story confidence") from exc
            evidence = " ".join(str(raw.get("evidence") or "").split()).strip()
            if entity_type not in _STORY_TYPES or not label or not 0 <= confidence <= 1:
                raise VisionUnavailable("AGY returned invalid story entity metadata")
            if entity_type == "person" and label not in allowed_people:
                raise VisionUnavailable("AGY returned unknown anonymous person in story graph")
            key = (entity_type, label)
            local_keys.add(key)
            entities.setdefault(key, {
                "type": entity_type,
                "label": label[:120],
                "description": description[:240],
                "source": "agy",
            })
            scene_entities.append({
                "scene_index": scene_id,
                "type": entity_type,
                "label": label[:120],
                "confidence": confidence,
                "evidence": evidence[:240],
            })

        raw_relations = item.get("relations") or []
        if not isinstance(raw_relations, list):
            raise VisionUnavailable("AGY returned invalid story relations")
        for raw in raw_relations:
            if not isinstance(raw, dict):
                raise VisionUnavailable("AGY returned invalid story relation")
            subject_type = str(raw.get("subject_type") or "").strip().casefold()
            object_type = str(raw.get("object_type") or "").strip().casefold()
            subject_label = " ".join(str(raw.get("subject_label") or "").split()).strip()
            object_label = " ".join(str(raw.get("object_label") or "").split()).strip()
            predicate = " ".join(str(raw.get("predicate") or "").split()).strip().casefold()
            try:
                confidence = float(raw.get("confidence"))
            except (TypeError, ValueError) as exc:
                raise VisionUnavailable("AGY returned invalid story relation confidence") from exc
            evidence = " ".join(str(raw.get("evidence") or "").split()).strip()
            subject_key = (subject_type, subject_label)
            object_key = (object_type, object_label)
            if (
                subject_type not in _STORY_TYPES
                or object_type not in _STORY_TYPES
                or not predicate
                or subject_key not in local_keys
                or object_key not in local_keys
                or not 0 <= confidence <= 1
            ):
                raise VisionUnavailable("AGY returned invalid story relation metadata")
            relations.append({
                "scene_index": scene_id,
                "subject_type": subject_type,
                "subject_label": subject_label[:120],
                "predicate": predicate[:80],
                "object_type": object_type,
                "object_label": object_label[:120],
                "confidence": confidence,
                "evidence": evidence[:240],
            })

    if seen_scenes != expected_scene_ids:
        raise VisionUnavailable("AGY returned unmatched story scene IDs")
    return {
        "entities": list(entities.values()),
        "scene_entities": scene_entities,
        "relations": relations,
    }


def extract_story_graph(
    scenes: list[dict],
    *,
    pool_dir: Path | None = None,
) -> dict:
    workers = pool_workers(pool_dir)
    expected_ids = {
        int(scene["index"])
        for scene in scenes
        if isinstance(scene.get("index"), int)
    }
    allowed_people = {
        str(label)
        for scene in scenes
        for label in scene.get("person_tracks") or []
        if str(label).startswith("Person ")
    }
    payload = [
        {
            "scene_id": int(scene["index"]),
            "start_seconds": float(scene.get("start_seconds") or 0.0),
            "end_seconds": float(scene.get("end_seconds") or 0.0),
            "dialogue": str(scene.get("text") or "")[:700],
            "visual_description": str(scene.get("visual_description") or "")[:240],
            "visual_tags": list(scene.get("visual_tags") or [])[:12],
            "visual_actions": list(scene.get("visual_actions") or [])[:12],
            "person_tracks": list(scene.get("person_tracks") or [])[:12],
        }
        for scene in scenes
        if isinstance(scene.get("index"), int)
    ]
    prompt = (
        "Build a compact story-memory graph from the supplied scene evidence. "
        "Return only JSON object {scenes:[{scene_id,entities,relations}]}. "
        "Entity types are exactly person, location, event, object. "
        "For person entities you MUST use only the supplied anonymous Person N labels and must never "
        "guess real-world names, actors, celebrities, or identities. Do not infer facts unsupported by "
        "dialogue or visual evidence. Each entity is {type,label,description,confidence,evidence}. "
        "Each relation is {subject_type,subject_label,predicate,object_type,object_label,confidence,evidence}. "
        "Only relate entities that are included in that same scene. Keep labels short and stable across scenes. "
        "SCENES: " + json.dumps(payload, ensure_ascii=False)
    )
    scheduler = pool_scheduler.PoolScheduler(workers)
    key = _resume_key("vision:story", json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def attempt(worker: tuple, unit: object) -> dict:
        _, url, root, token = worker
        request = urllib.request.Request(
            url,
            data=json.dumps({
                "prompt": prompt,
                "mode": "plan",
                "model": MODEL,
                "cwd": str(root),
            }).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=150) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            if exc.code == 429 or QUOTA_ERROR.search(f"{exc.reason} {detail}"):
                raise pool_scheduler.QuotaSignal("quota exhausted") from exc
            raise VisionUnavailable("AGY worker could not build story graph") from exc
        except (OSError, TimeoutError, json.JSONDecodeError) as exc:
            raise VisionUnavailable("AGY worker could not build story graph") from exc
        if not isinstance(result, dict) or result.get("ok") is not True or not isinstance(result.get("text"), str):
            raise VisionUnavailable("AGY worker did not return story graph output")
        return _parse_story_graph(result["text"], expected_ids, allowed_people)

    outcome = scheduler.run(key, attempt, units=(0,), rotate_on_error=False)
    if 0 not in outcome.data:
        raise VisionUnavailable("AGY story graph quota exhausted across configured accounts")
    return outcome.data[0]


def describe_candidates(
    source: Path,
    candidates: list[list[dict]],
    *,
    pool_dir: Path | None = None,
) -> dict[int, str]:
    return {
        index: observation["description"]
        for index, observation in describe_candidate_observations(
            source, candidates, pool_dir=pool_dir
        ).items()
    }
