from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from movie_review_factory import agy_vision, pipeline, pool_scheduler
from movie_review_factory.media_store import MediaStore
from movie_review_factory.models import JobConfig, MediaAsset, Shot


@pytest.fixture(autouse=True)
def _isolated_pool_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep AGY tests off the shared %LOCALAPPDATA% pool health file."""
    monkeypatch.setenv("MRF_POOL_STATE_FILE", str(tmp_path / "pool_health.json"))
    for name in (
        "MRF_AGY_RETRIES",
        "MRF_AGY_COOLDOWN_SECONDS",
        "MRF_AGY_MODEL",
        "MRF_AGY_EFFORT",
        "MRF_AGY_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def _write_pool_config(location: Path) -> None:
    pool = location / "pool"
    pool.mkdir()
    location.joinpath("setting.json").write_text(json.dumps({"agy": {"workers": {
        role: {"url": f"http://127.0.0.1:{7411 + number}", "root": str(pool),
               "secretRef": role, "allowedModes": ["plan"]}
        for number, role in enumerate(agy_vision.WORKER_ROLES)
    }}}), encoding="utf-8")
    location.joinpath("agy-pool-secrets.json").write_text(
        json.dumps({role: f"token-{role}" for role in agy_vision.WORKER_ROLES}), encoding="utf-8"
    )


def test_pool_status_tcp_probes_each_worker_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_pool_config(tmp_path)
    down = {7412}

    class _Conn:
        def __enter__(self) -> "_Conn":
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    def connect(address: tuple[str, int], timeout: float) -> _Conn:
        if address[1] in down:
            raise OSError("connection refused")
        return _Conn()

    monkeypatch.setattr(agy_vision.socket, "create_connection", connect)
    status = agy_vision.pool_status(pool_dir=tmp_path)
    assert status["configured"] is True
    assert status["total"] == 4
    assert status["reachable"] == 3
    assert {worker["port"]: worker["reachable"] for worker in status["workers"]} == {
        7411: True, 7412: False, 7413: True, 7414: True,
    }


def test_pool_status_reports_unconfigured_pool(tmp_path: Path) -> None:
    assert agy_vision.pool_status(pool_dir=tmp_path) == {
        "configured": False,
        "detail": "AGY pool is not configured",
        "workers": [],
        "reachable": 0,
        "total": 0,
    }


def test_agy_pool_describes_owned_candidate_frames_and_cleans_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    settings = tmp_path / "settings"
    settings.mkdir()
    (settings / "setting.json").write_text(json.dumps({"agy": {"workers": {"advisor": {
        "url": "http://127.0.0.1:7411", "root": str(pool), "secretRef": "advisor",
    }}}}), encoding="utf-8")
    (settings / "agy-pool-secrets.json").write_text(json.dumps({"advisor": "test-token"}), encoding="utf-8")
    source = tmp_path / "owned.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(agy_vision.shutil, "which", lambda name: "ffmpeg.exe")
    def extract(command: list[str], **kwargs: object) -> None:
        Path(command[-1]).write_bytes(b"frame-pixels")
    monkeypatch.setattr(agy_vision.subprocess, "run", extract)

    calls: list[str] = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def read(self):
            return json.dumps({"ok": True, "text": '[{"id":1,"description":"Red square"},{"id":2,"description":"Blue square"}]'}).encode()

    def worker(request, timeout):
        assert request.full_url == "http://127.0.0.1:7411/run"
        assert request.get_header("Authorization") == "Bearer test-token"
        body = json.loads(request.data)
        assert body["mode"] == "plan" and body["model"].startswith("gemini-")
        assert body["cwd"] == str(pool)
        assert all(Path(item["path"]).read_bytes() == b"frame-pixels" for item in json.loads(body["prompt"].split("FILES: ", 1)[1]))
        calls.append(body["prompt"])
        return Response()
    monkeypatch.setattr(agy_vision.urllib.request, "urlopen", worker)
    descriptions = agy_vision.describe_candidates(source, [[
        {"index": 1, "start_seconds": 0, "end_seconds": 2},
        {"index": 2, "start_seconds": 2, "end_seconds": 4},
    ]], pool_dir=settings)
    assert descriptions == {1: "Red square", 2: "Blue square"}
    assert len(calls) == 1
    assert not list(pool.glob("mrf-vision-*"))


def test_all_four_quotas_exhausted_reports_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    roles = ("advisor", "executor", "experiment", "reviewer")
    (tmp_path / "setting.json").write_text(json.dumps({"agy": {"workers": {
        role: {"url": f"http://127.0.0.1:{7411 + i}", "root": str(pool), "secretRef": role}
        for i, role in enumerate(roles)}}}), encoding="utf-8")
    (tmp_path / "agy-pool-secrets.json").write_text(json.dumps({role: role for role in roles}), encoding="utf-8")
    monkeypatch.setattr(agy_vision.shutil, "which", lambda name: "ffmpeg.exe")
    monkeypatch.setattr(agy_vision.subprocess, "run", lambda command, **kwargs: Path(command[-1]).write_bytes(b"pixels"))
    source = tmp_path / "owned.mp4"
    source.write_bytes(b"video")
    calls = []
    def worker(request, timeout):
        calls.append(request.full_url)
        raise agy_vision.urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)
    monkeypatch.setattr(agy_vision.urllib.request, "urlopen", worker)
    with pytest.raises(agy_vision.VisionUnavailable, match="quota exhausted across configured accounts"):
        agy_vision.describe_candidates(source, [[{"index": 1, "start_seconds": 0, "end_seconds": 2}]], pool_dir=tmp_path)
    assert len(calls) == 4
    assert not list(pool.glob("mrf-vision-*"))


def test_worker_without_image_output_tries_next_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    roles = ("advisor", "executor")
    (tmp_path / "setting.json").write_text(json.dumps({"agy": {"workers": {
        role: {"url": f"http://127.0.0.1:{7411 + i}", "root": str(pool), "secretRef": role}
        for i, role in enumerate(roles)}}}), encoding="utf-8")
    (tmp_path / "agy-pool-secrets.json").write_text(json.dumps({role: role for role in roles}), encoding="utf-8")
    source = tmp_path / "owned.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(agy_vision.shutil, "which", lambda name: "ffmpeg.exe")
    monkeypatch.setattr(agy_vision.subprocess, "run", lambda command, **kwargs: Path(command[-1]).write_bytes(b"pixels"))
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps({"ok": True, "text": '[{"id":1,"description":"colored bars"}]'}).encode()
    def worker(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise agy_vision.urllib.error.HTTPError(
                request.full_url, 500, "Internal Server Error", {},
                io.BytesIO(b'{"ok":false,"error":"agy returned no output"}'))
        return Response()
    monkeypatch.setattr(agy_vision.urllib.request, "urlopen", worker)
    result = agy_vision.describe_candidates(source, [[{"index": 1, "start_seconds": 0, "end_seconds": 2}]], pool_dir=tmp_path)
    assert result == {1: "colored bars"}
    assert len(calls) == 2


def test_agy_pool_rejects_unmatched_scene_ids() -> None:
    with pytest.raises(agy_vision.VisionUnavailable, match="scene IDs"):
        agy_vision._parse_descriptions('[{"id":99,"description":"guess"}]', {1})


def test_agy_observation_parser_keeps_grounded_visual_metadata() -> None:
    parsed = agy_vision._parse_observations(
        '[{"id":1,"description":"A woman runs beside a red car",'
        '"tags":["red car","street","red car"],'
        '"people":["woman"],"actions":["running"]}]',
        {1},
    )
    assert parsed == {
        1: {
            "description": "A woman runs beside a red car",
            "tags": ["red car", "street"],
            "people": ["woman"],
            "actions": ["running"],
        }
    }


def test_agy_vision_parses_json_after_rule_receipt_and_fence() -> None:
    response = (
        '[RULES] agent (always_on) + coding,pattern (viewed)\\n\\n'
        '```json\\n[{"id":1,"description":"A man wearing a cap",'
        '"tags":["cap"],"people":["man"],"actions":["standing"]}]\\n```'
    )
    assert agy_vision._parse_observations(response, {1})[1]["description"] == "A man wearing a cap"
    identity = (
        '[RULES] agent (always_on)\\n```json\\n'
        '[{"scene_id":1,"people":[]}]\\n```'
    )
    assert agy_vision._parse_identity_batch(identity, {1}, set()) == [{"scene_id": 1, "people": []}]


def test_identity_parser_keeps_anonymous_labels_and_rejects_real_names() -> None:
    parsed = agy_vision._parse_identity_batch(
        '[{"scene_id":1,"people":[{"label":"Person 1","description":"woman in red coat",'
        '"confidence":0.92,"evidence":"same red coat and dark hair"}]}]',
        {1},
        {"Person 1", "Person 2"},
    )
    assert parsed[0]["people"][0]["label"] == "Person 1"
    with pytest.raises(agy_vision.VisionUnavailable, match="anonymous person label"):
        agy_vision._parse_identity_batch(
            '[{"scene_id":1,"people":[{"label":"Jane Doe","description":"woman",'
            '"confidence":0.9,"evidence":"face"}]}]',
            {1},
            {"Person 1"},
        )


def test_story_graph_parser_uses_only_anonymous_people() -> None:
    parsed = agy_vision._parse_story_graph(
        '{"scenes":[{"scene_id":1,"entities":['
        '{"type":"person","label":"Person 1","description":"red coat","confidence":0.9,"evidence":"visible"},'
        '{"type":"location","label":"Hospital","description":"corridor","confidence":0.95,"evidence":"sign"}'
        '],"relations":[{"subject_type":"person","subject_label":"Person 1","predicate":"at",'
        '"object_type":"location","object_label":"Hospital","confidence":0.88,"evidence":"inside corridor"}]}]}',
        {1},
        {"Person 1"},
    )
    assert {item["label"] for item in parsed["entities"]} == {"Person 1", "Hospital"}
    assert parsed["relations"][0]["predicate"] == "at"
    with pytest.raises(agy_vision.VisionUnavailable, match="unknown anonymous person"):
        agy_vision._parse_story_graph(
            '{"scenes":[{"scene_id":1,"entities":['
            '{"type":"person","label":"Actor Name","description":"person","confidence":0.9,"evidence":"face"}'
            '],"relations":[]}]}',
            {1},
            {"Person 1"},
        )


def test_story_graph_quota_rotates_across_four_agy_accounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    roles = ("advisor", "executor", "experiment", "reviewer")
    (tmp_path / "setting.json").write_text(json.dumps({"agy": {"workers": {
        role: {"url": f"http://127.0.0.1:{7411 + i}", "root": str(pool), "secretRef": role}
        for i, role in enumerate(roles)
    }}}), encoding="utf-8")
    (tmp_path / "agy-pool-secrets.json").write_text(
        json.dumps({role: f"token-{role}" for role in roles}), encoding="utf-8"
    )
    calls: list[str] = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self):
            return json.dumps({"ok": True, "text": json.dumps({
                "scenes": [{
                    "scene_id": 1,
                    "entities": [{
                        "type": "location", "label": "Tunnel", "description": "dark tunnel",
                        "confidence": 0.9, "evidence": "visible walls",
                    }],
                    "relations": [],
                }]
            })}).encode()

    def worker(request, timeout):
        role = roles[int(request.full_url.split(":")[2].split("/")[0]) - 7411]
        calls.append(role)
        if role != "reviewer":
            raise agy_vision.urllib.error.HTTPError(
                request.full_url, 429, "Quota exceeded", {}, None
            )
        return Response()

    monkeypatch.setattr(agy_vision.urllib.request, "urlopen", worker)
    result = agy_vision.extract_story_graph([
        {
            "index": 1,
            "start_seconds": 0,
            "end_seconds": 10,
            "text": "",
            "person_tracks": [],
        }
    ], pool_dir=tmp_path)
    assert result["entities"][0]["label"] == "Tunnel"
    assert calls == ["advisor", "executor", "experiment", "reviewer"]


def test_scene_plan_passes_agy_visual_descriptions_into_claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "owned.mp4"
    source.write_bytes(b"video")
    pipeline.create_job(tmp_path, JobConfig(job_id="visual", source_video=source, content_agent="claude"))
    with MediaStore(tmp_path / "media_index.sqlite3") as store:
        store.migrate()
        store.replace_index(
            MediaAsset(path=source, duration_seconds=20),
            [
                Shot(media_asset_id=1, start_seconds=0, end_seconds=10, label="Scene 1"),
                Shot(media_asset_id=1, start_seconds=10, end_seconds=20, label="Scene 2"),
            ],
            [],
        )
    (tmp_path / "script.json").write_text(json.dumps({"sections": [
        {"title": "opening", "narration": "setup", "duration_seconds": 20},
    ]}), encoding="utf-8")
    scenes = [{"index": 1, "start_seconds": 0, "end_seconds": 10, "text": ""},
              {"index": 2, "start_seconds": 10, "end_seconds": 20, "text": ""}]
    (tmp_path / "scenes.json").write_text(json.dumps({
        "source_video": str(source), "duration_seconds": 20, "scenes": scenes,
    }), encoding="utf-8")
    monkeypatch.setattr(
        agy_vision,
        "describe_candidate_observations",
        lambda source, groups: {
            1: {"description": "a dark hallway", "tags": ["hallway"], "people": [], "actions": []},
            2: {"description": "a blue door opens", "tags": ["blue door"], "people": [], "actions": ["opening"]},
        },
    )
    monkeypatch.setattr(
        agy_vision,
        "track_anonymous_people",
        lambda source, scenes: {
            "tracks": [{"label": "Person 1", "description": "person in dark coat", "source": "agy"}],
            "appearances": [
                {"scene_index": 1, "label": "Person 1", "description": "person in dark coat", "confidence": 0.9, "evidence": "same coat"},
                {"scene_index": 2, "label": "Person 1", "description": "person in dark coat", "confidence": 0.88, "evidence": "same coat"},
            ],
        },
    )
    monkeypatch.setattr(
        agy_vision,
        "extract_story_graph",
        lambda scenes: {
            "entities": [
                {"type": "person", "label": "Person 1", "description": "person in dark coat", "source": "agy"},
                {"type": "location", "label": "Hallway", "description": "dark hallway", "source": "agy"},
            ],
            "scene_entities": [
                {"scene_index": 1, "type": "person", "label": "Person 1", "confidence": 0.9, "evidence": "visible"},
                {"scene_index": 1, "type": "location", "label": "Hallway", "confidence": 0.9, "evidence": "visible"},
            ],
            "relations": [
                {
                    "scene_index": 1,
                    "subject_type": "person",
                    "subject_label": "Person 1",
                    "predicate": "at",
                    "object_type": "location",
                    "object_label": "Hallway",
                    "confidence": 0.85,
                    "evidence": "Person 1 is in hallway",
                }
            ],
        },
    )
    monkeypatch.setattr(pipeline.semantic_search, "refresh_store_embeddings", lambda store: {"model": "fake"})
    monkeypatch.setattr(pipeline.semantic_search, "search_store", lambda store, query, limit=20: [])
    def select(**kwargs):
        by_index = {
            item["index"]: item
            for item in kwargs["context"]["scene_candidates"][0]
        }
        assert by_index[2]["visual_description"] == "a blue door opens"
        assert kwargs["allowed_tools"] == []
        return {"assignments": [{"shots": [{"start_scene_index": 2, "end_scene_index": 2, "rationale": "door opening"}]}], "notes": ""}
    monkeypatch.setattr(pipeline, "_run_reasoning_agent", select)
    pipeline._scene_plan(tmp_path, pipeline.load_manifest(tmp_path))
    plan = json.loads((tmp_path / "scene_plan.json").read_text(encoding="utf-8"))
    assert plan["visual_mode"] == "agy"
    assert plan["visual_descriptions"] == [
        {"scene_index": 1, "description": "a dark hallway"},
        {"scene_index": 2, "description": "a blue door opens"},
    ]
    assert plan["visual_observations"][1]["tags"] == ["blue door"]
    with MediaStore(tmp_path / "media_index.sqlite3") as store:
        persisted = store.list_visual_observations()
    assert [item["description"] for item in persisted] == [
        "a dark hallway", "a blue door opens"
    ]
    assert persisted[1]["actions"] == ["opening"]
    assert plan["identity_mode"] == "agy"
    assert plan["person_tracks"][0]["label"] == "Person 1"
    assert plan["story_mode"] == "agy"
    assert plan["story_summary"] == {
        "entity_count": 2,
        "appearance_count": 2,
        "relation_count": 1,
    }
    assert plan["clips"][0]["source_clip"] == {"start_seconds": 10.0, "end_seconds": 20.0}


def test_scene_plan_reports_unavailable_vision_without_inventing_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "owned.mp4"
    source.write_bytes(b"video")
    pipeline.create_job(tmp_path, JobConfig(job_id="fallback", source_video=source, content_agent="claude"))
    with MediaStore(tmp_path / "media_index.sqlite3") as store:
        store.migrate()
        store.replace_index(
            MediaAsset(path=source, duration_seconds=20),
            [Shot(media_asset_id=1, start_seconds=0, end_seconds=20, label="setup")],
            [],
        )
    (tmp_path / "script.json").write_text(json.dumps({"sections": [
        {"title": "opening", "narration": "setup", "duration_seconds": 20},
    ]}), encoding="utf-8")
    (tmp_path / "scenes.json").write_text(json.dumps({
        "source_video": str(source), "duration_seconds": 20,
        "scenes": [{"index": 1, "start_seconds": 0, "end_seconds": 20, "text": "setup"}],
    }), encoding="utf-8")
    def unavailable(*args):
        raise agy_vision.VisionUnavailable("AGY advisor pool is unavailable")
    monkeypatch.setattr(agy_vision, "describe_candidate_observations", unavailable)
    monkeypatch.setattr(agy_vision, "track_anonymous_people", unavailable)
    monkeypatch.setattr(agy_vision, "extract_story_graph", unavailable)
    monkeypatch.setattr(
        pipeline.semantic_search,
        "search_store",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            pipeline.semantic_search.EmbeddingUnavailable("model unavailable")
        ),
    )
    def select(**kwargs):
        assert all("visual_description" not in scene for scene in kwargs["context"]["scene_candidates"][0])
        return {"assignments": [{"shots": [{"start_scene_index": 1, "end_scene_index": 1, "rationale": "dialogue match"}]}], "notes": ""}
    monkeypatch.setattr(pipeline, "_run_reasoning_agent", select)
    pipeline._scene_plan(tmp_path, pipeline.load_manifest(tmp_path))
    plan = json.loads((tmp_path / "scene_plan.json").read_text(encoding="utf-8"))
    assert plan["visual_mode"] == "unavailable"
    assert plan["visual_descriptions"] == []
    assert plan["visual_error"] == "AGY advisor pool is unavailable"


def test_quota_exhaustion_rotates_across_four_agy_accounts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    roles = ("advisor", "executor", "experiment", "reviewer")
    settings = {"agy": {"workers": {role: {
        "url": f"http://127.0.0.1:{7411 + number}", "root": str(pool), "secretRef": role,
        "allowedModes": ["plan"],
    } for number, role in enumerate(roles)}}}
    (tmp_path / "setting.json").write_text(json.dumps(settings), encoding="utf-8")
    (tmp_path / "agy-pool-secrets.json").write_text(json.dumps({role: f"token-{role}" for role in roles}), encoding="utf-8")
    source = tmp_path / "owned.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(agy_vision.shutil, "which", lambda name: "ffmpeg.exe")
    monkeypatch.setattr(agy_vision.subprocess, "run", lambda command, **kwargs: Path(command[-1]).write_bytes(b"pixels"))
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps({"ok": True, "text": '[{"id":1,"description":"a person speaks"}]'}).encode()
    def worker(request, timeout):
        role = roles[int(request.full_url.split(":")[2].split("/")[0]) - 7411]
        assert request.get_header("Authorization") == f"Bearer token-{role}"
        calls.append(role)
        if role in ("advisor", "executor"):
            raise agy_vision.urllib.error.HTTPError(request.full_url, 500, "Quota exceeded", {}, None)
        return Response()
    monkeypatch.setattr(agy_vision.urllib.request, "urlopen", worker)
    result = agy_vision.describe_candidates(source, [[{"index": 1, "start_seconds": 0, "end_seconds": 2}]], pool_dir=tmp_path)
    assert result == {1: "a person speaks"}
    assert calls == ["advisor", "executor", "experiment"]
    assert not list(pool.glob("mrf-vision-*"))


def test_non_quota_worker_failure_does_not_rotate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    (tmp_path / "setting.json").write_text(json.dumps({"agy": {"workers": {
        role: {"url": f"http://127.0.0.1:{7411 + i}", "root": str(pool), "secretRef": role}
        for i, role in enumerate(("advisor", "executor"))}}}), encoding="utf-8")
    (tmp_path / "agy-pool-secrets.json").write_text(json.dumps({"advisor": "one", "executor": "two"}), encoding="utf-8")
    monkeypatch.setattr(agy_vision.shutil, "which", lambda name: "ffmpeg.exe")
    monkeypatch.setattr(agy_vision.subprocess, "run", lambda command, **kwargs: Path(command[-1]).write_bytes(b"pixels"))
    source = tmp_path / "owned.mp4"
    source.write_bytes(b"video")
    calls = []
    def worker(request, timeout):
        calls.append(request.full_url)
        raise agy_vision.urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)
    monkeypatch.setattr(agy_vision.urllib.request, "urlopen", worker)
    with pytest.raises(agy_vision.VisionUnavailable, match="worker"):
        agy_vision.describe_candidates(source, [[{"index": 1, "start_seconds": 0, "end_seconds": 2}]], pool_dir=tmp_path)
    assert len(calls) == 1
    assert not list(pool.glob("mrf-vision-*"))


def test_describe_resumes_failed_batch_without_restarting_completed_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    roles = ("advisor", "executor")
    (tmp_path / "setting.json").write_text(json.dumps({"agy": {"workers": {
        role: {"url": f"http://127.0.0.1:{7411 + i}", "root": str(pool), "secretRef": role}
        for i, role in enumerate(roles)}}}), encoding="utf-8")
    (tmp_path / "agy-pool-secrets.json").write_text(
        json.dumps({role: role for role in roles}), encoding="utf-8"
    )
    source = tmp_path / "owned.mp4"
    source.write_bytes(b"video")
    scenes = [
        {"index": index, "start_seconds": index - 1, "end_seconds": index}
        for index in range(1, 8)  # two batches: scenes 1-6, then scene 7
    ]
    monkeypatch.setattr(agy_vision.shutil, "which", lambda name: "ffmpeg.exe")
    extracted: list[str] = []

    def extract(command: list[str], **kwargs: object) -> None:
        extracted.append(Path(command[-1]).name)
        Path(command[-1]).write_bytes(b"pixels")

    monkeypatch.setattr(agy_vision.subprocess, "run", extract)
    clock = {"now": 4_000_000_000.0}
    monkeypatch.setattr(pool_scheduler, "_now", lambda: clock["now"])
    quota = {"on": True}
    requested: list[list[int]] = []

    class Response:
        def __init__(self, ids: list[int]) -> None:
            self.ids = ids

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"ok": True, "text": json.dumps([
                {"id": index, "description": f"scene {index}"} for index in self.ids
            ])}).encode()

    def worker(request, timeout: float) -> Response:
        body = json.loads(request.data)
        ids = sorted(item["id"] for item in json.loads(body["prompt"].split("FILES: ", 1)[1]))
        requested.append(ids)
        if quota["on"] and max(ids) > 6:
            raise agy_vision.urllib.error.HTTPError(
                request.full_url, 429, "Too Many Requests", {}, None
            )
        return Response(ids)

    monkeypatch.setattr(agy_vision.urllib.request, "urlopen", worker)

    with pytest.raises(agy_vision.VisionUnavailable, match="quota exhausted across configured accounts"):
        agy_vision.describe_candidates(source, [scenes], pool_dir=tmp_path)

    assert requested == [[1, 2, 3, 4, 5, 6], [7], [7]]
    assert extracted == [f"frame-{index}.jpg" for index in range(1, 8)]

    # Cooldown expired: only the batch that failed reaches a worker again.
    clock["now"] += 400
    quota["on"] = False
    requested.clear()
    result = agy_vision.describe_candidates(source, [scenes], pool_dir=tmp_path)

    assert result == {index: f"scene {index}" for index in range(1, 8)}
    assert all(isinstance(index, int) for index in result)
    assert requested == [[7]]
    assert extracted == [f"frame-{index}.jpg" for index in range(1, 8)] + ["frame-7.jpg"]
    assert not list(pool.glob("mrf-vision-*"))
