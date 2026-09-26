from __future__ import annotations

import builtins
import math
import random
from array import array
from pathlib import Path

import pytest

from movie_review_factory import semantic_search
from movie_review_factory.media_store import MediaStore
from movie_review_factory.models import MediaAsset, Shot, TranscriptSegment, VisualObservation


class _FakeEmbeddingModel:
    @staticmethod
    def _vector(text: str) -> list[float]:
        value = text.casefold()
        if any(token in value for token in ("xe", "car", "ô tô", "mưa", "rain")):
            return [1.0, 0.0, 0.0]
        if any(token in value for token in ("hải đăng", "lighthouse", "biển", "sea")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def passage_embed(self, texts):
        return iter(self._vector(text) for text in texts)

    def query_embed(self, texts):
        return iter(self._vector(text) for text in texts)


@pytest.fixture()
def fake_embeddings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MRF_EMBED_MODEL", "fake-multilingual")
    monkeypatch.setenv("MRF_EMBED_CACHE", str(tmp_path / "embed-cache"))
    monkeypatch.setattr(
        semantic_search,
        "_model",
        lambda name, cache, offline: _FakeEmbeddingModel(),
    )


class _CountingEmbeddingModel(_FakeEmbeddingModel):
    """Fake embedder that records every passage batch it is asked to embed."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    @property
    def embedded_texts(self) -> list[str]:
        return [text for batch in self.batches for text in batch]

    def passage_embed(self, texts):
        self.batches.append(list(texts))
        return super().passage_embed(texts)


@pytest.fixture()
def counting_embeddings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> _CountingEmbeddingModel:
    monkeypatch.setenv("MRF_EMBED_MODEL", "fake-multilingual")
    monkeypatch.setenv("MRF_EMBED_CACHE", str(tmp_path / "embed-cache"))
    model = _CountingEmbeddingModel()
    monkeypatch.setattr(semantic_search, "_model", lambda name, cache, offline: model)
    return model


def _seeded_store(store: MediaStore) -> tuple[int, int]:
    """Insert one visual shot and one transcript segment; return their ids."""
    asset = store.add_media_asset(MediaAsset(path=Path("movie.mp4"), duration_seconds=30))
    shot = store.add_shot(Shot(
        media_asset_id=asset.id, start_seconds=0, end_seconds=10, label="Scene A"
    ))
    assert shot.id is not None
    store.replace_visual_observations([
        VisualObservation(
            shot_id=shot.id,
            description="Một chiếc xe màu đỏ lao qua con đường ướt",
            tags=["đường phố"],
        )
    ])
    segment = store.add_transcript_segment(TranscriptSegment(
        media_asset_id=asset.id, start_seconds=20, end_seconds=22,
        text="Cuộc trò chuyện trong nhà",
    ))
    assert segment.id is not None
    return int(shot.id), int(segment.id)


def test_cosine_similarity_uses_real_float_vectors() -> None:
    first_dim, first = semantic_search._as_blob([1.0, 0.0, 0.0])
    same_dim, same = semantic_search._as_blob([0.9, 0.1, 0.0])
    other_dim, other = semantic_search._as_blob([0.0, 1.0, 0.0])

    assert semantic_search.cosine_similarity(first, first_dim, same, same_dim) > 0.99
    assert semantic_search.cosine_similarity(first, first_dim, other, other_dim) == pytest.approx(0.0)


def test_short_queries_require_stronger_similarity() -> None:
    assert semantic_search.minimum_score("tunnel") == semantic_search.SHORT_QUERY_MIN_SCORE
    assert semantic_search.minimum_score("red car") == semantic_search.SHORT_QUERY_MIN_SCORE
    assert semantic_search.minimum_score("red car in rain") == semantic_search.DEFAULT_MIN_SCORE


def test_advanced_query_parser_extracts_visual_constraints() -> None:
    parsed = semantic_search.parse_advanced_query(
        'xe chạy mưa person:"Person 1" action:running location:hospital object:"red bag" kind:visual'
    )
    assert parsed["text"] == "xe chạy mưa"
    assert parsed["person"] == ["Person 1"]
    assert parsed["action"] == ["running"]
    assert parsed["location"] == ["hospital"]
    assert parsed["object"] == ["red bag"]
    assert parsed["kind"] == ["visual"]


def test_similar_shots_uses_stored_vector_as_query() -> None:
    with MediaStore() as store:
        store.migrate()
        asset = store.add_media_asset(MediaAsset(path=Path("movie.mp4"), duration_seconds=30))
        shots = [
            store.add_shot(Shot(media_asset_id=asset.id, start_seconds=i * 10, end_seconds=(i + 1) * 10, label=str(i)))
            for i in range(3)
        ]
        _, a = semantic_search._as_blob([1.0, 0.0])
        _, near = semantic_search._as_blob([0.9, 0.1])
        _, far = semantic_search._as_blob([0.0, 1.0])
        store.replace_shot_embeddings([
            (int(shots[0].id), "source", "test", 2, a),
            (int(shots[1].id), "near", "test", 2, near),
            (int(shots[2].id), "far", "test", 2, far),
        ])
        original = semantic_search.model_name
        try:
            semantic_search.model_name = lambda: "test"
            result = semantic_search.similar_shots(store, int(shots[0].id), limit=2, min_score=0.0)
        finally:
            semantic_search.model_name = original
        assert [item["shot_id"] for item in result] == [shots[1].id, shots[2].id]


def test_store_semantic_search_ranks_visual_meaning_not_literal_words(
    fake_embeddings: None,
) -> None:
    with MediaStore() as store:
        store.migrate()
        asset = store.add_media_asset(MediaAsset(path=Path("movie.mp4"), duration_seconds=30))
        car = store.add_shot(Shot(
            media_asset_id=asset.id, start_seconds=0, end_seconds=10, label="Scene A"
        ))
        lighthouse = store.add_shot(Shot(
            media_asset_id=asset.id, start_seconds=10, end_seconds=20, label="Scene B"
        ))
        assert car.id is not None and lighthouse.id is not None
        store.replace_visual_observations([
            VisualObservation(
                shot_id=car.id,
                description="Một chiếc xe màu đỏ lao qua con đường ướt",
                tags=["đường phố"],
                actions=["lao nhanh"],
            ),
            VisualObservation(
                shot_id=lighthouse.id,
                description="Một ngọn hải đăng đứng bên biển",
                tags=["bờ biển"],
            ),
        ])
        store.add_transcript_segment(TranscriptSegment(
            media_asset_id=asset.id, start_seconds=20, end_seconds=22, text="Cuộc trò chuyện trong nhà"
        ))

        semantic_search.refresh_store_embeddings(store)
        results = semantic_search.search_store(store, "ô tô chạy trong cơn mưa", limit=3)

        assert len(results) == 1
        assert results[0]["kind"] == "visual"
        assert results[0]["shot_id"] == car.id
        assert results[0]["score"] == pytest.approx(1.0)
        assert store.list_shot_embeddings("fake-multilingual")[0]["dimension"] == 3


def test_person_track_identity_is_part_of_scene_embedding(
    fake_embeddings: None,
) -> None:
    with MediaStore() as store:
        store.migrate()
        asset = store.add_media_asset(MediaAsset(path=Path("movie.mp4"), duration_seconds=20))
        first = store.add_shot(Shot(
            media_asset_id=asset.id, start_seconds=0, end_seconds=10, label="Scene A"
        ))
        second = store.add_shot(Shot(
            media_asset_id=asset.id, start_seconds=10, end_seconds=20, label="Scene B"
        ))
        assert first.id is not None and second.id is not None
        store.replace_person_tracks(
            [{"label": "Person 1", "description": "red coat", "source": "agy"}],
            [
                {"label": "Person 1", "shot_id": first.id, "confidence": 0.95, "evidence": "red coat"},
                {"label": "Person 1", "shot_id": second.id, "confidence": 0.91, "evidence": "red coat"},
            ],
        )

        semantic_search.refresh_store_embeddings(store)
        docs = store.list_shot_embeddings("fake-multilingual")

        assert all("Person 1 red coat" in item["text"] for item in docs)


def test_new_records_are_embedded_and_stamped_with_hash_metadata(
    counting_embeddings: _CountingEmbeddingModel,
) -> None:
    with MediaStore() as store:
        store.migrate()
        shot_id, segment_id = _seeded_store(store)

        stats = semantic_search.refresh_store_embeddings(store)

        assert stats["embedded"] == 2
        assert stats["changed"] == 2
        assert stats["skipped"] == 0
        assert stats["embed_version"] == semantic_search.EMBED_VERSION == 1
        assert len(counting_embeddings.embedded_texts) == 2

        shot_row = store.list_shot_embeddings("fake-multilingual")[0]
        segment_row = store.list_transcript_embeddings("fake-multilingual")[0]
        assert shot_row["shot_id"] == shot_id
        assert segment_row["transcript_segment_id"] == segment_id
        for row in (shot_row, segment_row):
            assert row["embed_version"] == semantic_search.EMBED_VERSION
            assert row["content_hash"] == semantic_search.content_hash(row["text"])


def test_unchanged_records_skip_the_embedding_model(
    counting_embeddings: _CountingEmbeddingModel,
) -> None:
    with MediaStore() as store:
        store.migrate()
        _seeded_store(store)
        first = semantic_search.refresh_store_embeddings(store)
        embedded_after_first = len(counting_embeddings.embedded_texts)
        assert first["embedded"] == 2

        second = semantic_search.refresh_store_embeddings(store)

        assert second["embedded"] == 0
        assert second["changed"] == 0
        assert second["skipped"] == 2
        assert second["shot_count"] == 1
        assert second["transcript_count"] == 1
        # the fake embedder must not be called again for unchanged records
        assert len(counting_embeddings.embedded_texts) == embedded_after_first
        assert len(counting_embeddings.batches) == 2


def test_changed_content_reembeds_only_the_modified_record(
    counting_embeddings: _CountingEmbeddingModel,
) -> None:
    with MediaStore() as store:
        store.migrate()
        shot_id, segment_id = _seeded_store(store)
        semantic_search.refresh_store_embeddings(store)
        shot_vector = store.list_shot_embeddings("fake-multilingual")[0]["vector"]

        store.connection.execute(
            "UPDATE transcript_segments SET text = ? WHERE id = ?",
            ("Một cuộc trò chuyện mới hoàn toàn", segment_id),
        )
        store.connection.commit()
        stats = semantic_search.refresh_store_embeddings(store)

        assert stats["embedded"] == 1
        assert stats["changed"] == 1
        assert stats["skipped"] == 1
        assert counting_embeddings.embedded_texts[-1] == "Một cuộc trò chuyện mới hoàn toàn"
        transcript_row = store.list_transcript_embeddings("fake-multilingual")[0]
        assert transcript_row["text"] == "Một cuộc trò chuyện mới hoàn toàn"
        assert transcript_row["content_hash"] == semantic_search.content_hash(
            transcript_row["text"]
        )
        shot_row = store.list_shot_embeddings("fake-multilingual")[0]
        assert shot_row["shot_id"] == shot_id
        assert shot_row["vector"] == shot_vector


def test_bumping_embed_version_reembeds_every_record(
    counting_embeddings: _CountingEmbeddingModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with MediaStore() as store:
        store.migrate()
        _seeded_store(store)
        semantic_search.refresh_store_embeddings(store)
        embedded_after_first = len(counting_embeddings.embedded_texts)
        assert embedded_after_first == 2

        monkeypatch.setattr(
            semantic_search, "EMBED_VERSION", semantic_search.EMBED_VERSION + 1
        )
        stats = semantic_search.refresh_store_embeddings(store)

        assert stats["embedded"] == 2
        assert stats["changed"] == 2
        assert stats["skipped"] == 0
        assert stats["embed_version"] == semantic_search.EMBED_VERSION == 2
        assert len(counting_embeddings.embedded_texts) == embedded_after_first + 2
        rows = store.list_shot_embeddings("fake-multilingual") + store.list_transcript_embeddings(
            "fake-multilingual"
        )
        assert {row["embed_version"] for row in rows} == {2}


# --------------------------------------------------------------------------- #
# search_store scoring path (numpy exact, pure-python fallback)
# --------------------------------------------------------------------------- #
_SEARCH_DIMENSION = 384


class _EmbeddingRowsStore:
    """Store stub: ``search_store`` only reads the two embedding lists."""

    def __init__(self, shots: list[dict], segments: list[dict]) -> None:
        self._shots = shots
        self._segments = segments

    def list_shot_embeddings(self, model=None) -> list[dict]:
        return list(self._shots)

    def list_transcript_embeddings(self, model=None) -> list[dict]:
        return list(self._segments)


def _unit(values: list[float]) -> list[float]:
    norm = math.sqrt(math.fsum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _orthogonal_unit(rng: random.Random, base: list[float]) -> list[float]:
    noise = [rng.uniform(-1.0, 1.0) for _ in base]
    projection = math.fsum(a * b for a, b in zip(noise, base))
    return _unit([value - projection * anchor for value, anchor in zip(noise, base)])


def _vector_at_cosine(rng: random.Random, base: list[float], cosine: float) -> list[float]:
    """Unit vector whose cosine against ``base`` is (almost) exactly ``cosine``."""
    side = _orthogonal_unit(rng, base)
    offset = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return [cosine * anchor + offset * value for anchor, value in zip(base, side)]


def _patch_query(monkeypatch: pytest.MonkeyPatch, blob: bytes) -> None:
    monkeypatch.setattr(
        semantic_search, "embed_query", lambda text: (_SEARCH_DIMENSION, blob)
    )


def _random_rows(
    rng: random.Random, base: list[float], shots: int, segments: int
) -> tuple[list[dict], list[dict]]:
    shot_rows = [
        {
            "shot_id": index,
            "start_seconds": float(index),
            "end_seconds": float(index) + 3.5,
            "text": f"shot {index}",
            "dimension": _SEARCH_DIMENSION,
            "vector": semantic_search._as_blob(
                _vector_at_cosine(rng, base, rng.uniform(-0.9, 0.99))
            )[1],
        }
        for index in range(shots)
    ]
    segment_rows = [
        {
            "transcript_segment_id": index,
            "start_seconds": float(1000 + index),
            "end_seconds": float(1001 + index),
            "text": f"segment {index}",
            "speaker": "Speaker",
            "dimension": _SEARCH_DIMENSION,
            "vector": semantic_search._as_blob(
                _vector_at_cosine(rng, base, rng.uniform(-0.9, 0.99))
            )[1],
        }
        for index in range(segments)
    ]
    return shot_rows, segment_rows


def _row_identity(entry: tuple[str, dict]) -> tuple[str, int]:
    kind, row = entry
    key = "shot_id" if kind == "visual" else "transcript_segment_id"
    return kind, int(row[key])


def _result_identity(item: dict) -> tuple[str, int]:
    if item["kind"] == "visual":
        return item["kind"], int(item["shot_id"])
    return item["kind"], int(item["segment_id"])


def _reference_search(
    rows: list[tuple[str, dict]],
    query_blob: bytes,
    dimension: int,
    threshold: float,
    limit: int,
) -> list[tuple[int, float]]:
    """Pure-python reference: per-row fsum cosine, legacy filter and ordering."""
    scores: list[float] = []
    for _, row in rows:
        if int(row["dimension"]) != dimension:
            scores.append(-1.0)
            continue
        left = array("f")
        left.frombytes(query_blob)
        right = array("f")
        right.frombytes(row["vector"])
        dot = math.fsum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(math.fsum(a * a for a in left))
        right_norm = math.sqrt(math.fsum(a * a for a in right))
        if left_norm <= 0 or right_norm <= 0:
            scores.append(-1.0)
        else:
            scores.append(dot / (left_norm * right_norm))
    kept = [index for index, score in enumerate(scores) if score >= threshold]
    kept.sort(
        key=lambda index: (-scores[index], float(rows[index][1]["start_seconds"]), index)
    )
    return [(index, scores[index]) for index in kept[: max(1, limit)]]


def test_numpy_search_matches_pure_python_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = random.Random(20260926)
    base = _unit([rng.uniform(-1.0, 1.0) for _ in range(_SEARCH_DIMENSION)])
    _, query_blob = semantic_search._as_blob(base)
    shots, segments = _random_rows(rng, base, shots=300, segments=200)
    # rows stored with a foreign dimension must keep the legacy -1.0 score
    _, foreign = semantic_search._as_blob([1.0, 0.0, 0.0])
    shots.append({
        "shot_id": 9991,
        "start_seconds": 0.0,
        "end_seconds": 3.0,
        "text": "foreign dimension shot",
        "dimension": 3,
        "vector": foreign,
    })
    segments.append({
        "transcript_segment_id": 9992,
        "start_seconds": 0.0,
        "end_seconds": 3.0,
        "text": "foreign dimension segment",
        "speaker": None,
        "dimension": 3,
        "vector": foreign,
    })
    store = _EmbeddingRowsStore(shots, segments)
    _patch_query(monkeypatch, query_blob)

    query = "một chiếc xe lao nhanh trên phố"
    threshold = semantic_search.DEFAULT_MIN_SCORE
    assert semantic_search.minimum_score(query) == threshold

    rows = [("visual", row) for row in shots] + [("transcript", row) for row in segments]
    expected = _reference_search(rows, query_blob, _SEARCH_DIMENSION, threshold, 500)
    found = semantic_search.search_store(store, query, limit=500)

    assert 0 < len(found) < len(rows)
    assert [_result_identity(item) for item in found] == [
        _row_identity(rows[index]) for index, _ in expected
    ]
    for item, (index, score) in zip(found, expected):
        assert item["score"] == pytest.approx(score, abs=1e-5)
    assert all(item["score"] >= threshold for item in found)
    assert ("visual", 9991) not in [_result_identity(item) for item in found]
    assert ("transcript", 9992) not in [_result_identity(item) for item in found]

    top = semantic_search.search_store(store, query, limit=20)
    assert [_result_identity(item) for item in top] == [
        _result_identity(item) for item in found
    ][:20]


def test_score_ties_keep_original_index_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = random.Random(11)
    base = _unit([rng.uniform(-1.0, 1.0) for _ in range(_SEARCH_DIMENSION)])
    twin = _vector_at_cosine(rng, base, 0.9)
    _, query_blob = semantic_search._as_blob(base)
    # every row stores the very same vector -> the same score for all of them
    _, blob = semantic_search._as_blob(twin)
    shots = [
        {
            "shot_id": 1,
            "start_seconds": 4.0,
            "end_seconds": 8.0,
            "text": "duplicate a",
            "dimension": _SEARCH_DIMENSION,
            "vector": blob,
        },
        {
            "shot_id": 2,
            "start_seconds": 4.0,
            "end_seconds": 8.0,
            "text": "duplicate b",
            "dimension": _SEARCH_DIMENSION,
            "vector": blob,
        },
    ]
    segments = [
        {
            "transcript_segment_id": 3,
            "start_seconds": 4.0,
            "end_seconds": 6.0,
            "text": "duplicate c",
            "speaker": None,
            "dimension": _SEARCH_DIMENSION,
            "vector": blob,
        }
    ]
    store = _EmbeddingRowsStore(shots, segments)
    _patch_query(monkeypatch, query_blob)

    found = semantic_search.search_store(
        store, "một câu truy vấn đủ dài dùng ngưỡng thấp", limit=10, min_score=0.0
    )

    assert [_result_identity(item) for item in found] == [
        ("visual", 1),
        ("visual", 2),
        ("transcript", 3),
    ]
    assert len({item["score"] for item in found}) == 1


def test_score_ties_keep_the_legacy_start_time_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal scores still order by start time first; the original index only
    breaks a full tie (score + start time), exactly like the legacy
    ``results.sort(key=(-score, start_seconds))``."""
    rng = random.Random(12)
    base = _unit([rng.uniform(-1.0, 1.0) for _ in range(_SEARCH_DIMENSION)])
    twin = _vector_at_cosine(rng, base, 0.9)
    _, query_blob = semantic_search._as_blob(base)
    _, blob = semantic_search._as_blob(twin)
    shots = [{
        "shot_id": 5,
        "start_seconds": 5.0,
        "end_seconds": 9.0,
        "text": "later shot",
        "dimension": _SEARCH_DIMENSION,
        "vector": blob,
    }]
    segments = [{
        "transcript_segment_id": 7,
        "start_seconds": 1.0,
        "end_seconds": 3.0,
        "text": "earlier segment",
        "speaker": None,
        "dimension": _SEARCH_DIMENSION,
        "vector": blob,
    }]
    store = _EmbeddingRowsStore(shots, segments)
    _patch_query(monkeypatch, query_blob)

    found = semantic_search.search_store(
        store, "một câu truy vấn đủ dài dùng ngưỡng thấp", limit=10, min_score=0.0
    )

    assert [_result_identity(item) for item in found] == [
        ("transcript", 7),
        ("visual", 5),
    ]


def test_numpy_import_failure_falls_back_to_the_pure_python_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = random.Random(20260927)
    base = _unit([rng.uniform(-1.0, 1.0) for _ in range(_SEARCH_DIMENSION)])
    _, query_blob = semantic_search._as_blob(base)
    shots, segments = _random_rows(rng, base, shots=40, segments=25)
    store = _EmbeddingRowsStore(shots, segments)
    _patch_query(monkeypatch, query_blob)
    query = "một chiếc xe lao nhanh trên phố"

    with_numpy = semantic_search.search_store(store, query, limit=500)
    assert with_numpy

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("numpy disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    assert semantic_search._load_numpy() is None

    without_numpy = semantic_search.search_store(store, query, limit=500)

    assert [_result_identity(item) for item in without_numpy] == [
        _result_identity(item) for item in with_numpy
    ]
    for fallback, vectorized in zip(without_numpy, with_numpy):
        assert fallback["score"] == pytest.approx(vectorized["score"], abs=1e-5)
