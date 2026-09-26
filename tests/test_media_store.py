import sqlite3
from importlib.resources import files
from pathlib import Path

import pytest
from pydantic import ValidationError

from movie_review_factory.media_store import MediaStore
from movie_review_factory.models import (
    MediaAsset,
    SceneSelection,
    Shot,
    TranscriptSegment,
    VisualObservation,
)


def _migration_scripts() -> list[tuple[int, str]]:
    """Numbered SQL scripts shipped in the package, oldest first."""
    directory = files("movie_review_factory.migrations")
    scripts = [
        (int(item.name.split("_", 1)[0]), item.read_text(encoding="utf-8"))
        for item in directory.iterdir()
        if item.name.endswith(".sql")
    ]
    return sorted(scripts, key=lambda item: item[0])


def test_models_validate_time_ranges_and_required_text() -> None:
    with pytest.raises(ValidationError):
        Shot(media_asset_id=1, start_seconds=2, end_seconds=2)
    with pytest.raises(ValidationError):
        TranscriptSegment(
            media_asset_id=1, start_seconds=3, end_seconds=2, text="dialogue"
        )
    with pytest.raises(ValidationError):
        TranscriptSegment(
            media_asset_id=1, start_seconds=0, end_seconds=1, text=""
        )


def test_migrations_are_idempotent_and_foreign_keys_are_enforced() -> None:
    with MediaStore() as store:
        store.migrate()
        store.migrate()

        versions = store.connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [row[0] for row in versions] == [1, 2, 3, 4, 5, 6]
        assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1

        with pytest.raises(sqlite3.IntegrityError):
            store.add_shot(
                Shot(media_asset_id=999, start_seconds=0, end_seconds=1)
            )


def _embedding_columns(store: MediaStore, table: str) -> set[str]:
    rows = store.connection.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def test_migration_006_applies_to_existing_and_fresh_databases(tmp_path: Path) -> None:
    legacy_db = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(legacy_db)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    for version, sql in _migration_scripts():
        if version > 5:
            continue
        connection.executescript(
            "BEGIN IMMEDIATE;\n"
            + sql
            + f"\nINSERT INTO schema_migrations(version) VALUES ({version});\nCOMMIT;"
        )
    connection.execute(
        "INSERT INTO media_assets(path, duration_seconds) VALUES ('movie.mp4', 30)"
    )
    connection.execute(
        "INSERT INTO shots(media_asset_id, start_seconds, end_seconds, label) "
        "VALUES (1, 0, 10, 'legacy shot')"
    )
    connection.execute(
        "INSERT INTO shot_embeddings(shot_id, text, model, dimension, vector) "
        "VALUES (1, 'legacy text', 'test-model', 4, ?)",
        (b"\x00" * 16,),
    )
    connection.commit()
    connection.close()

    with MediaStore(legacy_db) as store:
        store.migrate()
        versions = store.connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [row[0] for row in versions] == [1, 2, 3, 4, 5, 6]
        assert {"content_hash", "embed_version"} <= _embedding_columns(
            store, "shot_embeddings"
        )
        assert {"content_hash", "embed_version"} <= _embedding_columns(
            store, "transcript_embeddings"
        )
        indexes = {
            row[1]
            for row in store.connection.execute(
                "PRAGMA index_list(shot_embeddings)"
            )
        }
        assert "shot_embeddings_hash_idx" in indexes
        # pre-006 rows carry no hash (NULL or '' default) so the next refresh sees a mismatch and re-embeds them
        legacy_row = store.list_shot_embeddings("test-model")[0]
        assert not legacy_row["content_hash"]
        assert legacy_row["embed_version"] == 0
        store.migrate()

    with MediaStore() as store:
        store.migrate()
        versions = store.connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [row[0] for row in versions] == [1, 2, 3, 4, 5, 6]
        assert {"content_hash", "embed_version"} <= _embedding_columns(
            store, "shot_embeddings"
        )
        assert {"content_hash", "embed_version"} <= _embedding_columns(
            store, "transcript_embeddings"
        )
        indexes = {
            row[1]
            for row in store.connection.execute(
                "PRAGMA index_list(transcript_embeddings)"
            )
        }
        assert "transcript_embeddings_hash_idx" in indexes


def test_crud_time_range_linkage_and_scene_selection() -> None:
    with MediaStore() as store:
        store.migrate()
        asset = store.add_media_asset(
            MediaAsset(path=Path("movie.mp4"), duration_seconds=120)
        )
        assert asset.id is not None
        assert store.get_media_asset(asset.id) == asset

        shot = store.add_shot(
            Shot(
                media_asset_id=asset.id,
                start_seconds=10,
                end_seconds=20,
                label="arrival",
            )
        )
        overlapping = store.add_transcript_segment(
            TranscriptSegment(
                media_asset_id=asset.id,
                start_seconds=18,
                end_seconds=22,
                text="We have arrived.",
                speaker="Guide",
            )
        )
        store.add_transcript_segment(
            TranscriptSegment(
                media_asset_id=asset.id,
                start_seconds=20,
                end_seconds=24,
                text="Boundary does not overlap.",
            )
        )

        assert shot.id is not None
        assert overlapping.id is not None
        assert store.transcript_for_shot(shot.id) == [overlapping]

        selection = store.add_scene_selection(
            SceneSelection(
                media_asset_id=asset.id,
                shot_id=shot.id,
                transcript_segment_id=overlapping.id,
                position=0,
                rationale="Opening beat",
            )
        )
        assert selection.id is not None


def test_fts_transcript_search_can_be_scoped_to_media() -> None:
    with MediaStore() as store:
        store.migrate()
        first = store.add_media_asset(
            MediaAsset(path=Path("first.mp4"), duration_seconds=60)
        )
        second = store.add_media_asset(
            MediaAsset(path=Path("second.mp4"), duration_seconds=60)
        )
        assert first.id is not None and second.id is not None

        wanted = store.add_transcript_segment(
            TranscriptSegment(
                media_asset_id=first.id,
                start_seconds=0,
                end_seconds=2,
                text="A mysterious lighthouse appears.",
            )
        )
        store.add_transcript_segment(
            TranscriptSegment(
                media_asset_id=second.id,
                start_seconds=0,
                end_seconds=2,
                text="Another lighthouse appears.",
            )
        )

        assert store.search_transcript("lighthouse", first.id) == [wanted]
        assert store.search_transcript("mysterious lighthouse", first.id) == [wanted]
        assert store.search_transcript("mysterious", second.id) == []


def test_scene_search_is_case_insensitive_and_treats_wildcards_literally() -> None:
    with MediaStore() as store:
        store.migrate()
        asset = store.add_media_asset(MediaAsset(path=Path("movie.mp4"), duration_seconds=60))
        assert asset.id is not None
        wanted = store.add_shot(Shot(media_asset_id=asset.id, start_seconds=2, end_seconds=4, label="Arrival at 100% Power"))
        store.add_shot(Shot(media_asset_id=asset.id, start_seconds=5, end_seconds=7, label="Departure"))

        assert store.search_shots("ARRIVAL", asset.id) == [wanted]
        assert store.search_shots("100%", asset.id) == [wanted]


def test_visual_observations_are_persistent_and_searchable() -> None:
    with MediaStore() as store:
        store.migrate()
        asset = store.add_media_asset(
            MediaAsset(path=Path("movie.mp4"), duration_seconds=60)
        )
        shot = store.add_shot(
            Shot(
                media_asset_id=asset.id,
                start_seconds=4,
                end_seconds=8,
                label="Silent street",
            )
        )
        assert shot.id is not None
        store.replace_visual_observations([
            VisualObservation(
                shot_id=shot.id,
                description="A woman in a red coat runs across a rainy street",
                tags=["red coat", "rain", "street"],
                people=["woman"],
                actions=["running"],
            )
        ])

        listing = store.list_visual_observations()
        assert listing[0]["description"].startswith("A woman")
        assert listing[0]["tags"] == ["red coat", "rain", "street"]
        assert [item["shot_id"] for item in store.search_visual_observations("rain")] == [shot.id]
        assert [item["shot_id"] for item in store.search_visual_observations("woman")] == [shot.id]
        assert store.search_visual_observations("lighthouse") == []


def test_replacing_visual_memory_invalidates_stale_shot_embeddings() -> None:
    with MediaStore() as store:
        store.migrate()
        asset = store.add_media_asset(MediaAsset(path=Path("movie.mp4"), duration_seconds=20))
        shot = store.add_shot(Shot(
            media_asset_id=asset.id, start_seconds=0, end_seconds=10, label="old label"
        ))
        assert shot.id is not None
        store.replace_shot_embeddings([
            (shot.id, "old visual", "test-model", 4, b"\x00" * 16)
        ])
        assert store.list_shot_embeddings("test-model")

        store.replace_visual_observations([
            VisualObservation(shot_id=shot.id, description="new visual")
        ])

        assert store.list_shot_embeddings("test-model") == []


def test_semantic_embeddings_and_person_tracks_are_persisted() -> None:
    with MediaStore() as store:
        store.migrate()
        asset = store.add_media_asset(MediaAsset(path=Path("movie.mp4"), duration_seconds=30))
        shot = store.add_shot(Shot(media_asset_id=asset.id, start_seconds=0, end_seconds=10, label="street"))
        segment = store.add_transcript_segment(TranscriptSegment(
            media_asset_id=asset.id, start_seconds=1, end_seconds=2, text="hello"
        ))
        assert shot.id is not None and segment.id is not None
        store.replace_person_tracks(
            [{"label": "Person 1", "description": "red coat", "source": "agy"}],
            [{"label": "Person 1", "shot_id": shot.id, "confidence": 0.9, "evidence": "same red coat"}],
        )
        vector = b"\x00" * 16
        store.replace_shot_embeddings([(shot.id, "rainy street", "test-model", 4, vector)])
        store.replace_transcript_embeddings([(segment.id, "hello", "test-model", 4, vector)])

        assert store.list_shot_embeddings("test-model")[0]["text"] == "rainy street"
        assert store.list_transcript_embeddings("test-model")[0]["text"] == "hello"
        assert store.person_labels_for_shot(shot.id) == ["Person 1"]
        tracks = store.list_person_tracks()
        assert tracks[0]["label"] == "Person 1"
        assert tracks[0]["appearances"][0]["confidence"] == pytest.approx(0.9)


def test_story_graph_is_normalized_and_queryable_by_scene() -> None:
    with MediaStore() as store:
        store.migrate()
        asset = store.add_media_asset(MediaAsset(path=Path("movie.mp4"), duration_seconds=30))
        shot = store.add_shot(Shot(
            media_asset_id=asset.id, start_seconds=0, end_seconds=10, label="arrival"
        ))
        assert shot.id is not None
        store.replace_story_graph(
            [
                {"type": "person", "label": "Person 1", "description": "red coat", "source": "agy"},
                {"type": "location", "label": "Hospital", "description": "hospital corridor", "source": "agy"},
                {"type": "event", "label": "Arrival", "description": "arrives at hospital", "source": "agy"},
            ],
            [
                {"type": "person", "label": "Person 1", "shot_id": shot.id, "confidence": 0.9, "evidence": "visible"},
                {"type": "location", "label": "Hospital", "shot_id": shot.id, "confidence": 0.95, "evidence": "signage"},
                {"type": "event", "label": "Arrival", "shot_id": shot.id, "confidence": 0.85, "evidence": "enters corridor"},
            ],
            [
                {
                    "subject_type": "person",
                    "subject_label": "Person 1",
                    "predicate": "arrives_at",
                    "object_type": "location",
                    "object_label": "Hospital",
                    "shot_id": shot.id,
                    "confidence": 0.88,
                    "evidence": "Person 1 enters hospital corridor",
                }
            ],
        )

        graph = store.list_story_graph()
        assert {item["label"] for item in graph["entities"]} == {"Person 1", "Hospital", "Arrival"}
        assert len(graph["scene_entities"]) == 3
        assert graph["relations"][0]["subject_label"] == "Person 1"
        assert graph["relations"][0]["predicate"] == "arrives_at"
        assert graph["relations"][0]["object_label"] == "Hospital"


def test_person_continuity_v2_tracks_alias_traits_and_ambiguity() -> None:
    with MediaStore() as store:
        store.migrate()
        asset = store.add_media_asset(MediaAsset(path=Path("movie.mp4"), duration_seconds=20))
        first = store.add_shot(Shot(media_asset_id=asset.id, start_seconds=0, end_seconds=10, label="A"))
        second = store.add_shot(Shot(media_asset_id=asset.id, start_seconds=10, end_seconds=20, label="B"))
        assert first.id and second.id
        store.replace_person_tracks(
            [{"label": "Person 1", "description": "dark hair", "appearance_summary": "dark hair", "source": "agy"}],
            [
                {"label": "Person 1", "shot_id": first.id, "description": "dark hair", "clothing": "red coat", "confidence": 0.94, "evidence": "same hair and coat"},
                {"label": "Person 1", "shot_id": second.id, "description": "dark hair", "clothing": "blue jacket", "confidence": 0.61, "ambiguous": True, "evidence": "partly obscured"},
            ],
        )
        track = store.list_person_tracks()[0]
        assert track["ambiguous"] is True
        assert track["mean_confidence"] == pytest.approx((0.94 + 0.61) / 2)
        assert track["continuity_confidence"] < track["mean_confidence"]
        assert {item["value"] for item in track["traits"] if item["trait_type"] == "clothing"} == {"red coat", "blue jacket"}
        updated = store.set_person_alias("Person 1", "Nam")
        assert updated["alias"] == "Nam"
        assert store.person_labels_for_shot(first.id) == ["Nam"]
        store.replace_person_tracks(
            [{"label": "Person 1", "description": "dark hair", "source": "agy"}],
            [{"label": "Person 1", "shot_id": first.id, "confidence": 0.96, "evidence": "new pass"}],
        )
        assert store.list_person_tracks()[0]["alias"] == "Nam"


def test_list_transcript_returns_every_segment_in_time_order() -> None:
    with MediaStore() as store:
        store.migrate()
        asset = store.add_media_asset(
            MediaAsset(path=Path("movie.mp4"), duration_seconds=60)
        )
        assert asset.id is not None
        store.add_transcript_segment(
            TranscriptSegment(
                media_asset_id=asset.id, start_seconds=5, end_seconds=6, text="second"
            )
        )
        store.add_transcript_segment(
            TranscriptSegment(
                media_asset_id=asset.id,
                start_seconds=1,
                end_seconds=2,
                text="first",
                speaker="Narrator",
            )
        )

        segments = store.list_transcript(asset.id)
        assert [segment.text for segment in segments] == ["first", "second"]
        assert segments[0].speaker == "Narrator"
