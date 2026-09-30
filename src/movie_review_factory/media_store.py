from __future__ import annotations

import json
import sqlite3
from importlib.resources import files
from pathlib import Path
from typing import Iterable, Iterator

from .models import MediaAsset, SceneSelection, Shot, TranscriptSegment, VisualObservation

_MIGRATION_PACKAGE = "movie_review_factory.migrations"


class MediaStore:
    """Small SQLite persistence boundary for media intelligence records."""

    def __init__(self, database: str | Path = ":memory:") -> None:
        self.connection = sqlite3.connect(database)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "MediaStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def migrate(self) -> None:
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        applied = {
            row[0]
            for row in self.connection.execute("SELECT version FROM schema_migrations")
        }
        migration_dir = files(_MIGRATION_PACKAGE)
        migrations: Iterator[tuple[int, str]] = (
            (int(item.name.split("_", 1)[0]), item.read_text(encoding="utf-8"))
            for item in sorted(migration_dir.iterdir(), key=lambda item: item.name)
            if item.name.endswith(".sql")
        )
        for version, sql in migrations:
            if version in applied:
                continue
            script = (
                "BEGIN IMMEDIATE;\n"
                + sql
                + f"\nINSERT INTO schema_migrations(version) VALUES ({version});\nCOMMIT;"
            )
            try:
                self.connection.executescript(script)
            except Exception:
                if self.connection.in_transaction:
                    self.connection.rollback()
                raise

    def add_media_asset(self, asset: MediaAsset) -> MediaAsset:
        cursor = self.connection.execute(
            "INSERT INTO media_assets(path, duration_seconds) VALUES (?, ?)",
            (str(asset.path), asset.duration_seconds),
        )
        self.connection.commit()
        return asset.model_copy(update={"id": cursor.lastrowid})

    def get_media_asset(self, asset_id: int) -> MediaAsset | None:
        row = self.connection.execute(
            "SELECT id, path, duration_seconds FROM media_assets WHERE id = ?",
            (asset_id,),
        ).fetchone()
        return MediaAsset.model_validate(dict(row)) if row else None

    def add_shot(self, shot: Shot) -> Shot:
        cursor = self.connection.execute(
            "INSERT INTO shots(media_asset_id, start_seconds, end_seconds, label) "
            "VALUES (?, ?, ?, ?)",
            (shot.media_asset_id, shot.start_seconds, shot.end_seconds, shot.label),
        )
        self.connection.commit()
        return shot.model_copy(update={"id": cursor.lastrowid})

    def add_transcript_segment(self, segment: TranscriptSegment) -> TranscriptSegment:
        cursor = self.connection.execute(
            "INSERT INTO transcript_segments("
            "media_asset_id, start_seconds, end_seconds, text, speaker, words"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                segment.media_asset_id,
                segment.start_seconds,
                segment.end_seconds,
                segment.text,
                segment.speaker,
                json.dumps([word.model_dump() for word in segment.words]),
            ),
        )
        self.connection.commit()
        return segment.model_copy(update={"id": cursor.lastrowid})

    def add_scene_selection(self, selection: SceneSelection) -> SceneSelection:
        cursor = self.connection.execute(
            "INSERT INTO scene_selections("
            "media_asset_id, shot_id, transcript_segment_id, position, rationale"
            ") VALUES (?, ?, ?, ?, ?)",
            (
                selection.media_asset_id,
                selection.shot_id,
                selection.transcript_segment_id,
                selection.position,
                selection.rationale,
            ),
        )
        self.connection.commit()
        return selection.model_copy(update={"id": cursor.lastrowid})

    def replace_index(
        self,
        asset: MediaAsset,
        shots: list[Shot],
        segments: list[TranscriptSegment],
    ) -> MediaAsset:
        with self.connection:
            self.connection.execute("DELETE FROM media_assets")
            cursor = self.connection.execute(
                "INSERT INTO media_assets(path, duration_seconds) VALUES (?, ?)",
                (str(asset.path), asset.duration_seconds),
            )
            asset_id = cursor.lastrowid
            self.connection.executemany(
                "INSERT INTO shots(media_asset_id, start_seconds, end_seconds, label) VALUES (?, ?, ?, ?)",
                ((asset_id, shot.start_seconds, shot.end_seconds, shot.label) for shot in shots),
            )
            self.connection.executemany(
                "INSERT INTO transcript_segments(media_asset_id, start_seconds, end_seconds, text, speaker, words) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    (
                        asset_id, segment.start_seconds, segment.end_seconds, segment.text,
                        segment.speaker, json.dumps([word.model_dump() for word in segment.words]),
                    )
                    for segment in segments
                ),
            )
        return asset.model_copy(update={"id": asset_id})

    def list_shots(self, media_asset_id: int | None = None) -> list[Shot]:
        sql = "SELECT id, media_asset_id, start_seconds, end_seconds, label FROM shots"
        params: tuple[object, ...] = ()
        if media_asset_id is not None:
            sql += " WHERE media_asset_id = ?"
            params = (media_asset_id,)
        sql += " ORDER BY start_seconds, id"
        rows = self.connection.execute(sql, params)
        return [Shot.model_validate(dict(row)) for row in rows]

    def search_shots(self, query: str, media_asset_id: int | None = None) -> list[Shot]:
        sql = (
            "SELECT id, media_asset_id, start_seconds, end_seconds, label FROM shots "
            "WHERE label LIKE ? ESCAPE '\\' COLLATE NOCASE"
        )
        escaped_query = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params: list[object] = [f"%{escaped_query}%"]
        if media_asset_id is not None:
            sql += " AND media_asset_id = ?"
            params.append(media_asset_id)
        sql += " ORDER BY start_seconds, id"
        rows = self.connection.execute(sql, params)
        return [Shot.model_validate(dict(row)) for row in rows]

    def get_shot(self, shot_id: int) -> Shot | None:
        row = self.connection.execute(
            "SELECT id, media_asset_id, start_seconds, end_seconds, label FROM shots WHERE id = ?",
            (shot_id,),
        ).fetchone()
        return Shot.model_validate(dict(row)) if row else None

    def replace_visual_observations(
        self, observations: list[VisualObservation]
    ) -> list[VisualObservation]:
        with self.connection:
            self.connection.execute("DELETE FROM shot_embeddings")
            self.connection.execute("DELETE FROM visual_observations")
            self.connection.executemany(
                "INSERT INTO visual_observations("
                "shot_id, description, tags, people, actions, source"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    (
                        item.shot_id,
                        item.description.strip(),
                        json.dumps(item.tags, ensure_ascii=False),
                        json.dumps(item.people, ensure_ascii=False),
                        json.dumps(item.actions, ensure_ascii=False),
                        item.source,
                    )
                    for item in observations
                ),
            )
        return observations

    @staticmethod
    def _visual_row(row: sqlite3.Row) -> dict:
        item = dict(row)
        for key in ("tags", "people", "actions"):
            try:
                value = json.loads(item[key])
            except (TypeError, json.JSONDecodeError):
                value = []
            item[key] = value if isinstance(value, list) else []
        return item

    def list_visual_observations(self) -> list[dict]:
        rows = self.connection.execute(
            "SELECT v.shot_id, v.description, v.tags, v.people, v.actions, v.source, "
            "s.start_seconds, s.end_seconds, s.label "
            "FROM visual_observations AS v "
            "JOIN shots AS s ON s.id = v.shot_id "
            "ORDER BY s.start_seconds, s.id"
        )
        return [self._visual_row(row) for row in rows]

    def search_visual_observations(self, query: str) -> list[dict]:
        phrase = '"' + query.replace('"', '""') + '"'
        rows = self.connection.execute(
            "SELECT v.shot_id, v.description, v.tags, v.people, v.actions, v.source, "
            "s.start_seconds, s.end_seconds, s.label "
            "FROM visual_observations_fts AS f "
            "JOIN visual_observations AS v ON v.shot_id = f.rowid "
            "JOIN shots AS s ON s.id = v.shot_id "
            "WHERE visual_observations_fts MATCH ? "
            "ORDER BY rank, s.start_seconds, s.id",
            (phrase,),
        )
        return [self._visual_row(row) for row in rows]

    def replace_shot_embeddings(
        self, rows: list[tuple[int, str, str, int, bytes]]
    ) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM shot_embeddings")
            self.connection.executemany(
                "INSERT INTO shot_embeddings(shot_id, text, model, dimension, vector) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def replace_transcript_embeddings(
        self, rows: list[tuple[int, str, str, int, bytes]]
    ) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM transcript_embeddings")
            self.connection.executemany(
                "INSERT INTO transcript_embeddings("
                "transcript_segment_id, text, model, dimension, vector"
                ") VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    @staticmethod
    def _sync_embeddings(
        connection: sqlite3.Connection,
        table: str,
        key_column: str,
        present_ids: Iterable[int],
        rows: list[tuple[int, str, str, int, bytes, str, int]],
    ) -> None:
        """Drop stale records and upsert only the rows that were re-embedded."""
        ids = [int(record_id) for record_id in present_ids]
        with connection:
            if ids:
                placeholders = ",".join("?" for _ in ids)
                connection.execute(
                    f"DELETE FROM {table} WHERE {key_column} NOT IN ({placeholders})",
                    tuple(ids),
                )
            else:
                connection.execute(f"DELETE FROM {table}")
            connection.executemany(
                f"INSERT INTO {table}("
                f"{key_column}, text, model, dimension, vector, content_hash, embed_version"
                ") VALUES (?, ?, ?, ?, ?, ?, ?) "
                f"ON CONFLICT({key_column}) DO UPDATE SET "
                "text = excluded.text, model = excluded.model, "
                "dimension = excluded.dimension, vector = excluded.vector, "
                "content_hash = excluded.content_hash, "
                "embed_version = excluded.embed_version, "
                "updated_at = CURRENT_TIMESTAMP",
                rows,
            )

    def sync_shot_embeddings(
        self,
        present_ids: Iterable[int],
        rows: list[tuple[int, str, str, int, bytes, str, int]],
    ) -> None:
        self._sync_embeddings(
            self.connection, "shot_embeddings", "shot_id", present_ids, rows
        )

    def sync_transcript_embeddings(
        self,
        present_ids: Iterable[int],
        rows: list[tuple[int, str, str, int, bytes, str, int]],
    ) -> None:
        self._sync_embeddings(
            self.connection,
            "transcript_embeddings",
            "transcript_segment_id",
            present_ids,
            rows,
        )

    def list_shot_embeddings(self, model: str | None = None) -> list[dict]:
        sql = (
            "SELECT e.shot_id, e.text, e.model, e.dimension, e.vector, "
            "e.content_hash, e.embed_version, "
            "s.start_seconds, s.end_seconds, s.label "
            "FROM shot_embeddings AS e JOIN shots AS s ON s.id = e.shot_id"
        )
        params: tuple[object, ...] = ()
        if model is not None:
            sql += " WHERE e.model = ?"
            params = (model,)
        sql += " ORDER BY s.start_seconds, s.id"
        return [dict(row) for row in self.connection.execute(sql, params)]

    def list_transcript_embeddings(self, model: str | None = None) -> list[dict]:
        sql = (
            "SELECT e.transcript_segment_id, e.text, e.model, e.dimension, e.vector, "
            "e.content_hash, e.embed_version, "
            "t.start_seconds, t.end_seconds, t.speaker "
            "FROM transcript_embeddings AS e "
            "JOIN transcript_segments AS t ON t.id = e.transcript_segment_id"
        )
        params: tuple[object, ...] = ()
        if model is not None:
            sql += " WHERE e.model = ?"
            params = (model,)
        sql += " ORDER BY t.start_seconds, t.id"
        return [dict(row) for row in self.connection.execute(sql, params)]

    def replace_person_tracks(
        self,
        tracks: list[dict],
        appearances: list[dict],
    ) -> None:
        existing_aliases = {
            str(row["label"]): str(row["alias"] or "")
            for row in self.connection.execute(
                "SELECT label, alias FROM person_tracks"
            )
        }
        with self.connection:
            self.connection.execute("DELETE FROM shot_embeddings")
            self.connection.execute("DELETE FROM person_tracks")
            track_ids: dict[str, int] = {}
            by_label: dict[str, list[dict]] = {}
            for item in appearances:
                by_label.setdefault(str(item.get("label") or ""), []).append(item)
            for track in tracks:
                label = str(track["label"])
                track_appearances = by_label.get(label, [])
                confidences = [float(item.get("confidence") or 0.0) for item in track_appearances]
                mean_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0.0
                )
                ambiguous = bool(track.get("ambiguous")) or any(
                    bool(item.get("ambiguous")) or float(item.get("confidence") or 0.0) < 0.65
                    for item in track_appearances
                )
                cursor = self.connection.execute(
                    "INSERT INTO person_tracks("
                    "label, description, source, alias, appearance_summary, mean_confidence, ambiguous"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        label,
                        str(track.get("description") or ""),
                        str(track.get("source") or "agy"),
                        str(track.get("alias") or existing_aliases.get(label, "")),
                        str(track.get("appearance_summary") or track.get("description") or ""),
                        mean_confidence,
                        int(ambiguous),
                    ),
                )
                track_ids[label] = int(cursor.lastrowid)
            self.connection.executemany(
                "INSERT INTO person_appearances("
                "person_track_id, shot_id, confidence, evidence, clothing, ambiguous"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    (
                        track_ids[str(item["label"])],
                        int(item["shot_id"]),
                        float(item.get("confidence") or 0.0),
                        str(item.get("evidence") or ""),
                        str(item.get("clothing") or ""),
                        int(bool(item.get("ambiguous"))),
                    )
                    for item in appearances
                    if str(item.get("label") or "") in track_ids
                ),
            )
            for label, items in by_label.items():
                track_id = track_ids.get(label)
                if not track_id:
                    continue
                descriptions = [
                    str(item.get("description") or "").strip()
                    for item in items
                    if str(item.get("description") or "").strip()
                ]
                clothing = [
                    str(item.get("clothing") or "").strip()
                    for item in items
                    if str(item.get("clothing") or "").strip()
                ]
                for trait_type, values in (("appearance", descriptions), ("clothing", clothing)):
                    seen: set[str] = set()
                    for value in values:
                        key = value.casefold()
                        if key in seen:
                            continue
                        seen.add(key)
                        matching = [item for item in items if str(item.get(
                            "description" if trait_type == "appearance" else "clothing"
                        ) or "").strip().casefold() == key]
                        shot_ids = [int(item["shot_id"]) for item in matching]
                        confidence = sum(float(item.get("confidence") or 0.0) for item in matching) / len(matching)
                        self.connection.execute(
                            "INSERT INTO person_traits("
                            "person_track_id, trait_type, value, first_shot_id, last_shot_id, confidence"
                            ") VALUES (?, ?, ?, ?, ?, ?)",
                            (track_id, trait_type, value, min(shot_ids), max(shot_ids), confidence),
                        )

    def list_person_tracks(self) -> list[dict]:
        tracks = [
            dict(row)
            for row in self.connection.execute(
                "SELECT id, label, alias, description, appearance_summary, "
                "mean_confidence, ambiguous, source "
                "FROM person_tracks ORDER BY id"
            )
        ]
        appearances = [
            dict(row)
            for row in self.connection.execute(
                "SELECT p.label, a.shot_id, a.confidence, a.evidence, "
                "a.clothing, a.ambiguous, s.start_seconds, s.end_seconds "
                "FROM person_appearances AS a "
                "JOIN person_tracks AS p ON p.id = a.person_track_id "
                "JOIN shots AS s ON s.id = a.shot_id "
                "ORDER BY p.id, s.start_seconds, s.id"
            )
        ]
        traits = [
            dict(row)
            for row in self.connection.execute(
                "SELECT p.label, t.trait_type, t.value, t.first_shot_id, "
                "t.last_shot_id, t.confidence FROM person_traits AS t "
                "JOIN person_tracks AS p ON p.id = t.person_track_id "
                "ORDER BY p.id, t.trait_type, t.id"
            )
        ]
        by_label: dict[str, list[dict]] = {}
        for item in appearances:
            by_label.setdefault(str(item["label"]), []).append(item)
        traits_by_label: dict[str, list[dict]] = {}
        for item in traits:
            traits_by_label.setdefault(str(item["label"]), []).append(item)
        for track in tracks:
            track["ambiguous"] = bool(track["ambiguous"])
            track["appearances"] = by_label.get(str(track["label"]), [])
            for appearance in track["appearances"]:
                appearance["ambiguous"] = bool(appearance["ambiguous"])
            weighted = [
                (
                    float(appearance.get("confidence") or 0.0),
                    0.85 ** (len(track["appearances"]) - index - 1),
                )
                for index, appearance in enumerate(track["appearances"])
            ]
            weight_sum = sum(weight for _, weight in weighted)
            track["continuity_confidence"] = (
                sum(confidence * weight for confidence, weight in weighted) / weight_sum
                if weight_sum else 0.0
            )
            track["traits"] = traits_by_label.get(str(track["label"]), [])
        return tracks

    def set_person_alias(self, label: str, alias: str) -> dict:
        alias = " ".join(alias.split()).strip()
        if len(alias) > 120:
            raise ValueError("person alias is too long")
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE person_tracks SET alias = ? WHERE label = ?",
                (alias, label),
            )
        if cursor.rowcount != 1:
            raise KeyError(label)
        return next(item for item in self.list_person_tracks() if item["label"] == label)

    def person_labels_for_shot(self, shot_id: int) -> list[str]:
        rows = self.connection.execute(
            "SELECT p.label, p.alias FROM person_appearances AS a "
            "JOIN person_tracks AS p ON p.id = a.person_track_id "
            "WHERE a.shot_id = ? ORDER BY p.id",
            (shot_id,),
        )
        return [
            str(row["alias"] or row["label"])
            for row in rows
        ]

    def replace_story_graph(
        self,
        entities: list[dict],
        scene_entities: list[dict],
        relations: list[dict],
    ) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM story_relations")
            self.connection.execute("DELETE FROM story_scene_entities")
            self.connection.execute("DELETE FROM story_entities")
            entity_ids: dict[tuple[str, str], int] = {}
            for entity in entities:
                key = (str(entity["type"]), str(entity["label"]))
                cursor = self.connection.execute(
                    "INSERT INTO story_entities(entity_type, label, description, source) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        key[0],
                        key[1],
                        str(entity.get("description") or ""),
                        str(entity.get("source") or "agy"),
                    ),
                )
                entity_ids[key] = int(cursor.lastrowid)

            self.connection.executemany(
                "INSERT INTO story_scene_entities(entity_id, shot_id, confidence, evidence) "
                "VALUES (?, ?, ?, ?)",
                (
                    (
                        entity_ids[(str(item["type"]), str(item["label"]))],
                        int(item["shot_id"]),
                        float(item.get("confidence") or 0.0),
                        str(item.get("evidence") or ""),
                    )
                    for item in scene_entities
                    if (str(item.get("type")), str(item.get("label"))) in entity_ids
                ),
            )

            self.connection.executemany(
                "INSERT INTO story_relations("
                "subject_entity_id, predicate, object_entity_id, shot_id, confidence, evidence"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    (
                        entity_ids[(str(item["subject_type"]), str(item["subject_label"]))],
                        str(item["predicate"]),
                        entity_ids[(str(item["object_type"]), str(item["object_label"]))],
                        int(item["shot_id"]),
                        float(item.get("confidence") or 0.0),
                        str(item.get("evidence") or ""),
                    )
                    for item in relations
                    if (
                        (str(item.get("subject_type")), str(item.get("subject_label"))) in entity_ids
                        and (str(item.get("object_type")), str(item.get("object_label"))) in entity_ids
                    )
                ),
            )

    def list_story_graph(self) -> dict:
        entities = [
            dict(row)
            for row in self.connection.execute(
                "SELECT id, entity_type AS type, label, description, source "
                "FROM story_entities ORDER BY entity_type, label"
            )
        ]
        scene_entities = [
            dict(row)
            for row in self.connection.execute(
                "SELECT e.entity_type AS type, e.label, se.shot_id, se.confidence, se.evidence, "
                "s.start_seconds, s.end_seconds "
                "FROM story_scene_entities AS se "
                "JOIN story_entities AS e ON e.id = se.entity_id "
                "JOIN shots AS s ON s.id = se.shot_id "
                "ORDER BY s.start_seconds, e.entity_type, e.label"
            )
        ]
        relations = [
            dict(row)
            for row in self.connection.execute(
                "SELECT se.entity_type AS subject_type, se.label AS subject_label, "
                "r.predicate, oe.entity_type AS object_type, oe.label AS object_label, "
                "r.shot_id, r.confidence, r.evidence, s.start_seconds, s.end_seconds "
                "FROM story_relations AS r "
                "JOIN story_entities AS se ON se.id = r.subject_entity_id "
                "JOIN story_entities AS oe ON oe.id = r.object_entity_id "
                "JOIN shots AS s ON s.id = r.shot_id "
                "ORDER BY s.start_seconds, r.id"
            )
        ]
        return {
            "entities": entities,
            "scene_entities": scene_entities,
            "relations": relations,
        }

    @staticmethod
    def _segment_from_row(row: sqlite3.Row) -> TranscriptSegment:
        """Build a TranscriptSegment from a row, decoding the JSON words column."""
        data = dict(row)
        raw = data.get("words")
        if isinstance(raw, str):
            data["words"] = json.loads(raw) if raw else []
        return TranscriptSegment.model_validate(data)

    def transcript_for_shot(self, shot_id: int) -> list[TranscriptSegment]:
        rows = self.connection.execute(
            "SELECT t.id, t.media_asset_id, t.start_seconds, t.end_seconds, "
            "t.text, t.speaker, t.words "
            "FROM shots AS s JOIN transcript_segments AS t "
            "ON t.media_asset_id = s.media_asset_id "
            "AND t.start_seconds < s.end_seconds "
            "AND t.end_seconds > s.start_seconds "
            "WHERE s.id = ? ORDER BY t.start_seconds, t.id",
            (shot_id,),
        )
        return [self._segment_from_row(row) for row in rows]

    def get_transcript_segment(self, segment_id: int) -> TranscriptSegment | None:
        row = self.connection.execute(
            "SELECT id, media_asset_id, start_seconds, end_seconds, text, speaker, words "
            "FROM transcript_segments WHERE id = ?",
            (segment_id,),
        ).fetchone()
        return self._segment_from_row(row) if row else None

    def list_transcript(
        self, media_asset_id: int | None = None
    ) -> list[TranscriptSegment]:
        sql = (
            "SELECT id, media_asset_id, start_seconds, end_seconds, text, speaker, words "
            "FROM transcript_segments"
        )
        params: tuple[object, ...] = ()
        if media_asset_id is not None:
            sql += " WHERE media_asset_id = ?"
            params = (media_asset_id,)
        sql += " ORDER BY start_seconds, id"
        rows = self.connection.execute(sql, params)
        return [self._segment_from_row(row) for row in rows]

    def search_transcript(
        self, query: str, media_asset_id: int | None = None
    ) -> list[TranscriptSegment]:
        sql = (
            "SELECT t.id, t.media_asset_id, t.start_seconds, t.end_seconds, "
            "t.text, t.speaker, t.words FROM transcript_segments_fts AS f "
            "JOIN transcript_segments AS t ON t.id = f.rowid "
            "WHERE transcript_segments_fts MATCH ?"
        )
        phrase = '"' + query.replace('"', '""') + '"'
        params: list[object] = [phrase]
        if media_asset_id is not None:
            sql += " AND t.media_asset_id = ?"
            params.append(media_asset_id)
        sql += " ORDER BY rank, t.start_seconds, t.id"
        rows = self.connection.execute(sql, params)
        return [self._segment_from_row(row) for row in rows]
