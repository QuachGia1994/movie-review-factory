CREATE TABLE media_assets (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    duration_seconds REAL NOT NULL CHECK (duration_seconds > 0)
) STRICT;

CREATE TABLE shots (
    id INTEGER PRIMARY KEY,
    media_asset_id INTEGER NOT NULL REFERENCES media_assets(id) ON DELETE CASCADE,
    start_seconds REAL NOT NULL CHECK (start_seconds >= 0),
    end_seconds REAL NOT NULL CHECK (end_seconds > start_seconds),
    label TEXT
) STRICT;

CREATE INDEX shots_media_time_idx
    ON shots (media_asset_id, start_seconds, end_seconds);

CREATE TABLE transcript_segments (
    id INTEGER PRIMARY KEY,
    media_asset_id INTEGER NOT NULL REFERENCES media_assets(id) ON DELETE CASCADE,
    start_seconds REAL NOT NULL CHECK (start_seconds >= 0),
    end_seconds REAL NOT NULL CHECK (end_seconds > start_seconds),
    text TEXT NOT NULL CHECK (length(trim(text)) > 0),
    speaker TEXT
) STRICT;

CREATE INDEX transcript_segments_media_time_idx
    ON transcript_segments (media_asset_id, start_seconds, end_seconds);

CREATE TABLE scene_selections (
    id INTEGER PRIMARY KEY,
    media_asset_id INTEGER NOT NULL REFERENCES media_assets(id) ON DELETE CASCADE,
    shot_id INTEGER NOT NULL REFERENCES shots(id) ON DELETE CASCADE,
    transcript_segment_id INTEGER REFERENCES transcript_segments(id) ON DELETE SET NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    rationale TEXT,
    UNIQUE (media_asset_id, position)
) STRICT;

CREATE INDEX scene_selections_shot_idx ON scene_selections (shot_id);
CREATE INDEX scene_selections_transcript_idx
    ON scene_selections (transcript_segment_id)
    WHERE transcript_segment_id IS NOT NULL;

CREATE VIRTUAL TABLE transcript_segments_fts USING fts5(
    text,
    content='transcript_segments',
    content_rowid='id'
);

CREATE TRIGGER transcript_segments_fts_insert AFTER INSERT ON transcript_segments BEGIN
    INSERT INTO transcript_segments_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER transcript_segments_fts_delete AFTER DELETE ON transcript_segments BEGIN
    INSERT INTO transcript_segments_fts(transcript_segments_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER transcript_segments_fts_update AFTER UPDATE OF text ON transcript_segments BEGIN
    INSERT INTO transcript_segments_fts(transcript_segments_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
    INSERT INTO transcript_segments_fts(rowid, text) VALUES (new.id, new.text);
END;
