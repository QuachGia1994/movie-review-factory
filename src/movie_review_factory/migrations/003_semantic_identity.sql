CREATE TABLE shot_embeddings (
    shot_id INTEGER PRIMARY KEY REFERENCES shots(id) ON DELETE CASCADE,
    text TEXT NOT NULL CHECK (length(trim(text)) > 0),
    model TEXT NOT NULL CHECK (length(trim(model)) > 0),
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    vector BLOB NOT NULL CHECK (length(vector) > 0),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

CREATE INDEX shot_embeddings_model_idx
    ON shot_embeddings (model, dimension);

CREATE TABLE transcript_embeddings (
    transcript_segment_id INTEGER PRIMARY KEY
        REFERENCES transcript_segments(id) ON DELETE CASCADE,
    text TEXT NOT NULL CHECK (length(trim(text)) > 0),
    model TEXT NOT NULL CHECK (length(trim(model)) > 0),
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    vector BLOB NOT NULL CHECK (length(vector) > 0),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

CREATE INDEX transcript_embeddings_model_idx
    ON transcript_embeddings (model, dimension);

CREATE TABLE person_tracks (
    id INTEGER PRIMARY KEY,
    label TEXT NOT NULL UNIQUE CHECK (length(trim(label)) > 0),
    description TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'agy',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

CREATE TABLE person_appearances (
    person_track_id INTEGER NOT NULL REFERENCES person_tracks(id) ON DELETE CASCADE,
    shot_id INTEGER NOT NULL REFERENCES shots(id) ON DELETE CASCADE,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    evidence TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (person_track_id, shot_id)
) STRICT;

CREATE INDEX person_appearances_shot_idx
    ON person_appearances (shot_id);
