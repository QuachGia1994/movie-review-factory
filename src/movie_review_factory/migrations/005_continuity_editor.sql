ALTER TABLE person_tracks ADD COLUMN alias TEXT NOT NULL DEFAULT '';
ALTER TABLE person_tracks ADD COLUMN appearance_summary TEXT NOT NULL DEFAULT '';
ALTER TABLE person_tracks ADD COLUMN mean_confidence REAL NOT NULL DEFAULT 0
    CHECK (mean_confidence >= 0 AND mean_confidence <= 1);
ALTER TABLE person_tracks ADD COLUMN ambiguous INTEGER NOT NULL DEFAULT 0
    CHECK (ambiguous IN (0, 1));

ALTER TABLE person_appearances ADD COLUMN clothing TEXT NOT NULL DEFAULT '';
ALTER TABLE person_appearances ADD COLUMN ambiguous INTEGER NOT NULL DEFAULT 0
    CHECK (ambiguous IN (0, 1));

CREATE TABLE person_traits (
    id INTEGER PRIMARY KEY,
    person_track_id INTEGER NOT NULL REFERENCES person_tracks(id) ON DELETE CASCADE,
    trait_type TEXT NOT NULL CHECK (trait_type IN ('appearance', 'clothing')),
    value TEXT NOT NULL CHECK (length(trim(value)) > 0),
    first_shot_id INTEGER REFERENCES shots(id) ON DELETE SET NULL,
    last_shot_id INTEGER REFERENCES shots(id) ON DELETE SET NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    UNIQUE(person_track_id, trait_type, value)
) STRICT;

CREATE INDEX person_traits_track_idx ON person_traits(person_track_id, trait_type);
