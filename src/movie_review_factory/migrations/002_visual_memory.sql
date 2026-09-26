CREATE TABLE visual_observations (
    shot_id INTEGER PRIMARY KEY REFERENCES shots(id) ON DELETE CASCADE,
    description TEXT NOT NULL CHECK (length(trim(description)) > 0),
    tags TEXT NOT NULL DEFAULT '',
    people TEXT NOT NULL DEFAULT '',
    actions TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'agy',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

CREATE VIRTUAL TABLE visual_observations_fts USING fts5(
    description,
    tags,
    people,
    actions,
    content='visual_observations',
    content_rowid='shot_id'
);

CREATE TRIGGER visual_observations_fts_insert AFTER INSERT ON visual_observations BEGIN
    INSERT INTO visual_observations_fts(rowid, description, tags, people, actions)
    VALUES (new.shot_id, new.description, new.tags, new.people, new.actions);
END;

CREATE TRIGGER visual_observations_fts_delete AFTER DELETE ON visual_observations BEGIN
    INSERT INTO visual_observations_fts(
        visual_observations_fts, rowid, description, tags, people, actions
    ) VALUES (
        'delete', old.shot_id, old.description, old.tags, old.people, old.actions
    );
END;

CREATE TRIGGER visual_observations_fts_update
AFTER UPDATE OF description, tags, people, actions ON visual_observations BEGIN
    INSERT INTO visual_observations_fts(
        visual_observations_fts, rowid, description, tags, people, actions
    ) VALUES (
        'delete', old.shot_id, old.description, old.tags, old.people, old.actions
    );
    INSERT INTO visual_observations_fts(rowid, description, tags, people, actions)
    VALUES (new.shot_id, new.description, new.tags, new.people, new.actions);
END;
