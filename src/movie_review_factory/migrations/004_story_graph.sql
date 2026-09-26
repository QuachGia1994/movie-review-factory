CREATE TABLE story_entities (
    id INTEGER PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK (
        entity_type IN ('person', 'location', 'event', 'object')
    ),
    label TEXT NOT NULL CHECK (length(trim(label)) > 0),
    description TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'agy',
    UNIQUE(entity_type, label)
) STRICT;

CREATE TABLE story_scene_entities (
    entity_id INTEGER NOT NULL REFERENCES story_entities(id) ON DELETE CASCADE,
    shot_id INTEGER NOT NULL REFERENCES shots(id) ON DELETE CASCADE,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    evidence TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (entity_id, shot_id)
) STRICT;

CREATE INDEX story_scene_entities_shot_idx
    ON story_scene_entities (shot_id);

CREATE TABLE story_relations (
    id INTEGER PRIMARY KEY,
    subject_entity_id INTEGER NOT NULL REFERENCES story_entities(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL CHECK (length(trim(predicate)) > 0),
    object_entity_id INTEGER NOT NULL REFERENCES story_entities(id) ON DELETE CASCADE,
    shot_id INTEGER NOT NULL REFERENCES shots(id) ON DELETE CASCADE,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    evidence TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX story_relations_shot_idx
    ON story_relations (shot_id);
CREATE INDEX story_relations_subject_idx
    ON story_relations (subject_entity_id);
CREATE INDEX story_relations_object_idx
    ON story_relations (object_entity_id);
