ALTER TABLE shot_embeddings ADD COLUMN content_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE shot_embeddings ADD COLUMN embed_version INTEGER NOT NULL DEFAULT 0
    CHECK (embed_version >= 0);

ALTER TABLE transcript_embeddings ADD COLUMN content_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE transcript_embeddings ADD COLUMN embed_version INTEGER NOT NULL DEFAULT 0
    CHECK (embed_version >= 0);

CREATE INDEX IF NOT EXISTS shot_embeddings_hash_idx
    ON shot_embeddings (content_hash, embed_version);

CREATE INDEX IF NOT EXISTS transcript_embeddings_hash_idx
    ON transcript_embeddings (content_hash, embed_version);
