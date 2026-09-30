-- Per-word timings for each transcript segment, stored as a JSON array of
-- {"word", "start", "end"}. Lets the media explorer time captions from the
-- actual spoken words instead of interpolating by character count. Additive and
-- rebuildable: old rows default to an empty array until the job is re-indexed.
ALTER TABLE transcript_segments ADD COLUMN words TEXT NOT NULL DEFAULT '[]';
