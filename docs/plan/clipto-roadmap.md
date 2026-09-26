# Clipto parity roadmap

Scope: unreleased Movie Review Factory development after persistent visual memory, multilingual vector search, and anonymous AGY person continuity.

## Target order

1. Real-movie validation harness (8–12 minute licensed/owned footage): compare lexical-only, visual-memory, semantic+identity retrieval; measure relevance, repetition, chronology, coverage, and manual override rate.
2. Scene-selection scoring v2: combine semantic, visual, dialogue, anonymous-person continuity, chronology, and diversity; penalize repeated/adjacent source ranges and narration mismatch.
3. Story-aware memory graph: Person N ↔ location ↔ event ↔ object ↔ scene relations with chronology.
4. Character continuity v2: richer appearance summaries, clothing-change history, ambiguity state, confidence decay, and optional manual rename; never infer real-world identity.
5. Advanced semantic visual query: multi-constraint text queries (person/action/location/object) and Person N co-occurrence.
6. Find similar scene: use an indexed scene embedding as the query vector.
7. Automatic B-roll replacement: replace repetitive/weak primary shots with semantically compatible alternatives.
8. Timeline editor: card/timeline reorder, trim, replace, lock, preview.
9. Section-level regenerate: regenerate visuals or scene choices for one narration section without rerunning the whole video.
10. Library search filters: project/person/duration/date/scene type/confidence/source filters plus saved search history.
11. Incremental embeddings: content hash/version metadata; only re-embed changed visual/transcript records.
12. Large-library ANN: move from linear cosine scan to HNSW/FAISS/sqlite-vector only after measured scale requires it.
13. AGY pool scheduler v2: advisor → executor → experiment → reviewer support with quota/health telemetry, cooldown, partial resume, and bounded retries.
14. Background indexing queue: import first, then transcript/visual/embedding/story indexing with progress while other projects remain usable.
15. Clean-Windows release gate: cold bootstrap with no preinstalled Node/Python/FFmpeg/FastEmbed model, then import → index → search → render → export.

## Execution policy

- Four-account AGY pool is runtime support for visual descriptions, anonymous continuity, and story-memory extraction; it is not counted as delegated subagent review.
- Keep all work unreleased until the real 8–12 minute validation and clean-Windows launch pass.
- Every new scorer/indexer must have deterministic fixture tests plus a runtime probe where external model behavior is involved.
- Preserve script approval, QA, and publish-handoff gates.
- Do not infer or persist real-world identity from faces; identity labels remain project-local aliases such as Person 1 unless the operator manually renames them.

## Current implementation status

Implemented and mechanism-verified:
- #1 validation harness: `mrf validate-scenes` compares lexical/visual and semantic+identity strategies on one indexed job without rendering; optional truth data reports top-1 and candidate hit rates.
- #2 scene-selection scoring v2: each candidate carries dialogue, visual/story, semantic, anonymous-person, chronology, diversity-penalty, and total scores. Repeat/adjacent-source penalties apply across section top choices while the Claude context stays capped at 12 candidates.
- #3 story-aware memory foundation: migration `004_story_graph.sql` stores normalized person/location/event/object entities, scene appearances, and relations. AGY extraction uses the existing advisor → executor → experiment → reviewer quota failover and refuses person labels outside the anonymous `Person N` roster.
- #4 character continuity v2: migration `005_continuity_editor.sql` adds project-local aliases, appearance summaries, clothing history, ambiguity, mean confidence, and recency-weighted confidence decay without real-world identity inference.
- #5 advanced visual query: Scene and Library Search accept person/action/location/object constraints, aliases, and anonymous-person co-occurrence facets.
- #6 similar-scene retrieval: an indexed scene embedding can be reused as the query vector.
- #7 automatic B-roll replacement: repeated, missing, or semantically weak primary shots can be replaced with compatible alternatives.
- #8 timeline editor: the local dashboard/API supports reorder, trim, replace, lock, and preview operations.
- #9 section-level regeneration: one narration section can regenerate scene choices while preserving locked clips and invalidating only downstream render state.
- #10 library filters and history: project/person/duration/date/scene-type/confidence/source filters and saved searches are persisted and exposed through the dashboard.
- #11 incremental embeddings: migration `006_incremental_embeddings.sql` adds `content_hash`/`embed_version` to shot and transcript embeddings; unchanged records skip the embedder, stats report `{embedded, skipped, changed}`; deterministic fake-embedder tests cover new/unchanged/changed/version-bump/migration-upgrade.
- #12 large-library ANN (measured first): `scripts/bench_ann.py` measures the real job DB plus synthetic 100–100k corpora. Pure-Python linear scan exceeds the 100ms p95 budget at ~289 vectors (one job: `135.7ms` at 383 vectors), so search switched to exact numpy scoring (already a fastembed dependency — no new package): `search_store` p95 `128ms → 1.17ms` at 383 vectors, `21.5ms` at 10k. HNSW/FAISS/sqlite-vector stay deferred until measured scale requires them.
- #13 AGY pool scheduler v2: `pool_scheduler.py` records per-role quota/health telemetry (`status`, `latency_ms`, `cooldown_until` → `%LOCALAPPDATA%\MovieReviewFactory\runtime\pool_health.json`), 300s cooldown (`MRF_AGY_COOLDOWN_SECONDS`) that skips only the cooled-down role, bounded retries (`MRF_AGY_RETRIES`, 0–3, `0.5s×attempt` backoff), and partial resume via `ScheduleResult(roles_used, skipped_roles, failures)`; AGY vision and agent failover run through the scheduler with public contracts unchanged.
- #14 background indexing queue: `pipeline.run_index()` runs ingest → transcript → scenes plus visual/story/embedding scene-memory (for the claude/agy modes) with no content stage, driven by an extracted `index_scene_memory` helper that keeps `_scene_plan` behaviour byte-for-byte. `webapp.JobsService` owns a single daemon FIFO worker: import auto-enqueues (`MRF_AUTO_INDEX`, default on), one job indexes at a time while other projects stay usable, per-stage progress is exposed on `status()`/`list_jobs()` and rendered live in the dashboard, and Run/Delete/Stop cooperatively pre-empt or cancel a pending index so nothing writes `media_index.sqlite3` concurrently. `mrf index <job>` and `POST /jobs/{id}/index` trigger the same path; a later `run` reuses the cached index. Deterministic tests cover run_index orchestration/progress/cancellation and the queue's FIFO order, idempotency, auto-enqueue, busy guards, and stop-cancel.
- #15 clean-Windows release gate (script): `scripts/release_gate.py` walks fixture → ingest → transcript → scenes → search → outline → script → scene_plan → tts → alignment → render → qa → export with a 13-step checklist, offline mode, standalone (no-repo-src) mode, and a shipped VAD-visible speech fixture `data/raw/mrf-gate-speech.mp4` (the bundled sample's music bed scores zero under Silero VAD at any length). This machine: online `18/18 PASS, exit 0`; offline `0 FAIL`.

Evidence:
- targeted roadmap #4–#10 suite: `82 passed, 1 skipped`;
- full suite: `297 passed, 6 skipped` (adds deterministic tests for #14 run_index and the FIFO indexing queue on top of the #11–#13, AGY prompt-budget fitting, numpy-vs-pure-python search equivalence, and claude transient-retry coverage);
- Python compile check passed for `src/movie_review_factory`;
- `git diff --check` exits `0` (Windows LF→CRLF notices only);
- live validation fixture: lexical/visual top-1 hit rate `0.50` vs semantic scorer v2 `1.00`; both candidate hit rates `1.00`;
- live AGY story probe grounded `Person 1`, hospital corridor, red bag, waiting area, events, and scene-local relations;
- four-worker story quota test reached reviewer after simulated advisor/executor/experiment quota exhaustion;
- rebuilt one-file bundle includes scorer/validation modules and migrations 001–006 plus `pool_scheduler.py`; offline self-test returned `root_ok=true`, `api_ok=true`;
- live AGY end-to-end run on a real 10-minute job (content_agent=agy): `research`, `outline`, `script`, and `scene_plan` (20 clips) all reached `ready` through the four-role pool after moving the runner prompt from argv to stdin (argv exceeds the 32,767-char Windows limit) and capping prompts at `MRF_AGY_PROMPT_MAX` (26,000 chars; measured AGY tolerance: 29k OK / 32k fail); script approval gate preserved (`tts` skipped until `approved=true`);
- release gate runs: online `18/18 PASS exit 0`, offline `12 PASS / 6 BOO (network) / 0 FAIL`.

Still open:
- #1 real 8–12 minute licensed/owned movie evaluation; synthetic fixtures prove mechanism only.
- #3–#10 quality and usability validation on real licensed film scenes, especially continuity aliases, B-roll replacement, and timeline editing.
- #14 background indexing queue: implemented and mechanism-verified (see above); real-workload tuning (a concurrency ceiling beyond one worker, retry/backoff on transient index failures) can follow once the #1 real-movie evaluation runs.
- #15 cold clean-Windows machine run: the gate script and checklist are ready and pass on this machine, but the true no-preinstalled-toolchain bootstrap on a fresh Windows box has not been executed yet.
- Release stays unreleased per execution policy until the real 8–12 minute validation and the clean-Windows launch pass.
