# Changelog

All notable changes to Movie Review Factory are documented here.

## [Unreleased]

### Added
- TTS connection check in the dashboard ("Kiểm tra kết nối"): a Settings button and `GET /api/tts/test` validate the active provider's API key without spending synthesis credits — ElevenLabs reports remaining character quota via `/v1/user/subscription`, FPT.AI confirms key presence (no free balance endpoint), Edge-TTS needs none (`tts_providers.check_provider`, injectable HTTP for offline tests).
- Offline, machine-bound licensing (`licensing.py`): `mrf serve` is gated by an Ed25519-signed, machine-fingerprinted license verified fully offline — an unlicensed machine shows an activation screen and mutating APIs return 402 until a valid key is entered (stored at `%LOCALAPPDATA%\MovieReviewFactory\license.key`, or supplied via `MRF_LICENSE`). The shipped app embeds only the public key; licenses are issued by the vendor-only `tools/issue_license.py` (not shipped in the installer). Adds `cryptography` as a core dependency, bundles the module in the one-file build, and documents the flow in `docs/licensing.md`.
- Read-only YouTube Studio retention advisory (`analytics.retention_advice`): averages recent measured `intro_drop`/`cta_drop`/`ctr_percent` and surfaces suggestions (tighten the hook, move the CTA, lift thumbnail CTR) into the creative-brief and hook-crafter flow and the dashboard analytics panel — strictly advisory, never auto-mutating the script (off switch `MRF_RETENTION_ADVICE=0`; a low-confidence flag under 3 samples). Adds a 4-step "lấy CSV từ YouTube Studio" guide modal.
- Pluggable studio-grade TTS providers (`tts_providers.py`): optional FPT.AI Voice and ElevenLabs alongside the default offline Edge-TTS, selectable per project (`tts_provider`/`tts_voice`, or `MRF_TTS_PROVIDER`/`MRF_TTS_VOICE`) with a dashboard picker and sample voices; provider API keys are read from env only (`MRF_FPTAI_API_KEY`/`MRF_ELEVENLABS_API_KEY`) and never persisted. Non-edge providers estimate word timing from the probed chunk duration; `voice.json` and the dashboard record the active engine/voice.
- Windows 1-click desktop installer (`scripts/package_windows.py` + `installer/movie-review-factory.iss`): stages embedded Python 3.12 + FFmpeg/ffprobe + yt-dlp + the app and builds a per-user Setup whose Desktop/Start shortcut starts the local server and opens `http://localhost:8765`; adds a `python -m movie_review_factory` entry point and build + code-signing/SmartScreen docs (`docs/windows-installer.md`).
- Overnight Batch Factory queue (`batch_queue.py` + dashboard "Hàng đợi hàng loạt"): paste many links to run sequentially to the script-review gate (approval gate intact — never auto-approved), fault-tolerant so one bad link is logged and skipped; an optional completion webhook pings Discord/Telegram (`MRF_DISCORD_WEBHOOK_URL`, or `MRF_TELEGRAM_BOT_TOKEN` + `MRF_TELEGRAM_CHAT_ID`).
- Download-from-link ingest via yt-dlp, gated on an explicit usage-rights confirmation (`link_download.py`): fetches <=1080p mp4 + matching vi/en subtitles into a project as `source.mp4`, exposes a dashboard "Kiểm tra link" metadata probe, and ships no IP-block/cookie/proxy evasion tooling by design.
- Copyright-safety clip transforms (`copyright_bypass.py`): per-clip horizontal flip, subtle zoom-crop and colour-grade applied in the render via `off`/`light`/`balanced`/`aggressive` profiles (`MRF_COPYRIGHT_BYPASS` / `cfg.copyright_bypass`) to reduce automated Content ID false-positives on reviewed footage.
- Hook teaser (`hook_crafter.py`): deterministic bilingual (EN/VI) most-dramatic-scene selection from `scenes.json` cuts a standalone 3-5s `hook.mp4` (+`hook.json`) with a punch-in zoom and optional impact SFX (`MRF_HOOK_SFX`); it never reads or modifies `final.mp4`.
- Full-frame watermark removal stage (`watermark_removal.py` + `mask_detection.py`): optional ProPainter-based reconstruction driven by a configured mask/band/box or color/temporal/external detectors, writing `source_clean.mp4` upstream of the render.
- Word-level transcript timing (migration `007_transcript_words.sql`): the transcript stage records per-word start/end and splits coarse VAD segments on sentence ends and silent gaps, so caption cues anchor to real speech instead of drifting by character-count interpolation.
- Visual rhythm helpers (`visual_rhythm.py`): optional Ken Burns motion (`MRF_KEN_BURNS`) and per-cut transition SFX (`MRF_TRANSITION_SFX`) in the render.
- Recorded creator product identity and a production roadmap for measured narration timing, editorial QA, recoverable edits, channel branding, delivery package, audience feedback and clean-Windows release checks.
- The creator dashboard now includes a reusable creative brief, exact-span recap/opinion tags with internal source references, named edit versions and conservative restore, decoded black/freeze/silence and loudness QA, optional rights-tracked voice/music/SFX mix, and approved-commentary 9:16 short export. Local series/brief/asset-rights planning and measured Studio analytics import modules are available; their dashboard integration and creator walkthrough remain in progress.
- Spoken-word TTS timing now drives two-line subtitle cues and measured section cuts in the renderer; QA checks caption geometry, source provenance, repeated or stretched footage, and voice/visual section drift, and actual midroll placement within the central 40–60% of narration. Ambiguous spoken-word chapter matches fail QA. The AGY midroll remains behind script approval.
- QA-gated handoff ZIP from the dashboard with MP4, SRT/VTT, chosen thumbnail, approved metadata, chapter timecodes from the actual render, upload notes, and SHA-256 revision/content manifest. Export does not upload or publish.
- Dashboard visual refresh: a modern token-based design system layered over the local web UI — refined dark palette plus a real light theme with a header toggle (persisted in `localStorage`, defaults to the OS `prefers-color-scheme`), elevation-by-lightness surfaces, an 8px spacing scale, accent-gradient primary buttons, focus-visible rings, pill status badges with a pulsing dot for running stages, segmented workspace/media tabs, elevated blurred dialogs, tabular-numeric timecodes, and themed scrollbars. The vertical sidebar (Create / Project / Library) is now a horizontal top control bar — a compact create dropdown, a horizontally-scrolling project switcher, and a library-search panel — so the workspace (Media Explorer, timeline, review) spans the full window width for a wider, more professional canvas; it collapses back to a stacked column on narrow screens. Every existing element id/class and all behaviour are unchanged (CSS/markup-only override), so scripts, tests, and the request builder keep working.
- Background indexing queue (roadmap #14): `pipeline.run_index()` runs source-only indexing (ingest → transcript → scenes plus visual/story/embedding scene-memory for the AGY modes) decoupled from content generation, and the local web app now owns a single FIFO worker that indexes imported projects one at a time. Import returns immediately and auto-enqueues indexing (opt out with `MRF_AUTO_INDEX=0`), the dashboard shows live per-stage index progress while other projects stay usable, and Run/Delete/Stop cooperatively pre-empt or cancel a pending index. New `mrf index <job>` CLI command and `POST /jobs/{id}/index` endpoint trigger the same work; a later `run` reuses the cached index. `_scene_plan` now shares the extracted `index_scene_memory` helper, so its behaviour is unchanged.
- Browser MP4 import with streamed upload progress, per-project source storage, and retry after failed import; direct `final.mp4` export after QA.
- Project and artifact cards with individual and multi-select Delete actions, Choose all, count-based confirmation, active-job preflight, and partial filesystem-error reporting; external source footage stays untouched.
- Media intelligence index: the `scenes` stage now writes a per-job `media_index.sqlite3` (SQLite STRICT tables + FTS5 full-text search over dialogue) covering the source asset, bounded scene shots including silent spans, and transcript segments, with idempotent numbered migrations.
- Media Explorer dashboard panel (clipto-style): full-text search across dialogue and scenes, a browsable full transcript timeline with speaker labels, click-to-seek into the source video at the exact timecode, and on-demand scene thumbnails extracted via FFmpeg and cached per shot.
- WebVTT transcript export: download the indexed transcript as `transcript.vtt` (web-native captions with speaker voice tags) straight from the Media Explorer, complementing the SRT captions the transcript stage already writes.
- Transcript-backed chat and deterministic highlight suggestions, with on-demand 9:16 highlight clip export.
- Persistent visual memory: AGY observations are stored per scene in SQLite/FTS5 with visible descriptions, tags, person labels, and actions; Media Explorer can search and display this visual evidence.
- Cross-project Library Search for transcript and visual-index matches, with direct project/timecode navigation.
- True multilingual semantic search using FastEmbed float32 embeddings and cosine similarity for scenes/transcript, with persisted model/dimension metadata and lexical fallback.
- Anonymous cross-scene person continuity from the AGY pool (`Person 1`, `Person 2`, ...), persisted per shot without real-world identity inference.
- Auditable scene-selection scoring v2 with semantic, visual, dialogue, anonymous-person, chronology, and diversity components plus repeat/adjacency penalties.
- `mrf validate-scenes` comparison reports for lexical/visual versus semantic+identity retrieval, with optional per-section ground truth and hit-rate metrics.
- Normalized story-memory graph storage for person/location/event/object entities, per-scene appearances, and grounded relations extracted through the four-account AGY pool.
- `content_agent=agy` mode (`--content-agent agy`, CLI help `scaffold | claude | agy`): jobs can run research, outline, script, scene-plan, and chat through the local Aki AGY pool (four loopback role workers, `POST /run`) instead of Claude Code, failing over to the next role on quota exhaustion, a busy worker, or unusable output; the dashboard offers an `AGY pool` option. Overrides: `MRF_AGY_MODEL` (default `gemini-3.7-flash-medium`), `MRF_AGY_EFFORT` (`low|medium|high`, default `medium`), `MRF_AGY_TIMEOUT_SECONDS` (default `130`).
- Opt-in live faster-whisper validation now proves CPU/int8 transcription, app-owned model caching, deterministic scene generation, and offline cache reuse on owned Vietnamese speech.
- Per-project cooperative cancellation, Windows Claude process-tree termination, resumable cancelled state, stale-running restart recovery, stop API/dashboard control, and race-safe idempotent stop handling.
- Incremental embeddings: migration `006_incremental_embeddings.sql` records `content_hash`/`embed_version` per shot/transcript embedding so unchanged records skip the embedder, with `{embedded, skipped, changed}` stats and version-bump re-embedding.
- Clean-machine release gate `scripts/release_gate.py` with a 13-step checklist (fixture → ingest → transcript → scenes → search → content → tts → alignment → render → qa → export), offline mode, standalone no-repo-src mode, and a shipped VAD-visible speech fixture `data/raw/mrf-gate-speech.mp4` (the bundled sample's music bed scores zero under Silero VAD); `scripts/bench_ann.py` measures search latency for the ANN decision.

### Changed
- The dashboard now shows the next action after import, draft script, render, or QA; script and metadata fields reload when pipeline artifacts first appear.
- Thumbnail generation now honors the job aspect ratio (1280x720 for 16:9, 720x1280 for 9:16).
- Claude content-agent invocation passes tools through a single `--tools` flag (empty for non-research stages) instead of the removed `--allowed-tools` duplicate.
- Source and final video previews stream with HTTP ranges and scoped media-session authorization; the token-protected dashboard shell opens before the fragment token is read, and malformed byte ranges are rejected before streaming. Media cache is cleared when its source index changes. Dialogue retrieval now ranks real words and returns no citation when no transcript matches.
- Claude scene retrieval is now visual-index-first and semantic-aware: AGY inspects all bounded scenes, anonymous `Person N` continuity and story-memory links are built before embedding refresh, and scorer v2 reranks the full bounded scene set while keeping the 12-candidate context limit. Cached observations/tracks/story graph provide a grounded fallback when a later AGY pass is unavailable.
- One-file packaging discovers numbered SQL migrations automatically instead of hard-coding a single migration; the full managed runtime now includes FastEmbed and an app-owned embedding-model cache with offline reuse.
- Semantic search scores with exact numpy matrix math (chunked, pure-Python fallback when numpy is absent): `search_store` p95 on the real job index drops from ~128ms to ~1.2ms at 383 vectors with unchanged ranking, tie-breaks, and `-1.0` dim-mismatch/zero-norm parity.
- AGY pool scheduler v2 (`pool_scheduler.py`): bounded retries with backoff (`MRF_AGY_RETRIES`, 0-3), a 300s role cooldown (`MRF_AGY_COOLDOWN_SECONDS`) that skips only the cooled-down role, partial resume via `ScheduleResult`, and per-role quota/health telemetry (status, latency_ms, cooldown_until) written to `%LOCALAPPDATA%\MovieReviewFactory\runtime\pool_health.json`. AGY prompts fit the measured ~26,000-char backend limit (`MRF_AGY_PROMPT_MAX`) with tiered scene/candidate context instead of failing the request.
- Ignore local scratch/dev artifacts (`_*_tmp.py`, dashboard logs, clipboard images, scratch reports) via `.gitignore` so they never enter a release.

### Removed
- Removed the unwired micro-speed bypass knob (`speed_factor` / `speed_filter`): clip video is concatenated silent while the independent TTS narration is the fixed-length master audio, so a video-only tempo shift would desync the two - it is intentionally not applied.

### Fixed
- Watermark color/temporal detection now streams decoded FFmpeg frames instead of materializing full-resolution PNG scratch frames, preventing multi-gigabyte disk spikes; partial masks and legacy extraction frames are removed on failure, and temporal detection stays bounded in memory.
- yt-dlp metadata/download now run under bounded timeouts (metadata 60s; download `MRF_DOWNLOAD_TIMEOUT`, default 1800s, `0` disables) plus a 30s per-socket read timeout on downloads, so a stalled connection or livestream can no longer hang the download thread indefinitely.
- The dashboard link-download form now sends the actual rights-confirmation checkbox value instead of a hardcoded `true`, keeping the client consent gate authoritative.
- The dashboard server forces UTF-8 on stdout/stderr so Vietnamese output can no longer crash a legacy Windows console (cp1258 `UnicodeEncodeError`).
- Long Edge TTS scripts synthesize in short, retryable chunks with measured per-chunk word offsets, FFmpeg concatenation and atomic output replacement; a failed chunk cannot replace the previous narration or its approved export.
- The transcript stage now skips with `source video has no audio track - nothing to transcribe` when the source container has no audio stream, instead of failing inside faster-whisper/PyAV with `tuple index out of range`.
- Claude content-agent failures surface a bounded `head…tail` diagnostic (≤400 chars, no full stdout flood) and transient HTTP 429/5xx advisor errors are retried with `5×(attempt+1)` backoff instead of failing the stage on the first hiccup.

### Security
- Hardened the yt-dlp invocation against argument injection: URLs are validated as http(s)-only and a `--` end-of-options guard is placed before the URL, so a value like `--exec=...` can no longer be parsed as a yt-dlp option.

## [0.2.0] - 2026-09-23

### Added
- One-file Windows distribution: `movie-review-factory.js` embeds the complete Python application runtime and opens the localhost dashboard from a single file.
- Zero-install first-run bootstrap for portable Node.js, managed Python 3.12, FFmpeg/FFprobe, faster-whisper, and edge-tts with integrity checks and cache reuse.
- Claude Code content-agent integration for research, outline, script, and indexed multi-shot scene planning while Python retains deterministic timestamp/media execution.
- Three-candidate thumbnail generation, dashboard preview/selection, and publish-handoff thumbnail metadata.
- Explicit script approval parity across CLI and web.

### Changed
- Scene planning supports multiple visual shots per narration section and deterministic exact-duration trim/loop behavior.
- Pipeline skipped stages are retryable, downstream edits invalidate stale artifacts, and pre-thumbnail manifests upgrade automatically.
- Publish handoff now requires approved script/metadata, a non-empty final render, and passing QA.
- One-file runtime caches toolchains separately from persistent jobs and supports offline relaunch after first provisioning.

### Fixed
- Whisper Windows execution explicitly uses CPU int8 mode.
- Script approval keeps `script.json` and `script.md` synchronized.
- Windows PATH merging preserves system commands when portable FFmpeg is injected.
- Zero-install bootstrap fails closed on missing cache in no-network mode and verifies downloaded Node, uv, and FFmpeg artifacts before use.

## [0.1.0] - 2026-09-20

### Added
- Initial resumable movie-review pipeline with typed job state, CLI control, and deterministic stage artifacts.
- Local web dashboard for job creation, status, script/metadata approval, render preview, and publish handoff.
- FFmpeg/FFprobe render and QA path with owned/synthetic verification fixtures and approval-gated publishing.
