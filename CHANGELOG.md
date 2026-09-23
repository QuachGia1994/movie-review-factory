# Changelog

All notable changes to Movie Review Factory are documented here.

## [Unreleased]

### Added
- Opt-in live faster-whisper validation now proves CPU/int8 transcription, app-owned model caching, deterministic scene generation, and offline cache reuse on owned Vietnamese speech.

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
