# Movie Review Factory — implementation plan

## Goal
Build a Windows-local AI movie review/recap factory where Claude Code generates reasoning-heavy content stages and deterministic local tooling owns media execution, state, and verification.

## Evidence reviewed
- zcbacxc/movie-narrator: 16-step pipeline covering research, script, TTS, alignment, scene detection/matching, BGM, subtitles, QA, render, validation, clip export; AGPL-3.0-or-later.
- keithhb33/AI-Movie-Shorts: subtitle-driven clip planning, ElevenLabs narration, FFmpeg timing/stretch/concat, 16:9 + 9:16.
- Minhal-Ahmed/CineRecap: Whisper → Gemini summary → TTS → video; simple reference implementation.
- yuchia329/yapper: resumable multi-stage movie commentary pipeline; useful for deterministic stage design.
- strangedeev/youtube-automation-agent: research/script/thumbnail/SEO/production/publish agent chain.
- Other useful reference: carlosmassaa/AI-Multimedia-Automation-Toolkit for multimodal retrieval, subtitles, cost tracking, and YouTube upload.

## Architecture decision
1. Claude Code CLI (`claude -p`) is the reasoning interface for research/outline/script and indexed multi-shot scene selection; the operator's Claude Code installation supplies authentication/model routing.
2. AkiMCP remains the machine-control plane: filesystem, shell, browser/CDP, task lifecycle, verification.
3. Media processing must be deterministic and local: FFmpeg/FFprobe + Whisper/faster-whisper; agents produce structured plans/artifacts rather than directly editing media.
4. Pipeline stages:
   ingest → metadata/research → transcript/scene index → story outline → review script → scene plan → TTS → alignment → render → QA → metadata → thumbnail → optional YouTube publish.
5. Agent boundaries:
   - researcher: factual/movie metadata research
   - story editor: outline + narration
   - scene editor: timestamp/clip plan
   - voice producer: TTS + alignment
   - render/QA worker: deterministic FFmpeg execution + validation
   - publisher: metadata/upload only after explicit approval
6. Human approval gates: script approval and final publish approval. Never auto-publish by default.
7. First target: Vietnamese 8–12 minute 16:9 review/recap, then 9:16 Shorts derivative.

## Legal/commercial guardrail
Use source footage only where the user has the necessary rights/licence or where the workflow is otherwise legally permitted. Do not build around downloading/re-uploading copyrighted full films. The generated narration/commentary should add original analysis rather than reproduce the film or source text.

## MVP deliverables
- Python project with typed stage contracts and resumable job state.
- CLI first; web UI after the pipeline is stable.
- JSON artifacts: research.json, scenes.json, script.json, voice.json, render.json, qa.json.
- Outputs: script.md, narration audio, SRT, final.mp4, 3 thumbnail candidates + thumbnail.jpg, thumbnails.json, YouTube metadata JSON.
- Test fixtures with synthetic/owned sample video; no copyrighted movie fixture checked into repo.
- README with Windows setup and Claude Code/AkiMCP agent workflow.

## Verification gates
- FFprobe confirms duration/codecs/audio.
- Subtitle timing does not exceed narration/video bounds.
- Render exit code 0 and output decodes.
- Audio/video duration drift within configured threshold.
- No missing stage artifacts.
- CI/unit tests green before any commit/push.
- YouTube publish is disabled in MVP until explicit user approval flow is implemented.

## Current status
- v0.1.0 is the published MVP baseline; v0.2.0 is the current release.
- CLI + local web application are implemented, including job creation/status, script review/approval, metadata review/approval, render preview, and a publish handoff gate that never uploads automatically.
- Deterministic media stages are implemented for ingest, faster-whisper transcription, scene indexing/planning, TTS, alignment, FFmpeg render, and FFprobe QA.
- Persona/system-prompt verification assets remain under `postman/`; live proxy precedence is not claimed beyond the reproducible Postman verification path.
- Release 0.2.0: faster-whisper uses CPU + int8 explicitly on Windows; script/metadata edits invalidate stale downstream outputs; thumbnail generation produces three clean 1280×720 candidates plus thumbnail.jpg; the dashboard previews/selects the primary thumbnail and selection invalidates only stale publish output; recoverable skipped stages retry; pre-thumbnail manifests auto-upgrade; jobs can explicitly select `content_agent=claude` for schema-constrained research/outline/script plus indexed multi-shot scene-selection reasoning; and the whole runtime packages into one generated `movie-review-factory.js`. On a fresh Windows machine, the JS bootstraps Node LTS through WSH/PowerShell, portable `uv`, managed CPython 3.12, the full Python app/Whisper/TTS dependency set, and portable FFmpeg/FFprobe with SHA256 verification. Runtime/toolchain caches live outside jobs, and a provisioned full cache has been verified to reopen with `--no-network`. Real Edge-TTS -> alignment -> FFmpeg render -> FFprobe QA has passed on the owned sample, and the project Claude schema path has passed against the currently configured endpoint; one transient Claude timeout was observed and is retried exactly once by default. The publish handoff still requires approved script/metadata, a non-empty final.mp4, and passing qa.json. Full suite passes 141 tests with 4 opt-in network tests skipped by default.
- Generated verification directories remain untracked and are not part of the release artifact.
- Zero-install distribution plus live Edge-TTS and Claude schema validation are complete. Claude Code 2.1.277 is authenticated and works through the current custom `ANTHROPIC_BASE_URL`; transient timeout behavior is fail-closed after one default retry. The remaining unverified network media boundary is faster-whisper model download/inference and reuse from the app-owned cache; Edge-TTS continues to require network at synthesis time.