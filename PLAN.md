# Movie Review Factory — implementation plan

## Goal
Build a Windows-local AI movie review/recap factory orchestrated by Claude Code through the claude-pm gateway/proxy, with AkiMCP as the local control/automation plane.

## Evidence reviewed
- zcbacxc/movie-narrator: 16-step pipeline covering research, script, TTS, alignment, scene detection/matching, BGM, subtitles, QA, render, validation, clip export; AGPL-3.0-or-later.
- keithhb33/AI-Movie-Shorts: subtitle-driven clip planning, ElevenLabs narration, FFmpeg timing/stretch/concat, 16:9 + 9:16.
- Minhal-Ahmed/CineRecap: Whisper → Gemini summary → TTS → video; simple reference implementation.
- yuchia329/yapper: resumable multi-stage movie commentary pipeline; useful for deterministic stage design.
- strangedeev/youtube-automation-agent: research/script/thumbnail/SEO/production/publish agent chain.
- Other useful reference: carlosmassaa/AI-Multimedia-Automation-Toolkit for multimodal retrieval, subtitles, cost tracking, and YouTube upload.

## Architecture decision
1. Claude Code via D:\LacViet\claude-pm is the agent/orchestration interface; gateway/proxy supplies the model backend.
2. AkiMCP remains the machine-control plane: filesystem, shell, browser/CDP, task lifecycle, verification.
3. Media processing must be deterministic and local: FFmpeg/FFprobe + Whisper/faster-whisper; agents produce structured plans/artifacts rather than directly editing media.
4. Pipeline stages:
   ingest → metadata/research → transcript/scene index → story outline → review script → scene plan → TTS → alignment → render → QA → thumbnail/metadata → optional YouTube publish.
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
- Outputs: script.md, narration audio, SRT, final.mp4, thumbnail prompt/asset, YouTube metadata JSON.
- Test fixtures with synthetic/owned sample video; no copyrighted movie fixture checked into repo.
- README with Windows setup and claude-pm/AkiMCP agent workflow.

## Verification gates
- FFprobe confirms duration/codecs/audio.
- Subtitle timing does not exceed narration/video bounds.
- Render exit code 0 and output decodes.
- Audio/video duration drift within configured threshold.
- No missing stage artifacts.
- CI/unit tests green before any commit/push.
- YouTube publish is disabled in MVP until explicit user approval flow is implemented.

## Current status
- New repo workspace created at D:\LacViet\movie-review-factory.
- Persona/system-prompt verification assets added under `postman/`.
- `claude-pm` wrapper now appends the local Nora persona file while preserving Claude Code's default system prompt.
- Live proxy precedence is not yet claimed as verified; the Postman collection is the reproducible verification path.
- No git remote/commit/push created.