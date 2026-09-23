# Agent workflow

Claude Code is the reasoning/orchestration client. Movie Review Factory invokes `claude` from PATH, or the executable/path configured by `MRF_CLAUDE_BIN`, in non-interactive print mode (`-p`) with JSON-schema structured output. Each call uses restricted + plan mode, an empty MCP config, and an explicit tool surface: WebSearch/WebFetch only for research, no tools for outline/script/scene-plan. Authentication/model routing come from the operator's Claude Code installation; provider credentials are not stored in this repo.

AkiMCP is the machine-control layer. Agents use AkiMCP for local filesystem, shell, browser/CDP, task lifecycle, and verification.

## Agent contracts

Researcher → research.json: movie brief, supported facts, source URLs, uncertainty. In `content_agent=claude` mode it may use Claude Code WebSearch/WebFetch; unresolved claims stay explicit.
Story editor → outline.json, script.json, script.md. Claude output is schema constrained; section timing and approval state remain deterministic in Python.
Scene editor → scene_plan.json with multiple validated visual shots per narration section. In `content_agent=claude` mode Claude selects only indexed start/end scene IDs for each shot; Python validates those IDs, resolves bounded timestamps, splits section duration across the shots, and writes top-level render clips. Claude never supplies raw timestamps.
Voice producer → narration audio + alignment metadata.
Render/QA worker → final MP4 + machine-readable QA report.
Thumbnail worker → three deterministic 1280×720 source-frame candidates. The operator can select a primary candidate in the dashboard; selection atomically refreshes `thumbnail.jpg` and invalidates only the downstream publish handoff.
Publisher → upload only after explicit human approval.

LLM agents choose WHAT; deterministic Python/FFmpeg/FFprobe/Whisper tooling owns stage contracts, timing, approval state, media execution, and validation. `scaffold` remains the default offline content mode; `claude` is an explicit per-job choice.
