# Agent workflow

Claude Code is the reasoning/orchestration client. Local launcher: D:\LacViet\claude-pm\clpm.cmd. The gateway/proxy supplies the model backend; provider API keys are not stored in this repo.

AkiMCP is the machine-control layer. Agents use AkiMCP for local filesystem, shell, browser/CDP, task lifecycle, and verification.

## Agent contracts

Researcher → research.json: metadata, sources, uncertainty.
Story editor → outline.json, script.json, script.md.
Scene editor → scene_plan.json with source timestamps and rationale.
Voice producer → narration audio + alignment metadata.
Render/QA worker → final MP4 + machine-readable QA report.
Publisher → upload only after explicit human approval.

LLM agents choose WHAT; deterministic FFmpeg/FFprobe/Whisper tooling decides HOW media is cut, encoded, timed, and validated.
