# Live production validation

Goal: verify the network-dependent production boundaries that offline tests intentionally exclude.

## Five lanes
1. Edge-TTS — synthesize real Vietnamese narration and verify non-empty decodable audio.
2. Alignment/render/QA — feed live narration through alignment, deterministic FFmpeg render, and FFprobe QA on owned sample media.
3. Claude runtime — exercise the project `run_claude_json()` path with schema output, not only CLI/auth probes.
4. Whisper boundary — keep model-download/inference opt-in because first-run model fetch is large and network-dependent.
5. Regression — preserve live checks as opt-in tests while default suite stays deterministic/offline.

## Constraints
- Network checks are opt-in and never required for the normal unit/CI suite.
- Use only synthetic/owned media.
- No automatic publish.
- Claude stays restricted and schema-constrained.
- No commit/push unless explicitly requested.

## Result
- Live Edge-TTS synthesized Vietnamese narration successfully with `vi-VN-HoaiMyNeural`; direct smoke produced a non-empty 5.52s MP3.
- Managed-runtime production smoke then ran approved script → Edge-TTS → alignment → deterministic FFmpeg render → FFprobe QA on the owned sample. Narration measured 5.688s, final.mp4 was non-empty (~2.19 MB), alignment used `voice.json`, and all 8 QA checks passed.
- Claude Code 2.1.277 is installed and OAuth auth status is valid. The project `run_claude_json()` schema probe succeeded through the currently configured custom `ANTHROPIC_BASE_URL`.
- One transient 60s Claude timeout was observed; repeated probes across repo root, an in-repo job directory, and an external temp directory all succeeded. `content_agent.py` now retries exactly one timeout by default (`MRF_CLAUDE_RETRIES`, range 0–3) while non-timeout launch/auth/schema failures still fail immediately.
- Added opt-in live integration tests for Claude and Edge-TTS production flow. Default suite stays offline/deterministic.
- Focused content-agent/one-file suite: 15 passed, 3 skipped. Live Claude opt-in test passed. Full suite: 141 passed, 4 skipped, one existing pytest-asyncio deprecation warning.
- Live faster-whisper model download/inference remains the only network-dependent media boundary not yet exercised; package import/bootstrap is already verified.
