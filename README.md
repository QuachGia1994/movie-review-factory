# Movie Review Factory

Windows-local AI movie review/recap pipeline. Claude Code CLI (`claude`) can generate research, outline, script, and multi-shot scene-selection reasoning; deterministic local tooling handles timestamps, media processing, state, and verification.

Pipeline: ingest → research → transcript/scene index → outline → script → scene plan → TTS → alignment → render → QA → metadata → thumbnail → publish.

MVP target: Vietnamese 8–12 minute 16:9 review/recap. The local dashboard previews three generated thumbnail candidates and lets the operator choose the primary image. Publishing is not automatic.

## One-file local web

Run `node scripts\build-onefile.mjs` to generate `movie-review-factory.js`. On 64-bit Windows, double-click that single file. First run bootstraps a private runtime under `%LOCALAPPDATA%\MovieReviewFactory`: Node LTS when Node is absent, managed Python 3.12 through portable `uv`, FFmpeg/FFprobe when absent, and the required Python app/Whisper/TTS packages. It then starts an ephemeral localhost port and opens the default browser. Jobs live in a `jobs` folder beside the JS file (falling back to LocalAppData when that folder is not writable), so replacing the JS with a newer build does not erase jobs.

A fresh machine only needs Windows Script Host + PowerShell (standard on supported Windows) and Internet access for first-run downloads. Node/Python/FFmpeg and the installed `faster-whisper`/`edge-tts` packages are cached afterward, so subsequent launches do not reinstall them. Downloaded Node, `uv`, and FFmpeg archives are SHA256-verified against upstream metadata. Claude Code remains optional: `content_agent=scaffold` works without it. Edge-TTS still needs network access when synthesizing speech. Faster-whisper stores its model under `%LOCALAPPDATA%\\MovieReviewFactory\\models\\whisper`; live CPU/int8 transcription plus offline model-cache reuse has been verified on Windows.

Useful checks:

```powershell
node movie-review-factory.js --self-test --no-browser
node movie-review-factory.js --self-test --no-browser --no-network
node movie-review-factory.js --no-browser
```

## Developer setup

Requires Python 3.11+ and FFmpeg/FFprobe.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[media,tts]"
mrf init-job .\jobs\demo --movie-title "The Matrix (1999)" --content-agent claude
mrf validate-job .\jobs\demo
mrf run .\jobs\demo --until script
mrf approve-script .\jobs\demo --confirm
mrf run .\jobs\demo --until thumbnail
```

`--content-agent scaffold` is the offline/default mode. `--content-agent claude` invokes `claude -p` with JSON-schema structured output for research, outline, script, and scene selection while keeping the script approval gate intact. Optional runtime overrides: `MRF_CLAUDE_BIN`, `MRF_CLAUDE_MODEL`, `MRF_CLAUDE_EFFORT`, `MRF_CLAUDE_TIMEOUT_SECONDS`, and `MRF_CLAUDE_RETRIES`. Claude retries one timeout by default; non-timeout CLI/auth/schema failures still fail immediately. Claude runs in restricted/plan mode with an empty MCP config; research exposes only WebSearch/WebFetch, while outline/script/scene-plan expose no tools.

See PLAN.md, docs/AGENT_WORKFLOW.md, and docs/REFERENCES.md.

Only process source footage you are authorized to use. The system is for original commentary/review, not re-uploading full copyrighted films.
