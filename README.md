# Movie Review Factory

Windows-local AI movie review/recap pipeline. Claude Code, through D:\LacViet\claude-pm, orchestrates reasoning agents; AkiMCP performs local filesystem/shell/browser automation and verification.

Pipeline: ingest → research → transcript/scene index → outline → script → scene plan → TTS → alignment → render → QA → metadata → publish.

MVP target: Vietnamese 8–12 minute 16:9 review/recap. Publishing is not automatic.

Requires Python 3.11+ and FFmpeg/FFprobe.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
mrf init-job .\jobs\demo
mrf validate-job .\jobs\demo
```

See PLAN.md, docs/AGENT_WORKFLOW.md, and docs/REFERENCES.md.

Only process source footage you are authorized to use. The system is for original commentary/review, not re-uploading full copyrighted films.
