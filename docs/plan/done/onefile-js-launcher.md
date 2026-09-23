# One-file JavaScript launcher

Goal: produce one portable `movie-review-factory.js` distribution file that opens the local web UI with no repo checkout or Python package installation step.

## Five work lanes
1. Packaging entrypoint — embed the complete runtime Python package and project metadata.
2. Runtime dependencies — auto-detect Python >=3.11; bootstrap only required lightweight core dependency; leave FFmpeg/Whisper/edge-tts/Claude as explicit optional runtime capabilities.
3. Windows launch UX — one file must run under both Windows Script Host double-click and Node, open the browser, use a free localhost port, and avoid duplicate servers.
4. Extraction/data lifecycle — content-hash runtime cache, atomic extraction, persistent jobs outside the cache so bundle upgrades never erase jobs.
5. Verification — deterministic build, Node self-test, real local HTTP smoke test, update/data-preservation checks.

## Constraints
- Final distribution artifact is one `.js` file.
- No media/test/sample/verification output is embedded.
- No automatic publish behavior is added.
- Runtime remains localhost-only.
- Claude never owns filesystem/media execution.
- No commit/push unless explicitly requested.

## Result
- Generated distribution artifact: `movie-review-factory.js` (~218 KB at this checkpoint).
- Embeds all seven `movie_review_factory` runtime modules plus `pyproject.toml` as base64 payloads.
- Windows Script Host branch finds Node and relaunches the same file; Node mode detects Python >=3.11, installs missing lightweight core dependencies (`pydantic`, `typer`), extracts the runtime atomically into a content-hash cache, starts localhost on a free port, and opens the browser.
- Jobs persist beside the JS file, with LocalAppData fallback when the directory is not writable. Runtime cache upgrades never delete jobs.
- Existing server instances are reused via a per-directory lock instead of starting duplicates.
- `--self-test` performs a real local HTTP smoke against `/` and `/api/jobs`, then shuts down cleanly.
- Tests prove deterministic rebuilds and isolated-copy startup/data preservation. Full suite: 136 passed; one existing pytest-asyncio deprecation warning.
- Direct `cscript.exe` execution could not be invoked through the current shell allowlist, although `cscript.exe` is present and the generated file is BOM-prefixed ES5-compatible for Windows Script Host parsing.
