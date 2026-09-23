# Zero-install Windows runtime

Goal: a fresh 64-bit Windows machine can double-click `movie-review-factory.js` and reach the localhost UI without preinstalling Node, Python, FFmpeg, or Python media/TTS packages.

## Five lanes
1. Node bootstrap — Windows Script Host + built-in PowerShell downloads/verifies latest Node LTS portable archive.
2. Python bootstrap — portable uv downloads/verifies itself, installs managed Python 3.12 and an isolated app venv.
3. FFmpeg bootstrap — download/verify release essentials, discover ffmpeg/ffprobe bin directory, inject it into child PATH.
4. Python runtime — install versioned app requirements into the managed venv once; include faster-whisper and edge-tts.
5. Verification — system-runtime fallback, managed-runtime cache behavior, isolated-copy HTTP smoke, data preservation, offline second launch.

## Constraints
- Distribution remains one generated JS file.
- Runtime/cache lives outside jobs and may be replaced without deleting jobs.
- HTTPS downloads are integrity-checked when the upstream publishes a checksum/digest.
- Claude Code stays optional.
- localhost-only; no automatic publish.
- no commit/push unless explicitly requested.

## Result
- WSH branch now bootstraps Node LTS portable when `node` is absent, verifies the archive against Node's official `SHASUMS256.txt`, caches it, and relaunches the same JS file with that Node executable.
- Node mode bootstraps portable `uv`, verifies its release digest, installs managed CPython 3.12 into the app cache, and creates an isolated venv.
- Default `full` runtime installs `pydantic`, `typer`, `faster-whisper`, and `edge-tts` with import verification before writing the cache marker; `core` remains available for lightweight verification.
- FFmpeg bootstrap prefers Gyan's official GitHub release mirror, reads the essentials ZIP SHA256 digest from the GitHub release asset, downloads/extracts the ZIP, verifies `ffmpeg.exe` and `ffprobe.exe`, and falls back to Gyan direct + `.sha256` metadata when release metadata is unavailable.
- Windows PATH merging is case-insensitive so adding portable FFmpeg does not hide system commands such as the Python launcher.
- `--no-network` fails closed on an empty cache and succeeds after the managed runtime has been provisioned.
- Verified live: portable Node WSH bootstrap passed; managed Python 3.12 core bootstrap passed; full faster-whisper + edge-tts bootstrap/import passed; portable FFmpeg bootstrap passed; full managed runtime reopened with `--no-network` and portable FFmpeg from cache.
- Default focused one-file tests: 5 passed, 2 opt-in network tests skipped. Full suite: 141 passed, 4 skipped, one existing pytest-asyncio deprecation warning.
- Edge-TTS still requires network when speech is synthesized, and faster-whisper may download its model on first transcription if that model is not already cached.
