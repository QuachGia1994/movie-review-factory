# Windows 1-click installer (.exe)

Ships Movie Review Factory as a double-click Windows installer so a
creator/agency can run it **without installing Python, FFmpeg, or using a
terminal**. The Setup bundles an embedded Python 3.12, `ffmpeg` + `ffprobe`,
`yt-dlp`, and the app itself; the Desktop / Start-menu shortcut starts the local
server and opens `http://localhost:8765`.

## What gets bundled

| Component | Source | Notes |
|-----------|--------|-------|
| CPython 3.12 (embeddable, amd64) | python.org | `import site` re-enabled so pip packages resolve |
| `ffmpeg.exe` + `ffprobe.exe` | gyan.dev release-essentials | put on the launcher's `PATH` |
| `yt-dlp.exe` | yt-dlp GitHub release | put on the launcher's `PATH` |
| `movie-review-factory` (+ `tts` extra) | this repo, `pip install .` | into the embedded interpreter |
| `mrf-launch.vbs` | generated | hidden-window launcher → `python -m movie_review_factory serve` |

The launcher stores jobs under `%LOCALAPPDATA%\MovieReviewFactory\jobs`, so an
uninstall never deletes the user's work.

## Prerequisites (build machine)

- Windows 10/11 x64.
- [Inno Setup 6](https://jrsoftware.org/isdl.php) (provides `ISCC.exe`).
- Internet access on first build (downloads are cached in `build/win/cache`).
- Python 3.11+ to run the packager itself.

## Build

```bat
:: full build: stage payload + produce the .exe (needs ISCC on PATH)
py -3 scripts\package_windows.py

:: stage the payload only (any OS), e.g. to inspect it
py -3 scripts\package_windows.py --skip-iscc

:: add the heavy ML extras (faster-whisper / fastembed) for offline transcription + search
py -3 scripts\package_windows.py --extras tts,media,semantic
```

Output:

- Payload: `build\win\payload\`
- Installer: `build\win\dist\movie-review-factory-setup-0.2.0.exe`

Useful flags: `--iscc "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"`,
`--python-version 3.12.7`, `--port 8765`, `--force-download`.

> The default build installs the base app + `tts` (Edge-TTS) only, keeping the
> installer small. `media`/`semantic` pull large wheels (faster-whisper, fastembed)
> and are opt-in via `--extras`.

## Code-signing & Windows SmartScreen

An **unsigned** installer triggers SmartScreen's *"Windows protected your PC —
Unknown publisher"* prompt; users must click *More info → Run anyway*. To remove
that friction:

1. **Get a code-signing certificate.** A standard OV certificate signs the binary
   but still needs reputation to build before SmartScreen trusts it. An **EV
   code-signing certificate** grants SmartScreen reputation immediately.
2. **Sign both the launcher-facing exe and the Setup exe** with `signtool` and a
   timestamp (so signatures stay valid after the cert expires):

   ```bat
   signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 ^
     "build\win\dist\movie-review-factory-setup-0.2.0.exe"
   ```

3. **Build reputation.** Even signed OV binaries may warn until enough installs
   accrue; distributing the same signed file consistently helps.
4. **Per-user install** (this script already sets `PrivilegesRequired=lowest`) so
   the user is not additionally blocked by a UAC admin prompt.

Signing is intentionally a separate manual step (it needs your private cert and
must run on a trusted machine); the packager does not embed secrets.

## Notes / limits

- ISCC only runs on Windows; on other OSes use `--skip-iscc` to stage the payload.
- The embeddable interpreter is amd64; ship a separate build for arm64 if needed.
- Copyright/Content-ID considerations for published output are a product/legal
  matter, independent of packaging.
