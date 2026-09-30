# Commercial release roadmap

Status: active · 2026-09-30. Scope: the last-mile items that turn the working pipeline into a sellable 1-click **paid** Windows desktop product. Sibling plans: [creator production roadmap](creator-production-roadmap.md) (production loop) and [Windows installer](../windows-installer.md) (build mechanics).

These are acceptance thresholds for a shippable paid build, **not market facts**. The `88%` / `$499–$2,500` figures in the external "AKITHINK" evaluation are market projections and are out of scope here — this plan tracks code and build state only. Legend (from the sibling roadmap): **DONE(code)** = implemented + automated tests pass; **PARTIAL** = code done but a real machine / human step still gates it; **GAP** = not built, or needs an owner decision / external system.

## Verified baseline · 2026-09-30 (read-only audit)
- Tests: `py -3 -m pytest tests -q` → **632 passed, 6 skipped, 0 failed** (638 collected, 123.98s). The 6 are opt-in network tests skipped by default, not failures.
- Installer **source** is complete: `scripts/package_windows.py` + `installer/movie-review-factory.iss` stage CPython 3.12 + `ffmpeg`/`ffprobe` + `yt-dlp` + the app + `mrf-launch.vbs`, per-user, v0.2.0. No compiled `.exe` exists in the tree → **GAP**.
- No licensing / machine-binding module exists anywhere in `src/` → **GAP**.
- No TTS "test connection" endpoint exists in `webapp.py` → **GAP** (`tts_providers.provider_api_key` / `missing_key_message` already exist and can back one).
- Correction: the installer's real output name is `movie-review-factory-setup-0.2.0.exe`; the external report's `MovieReviewFactory-Setup-0.2.0.exe` is inaccurate.

## Item 1 — Compile the 1-click `.exe` (Inno Setup)
Goal: produce the distributable `movie-review-factory-setup-0.2.0.exe`.
Status: **GAP** — source ready, binary never built.

Files touched (build only; no source edits expected):
- `scripts/package_windows.py` — packager, already complete; run it.
- `installer/movie-review-factory.iss` — Inno script, already complete.
- `docs/windows-installer.md` — prerequisites + code-signing guidance (reference).
- Output: `build/win/dist/movie-review-factory-setup-0.2.0.exe` (untracked build artifact).

Approach (no code): on a Windows 10/11 x64 host with Inno Setup 6 (`ISCC.exe` on PATH) and first-build internet access:
1. `py -3 scripts\package_windows.py` (add `--extras tts,media,semantic` for bundled offline Whisper/search; the base build is app + Edge-TTS only).
2. If ISCC is not on PATH: `--iscc "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"`.

Done criteria:
- Ships when: the `.exe` is produced, installs per-user with no admin/UAC prompt, the Desktop shortcut launches the hidden VBScript, the local server starts and the browser opens `http://localhost:8765`, and a job can be created on a machine with **no** Python/FFmpeg pre-installed.
- Remains **PARTIAL** until that clean-Windows first launch is actually observed (matches the sibling roadmap's open "true clean-Windows first bootstrap" gate).
- Hand-off (coding.B5 rung 6): compiling and the clean-machine launch genuinely need a Windows build host with ISCC + network — they cannot be settled by static reading. The exact command and expected output are above; a deviation (missing ISCC, download failure, shortcut fails to start the server) is the reopen signal.

Open decision (owner): code-signing. An unsigned installer triggers SmartScreen "Unknown publisher"; an **EV** cert grants reputation immediately, and signing is a separate manual step on a trusted machine (`signtool`, see installer doc). Decide whether v1 ships unsigned (with documented "More info → Run anyway") or blocks on a certificate.

## Item 2 — License key + machine binding
Goal: a paid build refuses to run without a valid, machine-bound license; copying the install folder to another machine does not work.
Status: **GAP** — nothing exists.

Files touched:
- New `src/movie_review_factory/licensing.py` — pure + injectable: `machine_fingerprint(reader=...)`, `verify_license(key, *, fingerprint, now=...) -> Result`; offline signature check (RSA or Ed25519) against an **embedded public key** (private key held only by the vendor).
- `src/movie_review_factory/cli.py` — gate the `serve` command (~line 177) before `webapp.run_server`; fail closed with a clear activation message.
- `src/movie_review_factory/webapp.py` — an activation route (`POST /api/license/activate`) plus a license-status line in the existing settings dialog; store the key at `%LOCALAPPDATA%\MovieReviewFactory\license.key` (same base as `jobs`, so uninstall keeps it out of job data).
- New `tests/test_licensing.py` — offline unit tests with a **test keypair** + injected fingerprint/clock: valid, bad signature, wrong machine, expired, tampered payload.
- New `docs/licensing.md` (or a section in the installer doc) — activation flow + how the vendor issues keys.

Approach (no code): read the Windows machine id from `HKLM\SOFTWARE\Microsoft\Cryptography\MachineGuid` (fallback: volume serial), normalize + hash it into the signed payload, verify the signature fully offline (the "RSA offline signature" idea) with no phone-home. Follow the `tts_providers` shape — pure functions with injected side-effect readers — so the whole module is unit-testable with zero real machine/network dependency (coding.C2 Result pattern; pattern.A6 keep the OS/registry detail behind a boundary).

Done criteria:
- **DONE(code)** when `licensing.py` + the `serve` gate + the activation endpoint + `test_licensing.py` pass offline, and a manually issued test key activates on this machine and is rejected once the injected fingerprint changes.
- **PARTIAL** until verified on a real second Windows machine (the copy is actually blocked).

Open decisions (owner — agent.B3 auth/billing boundary; do not invent silently): license **policy** (perpetual vs 1-year, seat count, offline-only vs periodic online re-check), the signing **keypair** and where the private key lives, and grace / transfer / deactivation handling. This plan deliberately does not choose these.

## Item 3 — "Test Connection" for TTS API keys (Web UI)
Goal: a Settings button that tells the user immediately whether the FPT.AI / ElevenLabs key is present, valid, and (where possible) in credit — so an out-of-credit key is not mistaken for a software bug.
Status: **GAP** — no endpoint; env-key handling already exists.

Files touched:
- `src/movie_review_factory/tts_providers.py` — add `check_provider(provider, *, api_key, http_request=...) -> Result`, reusing the existing injectable HTTP layer plus `provider_api_key` / `missing_key_message`.
- `src/movie_review_factory/webapp.py` — new `GET /api/tts/test?provider=...` in the `_route_get` dispatch (~line 2223) returning `{ok, detail}`; add a "Kiểm tra kết nối" button + status line next to the TTS provider `<select>` in the inline HTML (~lines 3175–3192) and wire a `fetch()` to it.
- `tests/test_tts_providers.py` + `tests/test_webapp.py` — offline tests with a fake `http_request`: key-missing, valid, invalid/401, network error.

Approach (no code; per-provider and cost-aware):
- **Edge-TTS**: no key → report "no key required".
- **ElevenLabs**: `GET /v1/user` (or `/v1/user/subscription`) — validates the key **and** returns remaining character quota at **zero TTS cost**.
- **FPT.AI**: no free balance endpoint; default to key-presence + reachability. A real 1-character synthesis would confirm billing but **costs credits** — keep it explicit/opt-in, never automatic.

Done criteria:
- **DONE(code)** when the endpoint + button + tests pass offline (fake HTTP), and a manual click with a real ElevenLabs key returns quota while a bad key returns a clear invalid state.
- Constraint (agent.B3): the default path must **not** spend paid API credits; any billable check (the FPT.AI synthesis probe) is opt-in and labeled as such in the UI.

## Sequencing & release gate
1. Item 3 (smallest, fully offline-testable) → 2. Item 2 (paid-copy protection) → 3. Item 1 (compile + clean-machine launch — the real ship gate).

Release gate (per project `CLAUDE.md` + sibling roadmap): `py -3 -m pytest tests -q` green, then `scripts/release_gate.py` (plus `--offline`). The compiled `.exe` clean-Windows first launch, and a signed binary if that route is chosen, are human / hand-off gates — not settled by tests alone.
