# Live faster-whisper validation

Goal: verify the final network-dependent media boundary: model download, CPU int8 inference, app-owned cache reuse, and transcript/scenes integration.

## Five lanes
1. Cache ownership — optional MRF_WHISPER_CACHE passed as faster-whisper download_root; developer behavior unchanged when unset.
2. Live inference — download/load the small model in managed Python and transcribe owned/synthetic Vietnamese speech.
3. Offline reuse — reuse the same app-owned model cache with HF_HUB_OFFLINE=1 / no launcher network.
4. Downstream integrity — real transcript JSON/SRT feeds deterministic scenes without invalid timestamps.
5. Verification/docs — opt-in network test, default offline suite, cache/runtime notes.

## Constraints
- Model/timestamps remain owned by deterministic Python stage logic.
- Model cache is outside jobs and survives JS bundle upgrades.
- Default CI must not download a Whisper model.
- Use synthetic/owned audio/video only.
- No commit/push unless explicitly requested.

## Result
- Managed CPython 3.12.14 with faster-whisper 1.2.1 downloaded `Systran/faster-whisper-small` into `%LOCALAPPDATA%\\MovieReviewFactory\\models\\whisper`.
- Live owned Vietnamese speech generated through Edge-TTS transcribed successfully with `device=cpu` and `compute_type=int8`.
- Evidence: 7 cache files, 2 transcript segments, 1 deterministic scene; every transcript timestamp stayed within the probed 7.632-second source duration.
- The same model cache reopened with `MRF_WHISPER_OFFLINE=1` + `HF_HUB_OFFLINE=1`; offline transcript text matched the online run.
- Added an opt-in pytest (`MRF_RUN_LIVE_WHISPER_TESTS=1`) that exercises live speech generation, model/cache use, offline reuse, and `transcript → scenes` integrity. Direct live pytest passed 1/1.
- Default suite remains network-free: 142 passed, 5 opt-in tests skipped, one existing pytest-asyncio deprecation warning.
