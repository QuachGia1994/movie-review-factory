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
