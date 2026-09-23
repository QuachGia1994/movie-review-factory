# Multi-clip visual planning

Goal: replace the current one-contiguous-source-range-per-narration-section contract with multiple validated source shots per section while keeping Claude limited to indexed scene choices and FFmpeg deterministic.

## Work lanes
1. Data contract — scene_plan shape, backward compatibility, artifact/state impact.
2. Render — flatten multi-shot plans into validated deterministic FFmpeg ranges.
3. Agent planner — schema/prompt/validation for ordered indexed scene selections only.
4. State/UX — editing/invalidation/status/API implications; avoid new manual gates.
5. Verification/docs — unit/E2E coverage, localization, README/workflow sync.

## Constraints
- Claude never supplies raw timestamps.
- Python owns validation, duration accounting, state and artifact writes.
- Preserve old single-range scene_plan compatibility where practical.
- No automatic publish.
- No commit/push unless explicitly requested.

## Result
- Scene-plan agent schema now supports 1–6 ordered shots per narration section using scene IDs only.
- Python validates every scene range, rejects duplicate shots inside a section, resolves timestamps, and expands sections into top-level render clips.
- Deterministic scaffold mode also spreads longer sections across representative scenes.
- Renderer now trims long source shots or loops short source shots to each clip's exact target duration before concat.
- Legacy scene plans with one top-level `source_clip` per clip remain render-compatible.
- Verification: pipeline tests 84/84; localization + real FFmpeg E2E 8/8; full suite 134/134; `pytest_asyncio` emits one deprecation warning only.
