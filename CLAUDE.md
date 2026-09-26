# Movie Review Factory — project facts

- Python 3.11+ package (`src/movie_review_factory`) shipped as a one-file Node launcher (`movie-review-factory.js`) built by `node scripts/build-onefile.mjs`; runtime profile `full` installs uv + a managed venv. Not published to any registry (PyPI 404, no CI workflows).
- Tests: `py -3 -m pytest tests -q` (no CI mirror to match — this is the gate). Release gate: `scripts/release_gate.py` (plus `--offline`).
- Content agents: `scaffold` (offline default), `claude` (`claude -p`), `agy` (local Aki AGY pool on 127.0.0.1, scheduler in `pool_scheduler.py`). Script approval gate must stay intact in every mode.
- **Declared embedded-migration exception (RULE-release B5):** numbered SQL migrations in `src/movie_review_factory/migrations/` are applied by `MediaStore.migrate()` when a job's `media_index.sqlite3` is opened — a single-process, per-job, derived database (rebuildable from job artifacts), not a shared service DB. There is no separate migration runner; expansion is additive-only (ADD COLUMN/index); old rows keep working via defaults; `schema_migrations` records applied versions and re-running is idempotent.
- Identities are anonymous by design (`Person N`); never infer real-world identity, and only process footage the user is authorized to use.
