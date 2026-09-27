# Compact project workspace

Scope: simplify the local Movie Review Factory dashboard while preserving import, stage approval, export, project deletion, and media editing behavior.

Reference evidence: Clipto's public Media Asset Library and Deepfinder screenshots place search, result cards and a player within a compact workspace; Meguri's MIT media library demonstrates a dense card grid and detail pane. These are layout references, not copied code or a claim about Clipto's private app.

Implementation:
1. Keep project selection, bulk delete, and import visible in a compact sidebar; advanced create controls are progressive disclosure.
2. Keep the current project, next action, run/stop, and progress visible at the top. Show only one of media, editor, review/export, or files at a time, with keyboard-accessible tab controls.
3. Bound project/result scrolling; preserve source player beside transcript on desktop and a compact mobile arrangement.
4. Preserve all existing element IDs, HTTP routes, validation gates, and status update handlers. Avoid rebuilding the editor while an input is focused.
5. Verify desktop and mobile viewport overflow, tab navigation, import/delete/export controls, full Python tests, one-file build and offline self-test.

Status: implemented. Verification (2026-09-27): `python -m pytest -q` 297 passed, 6 skipped; `node scripts/build-onefile.mjs` passed; `node movie-review-factory.js --self-test --no-browser --no-network` returned `ok: true`, root and API healthy. Live Chrome at a 627 CSS-pixel viewport: opened an existing project without changing it, switched all four tabs, observed no horizontal overflow, observed mobile project list collapse and QA-gated download hidden when not ready. A real end-to-end import/render/export with a new video and a 390-pixel viewport remain for a future validation run.
