# Noir & champagne reskin

Status: shipped (uncommitted on `feat/creator-roadmap-1-2-4`).

## Decision
Option B: CSS-only reskin of the existing `webapp.py` shell plus a launcher `--app` window. No framework, no build step, no new runtime dependency.

## Rejected
- Tauri desktop shell: Rust toolchain, signing and a second update channel for what is a window frame; `--app` gives the same chrome-less window.
- Vue/React rewrite: rewrites a working 800+ test UI for visuals only.
- Generator palette `#A16207` gold on black: too brown, fails AA as body text on dark panels; replaced by `#c9a961` (8.8:1 on `#0b0a08`), light theme `#8a6516` (5.3:1 on `#faf8f3`).
- Liquid Glass blur everywhere: `backdrop-filter` cost on weak GPUs while FFmpeg/Whisper run.

## What changed
- Tokens: bg `#0b0a08`, panel `#15130f`, text `#f2ede4`, dim `#b8ae9c`, muted `#948873`, accent `#c9a961`/hover `#e6c687`. One extra `<style>` block after the layout block, so it overrides without touching old rules.
- Type: Montserrat body, Cormorant headings (`font-size-adjust` keeps x-height close). 4 woff2 files (latin + vietnamese subsets) in `assets/fonts/`, served at `/assets/fonts/<name>.woff2` before auth (regex `[a-z-]+`, folder-confined, tested), bundled by `build-onefile.mjs`.
- Motion: busy bar only while `html[data-busy="1"]` (api() counter, 300ms delay so fast calls never flash); text sheen on scout/media loading states; skeleton sheen on empty busy media results; hover glint one-shot. All disabled by `prefers-reduced-motion`. No infinite animation while idle.
- Activation page (`ACTIVATION_HTML`): same tokens, fonts, gold title/button and clapper mark; `@font-face` rules live once in `_UI_FONT_FACES`, injected into both pages via the `/*@ui-fonts*/` placeholder. Status line has busy (sheen)/ok/err states and the button disables while checking.
- Initial "Đang tải…" in job list and highlights uses `.mrf-loading` sheen; replaced by real content on render.
- Icons: emoji brand mark and theme toggle replaced by inline SVG (CSS swaps moon/sun).
- Launcher: `findAppBrowser()` Edge then Chrome (ProgramFiles(x86), ProgramFiles, LOCALAPPDATA); `--app=<url> --user-data-dir=<appDataRoot>/app-window --window-size=1440,900`; fallback `cmd start`.

## Monitor
- GPU/CPU of the busy bar and sheen on weak PCs during encode; drop to static if reported.
- Font payload ~95KB first load (cached afterwards).
- `--app` window: downloads, `window.open` and file dialogs behave like Chrome but with its own profile (no user extensions/logins).
- Double-click launch with a real `--app` window not yet exercised end to end on a clean machine.
