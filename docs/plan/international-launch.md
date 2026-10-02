# International launch plan

Status: active · 2026-10-02. Business source of truth: [creator product identity](../biz/creator-product.md) (audience, offer, feature boundary, kill signals). Sibling plan for build mechanics: [commercial release roadmap](commercial-release-roadmap.md). Capacity: one developer, full time; pure software, no done-for-you service.

Legend: **GAP** = not started · **PARTIAL** = built, a real-world step still gates it · **DONE** = criteria met.

## M1 · weeks 1–4 — English works and payment is possible
Gate: every item below is DONE, or a kill signal from the biz doc fires.

| # | Item | Status | Done when |
|---|---|---|---|
| 1.1 | Payment provider applications (FastSpring first, plus one alternative), disclosing link download, clip transforms and watermark removal as they are | GAP | One provider approves the account for this product. Both refuse → kill signal |
| 1.2 | Rename `copyright_bypass` to a neutral name (e.g. `visual_variety`) and reword its CHANGELOG/UI copy; keep reading the old config key and `MRF_COPYRIGHT_BYPASS` as aliases so saved jobs and channel profiles still load | DONE | 13 tracked files updated (`git grep -i copyright_bypass` lists them); full test suite green; an old job config with the old key renders unchanged |
| 1.3 | English pipeline: prompts, pacing, wordplay, English TTS voices, caption rules | PARTIAL — code done (`narration_style.py`: English recap guide + ~155 wpm word targets in outline/script prompts, English scaffold, thumbnail and mid-roll CTA prompts; FPT.AI rejected for non-Vietnamese jobs; VieNeu allowed; captions already 2×42; render normalizes to -14 LUFS and extends short clips into unused footage instead of looping). 3/5 real videos rendered and QA-passed (`jobs/en-bike`, `en-ben10`, `en-vanishing`; Edge TTS, then VieNeu); 2 more films needed. VieNeu English has a Vietnamese accent and ~12% WER on names, so the benchmark should use Edge or ElevenLabs | 5 complete English videos, each under 10 manual fixes per 10 minutes |
| 1.4 | Blind comparison of the 5 videos against Recapo/Viralcade on the same films | GAP | At least 3/5 tie or win. Fails twice → kill signal |
| 1.5 | Compile the installer and launch it on a clean Windows machine (sibling roadmap Item 1) | GAP | Sibling roadmap Item 1 criteria met |
| 1.6 | Embed the real license public key; keep the private key offline (`docs/licensing.md`) | GAP | A vendor-issued key activates on a second machine |

## M2 · weeks 5–10 — Proof
| # | Item | Status | Done when |
|---|---|---|---|
| 2.1 | Vendor-run English channel on public-domain or licensed films only | GAP | 15–20 videos published with measured retention and growth |
| 2.2 | Automatic license issuance from the payment webhook (customer pastes machine code at checkout or after) | GAP | A test purchase delivers a working key with no manual step |
| 2.3 | Server-side update channel for yt-dlp, prompts and templates | GAP | An installed app picks up a new yt-dlp version without reinstalling |
| 2.4 | Self-serve onboarding: first-run wizard and a public-domain sample project | GAP | A new user exports a first video without contacting support |
| 2.5 | Landing page: benefit-first copy, Windows-only and minimum spec, ToS with user content responsibility, refund policy, demo from 2.1 | GAP | Live, with no Content ID claims anywhere |
| 2.6 | Founding sale: 20–50 seats at $499–699 with case-study consent | GAP | At least 10 sold, refunds under 10%, at least 3 customers publishing weekly |

## M3 · weeks 11–16 — Raise the price
| # | Item | Status | Done when |
|---|---|---|---|
| 3.1 | Document 3–5 customer case studies | GAP | Each has before/after output numbers and hours saved per video |
| 3.2 | Code signing certificate | GAP | Installer shows a named publisher instead of "Unknown publisher" |
| 3.3 | Switch pricing to $1,199+ perpetual plus ~25%/year updates | GAP | 3.1 done and support under ~5 hours/week |

## Out of scope
macOS build, multi-seat studio features, done-for-you setup.

## Sequencing
Week 1 starts 1.1 and 1.2 together (approval review takes time, and the rename must land before anything public). 1.3 → 1.4 run in parallel with them; 1.5 and 1.6 close M1. M2 starts only after the M1 gate.
