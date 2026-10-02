# Creator product identity
> updated 2026-10-02 · v0.2.0

## Audience and job
Primary: one solo creator running an English-language movie recap/commentary channel on Windows, producing 8–12 minute original commentary videos. Secondary: Vietnamese movie-review creators (the language the pipeline is already proven in). Out of scope for v1: multi-channel studios (multi-seat, headless queue, per-channel reporting) and macOS.

## Positioning
Windows-local review desk that links each draft claim, narration cue and chosen film shot to its source timecode, then lets the creator inspect and edit a playable review before export. Proposed differentiators versus per-video SaaS (Recapo, Viralcade), **unverified** until a blind comparison: no per-video vendor fee (the creator's own API keys, or local models), and source-grounded scripts with an editable timeline in one project.

## Offer
- Pure software. No done-for-you setup or 1:1 onboarding; support is docs plus async email/Discord.
- Windows only, stated on the sales page.
- AI mode: the creator's own API keys by default; local models are an optional "Unlimited mode" with a published minimum spec.
- License: offline, machine-bound (`docs/licensing.md`). The annual fee buys server-side value — yt-dlp and prompt updates, new templates and styles, scout data — not the activation check.
- Price ladder (provisional): founding license $499–699 for the first 20–50 seats in exchange for case-study consent; $1,199+ perpetual plus ~25%/year updates only after 3–5 documented case studies.

## Feature boundary
Owner decision 2026-10-02: the international build keeps link download (yt-dlp), clip transforms and watermark removal. Accepted risk: payment providers restrict products that enable infringement (Paddle's AUP names streaming downloaders and content copying, read 2026-10-02), so payment approval is the first launch gate. Applications disclose these features as they are. Messaging sells original commentary and never claims Content ID evasion; the usage-rights confirmation gate stays, and the user ToS places content responsibility on the user. Demo material uses public-domain or licensed films only.

## Revenue path
Paid desktop license as above. Kill signals: the English blind comparison fails twice, or no payment provider approves the product with these features — then return to the Vietnamese market first.

## Proof to seek
- M1: 5 complete English videos; at least 3/5 tie or win a blind comparison against a competitor on the same film; under 10 manual fixes per 10-minute output; a payment provider approves the account; the installer runs on a clean Windows machine.
- M2: the vendor's own English channel publishes 15–20 videos with measured retention and growth; at least 10 founding seats sold with refunds under 10%; at least 3 customers publishing weekly.
- M3: 3–5 documented case studies and support under ~5 hours/week before raising the price.
