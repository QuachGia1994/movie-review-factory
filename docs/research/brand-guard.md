# Brand guard: transparent logo and removal-resistant watermark

Status: decided and shipped (exploratory findings, 2026-10). Code: `branding.py`, `pipeline._render`, `scripts/render_brand_logo.py`.

## Goal chain

Visible brand on every review → viewers remember the channel → re-uploads still credit or expose the source → ownership claims are easy. Constraint: the brand must not hide the footage (the old overlay drew dark plates behind logo, name and chapter caption, and the logo itself sat on a dark tile).

## Facts and constraints

- Re-uploaders strip watermarks with a static mask: FFmpeg `delogo`, blur/box over a fixed rectangle, or AI inpainting on a fixed region. All three assume the mark stays put.
- Cropping/zooming removes edge marks cheaply; it cannot remove a mark that also appears inside the picture.
- A mid-frame watermark at high opacity ruins the film for real viewers (retention signal), so anything inside the picture must be faint.
- Narration is kept by re-uploaders (it is the product). The AGY mid-roll CTA already says the channel name, so audio is the strongest brand layer.
- No visible watermark is unremovable; the realistic goal is to raise removal cost and keep ownership proof.

## Decision

1. **Transparent logo**: speech-bubble screen (màn = screen/curtain, kể = telling) — white bubble ring with tail, curtain drapes tied back, teal play triangle, soft dark halo instead of a tile. One geometry renders `assets/man-ke.png` + `man-ke.svg` (`py -3 scripts/render_brand_logo.py`).
2. **No plates**: outside opt-in letterbox bands, the lockup and chapter caption use text stroke + soft halo only.
3. **Moving corner lockup** (`brand-mark.png`): hops top-left ↔ top-right every `MARK_HOP_SECONDS` = 45 s; a single static mask leaves the brand visible half the time, two masks double the visible damage.
4. **Ghost lockup** (`brand-ghost.png`): the lockup at `GHOST_OPACITY` = 16 % jumps to a per-job pseudo-random spot every `GHOST_HOP_SECONDS` = 11 s, between 16 % and 64 % of frame height (clear of corner mark and captions). Seed comes from the job id, so positions differ per video.
5. Jobs with a top band keep the static lockup inside the band; the ghost still applies. Off switch: `MRF_BRAND_GUARD=0` (static corner lockup, no ghost).

## Critique (kept)

- Steelman for a big centred watermark: hardest to remove. Rejected: hurts viewers more than thieves.
- Attack: a full crop + per-frame AI inpainting can still remove everything. Accepted: that costs real GPU time per video; the audio CTA still names the channel.
- Inversion (what makes the brand useless): the mark disappears into bright footage → white glyphs with dark halo read on both.
- Second-order: a moving mark can distract → hop interval is long (45 s) and the ghost is faint.

## Rejected

- High-opacity centre watermark (ruins the film).
- Invisible forensic watermark (needs a detector service; out of scope now).
- MP4 metadata tags (stripped by any re-encode).

## Assumptions to monitor

- Viewer complaints about the ghost → lower `GHOST_OPACITY` or lengthen the hop.
- Re-uploads found with the brand removed → consider audio fingerprint registration (Content ID) before forensic watermarking.
- Custom uploaded logos must be transparent PNG (upload already rejects fully opaque images).
