# Picture sync: keeping recap footage with the narration

Customer feedback: "long narration, slow clips, few shots, the line drifts away from the shot".

## Why it drifted

- A recap tells ~90 film minutes in ~10, a compression of ~9x. Section mode (Claude and
  other agents) gave each section at most 6 shots (15-30 s each); AGY beat mode gave one
  scene per 4 s+ sentence beat. A slot longer than its scene was filled by reading the film
  forward in real time (or looping), so the picture played at 1x while the voice moved 9x.
- Inside a section, shots were split by word weight, not by when the words are spoken, so
  cut points slid away from the sentence they illustrate.
- The Vietnamese prompt had no style guide, so scripts used long multi-event sentences that
  no single shot can match.

## What changed

1. `shot_planner.pace_shots` (runs in `_scene_plan` after mid-roll filling): every shot
   whose spoken slot exceeds the target (`MRF_SHOT_SECONDS`, default 3, clamped 1.5-12)
   becomes a montage. The matched scene stays first; the extra shots are unused indexed
   scenes sampled evenly between that scene and the next matched scene (if it is ahead and
   within 300 s), otherwise from the stretch right after it sized by the job's compression
   rate (film span / spoken time, clamped 1.5-12). Sub-shots share the original weight and a
   `beat` key. `scene_plan.json` records `pacing` (shot seconds, matched and montage counts).
2. Render timing: `_beat_timed_durations` maps each beat's token count onto the TTS word
   start times (`voice.json` `word_boundaries`, exact for Edge, chunk-estimated for VieNeu,
   FPT, ElevenLabs), so each beat starts on its first spoken word; a montage splits its
   beat by weight. Without per-beat narration or word timing it falls back to section bounds.
   `render.json` records `visual_timing_mode` and `beat_timed_sections`.
3. Vietnamese guide: film order with one on-screen moment per sentence, sentences of at most
   ~25 syllables, at most two commentary sentences in a row.

## Decision record

Decided: montage from scenes between matched anchors + word-timed beats - because it fixes
the compression lag without extra agent calls and never invents an order the matched scenes
do not already imply - rejected one AGY pick per 3 s sub-shot (3-5x more calls and latency),
speeding up footage (visibly odd, still desyncs), and only shortening the script (does not
fix the 1x playback) - reopen if viewers report montage shots that do not match the line;
then ask AGY to pick sub-shots only for beats over ~8 s.
