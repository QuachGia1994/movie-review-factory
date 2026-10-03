# YouTube packaging and retention structure

Status: implemented (`packaging.py`, metadata stage, outline/script prompts, thumbnail band).

## Source

Two top Vietnamese review videos were studied: a single-film horror review (Linh Miu Review, "Marui Video") and an anthology review (Pikachu Review Phim, "Chuyện kinh dị ngắn Nhật Bản 2"). Both share one packaging pattern and one narrative skeleton; the pipeline produced neither before this change.

## Gaps found

| Area | Competitors | Before |
|---|---|---|
| Title | `Review phim {thể loại} \| {câu gây tò mò}` | `[VI] Review/Recap – {job_id}` |
| Description | hook paragraph, chapters, subscribe link, fair-use notice, hashtags | bullet list of section titles + "Bản nháp" |
| Opening | one greeting line with channel name, 2-3 curiosity questions, "stay to the end", story by ~0:40 | picture-sync rules only |
| Sections | open loop at each section end, twist kept for the end | none |
| Ending | payoff, specific comment question, like/subscribe, next part tease | none (mid-roll CTA only) |
| Thumbnail | bottom band "Review Phim : **Tên phim**" with accent colour | headline + channel badge |
| Anthology | "Câu chuyện thứ N" sections, chapters per story | no mode |

## Decisions

- Packaging lives in `packaging.py` so the metadata stage, prompts and handoff share one source (chapters moved there from `handoff.py`).
- Retention rules go into a separate `story_structure` context key, not the Vietnamese `narration_style` guide: that guide stays picture-sync only so the beat planner is unaffected, and the structure rules can carry the channel greeting and series position.
- Greeting and subscribe link are per channel profile (`greeting`, `channel_url`); jobs without a profile fall back to the brand name.
- No clickbait the video cannot pay off: titles and hook must promise only what the video answers, and rumours are labelled as rumours. The analysis layer of the script stays, because it is what keeps the channel clear of reused-content limits.
- AGY writes titles, hook, comment question and tags only for Claude/AGY jobs; a deterministic draft always exists, so an AGY failure never blocks metadata. Metadata approval stays a manual gate.
- Speaking rate stays at the measured 240 syllables/min; competitors run ~265 but the TTS voices were measured at 240.
- The mid-roll CTA is kept; analytics decide whether to drop it per channel.

## Genre voice

`genre_tone.py` maps the free-text `genre` (accent-insensitive, VI or EN) to 12 voices: horror, thriller, crime, action, comedy, romance, drama, sci-fi, fantasy, animation, war, disaster. Each voice adds tagged rules (`[kinh dị] ...`) to `story_structure`: tone, sentence rhythm, vocabulary and what to avoid. It also sets a genre-specific fallback comment question and tells AGY packaging to match the mood. Blends such as "Kinh dị hài" lead with the first genre and borrow one rule from the second; unknown genres add nothing. `genre_voice` in the outline/script context names the chosen voice.

The same genre also sets the spoken voice:

| Genre | Rate | Pitch | Edge VI / EN voice | FPT.AI |
|---|---|---|---|---|
| horror | -10% | -6Hz | NamMinh / Guy | leminh |
| thriller | +4% | -2Hz | NamMinh / Guy | leminh |
| crime | -4% | -3Hz | NamMinh / Guy | leminh |
| action | +10% | +4Hz | NamMinh / Guy | leminh |
| comedy | +8% | +6Hz | HoaiMy / Aria | banmai |
| romance | -6% | +2Hz | HoaiMy / Aria | banmai |
| drama | -6% | -3Hz | HoaiMy / Aria | banmai |
| sci-fi | -2% | | NamMinh / Guy | leminh |
| fantasy | -4% | -2Hz | NamMinh / Guy | leminh |
| animation | +6% | +8Hz | HoaiMy / Aria | banmai |
| war | -8% | -5Hz | NamMinh / Guy | leminh |
| disaster | +6% | -2Hz | NamMinh / Guy | leminh |

- Precedence: a job `tts_voice` (or `MRF_TTS_VOICE` for network providers) beats the genre voice; an `MRF_TTS_EMOTION` preset replaces the genre prosody; `MRF_TTS_RATE/PITCH/VOLUME` override single values; `MRF_GENRE_VOICE=0` turns genre voice and prosody off.
- Edge applies rate and pitch; FPT.AI gets the `speed` header in steps of about 10% (-3..3); ElevenLabs gets `voice_settings.speed` (0.7-1.2); VieNeu keeps its default speed and voice. ElevenLabs voices are account-specific, so its voice is not switched.
- Script word targets, the beat planner and shot pacing use `240 x rate` syllables/min, so a slower horror voice gets ~10% fewer words and the video keeps its target length.

## Not done

- Per-section mood changes inside one video (one prosody per job).
- The genre speaking rates are starting values; they have not been measured against real renders like the 240 baseline.
- Automatic end-screen or card placement on YouTube (manual upload package only).
