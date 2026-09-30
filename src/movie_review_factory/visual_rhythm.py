"""Dynamic visual rhythm - Ken Burns pan/zoom and scene-transition SFX.

Reviewers never let the frame sit still: a slow, continuous pan/zoom (the "Ken
Burns" effect) keeps motion on screen, and a short whoosh/impact on every scene
cut punctuates the edit. Both are opt-in in the render stage (``MRF_KEN_BURNS``
and ``MRF_TRANSITION_SFX``) so the default, deterministic render is byte-for-byte
unchanged.

Everything here is pure string/number building so it is fully unit-testable
without invoking ffmpeg.
"""
from __future__ import annotations

KEN_BURNS_MAX_INTENSITY = 0.4
KEN_BURNS_MIN_INTENSITY = 0.02


def ken_burns_filter(
    width: int,
    height: int,
    fps: int,
    frames: int,
    *,
    index: int = 0,
    intensity: float = 0.12,
) -> str:
    """A ffmpeg filter fragment that applies a slow pan/zoom over ``frames``.

    Returns the geometry portion of a clip chain (no input/output pad labels) so
    it can replace the plain ``scale``/``pad`` step. The source is upscaled 2x
    first so the zoompan crop never shows padding, the zoom drifts toward
    ``1 + intensity`` across the clip, and the pan direction alternates by
    ``index`` so consecutive clips do not all drift the same way.
    """
    frames = max(1, int(frames))
    intensity = max(KEN_BURNS_MIN_INTENSITY, min(KEN_BURNS_MAX_INTENSITY, float(intensity)))
    max_zoom = round(1.0 + intensity, 3)
    step = round(intensity / frames, 6) or 0.0005
    over_w, over_h = width * 2, height * 2
    if index % 2 == 0:
        # Pan left -> right.
        x_expr = f"(iw-iw/zoom)/2*on/{frames}"
    else:
        # Pan right -> left.
        x_expr = f"(iw-iw/zoom)-(iw-iw/zoom)/2*on/{frames}"
    y_expr = "(ih-ih/zoom)/2"
    return (
        f"scale={over_w}:{over_h}:force_original_aspect_ratio=increase,"
        f"crop={over_w}:{over_h},"
        f"zoompan=z='min(zoom+{step},{max_zoom})':x='{x_expr}':y='{y_expr}':"
        f"d={frames}:s={width}x{height}:fps={fps}"
    )


def transition_cut_times(ranges: list[dict], *, intro_seconds: float = 0.0) -> list[float]:
    """Timeline seconds where a scene changes (a whoosh should hit).

    Excludes t=0 (the very start) and the tail after the final clip, so a whoosh
    only ever lands on an actual cut between two clips.
    """
    times: list[float] = []
    running = float(intro_seconds)
    for item in ranges[:-1]:
        running += float(item.get("duration_seconds", 0.0))
        times.append(round(running, 3))
    return times


def transition_sfx_filtergraph(
    audio_label: str,
    sfx_input_index: int,
    cut_times: list[float],
) -> tuple[str, str]:
    """Build the filtergraph that lays a whoosh on every cut over the base audio.

    A single SFX input is split N ways, each copy delayed to a cut time, then all
    are amixed over the base audio (``normalize=0`` keeps the narration level).
    Returns ``(filter_fragment, out_label)``; when there are no cuts the fragment
    is empty and the original ``audio_label`` is returned unchanged.
    """
    if not cut_times:
        return "", audio_label
    count = len(cut_times)
    base = audio_label if audio_label.startswith("[") else f"[{audio_label.split(':', 1)[0]}:a]"
    parts = ["[%d:a]asplit=%d%s" % (sfx_input_index, count, "".join(f"[s{k}]" for k in range(count)))]
    delayed: list[str] = []
    for k, seconds in enumerate(cut_times):
        ms = int(round(float(seconds) * 1000))
        parts.append(f"[s{k}]adelay={ms}|{ms}[d{k}]")
        delayed.append(f"[d{k}]")
    parts.append(
        f"{base}{''.join(delayed)}"
        f"amix=inputs={count + 1}:duration=first:dropout_transition=0:normalize=0[audiosfx]"
    )
    return ";".join(parts), "[audiosfx]"
