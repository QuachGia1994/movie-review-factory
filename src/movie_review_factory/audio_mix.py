"""Explicit, rights-tracked optional audio layers for a narrated review."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class AudioMix:
    input_args: tuple[str, ...]
    filters: tuple[str, ...]
    output_map: str
    provenance: tuple[dict, ...]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gain(value: object, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{label} must be between {minimum:g} and {maximum:g} dB")
    return number


def _asset(spec: object, label: str) -> tuple[Path, str]:
    if not isinstance(spec, Mapping):
        raise ValueError(f"{label} must contain path and rights_note")
    path = spec.get("path")
    note = spec.get("rights_note")
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{label}.path is required")
    if not isinstance(note, str) or not note.strip():
        raise ValueError(f"{label}.rights_note is required")
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise ValueError(f"{label}.path does not exist: {file}")
    return file, note.strip()


def build_audio_mix(
    config: Mapping[str, object] | None,
    overlay_count: int,
    duration_seconds: float | None,
) -> AudioMix:
    """Build FFmpeg arguments/filters; narration remains the first and master audio.

    The caller appends input_args after its visual overlay inputs, filters to its
    existing filter_complex, and maps output_map in place of 1:a:0. An empty
    config leaves original voice audio untouched.
    """
    if not isinstance(overlay_count, int) or overlay_count < 0:
        raise ValueError("overlay_count must be nonnegative")
    if duration_seconds is not None and (not math.isfinite(duration_seconds) or duration_seconds <= 0):
        raise ValueError("duration_seconds must be positive")
    if not config:
        return AudioMix((), (), "1:a:0", ())
    if not isinstance(config, Mapping):
        raise ValueError("audio config must be an object")

    voice_gain = _gain(config.get("voice_gain_db", 0), "voice_gain_db", -12, 12)
    music = config.get("music")
    effects = config.get("effects", [])
    if not isinstance(effects, list) or len(effects) > 16:
        raise ValueError("effects must be a list of at most 16 clips")
    if music is None and not effects and voice_gain == 0:
        return AudioMix((), (), "1:a:0", ())

    args: list[str] = []
    filters: list[str] = []
    provenance: list[dict] = []
    index = 2 + overlay_count
    if music is not None:
        path, note = _asset(music, "music")
        gain = _gain(music.get("gain_db", -18), "music.gain_db", -36, 0)
        args.extend(("-stream_loop", "-1", "-i", str(path)))
        provenance.append({"kind": "music", "path": str(path), "sha256": _digest(path),
                           "rights_note": note, "gain_db": gain})
        filters.append(
            f"[{index}:a:0]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"volume={gain:g}dB[music]"
        )
        index += 1

    effect_labels: list[str] = []
    for number, spec in enumerate(effects):
        path, note = _asset(spec, f"effects[{number}]")
        gain = _gain(spec.get("gain_db", -12), f"effects[{number}].gain_db", -36, 0)
        offset = spec.get("at_seconds", 0)
        if isinstance(offset, bool) or not isinstance(offset, (float, int)) or not math.isfinite(float(offset)):
            raise ValueError(f"effects[{number}].at_seconds must be finite")
        offset = float(offset)
        if offset < 0 or (duration_seconds is not None and offset >= duration_seconds):
            raise ValueError(f"effects[{number}].at_seconds falls outside narration")
        args.extend(("-i", str(path)))
        label = f"effect{number}"
        effect_labels.append(f"[{label}]")
        filters.append(
            f"[{index}:a:0]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"volume={gain:g}dB,adelay={round(offset * 1000)}:all=1[{label}]"
        )
        provenance.append({"kind": "effect", "path": str(path), "sha256": _digest(path),
                           "rights_note": note, "gain_db": gain, "at_seconds": offset})
        index += 1

    filters.insert(0, f"[1:a:0]aformat=sample_rates=48000:channel_layouts=stereo,"
                      f"volume={voice_gain:g}dB[voice_master]")
    if music is not None:
        filters.append("[voice_master]asplit=2[voice_mix][duck_key]")
        filters.append(
            "[music][duck_key]sidechaincompress="
            "threshold=0.125:ratio=4:attack=20:release=250[ducked_music]"
        )
        voice_label = "[voice_mix]"
        layer_labels = ["[ducked_music]", *effect_labels]
    else:
        voice_label = "[voice_master]"
        layer_labels = effect_labels

    if layer_labels:
        filters.append(
            f"{voice_label}{''.join(layer_labels)}amix="
            f"inputs={len(layer_labels) + 1}:duration=first:"
            "dropout_transition=0:normalize=0,alimiter=limit=0.95:level=false[audio]"
        )
    else:
        filters.append(f"{voice_label}alimiter=limit=0.95:level=false[audio]")
    return AudioMix(tuple(args), tuple(filters), "[audio]", tuple(provenance))
