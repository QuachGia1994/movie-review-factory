#!/usr/bin/env python3
"""R15 clean-Windows release gate.

Runs the whole local production chain on the current machine with a scaffold
content agent (no AGY pool, no Claude) and prints one PASS/FAIL/BOO row per
step to stdout:

    package-import -> fixture -> ingest -> research -> transcript -> scenes ->
    search -> outline -> script -> scene_plan -> approve-script -> tts ->
    alignment -> render -> qa -> export-video / export-subtitles /
    export-transcript

Status meanings (failures are never hidden, only classified):
    PASS  step ran and its own acceptance check held
    BOO   step was skipped; the exact reason is printed on the row
    FAIL  step ran and failed; the raw message is printed on the row

Exit codes:
    0  every step PASS (gate green)
    1  at least one FAIL
    2  no FAIL, but at least one BOO (gate not fully green)

Scope: this file writes only its own fixture under ``jobs/release-gate-fixture``
(guarded by a marker file so it can never delete a directory it did not
create). It never edits src, docs or git.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_ROOT_NAME = "release-gate-fixture"
GATE_MARKER = ".mrf-release-gate-root"
JOB_ID = "release-gate-job"
# Preferred gate source: a real Vietnamese speech clip (~9.7s, generated with
# edge-tts for R19). Measured with WhisperModel small int8 on this machine:
# vad_filter=True -> 2 segments (vad_filter=False -> 2 segments), so it is the
# only bundled source whose speech Silero VAD accepts.
GATE_SPEECH_VIDEO = REPO_ROOT / "data" / "raw" / "mrf-gate-speech.mp4"
SAMPLE_VIDEO = REPO_ROOT / "data" / "raw" / "ultracode-smoke-sample.mp4"
# Fallback: this bundled sample is only ~2.0s and its audio is bed music, so the
# fixture source is looped up to MIN_SOURCE_SECONDS before ingest (ffprobe/ingest
# must still see has_audio=true and a duration near this value) while the pipeline
# transcribes with vad_filter=True. Measured: looping alone does not recover
# segments - Silero VAD scores this sample's audio below threshold at 2.0s,
# 10.1s and 20.2s (0 segments in each case).
MIN_SOURCE_SECONDS = 10.0
APP_NAME = "MovieReviewFactory"
PROFILE_MODULES = ("pydantic", "typer", "faster_whisper", "fastembed", "edge_tts")
CORE_MODULES = ("pydantic", "typer")

PASS, FAIL, BOO = "PASS", "FAIL", "BOO"

STEP_ORDER = (
    "package-import", "fixture", "ingest", "research", "transcript", "scenes",
    "search", "outline", "script", "scene_plan", "approve-script",
    "tts", "alignment", "render", "qa",
    "export-video", "export-subtitles", "export-transcript",
)


# --- stdout -----------------------------------------------------------------

def emit(text: str = "") -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    print(text, flush=True)


def make_row(step: str, status: str, detail: str, seconds: float | None = None) -> dict:
    return {"step": step, "status": status, "detail": detail, "seconds": seconds}


def print_row(item: dict) -> None:
    tail = f"  [{item['seconds']:.1f}s]" if item.get("seconds") is not None else ""
    emit(f"[{item['status']:^4}] {item['step']:<17} {item['detail']}{tail}")


# --- environment discovery --------------------------------------------------

def app_root() -> Path:
    custom = os.environ.get("MRF_HOME", "").strip()
    if custom:
        return Path(custom)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / APP_NAME


def _valid_ffmpeg_dir(directory: Path) -> bool:
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    probe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    return (directory / exe).is_file() and (directory / probe).is_file()


def find_ffmpeg_bin() -> Path | None:
    """Locate an ffmpeg/ffprobe bin dir the same way the launcher does."""
    configured = os.environ.get("MRF_FFMPEG_BIN", "").strip()
    if configured and _valid_ffmpeg_dir(Path(configured)):
        return Path(configured)
    on_path = shutil.which("ffmpeg")
    if on_path:
        candidate = Path(on_path).resolve().parent
        if _valid_ffmpeg_dir(candidate):
            return candidate
    ffmpeg_root = app_root() / "toolchain" / "ffmpeg"
    if ffmpeg_root.is_dir():
        pattern = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        for candidate in sorted(ffmpeg_root.rglob(pattern)):
            if _valid_ffmpeg_dir(candidate.parent):
                return candidate.parent
    return None


def configure_env() -> dict:
    """Mirror the launcher's child environment (PATH, FFmpeg, model caches)."""
    info: dict[str, object] = {"ffmpeg_bin": None, "ffmpeg_source": "not found"}
    configured = os.environ.get("MRF_FFMPEG_BIN", "").strip()
    ffmpeg_bin = find_ffmpeg_bin()
    if ffmpeg_bin:
        current = os.environ.get("PATH", "")
        if str(ffmpeg_bin).lower() not in current.lower().split(os.pathsep):
            os.environ["PATH"] = str(ffmpeg_bin) + os.pathsep + current
        os.environ["MRF_FFMPEG_BIN"] = str(ffmpeg_bin)
        info["ffmpeg_bin"] = str(ffmpeg_bin)
        info["ffmpeg_source"] = "env MRF_FFMPEG_BIN" if configured else "PATH or toolchain cache"
    root = app_root()
    os.environ.setdefault("MRF_WHISPER_CACHE", str(root / "models" / "whisper"))
    os.environ.setdefault("MRF_EMBED_CACHE", str(root / "models" / "embeddings"))
    os.environ.setdefault("HF_HOME", str(root / "cache" / "huggingface"))
    return info


def package_candidates() -> list[Path]:
    """Repo source first, then launcher-extracted runtimes, newest first."""
    candidates: list[Path] = []
    source = REPO_ROOT / "src"
    if (source / "movie_review_factory" / "pipeline.py").is_file():
        candidates.append(source)
    runtime_root = app_root() / "runtime"
    if runtime_root.is_dir():
        runtimes = [
            entry for entry in runtime_root.iterdir()
            if entry.is_dir() and (entry / "movie_review_factory" / "pipeline.py").is_file()
        ]
        runtimes.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
        candidates.extend(runtimes)
    return candidates


def probe_package(root: Path) -> tuple[bool, str]:
    """Import movie_review_factory.pipeline from this root, in a subprocess."""
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import movie_review_factory.pipeline; print('import ok')" % str(root)
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, " | ".join(line.strip() for line in lines[-4:]) or f"exit {proc.returncode}"
    return True, "movie_review_factory.pipeline import ok"


PROBE_CODE = (
    "import importlib.util as u, json, sys; "
    "print(json.dumps({'exe': sys.executable, "
    "'version': '.'.join(str(v) for v in sys.version_info[:3]), "
    "'mods': {n: bool(u.find_spec(n)) for n in %r}}))" % (list(PROFILE_MODULES),)
)


def probe_python(python: str) -> dict | None:
    try:
        proc = subprocess.run(
            [python, "-c", PROBE_CODE],
            capture_output=True, text=True, timeout=90,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    info["requested"] = python
    return info


def managed_venv_python() -> Path:
    return app_root() / "toolchain" / "venv-py312" / "Scripts" / (
        "python.exe" if os.name == "nt" else "python"
    )


def resolve_python(args: argparse.Namespace) -> dict | None:
    """Pick the interpreter with the richest runtime profile available."""
    candidates = [args.python or os.environ.get("MRF_PYTHON", "").strip()]
    if not args.python:
        candidates += [sys.executable, str(managed_venv_python()), "python"]
    best: dict | None = None
    best_score = -1
    for candidate in [c for c in candidates if c]:
        info = probe_python(candidate)
        if not info:
            continue
        mods = info.get("mods", {})
        if not all(mods.get(name) for name in CORE_MODULES):
            score = 0
        elif all(mods.get(name) for name in PROFILE_MODULES):
            score = 2
        else:
            score = 1
        if score > best_score:
            best, best_score = info, score
        if score == 2:
            break
    if best is None or best_score <= 0:
        return None
    return best


def same_executable(left: str, right: str) -> bool:
    if not left or not right or not os.path.isfile(left):
        return False
    try:
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))
    except OSError:
        return False


# --- fixture ----------------------------------------------------------------

def reset_gate_root(gate_root: Path) -> None:
    """Recreate the gate's own jobs root; refuse to touch anything else."""
    if gate_root.exists():
        if not (gate_root / GATE_MARKER).is_file():
            raise RuntimeError(
                f"refusing to reuse {gate_root}: marker {GATE_MARKER} missing, "
                "so this directory was not created by release_gate.py"
            )
        shutil.rmtree(gate_root)
    gate_root.mkdir(parents=True)
    (gate_root / GATE_MARKER).write_text(
        "created by scripts/release_gate.py\n", encoding="utf-8"
    )


def probe_source(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not on PATH - FFmpeg is required for the gate fixture")
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr.strip()[:300]}")
    probe = json.loads(proc.stdout)
    streams = probe.get("streams", [])
    return {
        "duration": float(probe.get("format", {}).get("duration") or 0),
        "audio": any(s.get("codec_type") == "audio" for s in streams),
        "video": any(s.get("codec_type") == "video" for s in streams),
    }


def make_lavfi_source(target: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "no ffmpeg on PATH and the bundled sample has no audio - cannot build fixture"
        )
    proc = subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
         "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         str(target)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg lavfi fixture failed: {proc.stderr.strip()[-400:]}")
    return target


def make_looped_source(sample: Path, target: Path, duration: float,
                       min_seconds: float) -> tuple[Path, dict]:
    """Repeat ``sample`` with -stream_loop until it reaches ``min_seconds``.

    Stream copy is tried first (no re-encode); if the container comes out
    unusable or too short the fixture is re-encoded to h264/aac instead.
    Only ever writes ``target`` - never a file outside the gate fixture dir.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("no ffmpeg on PATH - cannot loop the short sample into a fixture")
    copies = max(2, int(math.ceil(min_seconds / duration)))
    stream_loop = copies - 1
    attempts = (
        ("-c:v", "copy", "-c:a", "copy"),
        ("-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"),
    )
    errors: list[str] = []
    for codecs in attempts:
        proc = subprocess.run(
            [ffmpeg, "-y", "-stream_loop", str(stream_loop), "-i", str(sample),
             *codecs, str(target)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not target.is_file() or target.stat().st_size <= 0:
            errors.append(f"{'/'.join(codecs[:2])}: {proc.stderr.strip()[-200:]}")
            continue
        try:
            info = probe_source(target)
        except RuntimeError as exc:
            errors.append(f"{'/'.join(codecs[:2])}: {exc}")
            continue
        if info["video"] and info["audio"] and info["duration"] >= min_seconds * 0.95:
            info["copies"] = copies
            info["codecs"] = "/".join(codecs[:2])
            return target, info
        errors.append(f"{'/'.join(codecs[:2])}: probe {info}")
    raise RuntimeError(f"ffmpeg loop fixture failed: {'; '.join(errors)}")


def resolve_source(gate_root: Path) -> tuple[Path, str]:
    """Return (source video, note).

    Candidate order: the shipped speech asset (VAD-visible, preferred), the
    bundled sample (looped up to ``MIN_SOURCE_SECONDS``), then a lavfi fixture.
    A candidate shorter than ``MIN_SOURCE_SECONDS`` is looped inside the gate
    fixture dir - never anywhere else.
    """
    target = gate_root / "fixture-source.mp4"
    for origin in (GATE_SPEECH_VIDEO, SAMPLE_VIDEO):
        if not (origin.is_file() and origin.stat().st_size > 0):
            continue
        try:
            info = probe_source(origin)
        except RuntimeError as exc:
            emit(f"       {origin.name} probe failed ({exc}); trying the next candidate")
            continue
        if not (info["video"] and info["audio"] and info["duration"] > 0):
            emit(f"       {origin.name} has no usable audio ({info}); trying the next candidate")
            continue
        if info["duration"] >= MIN_SOURCE_SECONDS:
            return origin, f"{origin} ({info['duration']:.3f}s, with audio)"
        # Shorter than the gate's minimum: loop it inside the fixture dir.
        copies = max(2, int(math.ceil(MIN_SOURCE_SECONDS / info["duration"])))
        emit(
            f"       {origin.name} is only {info['duration']:.3f}s; looping it x{copies} "
            "so the fixture source reaches the gate minimum"
        )
        try:
            looped, loop_info = make_looped_source(
                origin, target, info["duration"], MIN_SOURCE_SECONDS
            )
        except RuntimeError as exc:
            emit(f"       looping failed ({exc}); using {origin.name} as-is")
            return origin, (
                f"{origin} ({info['duration']:.3f}s, with audio, loop failed)"
            )
        return looped, (
            f"looped fixture {looped} ({loop_info['duration']:.3f}s, with audio, "
            f"{loop_info['copies']}x {loop_info['codecs']} of {info['duration']:.3f}s "
            f"{origin.name})"
        )
    emit(
        "       no usable bundled source (speech asset / sample); "
        "building a lavfi fixture"
    )
    make_lavfi_source(target)
    info = probe_source(target)
    return target, f"ffmpeg lavfi fixture {target} ({info['duration']:.3f}s, with audio)"


# --- computed gate steps ----------------------------------------------------

def step_search(job_dir, scenes_doc, MediaStore, semantic_search) -> tuple[str, str]:
    database = job_dir / "media_index.sqlite3"
    if not database.is_file():
        return BOO, "media_index.sqlite3 missing - the scenes index step produced no database"
    mode = scenes_doc.get("semantic_mode")
    if mode != "fastembed":
        return BOO, (
            "semantic index not embedded: semantic_mode="
            f"{mode!r} error={scenes_doc.get('semantic_error') or 'n/a'}"
        )
    try:
        with MediaStore(database) as store:
            store.migrate()
            results = semantic_search.search_store(
                store, "cảnh mở đầu phim", limit=5, min_score=0.0
            )
    except Exception as exc:  # noqa: BLE001 - the gate reports the raw failure
        return FAIL, f"search raised {type(exc).__name__}: {exc}"
    if not results:
        return FAIL, f"search_store returned 0 rows for model {semantic_search.model_name()}"
    top = results[0]
    return PASS, (
        f"{len(results)} rows from model {semantic_search.model_name()}; "
        f"top={top['kind']} score={float(top['score']):.3f}"
    )


def step_export_video(job_dir: Path, render_status: str, note: str = "") -> tuple[str, str]:
    final = job_dir / "final.mp4"
    if not final.is_file() or final.stat().st_size <= 0:
        detail = f"final.mp4 missing or empty (render status={render_status})"
        return BOO, f"{detail} | {note}" if note else detail
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return BOO, "ffprobe not on PATH - exported video cannot be validated"
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(final)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return FAIL, f"ffprobe failed on final.mp4: {proc.stderr.strip()[:300]}"
    probe = json.loads(proc.stdout)
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    duration = float(probe.get("format", {}).get("duration") or 0)
    if duration <= 0 or not video or not audio:
        return FAIL, (
            f"exported mp4 invalid: duration={duration} video={bool(video)} audio={bool(audio)}"
        )
    return PASS, (
        f"final.mp4 {duration:.3f}s {video.get('width')}x{video.get('height')} "
        f"{video.get('codec_name')}/{audio.get('codec_name')} "
        f"{final.stat().st_size} bytes at {final}"
    )


def step_export_captions(job_dir: Path, alignment_stage: dict, note: str = "") -> tuple[str, str]:
    """The narration subtitle track that the render burns into final.mp4."""
    captions = job_dir / "aligned.srt"
    if not captions.is_file():
        detail = (
            "aligned.srt missing (alignment stage: "
            f"{alignment_stage.get('status')}: {alignment_stage.get('message') or 'no message'})"
        )
        return BOO, f"{detail} | {note}" if note else detail
    text = captions.read_text(encoding="utf-8")
    if "-->" not in text:
        return FAIL, f"aligned.srt has no cue timing line: {text[:80]!r}"
    cues = [line for line in text.splitlines() if "-->" in line]
    if not cues:
        return FAIL, "aligned.srt parsed but contains no cues"
    return PASS, f"aligned.srt {len(cues)} cues, {len(text)} bytes"


def step_export_transcript(job_dir: Path, transcript_stage: dict, jobs_root: Path,
                           MediaStore, JobsService, note: str = "") -> tuple[str, str]:
    pending = transcript_stage.get("status") == "pending"
    database = job_dir / "media_index.sqlite3"
    if not database.is_file():
        detail = "media_index.sqlite3 missing - nothing to export subtitles from"
        return BOO, f"{detail} | {note}" if (note and pending) else detail
    try:
        with MediaStore(database) as store:
            segments = store.list_transcript(1)
    except Exception as exc:  # noqa: BLE001
        return FAIL, f"reading indexed transcript raised {type(exc).__name__}: {exc}"
    if not segments:
        detail = (
            "no indexed transcript segments to export (transcript stage: "
            f"{transcript_stage.get('status')}: {transcript_stage.get('message') or 'no message'})"
        )
        return BOO, f"{detail} | {note}" if (note and pending) else detail
    try:
        vtt = JobsService(jobs_root).transcript_export_path(JOB_ID, "vtt")
        text = vtt.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return FAIL, f"transcript export raised {type(exc).__name__}: {exc}"
    if not text.startswith("WEBVTT"):
        return FAIL, f"transcript.vtt is not valid WebVTT: {text[:60]!r}"
    return PASS, f"transcript.vtt {len(text)} bytes from {len(segments)} indexed segments"


# --- checklist --------------------------------------------------------------

CHECKLIST = [
    ("where node && where python && where ffmpeg",
     "no output for any of them (machine is clean)"),
    ("cscript //nologo movie-review-factory.js --wsh-self-test",
     'stdout starts with "MRF_WSH_OK " and '
     "%LOCALAPPDATA%\\MovieReviewFactory\\bootstrap\\node\\node.exe exists"),
    ("node movie-review-factory.js --self-test --no-browser",
     'exit 0 and stdout contains SELF-TEST {"ok": true'),
    ("node movie-review-factory.js --self-test --no-browser --no-network",
     "exit 0 (second launch served entirely from cache)"),
    ('powershell -c "Test-Path $env:LOCALAPPDATA\\MovieReviewFactory\\toolchain\\'
     'venv-py312\\Scripts\\python.exe"',
     "True"),
    ('powershell -c "& $env:LOCALAPPDATA\\MovieReviewFactory\\toolchain\\venv-py312\\'
     'Scripts\\python.exe -c \'import pydantic,typer,faster_whisper,fastembed,edge_tts\'"',
     "exit 0 (no ModuleNotFoundError)"),
    ('powershell -c "(Get-ChildItem $env:LOCALAPPDATA\\MovieReviewFactory\\toolchain\\'
     'ffmpeg -Recurse -Filter ffmpeg.exe | Measure-Object).Count"',
     ">= 1"),
    ('powershell -c "(Get-ChildItem $env:LOCALAPPDATA\\MovieReviewFactory\\models -Recurse '
     '-File | Measure-Object).Count"',
     "> 0 (Whisper + FastEmbed model cache present)"),
    ("copy scripts\\release_gate.py next to movie-review-factory.js (or keep it in the repo)",
     "file present"),
    ('"%LOCALAPPDATA%\\MovieReviewFactory\\toolchain\\venv-py312\\Scripts\\python.exe" '
     "scripts\\release_gate.py",
     "exit 0 and every STEP row = PASS (content_agent=scaffold, no AGY/Claude)"),
    ("ffprobe -v error -show_entries format=duration -of csv=p=0 "
     "jobs\\release-gate-fixture\\release-gate-job\\final.mp4",
     "a positive number (rendered deliverable exists)"),
    ('powershell -c "Get-Content jobs\\release-gate-fixture\\release-gate-job\\aligned.srt '
     '-TotalCount 3"',
     "first cue of aligned.srt: a cue number, a --> timing line and its text "
     "(mandatory - the narration subtitle track burned into final.mp4)",
     "conditional - only when the source has speech (the gate prefers "
     "data\\raw\\mrf-gate-speech.mp4, a real Vietnamese speech clip measured at "
     "2 segments with Whisper small int8 + vad_filter=True, looped to the gate "
     "minimum; the fallback sample's bed music yields 0 segments) "
     "AND the transcript stage indexed it (>= 1 segment): powershell -c \"Get-Content "
     'jobs\\release-gate-fixture\\release-gate-job\\transcript.vtt -TotalCount 1" '
     "-> first line WEBVTT (a source whose speech Whisper/VAD drops produces "
     "no transcript.vtt)"),
    ('"%LOCALAPPDATA%\\MovieReviewFactory\\toolchain\\venv-py312\\Scripts\\python.exe" '
     "scripts\\release_gate.py --offline",
     "steps through scene_plan PASS; tts/alignment/render/qa/export-video/"
     "export-subtitles/export-transcript = BOO with an explicit network reason, 0 FAIL"),
]


# --- gate run ---------------------------------------------------------------

def run_gate(args: argparse.Namespace) -> int:
    started = time.time()
    emit("=== R15 CLEAN-WINDOWS RELEASE GATE ===")
    emit(f"time            : {time.strftime('%Y-%m-%d %H:%M:%S %z')}")
    emit(f"machine         : {os.environ.get('COMPUTERNAME', '')} {sys.platform}")
    emit(f"python          : {sys.executable} ({sys.version.split()[0]})")

    rows: dict[str, dict] = {}
    repo_src = REPO_ROOT / "src"
    candidates = package_candidates()
    src_present = (repo_src / "movie_review_factory" / "pipeline.py").is_file()
    if src_present:
        emit("probing repo src import ...")
        src_ok, src_detail = probe_package(repo_src)
    else:
        src_ok, src_detail = True, "repo src not shipped with this gate - nothing to validate"
    selected: Path | None = repo_src if (src_present and src_ok) else None
    fallback_note = ""
    if selected is None:
        for candidate in candidates:
            if candidate == repo_src:
                continue
            emit(f"repo src unusable; probing runtime copy {candidate} ...")
            ok, detail = probe_package(candidate)
            if ok:
                selected = candidate
                fallback_note = f" | chain continues on runtime copy {candidate}"
                break
            fallback_note = f" | runtime copy also unusable: {detail}"
    if src_ok:
        detail = f"repo src: {src_detail}"
        if selected is not None and selected != repo_src:
            detail += f" | package source = {selected}"
        rows["package-import"] = make_row("package-import", PASS, detail)
    else:
        rows["package-import"] = make_row(
            "package-import", FAIL, f"repo src: {src_detail}{fallback_note}"
        )
    if selected is None:
        print_row(rows["package-import"])
        emit("[FAIL] environment       no importable movie_review_factory package found")
        return 1

    sys.path.insert(0, str(selected))
    env_info = configure_env()
    emit(f"package source  : {selected}{' (repo src)' if selected == repo_src else ' (launcher runtime)'}")
    emit(f"ffmpeg          : {env_info['ffmpeg_bin']} ({env_info['ffmpeg_source']})")
    emit(f"whisper cache   : {os.environ.get('MRF_WHISPER_CACHE')}")
    emit(f"embedding cache : {os.environ.get('MRF_EMBED_CACHE')}")
    emit("")

    from movie_review_factory import pipeline, semantic_search
    from movie_review_factory.media_store import MediaStore
    from movie_review_factory.models import JobConfig
    from movie_review_factory.webapp import JobsService

    profile = ", ".join(
        f"{name}={'yes' if importlib.util.find_spec(name) else 'NO'}"
        for name in PROFILE_MODULES
    )
    emit(f"runtime profile : {profile}")
    emit("content agent   : scaffold (AGY pool / Claude are never invoked)")
    emit("")

    gate_root = REPO_ROOT / "jobs" / GATE_ROOT_NAME
    job_dir = gate_root / JOB_ID
    blocked_reason: str | None = None
    manifest = None

    def stage_info(name: str) -> dict:
        if manifest is None:
            return {"stage": name, "status": "pending", "message": ""}
        stage = manifest.stage(name)
        if stage is None:
            return {"stage": name, "status": "pending", "message": "stage missing from manifest"}
        return stage.model_dump(mode="json")

    def pending_reason(stage_name: str) -> str:
        if stage_name == "tts" and args.offline:
            return "--offline requested: edge-tts synthesis not attempted (needs network)"
        return blocked_reason or "the run never reached this stage"

    def eval_stage(step: str, stage_name: str) -> None:
        info = stage_info(stage_name)
        status = info["status"]
        message = info["message"] or "(no message)"
        if status == "ready":
            item = make_row(step, PASS, message)
        elif status == "skipped":
            item = make_row(step, BOO, message)
        elif status in {"failed", "cancelled"}:
            item = make_row(step, FAIL, f"{status}: {message}")
        else:
            reason = pending_reason(stage_name)
            item = make_row(step, BOO, f"not attempted - {reason}")
        rows[step] = item

    # 1. fixture -------------------------------------------------------------
    t = time.time()
    emit("creating fixture job ...")
    try:
        reset_gate_root(gate_root)
        source, source_note = resolve_source(gate_root)
        config = JobConfig(
            job_id=JOB_ID,
            language="vi",
            target_minutes=1.0,
            aspect_ratio="16:9",
            source_video=source,
            movie_title="Release Gate Fixture",
            content_agent="scaffold",
        )
        pipeline.create_job(job_dir, config)
        agent = pipeline.load_manifest(job_dir).config.content_agent
        if agent != "scaffold":
            raise RuntimeError(f"content_agent came out as {agent!r}, expected 'scaffold'")
        rows["fixture"] = make_row(
            "fixture", PASS,
            f"job {job_dir}; {source_note}; content_agent=scaffold",
            time.time() - t,
        )
    except Exception as exc:  # noqa: BLE001
        rows["fixture"] = make_row(
            "fixture", FAIL, f"{type(exc).__name__}: {exc}", time.time() - t
        )
        blocked_reason = "fixture creation failed"

    # 2. phase A: ingest -> script -------------------------------------------
    if blocked_reason is None:
        t = time.time()
        emit("running phase A (ingest..script) ...")
        try:
            pipeline.run_job(job_dir, until="script")
        except Exception as exc:  # noqa: BLE001
            blocked_reason = f"phase A raised {type(exc).__name__}: {exc}"
        emit(f"phase A finished in {time.time() - t:.1f}s")
    else:
        emit("phase A not attempted: " + blocked_reason)
    manifest = pipeline.load_manifest(job_dir) if job_dir.exists() else None
    if manifest is not None:
        first_broken = next(
            (s for s in manifest.stages if s.status in {"failed", "cancelled"}), None
        )
        if first_broken is not None:
            blocked_reason = (
                f"run stopped at {first_broken.stage}: "
                f"{first_broken.status} - {first_broken.message}"
            )

    # 3. approve script + phase B: scene_plan -> qa ---------------------------
    if blocked_reason is None:
        script_stage = stage_info("script")
        if script_stage["status"] == "ready":
            t = time.time()
            try:
                data = pipeline.approve_script(job_dir)
                if not data.get("approved"):
                    raise RuntimeError("approve_script did not set approved=true")
                rows["approve-script"] = make_row(
                    "approve-script", PASS,
                    f"script.json approved for TTS "
                    f"({len(data.get('sections') or [])} sections)",
                    time.time() - t,
                )
            except Exception as exc:  # noqa: BLE001
                rows["approve-script"] = make_row(
                    "approve-script", FAIL, f"{type(exc).__name__}: {exc}", time.time() - t
                )
                blocked_reason = f"script approval failed: {rows['approve-script']['detail']}"
        else:
            rows["approve-script"] = make_row(
                "approve-script", BOO,
                f"script stage not ready ({script_stage['status']}: {script_stage['message']})",
            )
            blocked_reason = f"script stage not ready: {script_stage['message']}"

    if blocked_reason is None and args.offline:
        t = time.time()
        emit("running phase B (scene_plan only, --offline) ...")
        try:
            pipeline.run_job(job_dir, until="scene_plan")
        except Exception as exc:  # noqa: BLE001
            blocked_reason = f"phase B raised {type(exc).__name__}: {exc}"
        if blocked_reason is None:
            blocked_reason = (
                "--offline requested: edge-tts synthesis skipped, so "
                "alignment/render/qa cannot run"
            )
        emit(f"phase B finished in {time.time() - t:.1f}s")
    elif blocked_reason is None:
        t = time.time()
        emit("running phase B (scene_plan..qa) ...")
        try:
            pipeline.run_job(job_dir, until="qa")
        except Exception as exc:  # noqa: BLE001
            blocked_reason = f"phase B raised {type(exc).__name__}: {exc}"
        emit(f"phase B finished in {time.time() - t:.1f}s")
    else:
        emit("phase B not attempted: " + blocked_reason)
    manifest = pipeline.load_manifest(job_dir) if job_dir.exists() else manifest

    # 4. evaluate every step ---------------------------------------------------
    for step, stage_name in (
        ("ingest", "ingest"),
        ("research", "research"),
        ("transcript", "transcript"),
        ("scenes", "scenes"),
        ("outline", "outline"),
        ("script", "script"),
        ("scene_plan", "scene_plan"),
        ("tts", "tts"),
        ("alignment", "alignment"),
        ("render", "render"),
        ("qa", "qa"),
    ):
        eval_stage(step, stage_name)

    scenes_doc: dict = {}
    scenes_file = job_dir / "scenes.json"
    if scenes_file.is_file():
        try:
            scenes_doc = json.loads(scenes_file.read_text(encoding="utf-8"))
        except ValueError as exc:
            scenes_doc = {"semantic_mode": "unreadable", "semantic_error": str(exc)}
    status, detail = step_search(job_dir, scenes_doc, MediaStore, semantic_search)
    rows["search"] = make_row("search", status, detail)

    render_stage = stage_info("render")
    status, detail = step_export_video(
        job_dir, render_stage["status"],
        pending_reason("render") if render_stage["status"] == "pending" else "",
    )
    rows["export-video"] = make_row("export-video", status, detail)
    alignment_stage = stage_info("alignment")
    status, detail = step_export_captions(
        job_dir, alignment_stage,
        pending_reason("alignment") if alignment_stage["status"] == "pending" else "",
    )
    rows["export-subtitles"] = make_row("export-subtitles", status, detail)
    status, detail = step_export_transcript(
        job_dir, stage_info("transcript"), gate_root, MediaStore, JobsService,
        pending_reason("transcript"),
    )
    rows["export-transcript"] = make_row("export-transcript", status, detail)

    # 5. report ----------------------------------------------------------------
    emit("")
    emit("--- GATE RESULTS ---")
    emit(f"{'STAT':<5} {'STEP':<17} DETAIL")
    for step in STEP_ORDER:
        item = rows.get(step)
        if item is None:
            continue
        print_row(item)

    counts = {PASS: 0, FAIL: 0, BOO: 0}
    for step in STEP_ORDER:
        item = rows.get(step)
        if item:
            counts[item["status"]] += 1

    emit("")
    emit("--- SUMMARY ---")
    emit(f"steps           : {sum(counts.values())}  PASS={counts[PASS]}  "
         f"FAIL={counts[FAIL]}  BOO={counts[BOO]}")
    emit(f"job directory   : {job_dir}")
    emit("stages not run  : metadata, thumbnail, publish "
         "(outside roadmap #15 import->index->search->render->export scope)")
    if blocked_reason:
        emit(f"blocker         : {blocked_reason}")
    emit(f"wall clock      : {time.time() - started:.1f}s")

    emit("")
    emit("--- CLEAN-WINDOWS CHECKLIST "
         "(fresh Win10/11 x64, no Node/Python/FFmpeg preinstalled) ---")
    for index, entry in enumerate(CHECKLIST, start=1):
        emit(f"{index:>2}. run    : {entry[0]}")
        emit(f"    PASS   : {entry[1]}")
        if len(entry) > 2 and entry[2]:
            emit(f"    if     : {entry[2]}")

    emit("")
    emit("--- EXIT CODE ---")
    if counts[FAIL]:
        emit("1 - at least one FAIL")
        return 1
    if counts[BOO]:
        emit("2 - no FAIL, but at least one BOO (gate not fully green)")
        return 2
    emit("0 - every step PASS")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R15 clean-Windows release gate (prints PASS/FAIL/BOO per step)."
    )
    parser.add_argument(
        "--python", default=None,
        help="interpreter to use (default: auto-detect the fullest installed profile)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="do not attempt the network-dependent edge-tts step; record it as BOO",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    if os.environ.get("MRF_GATE_CHILD") == "1":
        return run_gate(args)

    chosen = resolve_python(args)
    if chosen is None:
        emit("=== R15 CLEAN-WINDOWS RELEASE GATE ===")
        emit("[FAIL] environment       no Python interpreter with pydantic + typer found "
             "(tried --python/MRF_PYTHON, current, managed venv, PATH)")
        return 1
    missing = [name for name, ok in chosen.get("mods", {}).items() if not ok]
    if missing:
        emit(f"note: interpreter {chosen['exe']} is missing {', '.join(missing)}; "
             "the matching steps will be reported BOO with that reason")
    if same_executable(chosen["exe"], sys.executable):
        return run_gate(args)

    emit(f"re-launching under {chosen['exe']} ({chosen.get('version')}) ...")
    env = os.environ.copy()
    env["MRF_GATE_CHILD"] = "1"
    proc = subprocess.run(
        [chosen["exe"], str(Path(__file__).resolve()), *sys.argv[1:]],
        env=env, cwd=os.getcwd(),
    )
    return int(proc.returncode or 0)


if __name__ == "__main__":
    sys.exit(main())
