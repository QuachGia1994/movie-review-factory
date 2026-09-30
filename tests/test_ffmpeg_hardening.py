"""Unit tests for the libx264 thread-cap + allocation-retry hardening helpers.

These lock in the behaviour that keeps the real-FFmpeg render tests from the
transient "x264 [error]: malloc of size N failed" / "Cannot allocate memory"
flake: a bounded thread cap on every encode and a single single-threaded retry.
"""
from __future__ import annotations

import pytest

import movie_review_factory.pipeline as pipeline


# --- ffmpeg_thread_cap ------------------------------------------------------

def test_thread_cap_honours_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRF_FFMPEG_THREADS", "3")
    assert pipeline.ffmpeg_thread_cap() == 3


def test_thread_cap_clamps_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRF_FFMPEG_THREADS", "999")
    assert pipeline.ffmpeg_thread_cap() == 16
    monkeypatch.setenv("MRF_FFMPEG_THREADS", "0")
    assert pipeline.ffmpeg_thread_cap() == 1


def test_thread_cap_ignores_bad_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRF_FFMPEG_THREADS", "not-a-number")
    assert 1 <= pipeline.ffmpeg_thread_cap() <= 8


def test_thread_cap_default_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MRF_FFMPEG_THREADS", raising=False)
    assert 1 <= pipeline.ffmpeg_thread_cap() <= 8


# --- is_ffmpeg_memory_error -------------------------------------------------

def test_memory_error_detects_x264_and_alloc_failures() -> None:
    assert pipeline.is_ffmpeg_memory_error("x264 [error]: malloc of size 7186688 failed")
    assert pipeline.is_ffmpeg_memory_error("Cannot allocate memory")
    assert pipeline.is_ffmpeg_memory_error("Out of memory")
    assert pipeline.is_ffmpeg_memory_error("libx264: malloc failed")


def test_memory_error_ignores_unrelated_failures() -> None:
    assert not pipeline.is_ffmpeg_memory_error("Invalid data found when processing input")
    assert not pipeline.is_ffmpeg_memory_error("")
    assert not pipeline.is_ffmpeg_memory_error(None)


# --- ffmpeg_command_single_thread -------------------------------------------

def test_single_thread_rewrites_existing_threads_without_mutating() -> None:
    cmd = ["ffmpeg", "-c:v", "libx264", "-threads", "8", "out.mp4"]
    reduced = pipeline.ffmpeg_command_single_thread(cmd)
    assert reduced == ["ffmpeg", "-c:v", "libx264", "-threads", "1", "out.mp4"]
    assert cmd[4] == "8"  # original list is untouched


def test_single_thread_inserts_before_output_when_missing() -> None:
    cmd = ["ffmpeg", "-c:v", "libx264", "out.mp4"]
    reduced = pipeline.ffmpeg_command_single_thread(cmd)
    assert reduced == ["ffmpeg", "-c:v", "libx264", "-threads", "1", "out.mp4"]


# --- video encoder selection ------------------------------------------------

def test_video_encoder_defaults_to_libx264(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MRF_VIDEO_ENCODER", raising=False)
    args, is_hardware = pipeline._video_encoder_args()
    assert is_hardware is False
    assert args[:2] == ["-c:v", "libx264"]


def test_video_encoder_forces_named_hardware_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRF_VIDEO_ENCODER", "nvenc")
    args, is_hardware = pipeline._video_encoder_args()
    assert is_hardware is True
    assert args[:2] == ["-c:v", "h264_nvenc"]


def test_video_encoder_auto_prefers_hardware_then_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRF_VIDEO_ENCODER", "auto")
    monkeypatch.setattr(pipeline, "_available_ffmpeg_encoders", lambda: {"h264_qsv", "libx264"})
    args, is_hardware = pipeline._video_encoder_args()
    assert (args[:2], is_hardware) == (["-c:v", "h264_qsv"], True)

    monkeypatch.setattr(pipeline, "_available_ffmpeg_encoders", lambda: {"libx264"})
    args, is_hardware = pipeline._video_encoder_args()
    assert (args[:2], is_hardware) == (["-c:v", "libx264"], False)
