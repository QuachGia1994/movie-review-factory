"""Offline unit tests for the read-only retention advisory engine.

These are pure (or read a temp ``analytics`` folder), so they run fully offline
with no Google/network mocking — which is exactly the point of keeping the
feedback loop advisory and local.
"""
from __future__ import annotations

import json
from pathlib import Path

from movie_review_factory import analytics


def _report(report_id: str, *, intro=None, cta=None, ctr=None) -> dict:
    return {
        "id": report_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "measurements": {
            "intro_drop_percentage_points": intro,
            "cta_drop_percentage_points": cta,
        },
        "ctr_percent": ctr,
    }


def test_high_intro_drop_flags_hook_suggestion():
    advice = analytics.retention_advice(reports=[_report("01", intro=42.0)])
    assert advice["enabled"] is True
    assert advice["advisory_only"] is True
    by_code = {s["code"]: s for s in advice["suggestions"]}
    assert "intro_hook_too_slow" in by_code
    assert by_code["intro_hook_too_slow"]["target"] == "hook"
    assert by_code["intro_hook_too_slow"]["severity"] == "high"


def test_cta_and_ctr_flag_their_suggestions():
    advice = analytics.retention_advice(reports=[
        _report("01", cta=30.0, ctr=3.2),
        _report("02", cta=28.0, ctr=3.8),
        _report("03", cta=26.0, ctr=3.5),
    ])
    codes = {s["code"] for s in advice["suggestions"]}
    assert "cta_placement" in codes
    assert "thumbnail_ctr_low" in codes
    assert "intro_hook_too_slow" not in codes  # no intro signal present
    assert advice["low_confidence"] is False  # 3 samples reaches the confidence floor


def test_values_below_threshold_produce_no_suggestions():
    advice = analytics.retention_advice(reports=[
        _report("01", intro=10.0, cta=5.0, ctr=9.0),
        _report("02", intro=12.0, cta=6.0, ctr=8.0),
        _report("03", intro=11.0, cta=4.0, ctr=7.0),
    ])
    assert advice["suggestions"] == []
    assert advice["signals"]["intro_drop_avg"] == 11.0


def test_averages_across_recent_reports_and_flags_low_confidence():
    advice = analytics.retention_advice(reports=[
        _report("01", intro=40.0),
        _report("02", intro=50.0),
    ])
    assert advice["signals"]["intro_drop_avg"] == 45.0
    assert advice["sample_size"] == 2
    assert advice["low_confidence"] is True  # fewer than 3 samples


def test_recent_window_limits_reports_considered():
    reports = [_report(f"{i:02d}", intro=40.0) for i in range(10)]
    advice = analytics.retention_advice(reports=reports, recent=3)
    assert advice["sample_size"] == 3


def test_off_switch_by_param_disables_engine():
    advice = analytics.retention_advice(reports=[_report("01", intro=99.0)], enabled=False)
    assert advice["enabled"] is False
    assert advice["suggestions"] == []


def test_off_switch_by_env(monkeypatch):
    monkeypatch.setenv(analytics.ADVICE_ENV_FLAG, "0")
    advice = analytics.retention_advice(reports=[_report("01", intro=99.0)])
    assert advice["enabled"] is False
    assert advice["suggestions"] == []


def test_env_truthy_keeps_engine_enabled(monkeypatch):
    monkeypatch.setenv(analytics.ADVICE_ENV_FLAG, "1")
    advice = analytics.retention_advice(reports=[_report("01", intro=99.0)])
    assert advice["enabled"] is True


def test_malformed_reports_are_ignored_safely():
    advice = analytics.retention_advice(reports=[
        "not a dict",
        {"measurements": "bad"},
        {"measurements": {"intro_drop_percentage_points": True}},  # bool must not count as a number
        _report("01", intro=40.0),
    ])
    assert advice["signals"]["intro_drop_avg"] == 40.0
    assert advice["sample_size"] == 3  # the string is dropped; the three dicts remain


def test_reads_measurements_from_analytics_folder(tmp_path: Path):
    folder = tmp_path / "analytics"
    folder.mkdir()
    (folder / "aaa.json").write_text(json.dumps(_report("aaa", intro=41.0)), encoding="utf-8")
    (folder / "bbb.json").write_text(json.dumps(_report("bbb", intro=43.0)), encoding="utf-8")
    advice = analytics.retention_advice(tmp_path)
    assert advice["sample_size"] == 2
    assert advice["signals"]["intro_drop_avg"] == 42.0
    assert any(s["code"] == "intro_hook_too_slow" for s in advice["suggestions"])


def test_no_data_yields_no_suggestions(tmp_path: Path):
    advice = analytics.retention_advice(tmp_path)
    assert advice["enabled"] is True
    assert advice["sample_size"] == 0
    assert advice["suggestions"] == []
