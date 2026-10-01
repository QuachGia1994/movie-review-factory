"""Multi-channel profile switcher: identity + job-creation defaults per channel."""

import io
import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from movie_review_factory import branding
from movie_review_factory.creator_library import CreatorLibrary
from movie_review_factory.models import ChannelProfile, ChannelSfx
from movie_review_factory.pipeline import load_manifest
from movie_review_factory.webapp import JobsService


def _png_bytes(color=(30, 120, 200, 180)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (128, 128), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_channel_profile_model_mirrors_jobconfig_constraints():
    profile = ChannelProfile.model_validate({"name": "  Màn Kể  ", "brand_top_band": 0.1})
    assert profile.name == "Màn Kể"          # trimmed
    assert profile.tts_provider == "edge"    # default preserved
    with pytest.raises(ValidationError):
        ChannelProfile.model_validate({"name": "X", "brand_top_band": 0.5})  # > 0.2
    with pytest.raises(ValidationError):
        ChannelProfile.model_validate({"name": "Tên\nSai"})  # control character


def test_first_saved_channel_auto_activates_and_syncs_identity(tmp_path):
    lib = CreatorLibrary(tmp_path)
    saved = lib.save_channel("kenh-a", {"name": "Kênh A", "tts_provider": "fptai", "tts_voice": "banmai"})
    assert saved["id"] == "kenh-a"
    listing = lib.list_channels()
    assert listing["active"] == "kenh-a"
    assert [item["id"] for item in listing["channels"]] == ["kenh-a"]
    # Saving the active profile syncs the shared brand identity render/thumbnail read.
    assert branding.load_settings(tmp_path)["name"] == "Kênh A"


def test_second_channel_does_not_steal_active(tmp_path):
    lib = CreatorLibrary(tmp_path)
    lib.save_channel("a", {"name": "Kênh A"})
    lib.save_channel("b", {"name": "Kênh B"})
    assert lib.list_channels()["active"] == "a"


def test_activate_switches_identity_and_materialises_logo(tmp_path):
    lib = CreatorLibrary(tmp_path)
    lib.save_channel("a", {"name": "Kênh A"})
    lib.save_channel("b", {"name": "Kênh B"})
    lib.set_channel_logo("b", _png_bytes())
    result = lib.activate_channel("b")
    assert result["active"] == "b"
    assert branding.load_settings(tmp_path)["name"] == "Kênh B"
    # The profile logo becomes the live brand logo the pipeline reads.
    assert branding.logo_path(tmp_path) == (tmp_path / ".brand" / "logo.png")
    assert (tmp_path / ".brand" / "logo.png").is_file()


def test_set_logo_marks_has_logo_and_stores_per_profile(tmp_path):
    lib = CreatorLibrary(tmp_path)
    lib.save_channel("a", {"name": "Kênh A"})
    record = lib.set_channel_logo("a", _png_bytes())
    assert record["has_logo"] is True
    assert branding.channel_logo_path(tmp_path, "a") is not None


def test_delete_guards_active_but_removes_others(tmp_path):
    lib = CreatorLibrary(tmp_path)
    lib.save_channel("a", {"name": "Kênh A"})
    lib.save_channel("b", {"name": "Kênh B"})
    with pytest.raises(ValueError, match="kích hoạt"):
        lib.delete_channel("a")  # active profile is protected
    lib.delete_channel("b")
    assert [item["id"] for item in lib.list_channels()["channels"]] == ["a"]
    with pytest.raises(KeyError):
        lib.get_channel("b")


def test_active_channel_defaults_excludes_identity_name(tmp_path):
    lib = CreatorLibrary(tmp_path)
    lib.save_channel("a", {
        "name": "Kênh A", "tts_provider": "elevenlabs",
        "brand_top_band": 0.08, "intro_seconds": 3,
    })
    defaults = lib.active_channel_defaults()
    assert "name" not in defaults
    assert defaults["tts_provider"] == "elevenlabs"
    assert defaults["brand_top_band"] == 0.08
    assert defaults["intro_seconds"] == 3


def test_backward_compatible_load_of_pre_channels_library(tmp_path):
    # A library written before the channels section still loads and accepts channels.
    (tmp_path / "_creator_library.json").write_text(
        json.dumps({"series": {}, "briefs": {}, "assets": {}}), encoding="utf-8"
    )
    lib = CreatorLibrary(tmp_path)
    assert lib.list_channels() == {"channels": [], "active": None}
    lib.save_channel("a", {"name": "Kênh A"})
    assert lib.list_channels()["active"] == "a"


def test_save_channel_maps_invalid_profile_to_value_error(tmp_path):
    lib = CreatorLibrary(tmp_path)
    with pytest.raises(ValueError):
        lib.save_channel("a", {"name": ""})  # empty name


def test_create_job_prefills_unset_fields_from_active_channel(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    service = JobsService(jobs)
    service.save_channel({
        "id": "a", "name": "Kênh A", "tts_provider": "fptai", "tts_voice": "banmai",
        "brand_top_band": 0.1, "intro_seconds": 2, "copyright_bypass": "aggressive",
    })

    service.create_job({"job_id": "vid1"})  # minimal payload inherits channel defaults
    cfg = load_manifest(jobs / "vid1").config
    assert cfg.tts_provider == "fptai"
    assert cfg.tts_voice == "banmai"
    assert cfg.brand_top_band == 0.1
    assert cfg.intro_seconds == 2
    assert cfg.copyright_bypass == "aggressive"

    # Explicit payload values override the channel default; unset fields still inherit.
    service.create_job({"job_id": "vid2", "tts_provider": "edge", "copyright_bypass": "off"})
    cfg2 = load_manifest(jobs / "vid2").config
    assert cfg2.tts_provider == "edge"
    assert cfg2.copyright_bypass == "off"
    assert cfg2.brand_top_band == 0.1


def test_jobs_service_activate_and_list_round_trip(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    service = JobsService(jobs)
    service.save_channel({"id": "a", "name": "Kênh A"})
    service.save_channel({"id": "b", "name": "Kênh B"})
    result = service.activate_channel("b")
    assert result["active"] == "b"
    assert service.list_channels()["active"] == "b"
    assert branding.load_settings(jobs)["name"] == "Kênh B"


# -- transition SFX palette (v1: insert at player time) ---------------------

def _audio_bytes() -> bytes:
    return b"ID3\x03\x00\x00\x00\x00\x00\x21fake-audio-bytes-for-tests" * 4


def test_channel_sfx_model_validation():
    sfx = ChannelSfx.model_validate({"slug": "whoosh-1", "label": "  Whoosh  ", "rights_note": "tự thu", "gain_db": -6})
    assert sfx.label == "Whoosh"
    assert sfx.gain_db == -6
    with pytest.raises(ValidationError):
        ChannelSfx.model_validate({"slug": "bad slug!", "label": "X", "rights_note": "r"})
    with pytest.raises(ValidationError):
        ChannelSfx.model_validate({"slug": "ok", "label": "X", "rights_note": ""})


def test_add_sfx_upload_and_active_palette(tmp_path):
    lib = CreatorLibrary(tmp_path)
    lib.save_channel("a", {"name": "Kênh A"})
    entry = lib.save_channel_sfx("a", {"slug": "whoosh", "label": "Whoosh", "rights_note": "tự thu", "gain_db": -8})
    assert entry["has_file"] is False
    assert lib.active_channel_sfx()["sfx"] == []  # not playable until a file is attached
    lib.set_channel_sfx_file("a", "whoosh", "audio/mpeg", _audio_bytes())
    assert branding.channel_sfx_path(tmp_path, "a", "whoosh") is not None
    palette = lib.active_channel_sfx()
    assert palette["channel"] == "Kênh A"
    assert [s["slug"] for s in palette["sfx"]] == ["whoosh"]


def test_save_channel_preserves_sfx_across_profile_edit(tmp_path):
    lib = CreatorLibrary(tmp_path)
    lib.save_channel("a", {"name": "Kênh A"})
    lib.save_channel_sfx("a", {"slug": "boom", "label": "Boom", "rights_note": "mua license"})
    lib.save_channel("a", {"name": "Kênh A", "tts_provider": "fptai"})  # editing defaults
    assert [s["slug"] for s in lib.get_channel("a")["sfx"]] == ["boom"]


def test_delete_sfx_removes_entry_and_file(tmp_path):
    lib = CreatorLibrary(tmp_path)
    lib.save_channel("a", {"name": "Kênh A"})
    lib.save_channel_sfx("a", {"slug": "whoosh", "label": "Whoosh", "rights_note": "tự thu"})
    lib.set_channel_sfx_file("a", "whoosh", "audio/wav", _audio_bytes())
    lib.delete_channel_sfx("a", "whoosh")
    assert lib.get_channel("a")["sfx"] == []
    assert branding.channel_sfx_path(tmp_path, "a", "whoosh") is None


def test_sfx_rejects_unknown_content_type(tmp_path):
    lib = CreatorLibrary(tmp_path)
    lib.save_channel("a", {"name": "Kênh A"})
    lib.save_channel_sfx("a", {"slug": "x", "label": "X", "rights_note": "r"})
    with pytest.raises(ValueError):
        lib.set_channel_sfx_file("a", "x", "video/mp4", _audio_bytes())


def test_channel_sfx_capped_at_eight(tmp_path):
    lib = CreatorLibrary(tmp_path)
    lib.save_channel("a", {"name": "Kênh A"})
    for i in range(8):
        lib.save_channel_sfx("a", {"slug": f"s{i}", "label": f"S{i}", "rights_note": "r"})
    with pytest.raises(ValueError, match="tối đa"):
        lib.save_channel_sfx("a", {"slug": "s8", "label": "S8", "rights_note": "r"})


def test_add_audio_mix_sfx_writes_valid_effect(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    service = JobsService(jobs)
    service.save_channel({"id": "a", "name": "Kênh A"})
    service.save_channel_sfx("a", {"slug": "whoosh", "label": "Whoosh", "rights_note": "tự thu", "gain_db": -6})
    service.save_channel_sfx_file("a", "whoosh", "audio/mpeg", _audio_bytes())
    service.create_job({"job_id": "vid1"})
    result = service.add_audio_mix_sfx("vid1", {"slug": "whoosh", "at_seconds": 12.5})
    effects = result["audio_mix"]["effects"]
    assert len(effects) == 1
    assert effects[0]["at_seconds"] == 12.5
    assert effects[0]["gain_db"] == -6
    assert effects[0]["rights_note"] == "tự thu"
    assert Path(effects[0]["path"]).is_file()
    service.add_audio_mix_sfx("vid1", {"slug": "whoosh", "at_seconds": 30})  # appends
    assert len(service.get_audio_mix("vid1")["audio_mix"]["effects"]) == 2


# -- mode (B): auto-place SFX at scene cuts from render.json ------------------

def _write_render(root: Path, durations, narration=100.0) -> None:
    (root / "render.json").write_text(
        json.dumps({"narration_duration_seconds": narration,
                    "clips": [{"duration_seconds": d} for d in durations]}),
        encoding="utf-8",
    )


def _sfx_service(tmp_path, slugs=("whoosh", "boom")):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    service = JobsService(jobs)
    service.save_channel({"id": "a", "name": "Kênh A"})
    for slug in slugs:
        service.save_channel_sfx("a", {"slug": slug, "label": slug, "rights_note": "tự thu", "gain_db": -8})
        service.save_channel_sfx_file("a", slug, "audio/mpeg", _audio_bytes())
    return service, jobs


def test_transition_cut_offsets_and_even_selection():
    from movie_review_factory import pipeline
    assert pipeline.transition_cut_offsets([10, 20, 30]) == [10.0, 30.0]
    assert pipeline.transition_cut_offsets([5]) == []
    assert pipeline.transition_cut_offsets([]) == []
    picked = pipeline.select_evenly([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)
    assert len(picked) == 3 and picked[0] == 1 and picked[-1] == 10
    assert pipeline.select_evenly([1, 2], 5) == [1, 2]
    assert pipeline.select_evenly([1, 2, 3], 0) == []


def test_place_transition_sfx_uses_render_offsets(tmp_path):
    service, jobs = _sfx_service(tmp_path)
    service.create_job({"job_id": "vid1"})
    _write_render(jobs / "vid1", [20, 30, 25, 25])
    result = service.place_transition_sfx("vid1")
    effects = result["audio_mix"]["effects"]
    assert result["placed"] == 3  # 4 clips -> 3 internal cuts
    assert [e["at_seconds"] for e in effects] == [20.0, 50.0, 75.0]
    assert all(e["source"] == "channel-transition" for e in effects)
    assert [Path(e["path"]).stem for e in effects] == ["whoosh", "boom", "whoosh"]  # cycles


def test_place_transition_sfx_requires_prior_render(tmp_path):
    service, jobs = _sfx_service(tmp_path, slugs=("whoosh",))
    service.create_job({"job_id": "vid1"})
    with pytest.raises(ValueError, match="dựng video"):
        service.place_transition_sfx("vid1")


def test_place_transition_sfx_preserves_manual_and_is_idempotent(tmp_path):
    service, jobs = _sfx_service(tmp_path, slugs=("whoosh",))
    service.create_job({"job_id": "vid1"})
    root = jobs / "vid1"
    service.add_audio_mix_sfx("vid1", {"slug": "whoosh", "at_seconds": 5})  # manual (no marker)
    _write_render(root, [20, 30, 50])            # first render produces cut timings
    service.place_transition_sfx("vid1")
    _write_render(root, [20, 30, 50])            # re-render (audio change invalidated it)
    service.place_transition_sfx("vid1")  # re-run must replace, not stack, auto effects
    effects = service.get_audio_mix("vid1")["audio_mix"]["effects"]
    manual = [e for e in effects if e.get("source") != "channel-transition"]
    auto = [e for e in effects if e.get("source") == "channel-transition"]
    assert len(manual) == 1
    assert len(auto) == 2  # 3 clips -> 2 cuts


def test_place_transition_sfx_respects_16_cap(tmp_path):
    service, jobs = _sfx_service(tmp_path, slugs=("whoosh",))
    service.create_job({"job_id": "vid1"})
    _write_render(jobs / "vid1", [1.0] * 40)  # 40 clips -> 39 cuts
    result = service.place_transition_sfx("vid1")
    assert result["placed"] == 16
    assert len(result["audio_mix"]["effects"]) == 16
