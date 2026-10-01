from movie_review_factory import editorial_qa


def _cue(index, start, end, text):
    return {"index": index, "start_seconds": start, "end_seconds": end, "text": text}


def _clip(start, end):
    return {"source_clip": {"start_seconds": start, "end_seconds": end}}


def test_caption_geometry_and_timing_identify_actual_overflow():
    checks = editorial_qa.caption_checks({
        "narration_seconds": 12,
        "cues": [
            _cue(1, 0, 4, "Dòng ngắn\nĐọc được"),
            _cue(2, 3.9, 10.5, "Một dòng dài hơn bốn mươi hai ký tự nên phải đánh dấu để sửa cho dễ đọc"),
            _cue(3, 10.5, 12.3, "Tràn cuối audio"),
        ],
    })
    by_name = {item["check"]: item for item in checks}
    assert by_name["caption_timing"]["value"]["invalid"] == [2, 3]
    assert by_name["caption_two_line_safe"]["value"]["overflow"] == [2]
    assert by_name["caption_readable_duration"]["value"]["outside_0.25_to_6.1_seconds"] == [2]
    # Duration is a readability judgement, not a hard error: it flags for human
    # review (passed) rather than failing QA, matching the other review checks.
    assert by_name["caption_readable_duration"]["passed"] is True
    assert by_name["caption_readable_duration"]["review_required"] is True


def test_repeated_footage_is_advisory_and_render_mismatch_fails():
    plan = {"clips": [_clip(0, 10), _clip(20, 30), _clip(2, 10)]}
    render = {"clips": [
        {"start_seconds": 0, "end_seconds": 10},
        {"start_seconds": 22, "end_seconds": 30},
        {"start_seconds": 2, "end_seconds": 10},
    ]}
    checks = {item["check"]: item for item in editorial_qa.clip_checks(plan, render)}
    assert checks["render_clip_provenance"]["passed"] is False
    assert checks["render_clip_provenance"]["value"]["mismatch"] == [2]
    repeat = checks["source_footage_reuse"]
    assert repeat["passed"] is True
    assert repeat["review_required"] is True
    assert repeat["value"]["repeated_pairs"][0]["first_clip"] == 1
    assert repeat["value"]["repeated_pairs"][0]["repeated_clip"] == 3


def test_stretched_footage_requires_editorial_review():
    clip = _clip(5, 10)
    clip["duration_seconds"] = 20
    checks = {item["check"]: item for item in editorial_qa.clip_checks(
        {"clips": [clip]}, {"clips": [{"start_seconds": 5, "end_seconds": 10}]}
    )}
    assert checks["render_clip_provenance"]["passed"] is True
    assert checks["stretched_footage"]["review_required"] is True
    assert checks["stretched_footage"]["value"]["looped_clips"][0]["clip"] == 1


def test_source_provenance_uses_windows_case_insensitive_paths():
    assert editorial_qa.provenance_checks({"source_video": "D:/Video/Ben.mp4"},
                                         "d:\\VIDEO\\ben.mp4")[0]["passed"] is True
    assert editorial_qa.provenance_checks({"source_video": "D:/Video/Other.mp4"},
                                         "D:/Video/Ben.mp4")[0]["passed"] is False


def test_source_provenance_matches_relative_and_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    absolute = str(tmp_path / "jobs" / "demo" / "source_clean.mp4")
    assert editorial_qa.provenance_checks({"source_video": "jobs\\demo\\source_clean.mp4"},
                                         absolute)[0]["passed"] is True


def test_rendered_chapter_cuts_match_measured_voice_sections():
    alignment = {"section_bounds": [
        {"section_index": 1, "start_seconds": 0},
        {"section_index": 2, "start_seconds": 5},
    ]}
    render = {"clips": [
        {"section_index": 1, "duration_seconds": 5},
        {"section_index": 2, "duration_seconds": 3},
    ]}
    assert editorial_qa.section_sync_checks(alignment, render)[0]["passed"]
    render["clips"][0]["duration_seconds"] = 4
    failure = editorial_qa.section_sync_checks(alignment, render)[0]
    assert not failure["passed"]
    assert failure["value"]["drift"][0]["section_index"] == 2
    assert not editorial_qa.section_sync_checks({"timing_mode": "tts_word_boundary"}, render)[0]["passed"]


def test_midroll_is_checked_against_actual_rendered_duration():
    clips = [{"type": "narration", "duration_seconds": 45},
             {"type": "midroll", "duration_seconds": 10},
             {"type": "narration", "duration_seconds": 45}]
    assert editorial_qa.midroll_checks({"clips": clips})[0]["value"]["fraction"] == 0.5
    assert editorial_qa.midroll_checks({"clips": clips})[0]["passed"]
    clips[0]["duration_seconds"] = 80
    assert not editorial_qa.midroll_checks({"clips": clips})[0]["passed"]


def test_source_text_bands_require_manual_watermark_review():
    check = editorial_qa.source_text_review_checks(0.1, 0)[0]
    assert check["passed"] is True
    assert check["review_required"] is True
    assert check["value"] == {"top_fraction": 0.1, "bottom_fraction": 0.0}
    assert editorial_qa.source_text_review_checks()[0]["review_required"] is False


def test_inspect_job_reads_artifacts_without_writing(tmp_path):
    import json
    data = {
        "alignment.json": {"narration_seconds": 8, "cues": [_cue(1, 0, 5, "Xin chào")]},
        "scene_plan.json": {"clips": [_clip(5, 10)]},
        "render.json": {"source_video": "D:/Video/Ben.mp4",
                        "clips": [{"start_seconds": 5, "end_seconds": 10}]},
    }
    for name, item in data.items():
        (tmp_path / name).write_text(json.dumps(item), encoding="utf-8")
    checks = editorial_qa.inspect_job(tmp_path, "D:/Video/Ben.mp4")
    assert all(item["passed"] for item in checks)
    assert len(list(tmp_path.iterdir())) == 3


def _ass(play_x, play_y, *, alignment=2, ml=144, mr=144, mv=140, font=45):
    return (
        "[Script Info]\n"
        f"PlayResX: {play_x}\nPlayResY: {play_y}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Arial,{font},&H00FFFFFF,&H000000FF,&H00000000,"
        f"&H64000000,0,0,0,0,100,100,0,0,1,2,1,{alignment},{ml},{mr},{mv},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Xin chào\n"
    )


def test_caption_geometry_passes_for_frame_sized_bottom_safe_ass(tmp_path):
    (tmp_path / "aligned.ass").write_text(_ass(1920, 1080), encoding="utf-8")
    checks = {c["check"]: c for c in
              editorial_qa.caption_geometry_checks(tmp_path, {"width": 1920, "height": 1080})}
    assert checks["caption_ass_playres"]["passed"] is True
    assert checks["caption_bottom_safe_area"]["passed"] is True
    assert checks["caption_bottom_safe_area"]["value"]["bottom_gap_fraction"] == 0.13


def test_caption_geometry_flags_libass_default_playres(tmp_path):
    # PlayRes stuck at the 384x288 libass default -> the exact subtitle regression.
    (tmp_path / "aligned.ass").write_text(_ass(384, 288, mv=40), encoding="utf-8")
    checks = {c["check"]: c for c in
              editorial_qa.caption_geometry_checks(tmp_path, {"width": 1920, "height": 1080})}
    assert checks["caption_ass_playres"]["passed"] is False


def test_caption_geometry_flags_caption_floated_out_of_safe_area(tmp_path):
    # Correct PlayRes but MarginV floats the block into the upper half.
    (tmp_path / "aligned.ass").write_text(_ass(1920, 1080, mv=760), encoding="utf-8")
    checks = {c["check"]: c for c in
              editorial_qa.caption_geometry_checks(tmp_path, {"width": 1920, "height": 1080})}
    assert checks["caption_ass_playres"]["passed"] is True
    assert checks["caption_bottom_safe_area"]["passed"] is False


def test_caption_geometry_skips_without_ass(tmp_path):
    assert editorial_qa.caption_geometry_checks(tmp_path, {"width": 1920, "height": 1080}) == []
