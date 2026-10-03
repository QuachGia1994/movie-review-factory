import pytest

from movie_review_factory import genre_tone, packaging


@pytest.mark.parametrize("text,expected", [
    ("Kinh dị", ["horror"]),
    ("kinh di", ["horror"]),
    ("Khoa học viễn tưởng", ["scifi"]),
    ("Kinh dị hài", ["horror", "comedy"]),
    ("Hài, kinh dị, tâm lý", ["comedy", "horror"]),
    ("Crime thriller", ["crime", "thriller"]),
    ("", []),
    ("phim đẹp", []),
])
def test_detect(text, expected):
    assert genre_tone.detect(text) == expected


def test_every_webapp_label_maps_to_a_voice():
    for label in genre_tone.GENRE_LABELS_VI:
        assert genre_tone.detect(label), label


def test_tables_cover_every_genre():
    for key in genre_tone._ALIASES:
        for table in (genre_tone._VI, genre_tone._EN):
            assert table[key]["rules"] and "{movie}" in table[key]["question"]


def test_structure_guide_appends_genre_voice():
    rules = packaging.structure_guide("vi", "single", {}, "Kinh dị")
    assert any(rule.startswith("[kinh dị]") for rule in rules)
    assert packaging.structure_guide("vi", "single", {}) == packaging.structure_guide("vi", "single", {}, "")
    blended = packaging.structure_guide("en", "single", {}, "horror comedy")
    assert any(rule.startswith("Blend of horror and comedy") for rule in blended)
    assert sum(rule.startswith("[comedy]") for rule in blended) == 1


def test_comment_question_follows_genre():
    assert "rợn người" in packaging.fallback_comment_question("vi", "Ring", "single", "kinh dị")
    assert "Ring" in packaging.fallback_comment_question("vi", "Ring", "single", "kinh dị")
    assert "số thứ tự" in packaging.fallback_comment_question("vi", "Ring", "compilation", "kinh dị")
    assert packaging.fallback_comment_question("en", "Heat", "single", "crime").startswith("Who did you suspect")
    assert packaging.fallback_comment_question("vi", "X", "single", "").startswith("Chi tiết nào")


def test_packaging_prompt_mentions_voice_only_for_known_genre():
    assert "horror voice" in packaging.packaging_prompt("vi", "Ring", "kinh dị", "Kenh", "single")
    assert "voice" not in packaging.packaging_prompt("vi", "Ring", "", "Kenh", "single")
