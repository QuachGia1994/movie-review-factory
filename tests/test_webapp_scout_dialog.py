import os
import re
import subprocess
import tempfile
import pytest
from movie_review_factory.webapp import INDEX_HTML

def test_webapp_index_html_js_syntax():
    """Verify that all embedded JavaScript blocks in INDEX_HTML parse without syntax errors."""
    scripts = re.findall(r"<script>(.*?)</script>", INDEX_HTML, re.DOTALL)
    assert len(scripts) >= 2, "Expected multiple script blocks in INDEX_HTML"

    node_available = False
    try:
        res = subprocess.run(["node", "-v"], capture_output=True, text=True)
        node_available = (res.returncode == 0)
    except FileNotFoundError:
        pass

    if node_available:
        for idx, code in enumerate(scripts):
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
                f.write(code)
                tmp_path = f.name
            try:
                res = subprocess.run(["node", "--check", tmp_path], capture_output=True, text=True)
                assert res.returncode == 0, f"Script block {idx} failed node --check:\n{res.stderr}"
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

def test_webapp_empty_and_toolbar_cleanliness():
    """Verify that empty state has no redundant open project button, header scout is hidden at home, and batch panel is uncollapsed."""
    # Top Toolbar button is present but hidden on empty home
    match = re.search(r'<button\s+id="openScoutHub"[^>]*>', INDEX_HTML)
    assert match is not None
    assert "hidden" in match.group(0), "openScoutHub should be hidden on empty home"

    # Landing actions: create from MP4, open project (loadJobs() disables it when none), auto scout.
    assert 'id="emptyCreateBtn"' in INDEX_HTML
    assert 'id="emptyScoutBtn"' in INDEX_HTML
    assert 'id="emptyProjectsBtn"' in INDEX_HTML, "emptyProjectsBtn (Mở project có sẵn) restored to empty actions"

    # Batch panel is flat div, not a details toggle
    assert '<details id="batchPanel"' not in INDEX_HTML
    assert '<div id="batchPanel"' in INDEX_HTML

    # Workspace tabs do not include tab-scout
    assert 'id="tab-scout"' not in INDEX_HTML, "tab-scout must be removed from workspace tabs"

def test_scout_cards_identify_the_actual_film():
    renderer_start = INDEX_HTML.index('async function loadScoutGems')
    renderer = INDEX_HTML[renderer_start:INDEX_HTML.index('grid.appendChild(card);', renderer_start)]
    assert '${escapeScoutHtml(gem.title)} (${gem.release_year})' not in renderer
    assert '${escapeScoutHtml(displayTitle)}</h3>' in renderer
    assert 'Mô tả gốc:</strong> ${escapeScoutHtml(sourceExcerpt)}' in renderer
    assert '${escapeScoutHtml(gem.summary)}' not in renderer
    assert '${escapeScoutHtml(gem.vietnamese_summary)}' in renderer
    assert 'Điểm kịch tính: ${gem.story_twist_index}' in renderer
    assert 'lượt bình chọn' in renderer


def _run_scout_helpers(expression: str) -> str:
    import json
    import shutil
    if not shutil.which("node"):
        pytest.skip("node is not installed")
    start = INDEX_HTML.index("function scoutIsUnlocalizedLive")
    helpers = INDEX_HTML[start:INDEX_HTML.index("function escapeScoutHtml", start)]
    result = subprocess.run(["node", "-e", helpers + f"\nprocess.stdout.write(JSON.stringify({expression}));"],
                            capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(result.stdout)


def test_scout_display_title_uses_real_upload_title_for_live_finds():
    generic = "Phim xưa độc lạ: Câu Chuyện Bí Ẩn"
    titles = _run_scout_helpers(
        "["
        "scoutDisplayTitle({id: 'yt-live-a', title: 'The House of the Dead (1978) | FULL VINTAGE HORROR MOVIE | John Ericson', vietnamese_title: '" + generic + "'}),"
        "scoutDisplayTitle({id: 'yt-live-b', title: 'Carnival of Souls (1962) Cult Film | Horror, Mystery | Full Length Movie', vietnamese_title: '" + generic + "'}),"
        "scoutDisplayTitle({id: 'yt-live-c', title: 'Night Train - Full Movie HD', vietnamese_title: '" + generic + "'}),"
        "scoutDisplayTitle({id: 'tmdb-1', title: 'The Vanishing (Spoorloos)', vietnamese_title: 'Biến Mất Không Dấu Vết (1988)'}),"
        "scoutSourceExcerpt({id: 'yt-live-d', title: 'X', summary: 'A '.repeat(200)}).length,"
        "scoutSourceExcerpt({id: 'bili-1', title: 'same', summary: 'same'}),"
        "scoutDisplayTitle({id: 'yt-live-e', localized: 'agy', title: 'Carnival of Souls (1962) | Full Movie', vietnamese_title: 'Lễ Hội Của Những Linh Hồn (1962)'}),"
        "scoutSourceExcerpt({id: 'yt-live-e', localized: 'agy', title: 'X', summary: 'Mary Henry survives'})"
        "]"
    )
    assert titles[:4] == [
        "The House of the Dead (1978)",
        "Carnival of Souls (1962) Cult Film",
        "Night Train",
        "Biến Mất Không Dấu Vết (1988)",
    ]
    assert titles[4] <= 218
    assert titles[5] == ""
    assert titles[6] == "Lễ Hội Của Những Linh Hồn (1962)"
    assert titles[7] == ""


def test_workspace_views_are_siblings_not_nested():
    from html.parser import HTMLParser

    void = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
    parents: dict[str, list[str]] = {}

    class Parser(HTMLParser):
        stack: list[tuple[str, str]] = []

        def handle_starttag(self, tag, attrs):
            if tag in void:
                return
            ident = dict(attrs).get("id") or ""
            if ident.startswith("view-"):
                parents[ident] = [i for _, i in self.stack if i.startswith("view-")]
            self.stack.append((tag, ident))

        def handle_endtag(self, tag):
            for k in range(len(self.stack) - 1, -1, -1):
                if self.stack[k][0] == tag:
                    del self.stack[k:]
                    return

    Parser().feed(INDEX_HTML[:INDEX_HTML.rfind("<script")])
    assert {"view-explore", "view-edit", "view-review", "view-review-content", "view-review-form", "view-files"} <= set(parents)
    assert all(not ancestors for ancestors in parents.values()), parents


def test_scout_ui_labels_are_vietnamese():
    scout_markup = INDEX_HTML[INDEX_HTML.index('<dialog id="scoutPanel"'):INDEX_HTML.index('</dialog>', INDEX_HTML.index('<dialog id="scoutConfigDialog"'))]
    assert 'Tự động săn phim độc lạ' in scout_markup
    assert 'tiềm năng lan truyền' in scout_markup
    assert 'Theo màu sắc' in scout_markup
    assert 'Theo thời gian' in scout_markup
    assert 'Nhóm AGY' in scout_markup
    for mixed_label in ('Hidden Gem', 'trending', 'Obscure', 'Twist', 'Rating', 'Viral:', 'AGY Pool'):
        assert mixed_label not in scout_markup


def test_webapp_scout_modal_and_download_flow():
    """Verify scout modal elements, theme-safe styling, and URL download integration."""
    # Scout Modal Dialog
    assert '<dialog id="scoutPanel"' in INDEX_HTML
    assert 'id="closeScoutHub"' in INDEX_HTML
    assert 'id="scoutRefreshBtn"' in INDEX_HTML
    assert 'id="scoutTopicSelect"' in INDEX_HTML
    assert 'id="scoutSourceSelect"' in INDEX_HTML
    assert 'id="scoutGrid"' in INDEX_HTML

    # Source retry card includes URL download and search fallback
    assert 'id="sourceRetryUrl"' in INDEX_HTML
    assert 'id="sourceRetryUrlBtn"' in INDEX_HTML
    assert 'id="sourceRetryFallbackSection"' in INDEX_HTML
    assert 'id="sourceRetrySearchFallbackBtn"' in INDEX_HTML

    # Scout card includes open link button and 1-click fallback button
    assert "Mở link ↗" in INDEX_HTML
    assert "🔍 Tìm YouTube" in INDEX_HTML
    assert "scout-fallback-btn" in INDEX_HTML

    # Scout card includes badges for duration and link health
    assert "phút" in INDEX_HTML
    assert "Sống" in INDEX_HTML
    assert "Cần kiểm tra" in INDEX_HTML

    # No hardcoded dark background #1a202c on scout cards
    assert "#1a202c" not in INDEX_HTML, "Scout cards should not hardcode #1a202c (breaks light theme)"
