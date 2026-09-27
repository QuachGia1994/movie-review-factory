import json

import pytest

from movie_review_factory.creator_library import CreatorLibrary
from movie_review_factory.models import JobConfig
from movie_review_factory.pipeline import create_job


def test_series_plan_links_existing_and_planned_jobs_without_changing_manifests(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    create_job(jobs / "ben", JobConfig(job_id="ben", movie_title="Ben 10"))
    before = (jobs / "ben" / "manifest.json").read_bytes()
    library = CreatorLibrary(jobs)
    series = library.save_series("alien-reviews", "Vũ trụ người ngoài hành tinh", [
        {"movie_title": "Ben 10", "job_id": "ben"},
        {"movie_title": "Phần tiếp theo", "job_id": None},
    ])
    assert series["entries"][0]["job_id"] == "ben"
    assert series["entries"][1]["job_id"] is None
    assert CreatorLibrary(jobs).list_series() == [series]
    assert (jobs / "ben" / "manifest.json").read_bytes() == before


def test_reusable_brief_keeps_job_config_independent(tmp_path):
    library = CreatorLibrary(tmp_path)
    saved = library.save_brief("Phân tích hài", {
        "review_thesis": "Tình bạn",
        "tone": "Hài hước",
        "spoiler_policy": "limited",
        "forbidden_claims": ["Không đoán kết thúc"],
    })
    assert saved["name"] == "Phân tích hài"
    copied = library.get_brief("Phân tích hài")
    copied["tone"] = "Bi quan"
    assert library.get_brief("Phân tích hài")["tone"] == "Hài hước"
    config = JobConfig(job_id="future", creative_brief=library.get_brief("Phân tích hài"))
    assert config.creative_brief.tone == "Hài hước"


def test_rights_records_are_manual_provenance_not_automatic_clearance(tmp_path):
    library = CreatorLibrary(tmp_path)
    record = library.record_asset_rights(
        "ben",
        r"D:\Video\Ben.mp4",
        source="Creator supplied source",
        usage="review clip",
        permission_status="unreviewed",
        evidence_note="Needs creator confirmation",
    )
    assert record["permission_status"] == "unreviewed"
    assert record["source"] == "Creator supplied source"
    assert CreatorLibrary(tmp_path).list_asset_rights("ben") == [record]
    with pytest.raises(ValueError, match="evidence"):
        library.record_asset_rights(
            "ben", "music.mp3", source="Stock library", usage="background music",
            permission_status="permitted", evidence_note="",
        )
    assert len(library.list_asset_rights("ben")) == 1


def test_search_projects_matches_metadata_brief_and_series_with_casefold(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    create_job(jobs / "ben", JobConfig(
        job_id="ben", movie_title="Ben 10",
        creative_brief={"review_thesis": "Tình bạn"},
    ))
    create_job(jobs / "other", JobConfig(job_id="other", movie_title="Khác"))
    (jobs / "ben" / "youtube_metadata.json").write_text(
        json.dumps({"title": "SWARM review", "description": "Màn Kể", "tags": ["alien"]}),
        encoding="utf-8",
    )
    library = CreatorLibrary(jobs)
    library.save_series("aliens", "Alien series", [{"movie_title": "Ben 10", "job_id": "ben"}])
    assert [hit["job_id"] for hit in library.search_projects("SWARM")] == ["ben"]
    assert [hit["job_id"] for hit in library.search_projects("Màn Kể")] == ["ben"]
    assert [hit["job_id"] for hit in library.search_projects("tình BẠN")] == ["ben"]
    assert [hit["job_id"] for hit in library.search_projects("alien series")] == ["ben"]
    assert [hit["job_id"] for hit in library.search_projects("alien")] == ["ben"]
    assert {hit["job_id"] for hit in library.search_projects("")} == {"ben", "other"}


def test_series_rejects_unknown_or_unsafe_job_id(tmp_path):
    library = CreatorLibrary(tmp_path)
    with pytest.raises(ValueError, match="unknown job"):
        library.save_series("series", "Series", [{"movie_title": "Missing", "job_id": "missing"}])
    with pytest.raises(ValueError, match="job_id"):
        library.record_asset_rights("../other", "file", source="x", usage="x")


def test_creator_library_http_flow_and_dashboard_controls(tmp_path):
    import threading
    import urllib.request
    from movie_review_factory.webapp import create_server

    server = create_server("127.0.0.1", 0, tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def post(path, body):
        request = urllib.request.Request(
            base + path, method="POST",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            html = response.read().decode("utf-8")
        for control in (
            'id="creatorProjectSearch"', 'id="seriesEntries"',
            'id="briefTemplateSelect"', 'id="rightsAssetPath"',
        ):
            assert control in html

        post("/api/jobs", {"job_id": "ben", "movie_title": "Ben 10"})
        post("/api/creator-library/series", {
            "series_id": "sci-fi", "title": "Review phim khoa học viễn tưởng",
            "entries": [{"movie_title": "Ben 10", "job_id": "ben"}],
        })
        assert get("/api/creator-library/series")["series"][0]["entries"][0]["job_id"] == "ben"

        post("/api/creator-library/briefs", {
            "name": "Kể chuyện hài", "brief": {"tone": "Hài hước", "spoiler_policy": "limited"},
        })
        assert get("/api/creator-library/briefs")["briefs"][0]["tone"] == "Hài hước"

        rights = post("/api/jobs/ben/rights", {
            "path": r"D:\Video\Ben.mp4", "source": "Creator supplied",
            "usage": "Review clip", "permission_status": "unreviewed",
            "evidence_note": "Creator to verify",
        })
        assert rights["permission_status"] == "unreviewed"
        assert get("/api/jobs/ben/rights")["rights"] == [rights]

        assert get("/api/creator-library/search?q=ben")["projects"][0]["job_id"] == "ben"
        assert get("/api/library-search?q=missing")["results"] == []
    finally:
        server.shutdown()
        server.server_close()
