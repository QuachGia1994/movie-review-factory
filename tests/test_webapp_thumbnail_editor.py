import json
import threading
import urllib.request
from pathlib import Path

from PIL import Image

from movie_review_factory.models import JobConfig
from movie_review_factory.pipeline import create_job, load_manifest, save_manifest
from movie_review_factory.webapp import JobsService, create_server


def _ready_job(jobs_root: Path) -> Path:
    root = jobs_root / "review"
    create_job(root, JobConfig(job_id="review"))
    candidates = []
    for index, color in enumerate(((30, 60, 90), (80, 70, 100), (100, 70, 60)), 1):
        name = f"thumbnail-{index}.jpg"
        Image.new("RGB", (1280, 720), color).save(root / name)
        candidates.append({"index": index, "file": name})
    (root / "thumbnail.jpg").write_bytes((root / "thumbnail-1.jpg").read_bytes())
    (root / "thumbnails.json").write_text(json.dumps({
        "candidates": candidates, "primary_candidate": "thumbnail-1.jpg",
        "primary_thumbnail": "thumbnail.jpg",
    }), encoding="utf-8")
    manifest = load_manifest(root)
    manifest.stage("thumbnail").mark("ready")
    save_manifest(root, manifest)
    return root


def test_service_edit_returns_three_variants_and_leaves_selected_thumbnail_alone(tmp_path: Path) -> None:
    root = _ready_job(tmp_path)
    original = (root / "thumbnail.jpg").read_bytes()
    result = JobsService(tmp_path).edit_thumbnail("review", "BEN 10 BÍ ẨN", "Màn Kể")
    assert len(result["variants"]) == 3
    assert (root / "thumbnail.jpg").read_bytes() == original
    assert all((root / v["preview_file"]).is_file() for v in result["variants"])


def test_http_edit_thumbnail_then_select_one_variant(tmp_path: Path) -> None:
    root = _ready_job(tmp_path)
    server = create_server("127.0.0.1", 0, tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        request = urllib.request.Request(base + "/api/jobs/review/thumbnails/edit",
            data=json.dumps({"headline": "BEN 10 BÍ ẨN", "channel_name": "Màn Kể"}).encode("utf-8"),
            method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            variants = json.load(response)["variants"]
        preview = variants[0]["preview_file"]
        with urllib.request.urlopen(base + "/api/jobs/review/artifacts/" + preview, timeout=5) as response:
            assert response.read(2) == b"\xff\xd8"
        choose = urllib.request.Request(base + "/api/jobs/review/thumbnails/select",
            data=json.dumps({"candidate": variants[0]["file"]}).encode("utf-8"),
            method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(choose, timeout=5) as response:
            assert json.load(response)["selected"] == variants[0]["file"]
        assert (root / "thumbnail.jpg").read_bytes() == (root / variants[0]["file"]).read_bytes()
        with urllib.request.urlopen(base, timeout=5) as response:
            html = response.read().decode("utf-8")
        assert 'id="thumbnailHeadline"' in html
        assert 'id="thumbnailVariantGrid"' in html
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
