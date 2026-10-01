"""Offline tests for the Windows installer packager's pure helpers and the .iss.

The packager's network / ISCC steps need Windows + Inno Setup + internet and are
out of scope here; these cover the deterministic pieces (download plan, launcher
script, ._pth patch) and the installer script's key invariants.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "package_windows.py"
ISS = REPO / "installer" / "movie-review-factory.iss"


def _load():
    spec = importlib.util.spec_from_file_location("package_windows", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


pkg = _load()


def test_download_plan_covers_runtime_tools(tmp_path):
    plan = pkg.download_plan(tmp_path, python_version="3.12.7")
    by_name = {item["name"]: item for item in plan}
    assert set(by_name) == {"python-embed", "get-pip", "ffmpeg", "yt-dlp"}
    assert "3.12.7" in by_name["python-embed"]["url"]
    assert by_name["python-embed"]["dest"].parent == tmp_path
    assert by_name["yt-dlp"]["url"].endswith("yt-dlp.exe")


def test_launcher_starts_server_and_opens_dashboard():
    vbs = pkg.render_launcher_vbs(port=8765)
    assert "-m movie_review_factory serve" in vbs
    assert "http://localhost:" in vbs
    assert "python\\python.exe" in vbs
    assert ", 0, False" in vbs           # hidden window, non-blocking
    assert "%LOCALAPPDATA%" in vbs        # jobs stay in the user profile


def test_launcher_respects_custom_port():
    assert "thePort = 9090" in pkg.render_launcher_vbs(port=9090)


def test_patch_embed_pth_enables_site_and_is_idempotent():
    patched = pkg.patch_embed_pth("python312.zip\n#import site\n")
    assert "import site" in patched
    assert "#import site" not in patched
    assert "Lib\\site-packages" in patched
    assert pkg.patch_embed_pth(patched).count("import site") == 1


def test_download_rejects_corrupt_cache_and_checksum_mismatch(tmp_path, monkeypatch):
    dest = tmp_path / "artifact.bin"
    dest.write_bytes(b"corrupt-cache")
    payload = b"partial-or-wrong"

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(pkg.urllib.request, "urlopen", lambda *args, **kwargs: Response(payload))
    expected = hashlib.sha256(b"expected").hexdigest()
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        pkg.download("https://example.invalid/artifact", dest, sha256=expected)
    assert dest.read_bytes() == b"corrupt-cache"
    assert not (tmp_path / "artifact.bin.part").exists()


@pytest.mark.parametrize("member", ["../escape", "/absolute", "C:/drive", "//server/share"])
def test_safe_extract_rejects_zip_traversal(tmp_path, member):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, b"bad")
    with zipfile.ZipFile(archive_path) as archive, pytest.raises(RuntimeError, match="unsafe ZIP"):
        pkg.safe_extract_zip(archive, tmp_path / "out")


def test_iss_is_per_user_and_wires_the_launcher():
    text = ISS.read_text(encoding="utf-8")
    assert "PrivilegesRequired=lowest" in text
    assert "OutputBaseFilename=movie-review-factory-setup" in text
    assert "mrf-launch.vbs" in text
    assert "{autodesktop}" in text
