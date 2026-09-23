"""Tests for the generated one-file JavaScript distribution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build-onefile.mjs"
BUNDLE = REPO_ROOT / "movie-review-factory.js"
NODE = shutil.which("node")
CSCRIPT = shutil.which("cscript.exe")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is required for one-file bundle tests")


def _build() -> dict:
    result = subprocess.run(
        [NODE, str(BUILD_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def test_onefile_build_is_deterministic_and_resolves_placeholders() -> None:
    first = _build()
    first_bytes = BUNDLE.read_bytes()
    first_digest = hashlib.sha256(first_bytes).hexdigest()

    second = _build()
    second_bytes = BUNDLE.read_bytes()
    second_digest = hashlib.sha256(second_bytes).hexdigest()

    assert first["hash"] == second["hash"]
    assert first_digest == second_digest
    assert first_bytes == second_bytes
    assert not first_bytes.startswith(b"\xef\xbb\xbf")
    assert all(byte < 128 for byte in first_bytes)
    assert b"__MRF_VERSION__" not in first_bytes
    assert b"__MRF_HASH__" not in first_bytes
    assert b"__MRF_BUNDLE__" not in first_bytes
    assert set(first["embeddedFiles"]) == {
        "movie_review_factory/__init__.py",
        "movie_review_factory/cli.py",
        "movie_review_factory/content_agent.py",
        "movie_review_factory/localization.py",
        "movie_review_factory/models.py",
        "movie_review_factory/pipeline.py",
        "movie_review_factory/webapp.py",
        "pyproject.toml",
    }


def test_onefile_runs_from_an_isolated_copy_and_preserves_jobs(tmp_path: Path) -> None:
    _build()
    standalone = tmp_path / "MovieReviewFactory.js"
    shutil.copy2(BUNDLE, standalone)

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    marker = jobs / "keep-me.txt"
    marker.write_text("persistent", encoding="utf-8")

    runtime_home = tmp_path / "runtime-home"
    env = os.environ.copy()
    env["MRF_HOME"] = str(runtime_home)
    env["MRF_NO_AUTO_INSTALL"] = "1"

    result = subprocess.run(
        [
            NODE,
            str(standalone),
            "--self-test",
            "--no-browser",
            "--system-runtime",
            "--runtime-profile",
            "core",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert '"ok": true' in result.stdout
    assert '"root_ok": true' in result.stdout
    assert '"api_ok": true' in result.stdout
    assert marker.read_text(encoding="utf-8") == "persistent"

    runtime_dirs = list((runtime_home / "runtime").iterdir())
    assert len(runtime_dirs) == 1
    package_dir = runtime_dirs[0] / "movie_review_factory"
    assert (package_dir / "pipeline.py").is_file()
    assert (package_dir / "webapp.py").is_file()


def test_onefile_contains_zero_install_bootstrap_contract() -> None:
    _build()
    text = BUNDLE.read_text(encoding="utf-8-sig")
    assert "https://nodejs.org/dist/index.json" in text
    assert "https://api.github.com/repos/astral-sh/uv/releases/latest" in text
    assert "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" in text
    assert "faster-whisper>=1.1,<2" in text
    assert "edge-tts>=7,<8" in text
    assert "Node archive checksum mismatch" in text
    assert "uv archive checksum mismatch" in text
    assert "FFmpeg archive checksum mismatch" in text
    assert "MRF_WSH_FORCE_PORTABLE_NODE" in text
    assert "MRF_FORCE_PORTABLE_FFMPEG" in text
    assert "UV_PYTHON_INSTALL_DIR" in text
    assert "--no-network" in text


def test_zero_install_no_network_fails_closed_on_empty_cache(tmp_path: Path) -> None:
    _build()
    standalone = tmp_path / "MovieReviewFactory.js"
    shutil.copy2(BUNDLE, standalone)
    empty_home = tmp_path / "empty-runtime"

    result = subprocess.run(
        [
            NODE,
            str(standalone),
            "--self-test",
            "--no-browser",
            "--runtime-profile",
            "core",
            "--no-network",
            "--home",
            str(empty_home),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "Thi" in result.stderr
    assert "uv portable" in result.stderr
    assert not (empty_home / "toolchain" / "venv-py312").exists()


@pytest.mark.skipif(CSCRIPT is None, reason="Windows Script Host is unavailable")
def test_windows_script_host_relaunch_probe() -> None:
    _build()
    result = subprocess.run(
        [CSCRIPT, "//nologo", str(BUNDLE), "--wsh-self-test"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "MRF_WSH_OK" in result.stdout
    assert "node" in result.stdout.lower()


@pytest.mark.skipif(CSCRIPT is None, reason="Windows Script Host is unavailable")
@pytest.mark.skipif(
    os.environ.get("MRF_RUN_ZERO_INSTALL_NETWORK_TESTS") != "1",
    reason="network bootstrap test is opt-in",
)
def test_wsh_bootstraps_portable_node_when_system_node_is_forced_off(
    tmp_path: Path,
) -> None:
    _build()
    standalone = tmp_path / "MovieReviewFactory.js"
    shutil.copy2(BUNDLE, standalone)

    runtime_home = tmp_path / "runtime-home"
    env = os.environ.copy()
    env["MRF_HOME"] = str(runtime_home)
    env["MRF_WSH_FORCE_PORTABLE_NODE"] = "1"

    result = subprocess.run(
        [CSCRIPT, "//nologo", str(standalone), "--wsh-self-test"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "MRF_WSH_OK" in result.stdout
    node_exe = runtime_home / "bootstrap" / "node" / "node.exe"
    assert node_exe.is_file()

    version = subprocess.run(
        [str(node_exe), "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert version.stdout.strip().startswith("v")


@pytest.mark.skipif(
    os.environ.get("MRF_RUN_ZERO_INSTALL_NETWORK_TESTS") != "1",
    reason="network bootstrap test is opt-in",
)
def test_full_managed_runtime_bootstraps_then_reopens_without_network(
    tmp_path: Path,
) -> None:
    _build()
    standalone = tmp_path / "MovieReviewFactory.js"
    shutil.copy2(BUNDLE, standalone)

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    marker = jobs / "keep-me.txt"
    marker.write_text("persistent", encoding="utf-8")
    runtime_home = tmp_path / "runtime-home"

    first = subprocess.run(
        [
            NODE,
            str(standalone),
            "--self-test",
            "--no-browser",
            "--runtime-profile",
            "full",
            "--portable-ffmpeg",
            "--home",
            str(runtime_home),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert first.returncode == 0, first.stderr or first.stdout
    assert "[managed]" in first.stdout
    assert "Runtime profile: full" in first.stdout
    assert '"ok": true' in first.stdout

    python_exe = (
        runtime_home / "toolchain" / "venv-py312" / "Scripts" / "python.exe"
    )
    ffmpeg_exe = runtime_home / "toolchain" / "ffmpeg" / "bin" / "ffmpeg.exe"
    ffprobe_exe = runtime_home / "toolchain" / "ffmpeg" / "bin" / "ffprobe.exe"
    assert python_exe.is_file()
    assert ffmpeg_exe.is_file()
    assert ffprobe_exe.is_file()

    imports = subprocess.run(
        [
            str(python_exe),
            "-c",
            "import pydantic,typer,faster_whisper,edge_tts; print('ok')",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert imports.stdout.strip() == "ok"

    second = subprocess.run(
        [
            NODE,
            str(standalone),
            "--self-test",
            "--no-browser",
            "--runtime-profile",
            "full",
            "--portable-ffmpeg",
            "--no-network",
            "--home",
            str(runtime_home),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert second.returncode == 0, second.stderr or second.stdout
    assert '"ok": true' in second.stdout
    assert marker.read_text(encoding="utf-8") == "persistent"
