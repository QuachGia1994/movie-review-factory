from __future__ import annotations

from pathlib import Path

from movie_review_factory import tts_providers, webapp


def test_vieneu_available_falls_back_to_isolated_probe():
    def missing(_name):
        raise ImportError("vieneu")

    assert tts_providers.vieneu_available(
        import_module=missing, isolated_check=lambda: True,
    ) is True
    assert tts_providers.vieneu_available(
        import_module=missing, isolated_check=lambda: False,
    ) is False


def test_vieneu_env_path_is_overrideable(monkeypatch, tmp_path: Path):
    custom = tmp_path / "managed-vieneu"
    monkeypatch.setenv(tts_providers.VIENEU_ENV_ENV, str(custom))
    assert tts_providers.vieneu_env_dir() == custom
    expected = custom / ("Scripts/python.exe" if tts_providers.os.name == "nt" else "bin/python")
    assert tts_providers.vieneu_env_python() == expected


def test_isolated_probe_uses_argument_list_timeout_and_no_shell(tmp_path: Path):
    python = tmp_path / "python.exe"
    python.write_bytes(b"")
    calls = []

    class Result:
        returncode = 0

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return Result()

    assert tts_providers._isolated_vieneu_available(python, run=fake_run)
    args, kwargs = calls[0]
    assert args == [str(python), "-c", "import vieneu"]
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 30


def test_installer_command_is_bounded_and_shell_free(monkeypatch):
    calls = []

    class Result:
        returncode = 0

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return Result()

    monkeypatch.setattr(webapp.pipeline.subprocess, "run", fake_run)
    webapp.JobsService._run_installer_command(["python", "-c", "pass"], timeout=42)
    assert calls[0][0] == ["python", "-c", "pass"]
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["timeout"] == 42


def test_ui_exposes_vieneu_provider_and_install_behavior():
    assert 'option value="vieneu"' in webapp.INDEX_HTML
    assert "/api/system/install-package" in webapp.INDEX_HTML
    assert "package: 'vieneu'" in webapp.INDEX_HTML
