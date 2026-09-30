"""Offline unit tests for the machine-bound license module.

Uses a throwaway Ed25519 test keypair (via the injectable public-key override),
a fake machine fingerprint, and a tmp LOCALAPPDATA - so nothing touches the real
machine, the network, or the vendor's key.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from movie_review_factory import licensing

FAKE_FP = "fingerprint-abc"


@pytest.fixture
def keypair(monkeypatch):
    private, public = licensing.generate_keypair()
    monkeypatch.setattr(licensing, "LICENSE_PUBLIC_KEY_B64", licensing.b64encode(public))
    return private, public


def _issue(private, *, machine=FAKE_FP, name="Test", expires=None) -> str:
    payload = licensing.build_payload(machine=machine, name=name, expires=expires)
    signature = licensing.ed25519_sign(private, licensing.payload_bytes(payload))
    return licensing.encode_license(payload, signature)


def test_machine_fingerprint_is_stable_and_hashed():
    fp1 = licensing.machine_fingerprint(reader=lambda: "guid:ABC")
    fp2 = licensing.machine_fingerprint(reader=lambda: "guid:ABC")
    assert fp1 == fp2
    assert fp1 != "guid:ABC"  # hashed, never the raw id
    assert len(fp1) == 64
    assert licensing.machine_fingerprint(reader=lambda: "guid:OTHER") != fp1


def test_machine_fingerprint_empty_raises():
    with pytest.raises(licensing.LicenseError):
        licensing.machine_fingerprint(reader=lambda: "")


def test_valid_license_verifies_for_its_machine(keypair):
    private, _ = keypair
    status = licensing.verify_license(_issue(private), fingerprint=FAKE_FP)
    assert status.ok is True
    assert status.name == "Test"
    assert status.machine == FAKE_FP


def test_license_rejected_on_other_machine(keypair):
    private, _ = keypair
    status = licensing.verify_license(_issue(private), fingerprint="other-machine")
    assert status.ok is False
    assert "máy" in status.reason


def test_expired_license_is_rejected(keypair):
    private, _ = keypair
    past = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(licensing._ISO)
    status = licensing.verify_license(_issue(private, expires=past), fingerprint=FAKE_FP)
    assert status.ok is False
    assert "hết hạn" in status.reason


def test_future_dated_term_still_valid(keypair):
    private, _ = keypair
    future = (datetime.now(timezone.utc) + timedelta(days=365)).strftime(licensing._ISO)
    status = licensing.verify_license(_issue(private, expires=future), fingerprint=FAKE_FP)
    assert status.ok is True


def test_tampered_payload_fails_signature(keypair):
    private, _ = keypair
    _payload_b64, sig_b64 = _issue(private).split(".", 1)
    forged_payload = licensing.build_payload(machine=FAKE_FP, name="Hacker")
    forged = licensing.b64encode(licensing.payload_bytes(forged_payload)) + "." + sig_b64
    status = licensing.verify_license(forged, fingerprint=FAKE_FP)
    assert status.ok is False
    assert "chữ ký" in status.reason


def test_empty_and_malformed_licenses(keypair):
    assert licensing.verify_license("", fingerprint=FAKE_FP).ok is False
    assert licensing.verify_license("not-a-license", fingerprint=FAKE_FP).ok is False
    assert licensing.verify_license("@@@.@@@", fingerprint=FAKE_FP).ok is False


def test_missing_public_key_blocks(monkeypatch):
    monkeypatch.setattr(licensing, "LICENSE_PUBLIC_KEY_B64", "")
    status = licensing.verify_license("a.b", fingerprint=FAKE_FP)
    assert status.ok is False
    assert "public key" in status.reason


def test_activate_persists_only_when_valid(keypair, monkeypatch, tmp_path):
    private, _ = keypair
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv(licensing.LICENSE_ENV, raising=False)

    bad = licensing.activate("garbage", fingerprint=FAKE_FP)
    assert bad.ok is False
    assert not licensing.license_path().exists()

    good = licensing.activate(_issue(private), fingerprint=FAKE_FP)
    assert good.ok is True
    assert licensing.license_path().exists()
    assert licensing.current_status(fingerprint=FAKE_FP).ok is True


def test_env_license_overrides_file(keypair, monkeypatch, tmp_path):
    private, _ = keypair
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv(licensing.LICENSE_ENV, _issue(private))
    assert licensing.current_status(fingerprint=FAKE_FP).ok is True
