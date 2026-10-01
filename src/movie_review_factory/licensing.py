"""Offline, machine-bound license verification (Ed25519).

The shipped app embeds only the PUBLIC key, so it can verify a license but can
never mint one. Licenses are issued offline by ``tools/issue_license.py`` (which
is NOT shipped in the installer) using the matching private key kept by the
vendor. Everything here is pure and injectable - the machine-id reader, the clock
and the signature verifier are all parameters - so the module is unit-testable
offline with a throwaway test keypair and no registry/network access.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

PRODUCT = "movie-review-factory"

# Ed25519 public key (base64url); minted by tools/issue_license.py and pasted in at the vendor build. Empty here means no license verifies, so the app stays hard-blocked. See docs/licensing.md.
LICENSE_PUBLIC_KEY_B64 = ""

# Optional override for headless / automated runs; the dashboard normally writes the license file instead.
LICENSE_ENV = "MRF_LICENSE"

_ISO = "%Y-%m-%dT%H:%M:%SZ"


class LicenseError(RuntimeError):
    """The machine identity could not be read."""


@dataclass(frozen=True)
class LicenseStatus:
    ok: bool
    reason: str = ""
    name: str = ""
    expires: str | None = None
    machine: str = ""

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "name": self.name,
            "expires": self.expires,
            "machine": self.machine,
        }


# -- base64url helpers --------------------------------------------------------

def b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64decode(text: str) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


# -- machine fingerprint ------------------------------------------------------

def machine_fingerprint(*, reader: Callable[[], str] | None = None) -> str:
    """Stable, hashed id for this machine. ``reader`` is injectable for tests."""
    raw = ((reader or _read_machine_id)() or "").strip()
    if not raw:
        raise LicenseError("không đọc được định danh máy")
    return hashlib.sha256(("mrf:" + raw).encode("utf-8")).hexdigest()


def _read_machine_id() -> str:
    guid = _windows_machine_guid()
    if guid:
        return "guid:" + guid
    serial = _windows_volume_serial()
    if serial:
        return "vol:" + serial
    return "host:" + (platform.node() or "unknown")


def _windows_machine_guid() -> str:
    try:
        import winreg
    except ImportError:
        return ""
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
        return str(value).strip()
    except OSError:
        return ""


def _windows_volume_serial() -> str:
    try:
        import ctypes

        serial = ctypes.c_uint(0)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(  # type: ignore[attr-defined]
            ctypes.c_wchar_p("C:\\"), None, 0, ctypes.byref(serial), None, None, None, 0
        )
        return f"{serial.value:08X}" if ok else ""
    except (OSError, AttributeError, ValueError):
        return ""


# -- Ed25519 (cryptography imported lazily so the module loads without it) ----

def _ed25519_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
        return True
    except (InvalidSignature, ValueError):
        return False


def ed25519_sign(private_key: bytes, message: bytes) -> bytes:
    """Sign ``message`` with a raw 32-byte private key (issuer-side helper)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    return Ed25519PrivateKey.from_private_bytes(private_key).sign(message)


def generate_keypair() -> tuple[bytes, bytes]:
    """Return ``(private_raw, public_raw)`` 32-byte Ed25519 keys (issuer-side)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    private = Ed25519PrivateKey.generate()
    private_raw = private.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
    )
    public_raw = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return private_raw, public_raw


# -- license payload + encoding -----------------------------------------------

def build_payload(*, machine: str, name: str = "", expires: str | None = None,
                  issued: str | None = None) -> dict:
    """A license payload. ``expires=None`` means a perpetual license."""
    return {
        "product": PRODUCT,
        "machine": machine,
        "name": name,
        "seats": 1,
        "issued": issued or datetime.now(timezone.utc).strftime(_ISO),
        "expires": expires,
    }


def payload_bytes(payload: dict) -> bytes:
    """Canonical bytes that get signed (sorted keys, compact separators)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encode_license(payload: dict, signature: bytes) -> str:
    """``<payload_b64>.<signature_b64>`` - the string a customer pastes in."""
    return b64encode(payload_bytes(payload)) + "." + b64encode(signature)


# -- verification -------------------------------------------------------------

def verify_license(
    license_text: str,
    *,
    fingerprint: str,
    now: datetime | None = None,
    public_key_b64: str | None = None,
    verify_signature: Callable[[bytes, bytes, bytes], bool] | None = None,
) -> LicenseStatus:
    """Verify signature + machine binding + expiry. Never raises for bad input."""
    key_b64 = LICENSE_PUBLIC_KEY_B64 if public_key_b64 is None else public_key_b64
    verify = verify_signature or _ed25519_verify
    text = (license_text or "").strip()
    if not text:
        return LicenseStatus(False, "chưa kích hoạt: chưa có license", machine=fingerprint)
    if not key_b64:
        return LicenseStatus(False, "bản dựng thiếu public key license", machine=fingerprint)
    try:
        payload_b64, signature_b64 = text.split(".", 1)
        raw_payload = b64decode(payload_b64)
        signature = b64decode(signature_b64)
        public_key = b64decode(key_b64)
    except (ValueError, binascii.Error):
        return LicenseStatus(False, "license sai định dạng", machine=fingerprint)
    if not verify(public_key, raw_payload, signature):
        return LicenseStatus(False, "chữ ký license không hợp lệ", machine=fingerprint)
    try:
        payload = json.loads(raw_payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return LicenseStatus(False, "license sai định dạng (payload)", machine=fingerprint)
    if payload.get("product") != PRODUCT:
        return LicenseStatus(False, "license không dành cho sản phẩm này", machine=fingerprint)
    if payload.get("machine") != fingerprint:
        return LicenseStatus(False, "license không khớp máy này", machine=fingerprint)
    expires = payload.get("expires")
    if expires:
        moment = now or datetime.now(timezone.utc)
        try:
            deadline = datetime.strptime(expires, _ISO).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return LicenseStatus(False, "license có hạn dùng sai định dạng", machine=fingerprint)
        if moment > deadline:
            return LicenseStatus(False, f"license đã hết hạn ({expires})", machine=fingerprint)
    return LicenseStatus(True, "ok", name=str(payload.get("name") or ""),
                         expires=expires, machine=fingerprint)


# -- license file storage -----------------------------------------------------

def license_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "MovieReviewFactory"


def license_path() -> Path:
    return license_dir() / "license.key"


def load_license_text() -> str:
    env = os.environ.get(LICENSE_ENV, "").strip()
    if env:
        return env
    try:
        return license_path().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def save_license_text(text: str) -> Path:
    directory = license_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = license_path()
    temporary = directory / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write((text or "").strip())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


# -- high-level entry points --------------------------------------------------

def current_status(*, license_text: str | None = None, fingerprint: str | None = None,
                   now: datetime | None = None) -> LicenseStatus:
    """Verify the machine's stored (or supplied) license."""
    fp = machine_fingerprint() if fingerprint is None else fingerprint
    text = load_license_text() if license_text is None else license_text
    return verify_license(text, fingerprint=fp, now=now)


def activate(license_text: str, *, fingerprint: str | None = None,
             now: datetime | None = None) -> LicenseStatus:
    """Verify a pasted license for this machine and persist it only if valid."""
    fp = machine_fingerprint() if fingerprint is None else fingerprint
    status = verify_license(license_text, fingerprint=fp, now=now)
    if status.ok:
        save_license_text(license_text)
    return status
