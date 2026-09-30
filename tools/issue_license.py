#!/usr/bin/env python3
"""Vendor-side license issuer - NOT shipped in the installer.

One-time bootstrap:
  py -3 tools/issue_license.py keygen
      Generate an Ed25519 keypair. Paste the printed PUBLIC key into
      src/movie_review_factory/licensing.py (LICENSE_PUBLIC_KEY_B64) and store the
      PRIVATE key OFFLINE (a password manager / encrypted note) - never commit it.

Per customer:
  py -3 tools/issue_license.py issue --private <PRIV_B64> --machine <FINGERPRINT> [--name NAME] [--days N]
      Sign a license bound to one machine fingerprint. Perpetual unless --days is
      given. Prints the license string to hand to the customer.

Get a customer's machine fingerprint from the activation screen, or on their box:
  py -3 -c "from movie_review_factory import licensing; print(licensing.machine_fingerprint())"

This tool lives under tools/ and is never installed into the shipped app (the
installer only pip-installs the movie_review_factory package), so the signing
capability never leaves the vendor's machine.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_review_factory import licensing


def _keygen(_args) -> None:
    private_raw, public_raw = licensing.generate_keypair()
    print("PUBLIC KEY  -> paste into licensing.LICENSE_PUBLIC_KEY_B64:")
    print("  " + licensing.b64encode(public_raw))
    print()
    print("PRIVATE KEY -> store OFFLINE, never commit:")
    print("  " + licensing.b64encode(private_raw))


def _issue(args) -> None:
    private = licensing.b64decode(args.private)
    expires = None
    if args.days:
        expires = (datetime.now(timezone.utc) + timedelta(days=args.days)).strftime(licensing._ISO)
    payload = licensing.build_payload(machine=args.machine, name=args.name or "", expires=expires)
    signature = licensing.ed25519_sign(private, licensing.payload_bytes(payload))
    print(licensing.encode_license(payload, signature))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Movie Review Factory license issuer (vendor-only).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("keygen", help="generate an Ed25519 keypair")
    issue = sub.add_parser("issue", help="sign a license for one machine")
    issue.add_argument("--private", required=True, help="base64url private key from keygen")
    issue.add_argument("--machine", required=True, help="customer machine fingerprint")
    issue.add_argument("--name", default="", help="customer / license label")
    issue.add_argument("--days", type=int, default=0, help="term in days (0 = perpetual)")
    args = parser.parse_args(argv)
    {"keygen": _keygen, "issue": _issue}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
