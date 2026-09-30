# License activation (Ed25519, offline, machine-bound)

Movie Review Factory's `serve` command is license-gated: on an unlicensed machine the dashboard shows an **activation screen** and every mutating API returns `402` until a valid, machine-bound key is entered. Verification is **fully offline** (no phone-home) using an Ed25519 signature; the shipped app embeds only the **public** key, so it can verify a license but can never mint one.

## Policy (current)
- Perpetual by default (`expires: null`); a term is supported by issuing with `--days N`.
- **1 machine per key** (bound to a hashed machine fingerprint).
- No version lock.
- Pure offline: **no revocation** once issued. To "transfer", issue a new key for the new machine's fingerprint (the old key keeps working on the old machine — it cannot be killed remotely offline). Time-box high-risk sales with `--days`.

## One-time vendor bootstrap
1. `py -3 tools/issue_license.py keygen`
2. Paste the printed **PUBLIC KEY** into `src/movie_review_factory/licensing.py` → `LICENSE_PUBLIC_KEY_B64`.
3. Store the **PRIVATE KEY** offline (password manager / encrypted note). Never commit it. `tools/` is not shipped in the installer, so the signing capability stays on your machine.

Until a real public key is embedded, `serve` stays hard-blocked (this is intended).

## Issue a license to a customer
1. Get the customer's machine fingerprint — from the activation screen, or on their machine:
   `py -3 -c "from movie_review_factory import licensing; print(licensing.machine_fingerprint())"`
2. `py -3 tools/issue_license.py issue --private <PRIV_B64> --machine <FINGERPRINT> --name "Studio X"` (add `--days 365` for a 1-year term), then send them the printed license string.

## Customer activation
- Launch the app (`mrf serve` or the desktop shortcut). The activation screen shows their machine code and a box to paste the key.
- On success the key is saved to `%LOCALAPPDATA%\MovieReviewFactory\license.key` and the dashboard opens. `MRF_LICENSE` (env var) can supply the key for headless runs instead of the file.

## Honest limitation
The app is readable Python inside an embeddable interpreter, so a determined user can still patch out the check locally. Ed25519 (asymmetric) meaningfully raises the bar over a shared secret: keys **cannot be forged** and a single key **cannot be shared across machines**. It is copy-protection, not DRM.
