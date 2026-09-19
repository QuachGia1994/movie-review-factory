# Postman persona verification

Import `persona-system-verification.collection.json` into Postman and set `api_key` locally. The collection compares a baseline request with requests carrying a first-class `system` field.

Expected checks:
- A establishes the gateway's baseline identity behavior.
- B verifies persona injection through `system`.
- C checks repeated identity probing without unsafe behavior.
- D checks that persona configuration does not weaken safety.
- E checks ordinary technical quality.

The collection deliberately does not embed a real API key.
