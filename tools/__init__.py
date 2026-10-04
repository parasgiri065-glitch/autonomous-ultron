"""Reference tool implementations for Ultron Phase 1.

Every tool here obeys the same contract:

1. Read a JSON object of inputs from **stdin**.
2. Do the work using only declared permissions (the sandbox enforces the rest).
3. Print a JSON envelope to **stdout** and nothing else:

   ``{"ok": true, "result": {...}, "meta": {...}}``

No secrets are ever available: the sandbox forwards an allowlist of non-secret
env vars only. Tool results must match the manifest's ``outputs`` types or the
verifier rejects them (and the run stops, so bad output is never cached).
"""

from __future__ import annotations

__all__ = ["_io"]
