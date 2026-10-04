"""http_fetch: single-URL fetch. Reference implementation of a MEDIUM-risk tool.

Because it declares ``network:http`` **and** ``risk: medium``, the policy gate
will not run it unattended: it produces an approval request instead. Use it to
exercise the human-in-the-loop path (see tests/test_smoke.py).
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from tools._io import main_guard, optional, require

ALLOWED_SCHEMES = {"http", "https"}
DEFAULT_MAX_BYTES = 50_000


def _mock(path: str, url: str) -> dict[str, Any]:
    import json

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    entry = data.get(url) or next(iter(data.values()), {})
    return dict(entry)


def run(payload: dict[str, Any]) -> dict[str, Any]:
    url = require(payload, "url", str).strip()
    max_bytes = int(optional(payload, "max_bytes", int, DEFAULT_MAX_BYTES))
    max_bytes = max(256, min(max_bytes, 500_000))

    scheme = urlparse(url).scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"scheme {scheme!r} not allowed")

    mock_path = os.environ.get("ULTRON_WEB_MOCK", "").strip()
    if mock_path:
        entry = _mock(mock_path, url)
        text = str(entry.get("text", ""))
        return {
            "status": int(entry.get("status", 200)),
            "url": url,
            "text": text[:max_bytes],
            "truncated": len(text) > max_bytes,
        }

    import httpx

    with httpx.Client(timeout=15.0, follow_redirects=True) as client:
        resp = client.get(url)
    text = resp.text[:max_bytes]
    return {
        "status": resp.status_code,
        "url": str(resp.url),
        "text": text,
        "truncated": len(resp.text) > max_bytes,
    }


if __name__ == "__main__":
    main_guard(run)
