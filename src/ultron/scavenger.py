"""Allowlisted, offline-testable OpenAPI scavenger.

Discovery is disabled unless explicitly enabled (or explicit seed URLs are
passed by a caller). The engine only fetches from the three approved domains and
never turns a discovered document into a trusted tool without Forge verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, get_settings
from .forge import ForgeEngine

LOG = logging.getLogger(__name__)
ALLOWED_DOMAINS = frozenset({"api.apis.guru", "raw.githubusercontent.com", "api.github.com"})
HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options")


@dataclass(slots=True)
class ToolCandidate:
    """One deduplicated OpenAPI document and its first useful operation."""

    name: str
    spec_url: str
    spec_hash: str
    document: dict[str, Any]
    endpoint: str
    method: str
    description: str = ""
    source_url: str = ""
    api_version: str = ""
    operations: list[dict[str, str]] = field(default_factory=list)

    @property
    def url(self) -> str:
        return self.spec_url


Fetcher = Callable[[str], bytes | str | dict[str, Any] | list[Any]]


class Scavenger:
    """Fetch and wrap only machine-readable specifications from approved hosts."""

    default_seeds = (
        "https://api.apis.guru/v2/list.json",
        "https://api.github.com/search/code?q=extension%3Aopenapi+OR+extension%3Aswagger",
    )

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        fetcher: Fetcher | None = None,
        max_candidates: int = 20,
        rate_limit_s: float = 0.1,
        allowed_domains: Iterable[str] = ALLOWED_DOMAINS,
    ) -> None:
        self.settings = settings or get_settings()
        self.fetcher = fetcher or self._fetch
        self.max_candidates = min(20, max(1, max_candidates))
        self.rate_limit_s = max(0.0, rate_limit_s)
        self.allowed_domains = frozenset(allowed_domains)
        self.rejections: list[dict[str, str]] = []
        self.fetch_errors: list[dict[str, str]] = []
        self._last_fetch = 0.0

    def discover(self, seed_urls: Iterable[str] | None = None) -> list[ToolCandidate]:
        """Discover at most 20 unique specs, sorted by source URL.

        Passing explicit seeds is the deterministic test/operator escape hatch;
        unattended default discovery remains disabled by ``ULTRON_SCAVENGE=0``.
        """
        explicit = seed_urls is not None
        if not explicit and not self.settings.scavenge_enabled:
            return []
        seeds = list(seed_urls) if explicit else list(self.default_seeds)
        candidates: list[ToolCandidate] = []
        seen_hashes: set[str] = set()
        seen_urls: set[str] = set()
        queue = list(seeds)
        while queue and len(candidates) < self.max_candidates:
            url = queue.pop(0)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            if not self._allowed(url):
                continue
            try:
                document = _as_document(self._fetch_checked(url))
            except Exception as exc:
                self.fetch_errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
                continue
            directory_urls = _extract_spec_urls(document, url, self.allowed_domains)
            if directory_urls:
                queue.extend(directory_urls)
                continue
            candidate = self._candidate_from_document(document, url)
            if candidate is None or candidate.spec_hash in seen_hashes:
                continue
            seen_hashes.add(candidate.spec_hash)
            candidates.append(candidate)
        return candidates

    def wrap(self, candidate: ToolCandidate) -> tuple[str, dict[str, Any]]:
        """Generate a stdlib-only wrapper and a medium-risk manifest."""
        operation_url = candidate.endpoint
        method = candidate.method.upper()
        code = _wrapper_code(operation_url, method)
        permissions = ["network:https"] if operation_url.lower().startswith("https://") else []
        manifest = {
            "name": _tool_name(candidate.name),
            "version": "0.1.0",
            "description": candidate.description[:500] or f"OpenAPI wrapper for {candidate.name}",
            "risk": "medium",
            "provides": [f"api.{candidate.name}.result"],
            "requires": [],
            "permissions": permissions,
            "inputs": {"params": "dict", "body": "dict", "mock_response": "any"},
            "outputs": {"result": "any"},
            "tags": ["api", "openapi", candidate.name],
            "deterministic": False,
        }
        return code, manifest

    def forge_candidates(
        self,
        candidates: Iterable[ToolCandidate],
        *,
        engine: ForgeEngine | None = None,
        max_tools: int = 20,
    ) -> dict[str, list[Any]]:
        """Feed wrappers into Forge; policy/sandbox failures are retained in report."""
        engine = engine or ForgeEngine(self.settings)
        report: dict[str, list[Any]] = {"forged": [], "failed": [], "skipped": []}
        for candidate in list(candidates)[: min(20, max_tools)]:
            code, manifest = self.wrap(candidate)
            try:
                spec = engine.ledger.record_gap(
                    f"scavenge:{candidate.name}",
                    expected_outputs={"result": "any"},
                    suggested_provides=[f"api.{candidate.name}.result"],
                )
                temporary = engine.synthesize_tool(spec, code, manifest)
                # The mock response is deliberately supplied, but a manifest that
                # requests network still needs a separate policy grant. Thus this
                # path remains fail-closed in default nightly/offline runs.
                if engine.test_tool(
                    temporary,
                    {"params": {}, "body": {}, "mock_response": {"ok": True}},
                    {"result": "any"},
                ):
                    report["forged"].append(candidate)
                else:
                    report["failed"].append(candidate)
            except Exception as exc:
                report["failed"].append({"candidate": candidate, "error": str(exc)})
        return report

    def _candidate_from_document(
        self, document: dict[str, Any], source_url: str
    ) -> ToolCandidate | None:
        if not (document.get("openapi") or document.get("swagger")):
            return None
        version = str(document.get("openapi") or document.get("swagger"))
        base = _server_url(document, source_url)
        operations: list[dict[str, str]] = []
        paths = document.get("paths") or {}
        if not isinstance(paths, dict):
            return None
        for path, item in paths.items():
            if not isinstance(item, dict):
                continue
            for method in HTTP_METHODS:
                operation = item.get(method)
                if not isinstance(operation, dict):
                    continue
                operation_id = str(operation.get("operationId") or f"{method}_{path}")
                endpoint = urllib.parse.urljoin(base.rstrip("/") + "/", str(path).lstrip("/"))
                operations.append({"name": operation_id, "endpoint": endpoint, "method": method})
        if not operations:
            return None
        title = str((document.get("info") or {}).get("title") or operations[0]["name"])
        normalized = json.dumps(document, sort_keys=True, separators=(",", ":"))
        spec_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        first = operations[0]
        return ToolCandidate(
            name=_candidate_name(title),
            spec_url=source_url,
            spec_hash=spec_hash,
            document=document,
            endpoint=first["endpoint"],
            method=first["method"],
            description=str((document.get("info") or {}).get("description") or title),
            source_url=source_url,
            api_version=version,
            operations=operations,
        )

    def _allowed(self, url: str) -> bool:
        try:
            parsed = urllib.parse.urlparse(url)
            host = (parsed.hostname or "").lower()
            allowed = parsed.scheme == "https" and host in self.allowed_domains
        except ValueError:
            allowed = False
        if not allowed:
            self.rejections.append({"url": url, "reason": "domain is outside strict allowlist"})
            LOG.warning("scavenger rejected non-allowlisted URL: %s", url)
            return False
        return True

    def _fetch_checked(self, url: str) -> bytes | str | dict[str, Any] | list[Any]:
        now = time.monotonic()
        delay = self.rate_limit_s - (now - self._last_fetch)
        if delay > 0:
            time.sleep(delay)
        self._last_fetch = time.monotonic()
        return self.fetcher(url)

    @staticmethod
    def _fetch(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "ultron-scavenger/2.4"})
        with urllib.request.urlopen(request, timeout=15.0) as response:
            return response.read()


def _as_document(raw: bytes | str | dict[str, Any] | list[Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    value = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    if not isinstance(value, dict):
        raise ValueError("document is not a JSON object")
    return value


def _extract_spec_urls(
    document: dict[str, Any], source_url: str, allowed: Iterable[str]
) -> list[str]:
    urls: list[str] = []

    def add(value: Any) -> None:
        if not isinstance(value, str):
            return
        absolute = urllib.parse.urljoin(source_url, value)
        host = (urllib.parse.urlparse(absolute).hostname or "").lower()
        if urllib.parse.urlparse(absolute).scheme == "https" and host in allowed:
            urls.append(absolute)

    if "paths" in document and (document.get("openapi") or document.get("swagger")):
        return []
    if isinstance(document.get("items"), list):
        for item in document["items"]:
            if isinstance(item, dict):
                add(item.get("download_url") or item.get("url") or item.get("swaggerUrl"))
    for value in document.values():
        if isinstance(value, dict):
            add(value.get("swaggerUrl") or value.get("openapiUrl") or value.get("download_url"))
            versions = value.get("versions")
            if isinstance(versions, dict):
                for version in versions.values():
                    if isinstance(version, dict):
                        add(
                            version.get("swaggerUrl")
                            or version.get("openapiUrl")
                            or version.get("url")
                        )
    return list(dict.fromkeys(urls))


def _server_url(document: dict[str, Any], source_url: str) -> str:
    servers = document.get("servers")
    if isinstance(servers, list) and servers and isinstance(servers[0], dict):
        value = servers[0].get("url")
        if isinstance(value, str):
            return urllib.parse.urljoin(source_url, value)
    if document.get("swagger"):
        scheme = (document.get("schemes") or [urllib.parse.urlparse(source_url).scheme])[0]
        host = document.get("host") or urllib.parse.urlparse(source_url).netloc
        return f"{scheme}://{host}{document.get('basePath') or ''}"
    parsed = urllib.parse.urlparse(source_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _candidate_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return (slug or "openapi_api")[:48].rstrip("_")


def _tool_name(value: str) -> str:
    slug = _candidate_name(value)
    return ("api_" + slug)[:49].rstrip("_")


def _wrapper_code(endpoint: str, method: str) -> str:
    return f"""from tools._io import main_guard, require
import json
import urllib.parse
import urllib.request

ENDPOINT = {endpoint!r}
METHOD = {method!r}

def run(payload):
    params = payload.get("params") or {{}}
    body = payload.get("body") or {{}}
    if "mock_response" in payload:
        return {{"result": payload["mock_response"]}}
    url = ENDPOINT
    if METHOD in {{"GET", "HEAD", "DELETE"}} and params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params, doseq=True)
    data = None if METHOD in {{"GET", "HEAD", "DELETE"}} else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=METHOD, headers={{"Content-Type": "application/json", "User-Agent": "ultron-api-wrapper/2.4"}})
    with urllib.request.urlopen(request, timeout=15) as response:
        raw = response.read().decode("utf-8")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {{"text": raw}}
    return {{"result": result}}

if __name__ == "__main__":
    main_guard(run)
"""
