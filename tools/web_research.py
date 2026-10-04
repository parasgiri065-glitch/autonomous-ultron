"""web_research: gather sources for a query and summarize them *extractively*.

Reference implementation for ``tools/examples/web_research.json``.

Cost story: the summary is built by deterministic sentence selection, not by an
LLM. The planner maps "research X" to this tool, the sandbox gets
``network:http`` only if the manifest asked for it and the policy gate agreed,
and the whole envelope is cached, so the second identical query costs $0 and
makes zero HTTP calls.

Modes
-----
* ``ULTRON_WEB_MOCK=/path/mock.json`` -- offline fixture mode (used by tests and
  by CI, where the sandbox has no network at all). Wins over everything else.
* ``ULTRON_EVAL_LIVE=1`` -- live HTTP via httpx *with* a URL-keyed SQLite page
  cache (``ULTRON_WEB_CACHE``) so a page already fetched for another query is
  never paid for twice. Opt-in, local/manual only; never set by CI.
* otherwise -- live HTTP via httpx, no page cache (the Phase 1 behaviour).
"""

from __future__ import annotations

import json
import os
import re
from html.parser import HTMLParser
from typing import Any, ClassVar
from urllib.parse import quote_plus, urlparse

from tools import _webcache
from tools._io import main_guard, optional, require, text_sentences

SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/?q={query}"
DEFAULT_MAX_SOURCES = 3
MAX_MAX_SOURCES = 8
MAX_BYTES = 200_000
USER_AGENT = "ultron-phase1-research/0.1 (+https://github.com/autonomous-ultron)"


class _TextExtractor(HTMLParser):
    """Minimal HTML -> text. No bs4 needed inside the sandbox."""

    SKIP: ClassVar[set[str]] = {"script", "style", "noscript", "head", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._depth = 0
        self._chunks: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._depth += 1
        if tag == "a":
            href = dict(attrs).get("href") or ""
            if href.startswith("http"):
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._depth:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._depth:
            text = data.strip()
            if text:
                self._chunks.append(text)

    @property
    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def html_to_text(html: str) -> tuple[str, list[str]]:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text, parser.links


# --------------------------------------------------------------------- sources
def _mock_documents(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    docs = data.get("documents", data) if isinstance(data, dict) else data
    if not isinstance(docs, list):
        raise ValueError("mock file must contain a list of documents")
    return [dict(d) for d in docs]


def _score(doc: dict[str, str], terms: set[str]) -> float:
    blob = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
    hits = sum(1 for t in terms if t in blob)
    return hits / max(len(terms), 1)


def _select(docs: list[dict[str, str]], query: str, max_sources: int) -> list[dict[str, str]]:
    terms = {t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) > 2}
    ranked = sorted(docs, key=lambda d: (-_score(d, terms), d.get("url", "")))
    relevant = [d for d in ranked if _score(d, terms) > 0] or ranked
    return relevant[:max_sources]


def _fetch_raw(client: Any, url: str, cache: _webcache.WebCache | None) -> str:
    """GET ``url`` as text, through the URL-keyed cache when one is active."""
    if cache is not None:
        hit = cache.get(url)
        if hit is not None:
            return hit
    resp = client.get(url)
    resp.raise_for_status()
    text = resp.text[:MAX_BYTES]
    if cache is not None:
        cache.set(url, text)
    return text


def _live_search(
    query: str,
    max_sources: int,
    cache: _webcache.WebCache | None = None,
) -> list[dict[str, str]]:
    import httpx  # imported lazily so offline mode never touches the network

    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(timeout=15.0, follow_redirects=True, headers=headers) as client:
        _, links = html_to_text(
            _fetch_raw(client, SEARCH_ENDPOINT.format(query=quote_plus(query)), cache)
        )
        candidates: list[str] = []
        for link in links:
            if "duckduckgo.com" in urlparse(link).netloc:
                continue
            if link not in candidates:
                candidates.append(link)
            if len(candidates) >= max_sources:
                break
        docs: list[dict[str, str]] = []
        for url in candidates:
            try:
                text, _ = html_to_text(_fetch_raw(client, url, cache))
            except Exception as exc:
                docs.append({"url": url, "title": url, "text": f"(unavailable: {exc})"})
                continue
            docs.append({"url": url, "title": url, "text": text[:20_000]})
        return docs


# -------------------------------------------------------------------- summary
def summarize(query: str, docs: list[dict[str, str]], sentences_per_source: int = 2) -> str:
    """Extractive, deterministic summary: top matching sentences per source."""
    terms = {t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) > 2}
    parts: list[str] = []
    for doc in docs:
        sentences = text_sentences(doc.get("text", ""))
        scored = sorted(
            sentences,
            key=lambda s: (-sum(1 for t in terms if t in s.lower()), len(s)),
        )
        picked = [s for s in scored if sum(1 for t in terms if t in s.lower()) > 0][
            :sentences_per_source
        ]
        if not picked:
            picked = sentences[:1]
        if picked:
            parts.append(f"[{doc.get('url', 'source')}] " + " ".join(picked))
    return " ".join(parts).strip()


def confidence_for(docs: list[dict[str, str]], query: str, summary: str) -> float:
    """Heuristic 0..1 confidence. Deliberately conservative in Phase 1.

    No LLM, no self-reported certainty: if we only found one source, or no source
    matched the query terms, confidence stays low and the agent should say so.
    """
    if not docs or not summary:
        return 0.0
    terms = {t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) > 2}
    coverage = sum(1 for t in terms if t in summary.lower()) / max(len(terms), 1)
    breadth = min(len(docs), 3) / 3
    return round(min(0.9, 0.4 * coverage + 0.5 * breadth), 3)


def run(payload: dict[str, Any]) -> dict[str, Any]:
    query = require(payload, "query", str).strip()
    if not query:
        raise ValueError("query must not be empty")
    max_sources = int(optional(payload, "max_sources", int, DEFAULT_MAX_SOURCES))
    max_sources = max(1, min(max_sources, MAX_MAX_SOURCES))

    mock_path = os.environ.get("ULTRON_WEB_MOCK", "").strip()
    cache = None
    if mock_path:
        docs = _select(_mock_documents(mock_path), query, max_sources)
        mode = "mock"
    else:
        # Fixtures win over live mode: an eval run stays hermetic even if the
        # operator has ULTRON_EVAL_LIVE=1 exported in their shell.
        cache = _webcache.maybe_cache()
        docs = _live_search(query, max_sources, cache)
        mode = "live"

    summary = summarize(query, docs)
    meta: dict[str, Any] = {"mode": mode, "documents": len(docs)}
    if cache is not None:  # live runs only: mock mode adds nothing to report
        meta["cache"] = cache.stats()
    return {
        "summary": summary,
        "sources": [d.get("url", "") for d in docs],
        "confidence": confidence_for(docs, query, summary),
        "_meta": meta,
    }


if __name__ == "__main__":
    main_guard(run)
